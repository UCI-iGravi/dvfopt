"""GPU-accelerated penalty → log-barrier 3D solver for the 6-tetrahedron
constraint via PyTorch autograd.

Same two-phase homotopy as the numpy/scipy 3D tet path (which goes
through :class:`dvfopt.constraints.Tet6Constraint3D` +
:class:`dvfopt.strategies.BarrierStrategy`), but every forward + gradient
runs on a single ``(3, D, H, W)`` torch tensor:

* **Phase 1 — exterior quadratic penalty.** ``F = anchor(phi - phi_init)
  + lam * sum_k max(0, target - V_k)^2`` over a ``lam_schedule``.
* **Phase 2 — log-barrier interior point.** Once every tet is feasible,
  switch to ``F = anchor + (-mu * sum_k log(V_k - threshold))`` over a
  decreasing ``mu_schedule``.

Both phases use :class:`torch.optim.LBFGS`, so iterates stay on-device.
This is a focused first cut — full-grid only (no windowed / active-set
machinery; see :mod:`dvfopt.core.iterative3d_barrier_torch` for that
pattern, which can be ported here in a follow-up).

Public entry: :func:`iterative_3d_tet_barrier_torch`.
"""

from __future__ import annotations

import time

import numpy as np

from dvfopt._defaults import _resolve_params

# Torch is in the [benchmarks] extra. Defer the import so this module
# can be imported on a torch-less install (matches the pattern used by
# iterative2d_barrier.py after PR #11).
try:
    import torch
except ImportError:
    torch = None


_DEFAULT_LAM_SCHEDULE = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
_DEFAULT_MU_SCHEDULE = (1e-1, 1e-2, 1e-3, 1e-4)


def _anchor_term_torch(diff, kind, eps_l1):
    """Smoothed anchor objective on a torch tensor.

    Mirrors :func:`dvfopt.core._barrier_core.anchor_term` but returns
    just the scalar value (autograd handles the gradient).
    """
    if kind == 'l2':
        return 0.5 * (diff * diff).sum()
    if kind == 'l1':
        s = torch.sqrt(diff * diff + eps_l1 * eps_l1)
        return (s - eps_l1).sum()
    if kind == 'none':
        return torch.zeros((), dtype=diff.dtype, device=diff.device)
    raise ValueError(f'unknown anchor kind: {kind!r}')


def iterative_3d_tet_barrier_torch(
    deformation,
    *,
    threshold=None,
    margin=1e-3,
    lam_schedule=_DEFAULT_LAM_SCHEDULE,
    mu_schedule=_DEFAULT_MU_SCHEDULE,
    max_iter=200,
    anchor='l2',
    eps_l1=1e-4,
    device=None,
    dtype=None,
    verbose=1,
    record_history=False,
):
    """GPU/CPU penalty → log-barrier solver for ``V_k(phi) >= threshold``
    on every tetrahedron of every voxel cell.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
    threshold : float, optional
        Lower bound on per-tet volume. Defaults to
        ``DEFAULT_PARAMS['threshold']`` (0.01).
    margin : float
        Safety margin above ``threshold`` used by the penalty target.
    lam_schedule, mu_schedule : sequence of float
        Continuation schedules. Penalty step ``k`` minimises with
        ``lam = lam_schedule[k]``; the run advances when a step
        decreases the violation. Phase 2 starts once every tet
        clears ``threshold`` and runs through ``mu_schedule``.
    max_iter : int
        Per-phase L-BFGS iteration cap.
    anchor : {'l1', 'l2', 'none'}
        Anchor objective against ``deformation``.
    eps_l1 : float
        Smoothing constant for the L1 anchor.
    device : str | torch.device | None
        Default: ``'cuda'`` if available, else ``'cpu'``.
    dtype : torch.dtype | None
        Default: ``torch.float32`` (resolved inside; torch may be None
        at import time).
    verbose : int
        ``0`` silent, ``1`` per-step log, ``2`` adds inner-LBFGS log.
    record_history : bool
        If True, returns ``(phi, history)`` with per-step stats.

    Returns
    -------
    phi_corrected : ndarray, shape ``(3, D, H, W)`` — channels ``[dz, dy, dx]``.
    history : list of dict, only if ``record_history=True``.
    """
    if torch is None:
        raise ImportError(
            'iterative_3d_tet_barrier_torch requires torch (optional dependency). '
            "Install with: pip install -e '.[benchmarks]'"
        )
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    p = _resolve_params(threshold=threshold)
    threshold_f = float(p['threshold'])
    target_f = threshold_f + margin

    deformation = np.asarray(deformation, dtype=np.float64)
    if deformation.ndim != 4 or deformation.shape[0] != 3:
        raise ValueError(f'expected (3, D, H, W) input; got shape {deformation.shape}')

    phi_init = torch.tensor(deformation, dtype=dtype, device=device)
    phi_var = phi_init.clone().requires_grad_(True)

    # Per-tet volumes via the on-device forward we already built and
    # tested against the numpy version + analytical adjoint.
    from dvfopt.jacobian.tetrahedron_sign_torch import six_tet_volumes_3d_torch

    def volumes(phi):
        return six_tet_volumes_3d_torch(phi)

    history: list[dict] = []
    t0 = time.time()

    # Initial diagnostics.
    with torch.no_grad():
        V0 = volumes(phi_init)
        init_neg = int((V0 <= 0).sum().item())
        init_min = float(V0.min().item())
    if verbose >= 1:
        print(
            f'[3d-tet-barrier-torch init] grid {tuple(deformation.shape[1:])}  '
            f'threshold={threshold_f}  margin={margin}  anchor={anchor}  '
            f'device={device} dtype={dtype}'
        )
        print(f'[init] tet neg={init_neg}  min={init_min:+.5f}')

    # -----------------------------------------------------------------
    # Phase 1: exterior quadratic penalty.
    # -----------------------------------------------------------------
    feasible = init_neg == 0
    for step, lam in enumerate(lam_schedule):
        if feasible:
            break

        def closure():
            opt.zero_grad()
            V = volumes(phi_var)
            viol = torch.clamp(target_f - V, min=0.0)
            penalty = lam * (viol * viol).sum()
            anchor_v = _anchor_term_torch(phi_var - phi_init, anchor, eps_l1)
            loss = anchor_v + penalty
            loss.backward()
            return loss

        opt = torch.optim.LBFGS(
            [phi_var],
            max_iter=max_iter,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-10,
            tolerance_change=1e-12 if dtype == torch.float64 else 1e-9,
        )
        opt.step(closure)

        with torch.no_grad():
            V = volumes(phi_var)
            n_neg = int((V <= 0).sum().item())
            cur_min = float(V.min().item())
            l2 = float(torch.linalg.norm(phi_var - phi_init).item())
        if verbose >= 1:
            print(
                f'[penalty step {step}] lam={lam:.0e}  '
                f'n_neg={n_neg}  min={cur_min:+.6f}  L2(phi-init)={l2:.5f}'
            )
        if record_history:
            # ``min_T`` is the canonical schema key for "minimum constraint
            # value at this phase" across the package; emitting ``min_V``
            # silently dropped this through SolveInfo.from_legacy_history.
            history.append(
                dict(phase='penalty', step=step, lam=lam, n_neg=n_neg, min_T=cur_min, l2=l2)
            )
        if cur_min >= target_f:
            feasible = True

    # -----------------------------------------------------------------
    # Phase 2: log-barrier interior point.
    # -----------------------------------------------------------------
    if feasible:
        for step, mu in enumerate(mu_schedule):

            def closure():
                opt.zero_grad()
                V = volumes(phi_var)
                slack = V - threshold_f
                # Guard against any iterate that briefly crosses
                # threshold during the line search.
                active = slack > 0
                if not bool(active.all().item()):
                    safe = torch.where(active, slack, torch.ones_like(slack))
                    viol = torch.clamp(-slack + 1e-12, min=0.0) * (~active).to(slack.dtype)
                    bar = -mu * (torch.log(safe) * active.to(slack.dtype)).sum()
                    penalty = 1e8 * (viol * viol).sum()
                    anchor_v = _anchor_term_torch(phi_var - phi_init, anchor, eps_l1)
                    loss = anchor_v + bar + penalty
                else:
                    bar = -mu * torch.log(slack).sum()
                    anchor_v = _anchor_term_torch(phi_var - phi_init, anchor, eps_l1)
                    loss = anchor_v + bar
                loss.backward()
                return loss

            opt = torch.optim.LBFGS(
                [phi_var],
                max_iter=max_iter,
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-10,
                tolerance_change=1e-12 if dtype == torch.float64 else 1e-9,
            )
            opt.step(closure)

            with torch.no_grad():
                V = volumes(phi_var)
                n_neg = int((V <= 0).sum().item())
                cur_min = float(V.min().item())
                l2 = float(torch.linalg.norm(phi_var - phi_init).item())
            if verbose >= 1:
                print(
                    f'[barrier  step {step}] mu={mu:.0e}  '
                    f'n_neg={n_neg}  min={cur_min:+.6f}  L2(phi-init)={l2:.5f}'
                )
            if record_history:
                history.append(
                    dict(phase='barrier', step=step, mu=mu, n_neg=n_neg, min_T=cur_min, l2=l2)
                )

    wall = time.time() - t0
    with torch.no_grad():
        V_final = volumes(phi_var)
        n_neg = int((V_final <= 0).sum().item())
        min_V = float(V_final.min().item())
    if verbose >= 1:
        ok = n_neg == 0 and min_V >= threshold_f - 1e-5
        print(
            f'[3d-tet-barrier-torch done] feasible={ok}  '
            f'n_neg={n_neg}  min={min_V:+.6f}  ({wall:.2f}s)'
        )

    phi_corr = phi_var.detach().cpu().numpy().astype(np.float64)
    if record_history:
        return phi_corr, history
    return phi_corr


__all__ = ['iterative_3d_tet_barrier_torch']
