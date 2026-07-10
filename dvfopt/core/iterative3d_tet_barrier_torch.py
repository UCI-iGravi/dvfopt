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

Two modes:

* **Full-grid** (default, ``windowed=False``). One pass over the whole
  volume — simplest and the right choice when the fold density is
  uniform.
* **Windowed** (``windowed=True``). Detect connected components of
  folded cells via scipy CCL, build a padded corner-bbox per
  component, run penalty→barrier on each patch with a frozen
  Dirichlet ring, repeat until all folds are cleared. Cuts memory and
  compute drastically when folds are spatially clustered (the common
  case). The pattern is the same one used by
  :mod:`dvfopt.core.iterative3d_barrier_torch` for the 3D Jdet path.

The batched-non-overlapping-patches optimization from the Jdet path is
not yet ported — each patch is optimized independently here.

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


# 26-connectivity CCL structuring element (3D). Cached at module load.
_CCL_26 = None


def _ccl_structure():
    """3D 26-connectivity structuring element for ``scipy.ndimage.label``."""
    global _CCL_26
    if _CCL_26 is None:
        _CCL_26 = np.ones((3, 3, 3), dtype=np.uint8)
    return _CCL_26


def _patch_corner_bbox(cz, cy, cx, pad, D, H, W):
    """Convert fold-cell coords to a corner-space bbox with padding.

    Cell ``(cz, cy, cx)`` uses corners ``(cz..cz+1, cy..cy+1, cx..cx+1)``,
    so the corner-space range for a set of cells is
    ``cells.min()..cells.max()+1`` per axis. Padding adds ``pad`` voxels
    of frozen-boundary expansion, clipped to the global grid.

    Returns
    -------
    (z0, z1, y0, y1, x0, x1) inclusive corner-space indices.
    """
    z0 = max(0, int(cz.min()) - pad)
    z1 = min(D - 1, int(cz.max()) + 1 + pad)
    y0 = max(0, int(cy.min()) - pad)
    y1 = min(H - 1, int(cy.max()) + 1 + pad)
    x0 = max(0, int(cx.min()) - pad)
    x1 = min(W - 1, int(cx.max()) + 1 + pad)
    return z0, z1, y0, y1, x0, x1


def _optimize_patch_3d_tet_torch(
    phi_full,
    z0,
    z1,
    y0,
    y1,
    x0,
    x1,
    threshold_f,
    target_f,
    lam_schedule,
    mu_schedule,
    max_iter,
    anchor,
    eps_l1,
    dtype,
    device,
):
    """Run penalty → barrier on a single patch, splicing result into ``phi_full``.

    The patch spans corners ``(z0..z1, y0..y1, x0..x1)`` inclusive. The
    outermost layer of corners within the patch is FROZEN to the
    *current* ``phi_full`` state (matching the existing 3D Jdet windowed
    solver: subsequent outer rounds don't have to fight each other's
    boundary choices), except for faces coinciding with the volume
    boundary, which stay free; interior corners are free.

    Mutates ``phi_full`` in place (overwrites the patch range with the
    optimized result). Other regions of ``phi_full`` are untouched.

    Returns ``(lam_steps, mu_steps)`` for verbose reporting.
    """
    from dvfopt.jacobian.tetrahedron_sign_torch import six_tet_volumes_3d_torch

    # Patch corner shape (inclusive).
    Dp = z1 - z0 + 1
    Hp = y1 - y0 + 1
    Wp = x1 - x0 + 1
    if Dp < 2 or Hp < 2 or Wp < 2:
        # Degenerate patch — no cells to optimize. Skip.
        return 0, 0

    # Start the variable at the CURRENT full-grid state — any prior outer
    # iter may have moved the boundary corners; we want those to be the
    # locked values for this run. The boundary-Dirichlet reference is
    # taken from the same snapshot so the frozen ring stays put.
    phi_patch_var = (
        phi_full[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1].detach().clone().requires_grad_(True)
    )
    phi_init_patch = phi_patch_var.detach().clone()

    # Build the frozen-corner mask. Patch faces that have exterior context
    # form the "Dirichlet ring": their values are pinned to phi_init_patch
    # during optimization. Faces coinciding with the volume boundary stay
    # FREE — they have no exterior context to stay consistent with (same
    # convention as ``_patch_frozen_mask`` in iterative3d_barrier.py and
    # every other windowed mask in the package).
    D_full, H_full, W_full = phi_full.shape[1], phi_full.shape[2], phi_full.shape[3]
    frozen = torch.zeros((Dp, Hp, Wp), dtype=torch.bool, device=device)
    if z0 > 0:
        frozen[0, :, :] = True
    if z1 < D_full - 1:
        frozen[-1, :, :] = True
    if y0 > 0:
        frozen[:, 0, :] = True
    if y1 < H_full - 1:
        frozen[:, -1, :] = True
    if x0 > 0:
        frozen[:, :, 0] = True
    if x1 < W_full - 1:
        frozen[:, :, -1] = True
    frozen_b = frozen.unsqueeze(0).expand(3, Dp, Hp, Wp)  # broadcast over channel

    def _anchor_v(phi_eff):
        diff = phi_eff - phi_init_patch
        if anchor == 'l2':
            return 0.5 * (diff * diff).sum()
        if anchor == 'l1':
            s = torch.sqrt(diff * diff + eps_l1 * eps_l1)
            return (s - eps_l1).sum()
        return torch.zeros((), dtype=diff.dtype, device=diff.device)

    lam_steps = 0
    mu_steps = 0
    feasible = False

    # -----------------------------------------------------------------
    # Phase 1: exterior quadratic penalty on patch.
    # -----------------------------------------------------------------
    for lam in lam_schedule:
        if feasible:
            break

        def closure(lam_=lam):
            opt.zero_grad()
            phi_eff = torch.where(frozen_b, phi_init_patch, phi_patch_var)
            V = six_tet_volumes_3d_torch(phi_eff)
            viol = torch.clamp(target_f - V, min=0.0)
            penalty = lam_ * (viol * viol).sum()
            loss = _anchor_v(phi_eff) + penalty
            loss.backward()
            return loss

        opt = torch.optim.LBFGS(
            [phi_patch_var],
            max_iter=max_iter,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-10,
            tolerance_change=1e-12 if dtype == torch.float64 else 1e-9,
        )
        opt.step(closure)
        lam_steps += 1

        with torch.no_grad():
            phi_eff = torch.where(frozen_b, phi_init_patch, phi_patch_var)
            V = six_tet_volumes_3d_torch(phi_eff)
            if float(V.min().item()) >= target_f:
                feasible = True

    # -----------------------------------------------------------------
    # Phase 2: log-barrier interior point on patch.
    # -----------------------------------------------------------------
    if feasible:
        for mu in mu_schedule:

            def closure(mu_=mu):
                opt.zero_grad()
                phi_eff = torch.where(frozen_b, phi_init_patch, phi_patch_var)
                V = six_tet_volumes_3d_torch(phi_eff)
                slack = V - threshold_f
                active = slack > 0
                if not bool(active.all().item()):
                    safe = torch.where(active, slack, torch.ones_like(slack))
                    viol = torch.clamp(-slack + 1e-12, min=0.0) * (~active).to(slack.dtype)
                    bar = -mu_ * (torch.log(safe) * active.to(slack.dtype)).sum()
                    penalty = 1e8 * (viol * viol).sum()
                    loss = _anchor_v(phi_eff) + bar + penalty
                else:
                    bar = -mu_ * torch.log(slack).sum()
                    loss = _anchor_v(phi_eff) + bar
                loss.backward()
                return loss

            opt = torch.optim.LBFGS(
                [phi_patch_var],
                max_iter=max_iter,
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-10,
                tolerance_change=1e-12 if dtype == torch.float64 else 1e-9,
            )
            opt.step(closure)
            mu_steps += 1

    # Splice the optimized patch (interior only — boundary is by
    # construction unchanged) back into the full grid.
    with torch.no_grad():
        phi_eff = torch.where(frozen_b, phi_init_patch, phi_patch_var)
        phi_full[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = phi_eff.detach()

    return lam_steps, mu_steps


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
    windowed=False,
    pad=2,
    max_outer_iter=20,
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
    windowed : bool
        When ``True``, detect connected components of folded cells via
        scipy CCL, build a padded corner-bbox per component, and run
        penalty→barrier on each patch with a frozen Dirichlet ring.
        Repeats up to ``max_outer_iter`` times until no folds remain.
        Drastically reduces memory + compute on volumes where folds are
        spatially clustered (the common case). When ``False`` (default),
        runs full-grid in one pass (the Phase 2 implementation).
    pad : int
        Voxels of corner-space expansion around each component's bbox
        for the frozen boundary ring. Only used when ``windowed=True``.
    max_outer_iter : int
        Cap on outer iterations (CCL → per-patch optimize → CCL ...).
        Only used when ``windowed=True``.

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
            f'device={device} dtype={dtype}  '
            f'mode={"windowed" if windowed else "full-grid"}'
        )
        print(f'[init] tet neg={init_neg}  min={init_min:+.5f}')

    # -----------------------------------------------------------------
    # Windowed path: per-component patch optimization with frozen
    # boundary ring. Detect → CCL → optimize-each → repeat.
    # -----------------------------------------------------------------
    if windowed:
        from scipy.ndimage import label as cc_label

        phi_full = phi_init.detach().clone()
        struct = _ccl_structure()

        for outer_iter in range(max_outer_iter):
            with torch.no_grad():
                V = volumes(phi_full)
                fold_cells_mask = (V.min(dim=0).values < threshold_f).cpu().numpy()
                # ``n_neg`` matches the canonical schema used by the
                # final-stats logging below and by the rest of the
                # wallbreaker family (tets with V <= 0). The
                # cells-below-threshold count is preserved under
                # ``n_fold_cells`` for windowed-mode diagnostics.
                n_neg = int((V <= 0).sum().item())
            n_fold_cells = int(fold_cells_mask.sum())
            if verbose >= 1:
                print(
                    f'[windowed outer {outer_iter}] fold cells: {n_fold_cells}  '
                    f'tets with V<=0: {n_neg}'
                )
            if record_history:
                history.append(
                    dict(
                        phase=f'outer_{outer_iter}_pre',
                        step=outer_iter,
                        n_neg=n_neg,
                        n_fold_cells=n_fold_cells,
                        min_T=float(V.min().item()),
                    )
                )
            if n_fold_cells == 0:
                break

            labeled, n_comp = cc_label(fold_cells_mask, structure=struct)
            for comp_id in range(1, n_comp + 1):
                coords = np.where(labeled == comp_id)
                if coords[0].size == 0:
                    continue
                z0, z1, y0, y1, x0, x1 = _patch_corner_bbox(
                    coords[0],
                    coords[1],
                    coords[2],
                    pad,
                    phi_full.shape[1],
                    phi_full.shape[2],
                    phi_full.shape[3],
                )
                lam_steps, mu_steps = _optimize_patch_3d_tet_torch(
                    phi_full,
                    z0,
                    z1,
                    y0,
                    y1,
                    x0,
                    x1,
                    threshold_f,
                    target_f,
                    lam_schedule,
                    mu_schedule,
                    max_iter,
                    anchor,
                    eps_l1,
                    dtype,
                    device,
                )
                if verbose >= 2:
                    print(
                        f'  comp {comp_id:3d}: bbox ({z0}-{z1}, {y0}-{y1}, {x0}-{x1})  '
                        f'lam_steps={lam_steps} mu_steps={mu_steps}'
                    )

        # Final stats + return.
        with torch.no_grad():
            V_final = volumes(phi_full)
            n_neg = int((V_final <= 0).sum().item())
            min_V = float(V_final.min().item())
        wall = time.time() - t0
        if verbose >= 1:
            ok = n_neg == 0 and min_V >= threshold_f - 1e-5
            print(
                f'[3d-tet-barrier-torch done] feasible={ok}  '
                f'n_neg={n_neg}  min={min_V:+.6f}  ({wall:.2f}s, windowed)'
            )
        if record_history:
            history.append(dict(phase='final', step=-1, n_neg=n_neg, min_T=min_V, wall_s=wall))
        phi_corr = phi_full.detach().cpu().numpy().astype(np.float64)
        if record_history:
            return phi_corr, history
        return phi_corr

    # -----------------------------------------------------------------
    # Phase 1: exterior quadratic penalty.
    # -----------------------------------------------------------------
    # Gate on the barrier's actual requirement (min volume clearing the
    # threshold+margin target), not merely "no non-positive tets": a field
    # with 0 < min V < threshold must still run the graduated lam schedule,
    # otherwise the barrier phase immediately falls into the emergency
    # 1e8*viol^2 branch. Matches the windowed mode and the other barrier
    # solvers, which all gate on init_min >= target.
    feasible = init_min >= target_f
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
