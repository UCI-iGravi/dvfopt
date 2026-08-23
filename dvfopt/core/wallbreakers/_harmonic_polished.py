"""Harmonic seed + ALM + log-barrier L2 polish anchored to ``phi_in``.

This is the **always-feasibility baseline** (``m10`` in the
wall_breakers experiment suite). On the original B0039 DVF it achieves
528/528 feasibility — the log-barrier polish anchored to ``phi_in``
mathematically cannot lose feasibility because the barrier is +inf at
``T_k <= threshold`` and scipy L-BFGS-B's line search shrinks the step
on any +inf evaluation.

Pipeline
--------
1.  Harmonic extension (:func:`harmonic_extension_2d`) produces a
    feasible seed by replacing dilated fold cores with smooth Laplacian
    completions of the ring.
2.  If the seed is not yet feasible (rare on full slices but common on
    tight crops), run a short ALM (:func:`augmented_lagrangian_2d`) to
    push it across the wall.
3.  scipy L-BFGS-B with::

        F = 0.5 ||phi - phi_in||^2  -  mu * sum log(T_k - threshold)

    and a ``mu_schedule = 1e-1 -> 1e-5``. The barrier cannot lose
    feasibility; all the polish does is slide along the feasible
    manifold toward the input.

Promoted from ``notebooks/experiments/wall_breakers/methods/m10_harmonic_l2_polished.py``.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt._logging import log_info, log_warning
from dvfopt.core.primitives.tri import tri_areas_flat as _tri_areas_flat
from dvfopt.core.wallbreakers._alm import augmented_lagrangian_2d
from dvfopt.core.wallbreakers._common import (
    barrier_anchored_objective as _barrier_anchored,
)
from dvfopt.core.wallbreakers._common import (
    min_tri as _min_tri,
)
from dvfopt.core.wallbreakers._common import (
    resolved_safety_margin,
)
from dvfopt.core.wallbreakers._harmonic import harmonic_extension_2d
from dvfopt.objectives import L2Objective, NoneObjective, Objective, _kind_eps


def iterative_2d_tri_harmonic_polished(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    margin: float = 1e-3,
    ring_pad: int = 2,
    max_grow_iters: int = 8,
    mu_schedule: tuple = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    inner_maxiter: int = 300,
    objective: Objective | None = None,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    record_history: bool = False,
    step_callback=None,
):
    """3-stage feasibility-guaranteed 2-triangle solver.

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)`` or ``(3, 1, H, W)``
        Input deformation field.
    threshold : float
        Lower bound for both triangle areas.
    margin : float
        Safety margin used during the ALM stage to land strictly above
        the polish's log-barrier singularity.
    ring_pad : int
        Initial cell-dilation of the harmonic patch ring.
    max_grow_iters : int
        Maximum extra dilation rounds inside the harmonic-extension
        stage when the current patch still has residual folds. Default
        ``8`` matches the original manuscript setting. Bump if your
        slice has large non-convex fold cores whose dilated boundary
        ring isn't convex at the default — the wallbreaker will then
        try a larger ring before falling back to "best-effort patch".
    mu_schedule : sequence of float
        Continuation schedule for the barrier polish (smaller ``mu`` ->
        closer to the L2-optimum, but a wall at the constraint surface).
    inner_maxiter : int
        L-BFGS-B ``maxiter`` per ``mu`` step.
    objective : Objective or None
        Anchor objective. ``None`` (default) means
        :class:`~dvfopt.objectives.L2Objective` — the manuscript setting.
    time_budget_s : float
        Wall-time budget. The function returns whatever it has when this
        is exhausted.
    verbose : int
    record_history : bool
        If False (default), returns ``phi`` (ndarray of shape ``(2, H, W)``).
        If True, returns ``(phi, info)`` where ``info`` is a dict with
        per-stage statistics.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    info : dict, only if ``record_history=True``
    """
    objective = objective or L2Objective()
    anchor, eps_l1 = _kind_eps(objective)
    # Coerce input shape.
    if phi_in.ndim == 4:
        if phi_in.shape[0] == 3:
            phi_in = np.stack([phi_in[1, 0], phi_in[2, 0]])
        else:
            phi_in = phi_in[:, 0]
    if phi_in.dtype != np.float64:
        phi_in = phi_in.astype(np.float64)

    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    H, W = phi_in.shape[1], phi_in.shape[2]
    t0 = time.time()
    info: dict = {}

    def _fire(stage: str, phi):
        """Forward an intermediate phi snapshot to ``step_callback`` —
        the GUI consumes these for the history slider so the user can
        scrub through M10's internal pipeline. Silent on consumer
        exceptions; KeyboardInterrupt propagates as the documented
        stop signal."""
        if step_callback is None:
            return
        try:
            step_callback({'phi': np.asarray(phi).copy(), 'stage': stage})
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log_warning(f'step_callback raised {type(exc).__name__}: {exc}; continuing')

    # Stage 1: harmonic extension.
    seed, r1_info = harmonic_extension_2d(
        phi_in,
        threshold=threshold,
        ring_pad=ring_pad,
        max_grow_iters=max_grow_iters,
        margin=0.0,
        record_history=True,
    )
    seed_min = _min_tri(seed)
    seed_L2 = float(np.linalg.norm((seed - phi_in).ravel()))
    info['stage1_harmonic'] = dict(
        min_T=seed_min, L2=seed_L2, wall=time.time() - t0, patches=r1_info.get('patches')
    )
    if verbose:
        log_info(
            f'  stage1 harmonic min_T={seed_min:+.5f}  L2={seed_L2:.1f}  ({time.time() - t0:.1f}s)',
        )
    _fire('stage1_harmonic', seed)

    # Stage 2: ALM cleanup if seed not strictly interior.
    safety_margin = resolved_safety_margin(margin)
    if seed_min < threshold + safety_margin:
        remaining = max(60.0, time_budget_s * 0.35)
        seed = augmented_lagrangian_2d(
            seed,
            threshold=threshold + safety_margin,
            margin=1e-4,
            objective=objective,
            outer_max=30,
            inner_maxiter=150,
            time_budget_s=remaining,
            verbose=0,
        )
        info['stage2_alm'] = dict(
            min_T=_min_tri(seed),
            L2=float(np.linalg.norm((seed - phi_in).ravel())),
            wall=time.time() - t0,
        )
        if verbose:
            log_info(
                f'  stage2 ALM     min_T={_min_tri(seed):+.5f}  '
                f'L2={info["stage2_alm"]["L2"]:.1f}  '
                f'({time.time() - t0:.1f}s)',
            )
        _fire('stage2_alm', seed)
    else:
        info['stage2_alm'] = {'skipped': 'seed-already-feasible'}

    cur_min = _min_tri(seed)
    if cur_min <= threshold + 1e-6:
        seed_bump = augmented_lagrangian_2d(
            seed,
            threshold=threshold + safety_margin,
            margin=1e-4,
            objective=NoneObjective(),
            outer_max=10,
            inner_maxiter=100,
            time_budget_s=60.0,
            verbose=0,
        )
        if _min_tri(seed_bump) > _min_tri(seed):
            seed = seed_bump
        cur_min = _min_tri(seed)
        info['stage2b_alm_bump'] = dict(
            min_T=cur_min, L2=float(np.linalg.norm((seed - phi_in).ravel()))
        )

    if cur_min <= threshold:
        info['stage3_polish'] = {'skipped': f'min_T={cur_min:.5f} not strict-feasible'}
        info['final_min_T'] = cur_min
        return (seed, info) if record_history else seed

    # Stage 3: log-barrier L2 polish anchored to phi_in.
    phi_in_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = np.concatenate([seed[0].ravel(), seed[1].ravel()])
    polish_log = []
    for mu in mu_schedule:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(
            _barrier_anchored,
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            args=(phi_in_flat, H, W, threshold, mu, anchor, eps_l1),
            options=dict(maxiter=inner_maxiter, ftol=1e-12, gtol=1e-9),
        )
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        min_T = float(T.min())
        phi_cur = np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)])
        L2 = float(np.linalg.norm((phi_cur - phi_in).ravel()))
        polish_log.append(dict(mu=mu, min_T=min_T, L2=L2, nit=int(res.nit), wall=time.time() - t0))
        if verbose:
            log_info(
                f'  stage3 mu={mu:.1e}  min_T={min_T:+.5f}  L2={L2:.1f}  '
                f'nit={res.nit}  ({time.time() - t0:.1f}s)',
            )
        # Emit per-µ snapshot so the GUI can scrub through the polish.
        _fire(f'stage3_polish_mu={mu:g}', phi_cur)

    phi_out = np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)])
    info['stage3_polish'] = dict(
        steps=len(polish_log),
        final_min_T=_min_tri(phi_out),
        final_L2=float(np.linalg.norm((phi_out - phi_in).ravel())),
        log_last3=polish_log[-3:],
    )
    info['final_min_T'] = _min_tri(phi_out)
    return (phi_out, info) if record_history else phi_out
