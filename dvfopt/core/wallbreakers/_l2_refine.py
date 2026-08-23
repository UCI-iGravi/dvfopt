"""Soft-quadratic-penalty L-BFGS-B refinement of an existing feasible seed.

Replaces a log barrier (which freezes its active set in 1-2 iterations
and can't reduce L2 further) with a one-sided quadratic penalty that
is exactly zero for cells with ``T_k > threshold``. Non-active cells
contribute no constraint gradient and L-BFGS-B can move them freely
toward ``phi_in`` (large L2 reduction possible); active cells get a
smooth quadratic kickback proportional to ``lambda`` that pins them at
the boundary as ``lambda`` is annealed up.

Promoted from ``notebooks/experiments/wall_breakers/methods/m12_l2_refine.py``.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from dvfopt._logging import log_info
from dvfopt.core.primitives.tri import (
    tri_areas_flat as _tri_areas_flat,
)
from dvfopt.core.primitives.tri import (
    tri_grad_T_v as _tri_grad_T_v,
)
from dvfopt.objectives import L2Objective, Objective, _kind_eps

# Optional fused JIT path for the L1-anchored objective — the common
# case for cluster_slp's m14_fast inner. Combines anchor + T-area
# computation + violation + gradient scatter into a single fused
# loop, avoiding three separate numpy passes. Auto-detected at import.
try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False)
    def _soft_pen_l1_fused_kernel(dy, dx, dy_in, dx_in, H, W, threshold, lam, eps_l1):
        """Fused L1-anchor + T-areas + viol + grad in one kernel.

        Returns (val, g_dy, g_dx). Anchor val and grad use smoothed L1
        (sqrt(diff^2 + eps^2) - eps). Constraint contribution skips
        cells where both T1 and T2 satisfy the threshold (the common
        case during late lambda annealing)."""
        g_dy = np.zeros((H, W))
        g_dx = np.zeros((H, W))
        val = 0.0
        # Smoothed-L1 anchor on every grid vertex.
        for i in range(H):
            for j in range(W):
                diff_y = dy[i, j] - dy_in[i, j]
                diff_x = dx[i, j] - dx_in[i, j]
                s_y = np.sqrt(diff_y * diff_y + eps_l1 * eps_l1)
                s_x = np.sqrt(diff_x * diff_x + eps_l1 * eps_l1)
                val += (s_y - eps_l1) + (s_x - eps_l1)
                g_dy[i, j] = diff_y / s_y
                g_dx[i, j] = diff_x / s_x
        # Constraint violation + adjoint in one pass.
        for i in range(H - 1):
            for j in range(W - 1):
                x_tl = j + dx[i, j]
                y_tl = i + dy[i, j]
                x_tr = (j + 1) + dx[i, j + 1]
                y_tr = i + dy[i, j + 1]
                x_bl = j + dx[i + 1, j]
                y_bl = (i + 1) + dy[i + 1, j]
                x_br = (j + 1) + dx[i + 1, j + 1]
                y_br = (i + 1) + dy[i + 1, j + 1]
                # T1 (A=TR, B=BL, C=BR).
                ABx_1 = x_bl - x_tr
                ABy_1 = y_bl - y_tr
                ACx_1 = x_br - x_tr
                ACy_1 = y_br - y_tr
                T1 = -0.5 * (ABx_1 * ACy_1 - ABy_1 * ACx_1)
                # T2 (A=TL, B=BL, C=TR).
                ABx_2 = x_bl - x_tl
                ABy_2 = y_bl - y_tl
                ACx_2 = x_tr - x_tl
                ACy_2 = y_tr - y_tl
                T2 = -0.5 * (ABx_2 * ACy_2 - ABy_2 * ACx_2)
                v1 = max(0.0, threshold - T1)
                v2 = max(0.0, threshold - T2)
                if v1 == 0.0 and v2 == 0.0:
                    continue
                val += lam * (v1 * v1 + v2 * v2)
                if v1 != 0.0:
                    c1 = -2.0 * lam * 0.5 * v1
                    g_dx[i, j + 1] += c1 * (y_br - y_bl)
                    g_dy[i, j + 1] += c1 * (x_bl - x_br)
                    g_dx[i + 1, j] += -c1 * (y_br - y_tr)
                    g_dy[i + 1, j] += c1 * (x_br - x_tr)
                    g_dx[i + 1, j + 1] += c1 * (y_bl - y_tr)
                    g_dy[i + 1, j + 1] += -c1 * (x_bl - x_tr)
                if v2 != 0.0:
                    c2 = -2.0 * lam * 0.5 * v2
                    g_dx[i, j] += c2 * (y_tr - y_bl)
                    g_dy[i, j] += c2 * (x_bl - x_tr)
                    g_dx[i + 1, j] += -c2 * (y_tr - y_tl)
                    g_dy[i + 1, j] += c2 * (x_tr - x_tl)
                    g_dx[i, j + 1] += c2 * (y_bl - y_tl)
                    g_dy[i, j + 1] += -c2 * (x_bl - x_tl)
        return val, g_dy, g_dx


def _soft_pen_objective(phi_flat, phi_in_flat, H, W, threshold, lam, anchor, eps_l1):
    """L1/L2-anchored soft-quadratic-penalty objective + gradient.

    Uses a fused Numba JIT kernel for the common L1-anchored case
    (cluster_slp's m14_fast inner default). Falls back to the
    numpy-on-pre-JIT'd-primitives path otherwise. The fused kernel is
    2-3x faster on this hot inner loop because it folds three
    separate numpy passes (anchor / area / viol-grad) into one and
    skips inactive cells."""
    if anchor == 'l1' and _HAVE_NUMBA:
        HW = H * W
        dy = np.ascontiguousarray(phi_flat[:HW].reshape(H, W))
        dx = np.ascontiguousarray(phi_flat[HW:].reshape(H, W))
        dy_in = np.ascontiguousarray(phi_in_flat[:HW].reshape(H, W))
        dx_in = np.ascontiguousarray(phi_in_flat[HW:].reshape(H, W))
        val, g_dy, g_dx = _soft_pen_l1_fused_kernel(
            dy, dx, dy_in, dx_in, H, W, threshold, lam, eps_l1
        )
        grad = np.concatenate([g_dy.ravel(), g_dx.ravel()])
        return val, grad
    diff = phi_flat - phi_in_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff)
        grad = diff.copy()
    elif anchor == 'l1':
        s = np.sqrt(diff * diff + eps_l1 * eps_l1)
        val = float((s - eps_l1).sum())
        grad = diff / s
    else:
        val = 0.0
        grad = np.zeros_like(diff)
    T = _tri_areas_flat(phi_flat, H, W)
    viol = np.maximum(0.0, threshold - T)
    if viol.any():
        val += lam * float((viol * viol).sum())
        grad = grad - 2.0 * lam * _tri_grad_T_v(phi_flat, H, W, viol)
    return val, grad


def l2_refine_2d(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    seed: np.ndarray = None,
    margin: float = 1e-3,
    objective: Objective | None = None,
    lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8, 1e10),
    inner_maxiter: int = 2000,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    require_feasibility: bool = True,
    record_history: bool = False,
):
    """Anneal soft-penalty L-BFGS-B from ``seed`` while anchoring to ``phi_in``.

    Parameters
    ----------
    seed : ndarray, shape ``(2, H, W)``, optional
        Starting point. If ``None``, runs
        :func:`iterative_2d_tri_harmonic_polished` (m10) first.
    require_feasibility : bool
        If True, escalates ``lam`` until every ``T_k >= threshold``.
    record_history : bool
        If True, returns ``(phi, info)`` instead of just ``phi``.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    info : dict, only if ``record_history=True``
    """
    from dvfopt._defaults import DEFAULT_PARAMS
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

    objective = objective or L2Objective()
    anchor, eps_l1 = _kind_eps(objective)
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    H, W = phi_in.shape[1], phi_in.shape[2]
    t0 = time.time()
    info: dict = {}

    if seed is None:
        from dvfopt.core.wallbreakers._harmonic_polished import iterative_2d_tri_harmonic_polished

        seed = iterative_2d_tri_harmonic_polished(
            phi_in,
            threshold=threshold,
            margin=margin,
            objective=objective,
            time_budget_s=time_budget_s * 0.5,
            verbose=verbose,
        )
        info['seed'] = dict(
            min_T=float(np.minimum(*_triangle_areas_2d(seed[0], seed[1])).min()),
            L2_to_input=float(np.linalg.norm((seed - phi_in).ravel())),
            wall=time.time() - t0,
        )

    phi_in_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = np.concatenate([seed[0].ravel(), seed[1].ravel()])

    log = []
    for lam in lam_schedule:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(
            _soft_pen_objective,
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            args=(phi_in_flat, H, W, threshold + margin, lam, anchor, eps_l1),
            options=dict(maxiter=inner_maxiter, ftol=1e-12, gtol=1e-9),
        )
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        min_T = float(T.min())
        phi_cur = np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)])
        L2 = float(np.linalg.norm((phi_cur - phi_in).ravel()))
        log.append(dict(lam=lam, min_T=min_T, L2=L2, nit=int(res.nit), wall=time.time() - t0))
        if verbose:
            log_info(
                f'  refine lam={lam:.0e}  min_T={min_T:+.5f}  L2={L2:.1f}  '
                f'nit={res.nit}  ({time.time() - t0:.1f}s)',
            )

    if require_feasibility:
        T = _tri_areas_flat(phi_flat, H, W)
        lam = lam_schedule[-1]
        while T.min() < threshold and time.time() - t0 < time_budget_s:
            lam *= 10.0
            res = minimize(
                _soft_pen_objective,
                phi_flat,
                jac=True,
                method='L-BFGS-B',
                args=(phi_in_flat, H, W, threshold + margin, lam, anchor, eps_l1),
                options=dict(maxiter=inner_maxiter, ftol=1e-12, gtol=1e-9),
            )
            phi_flat = res.x
            T = _tri_areas_flat(phi_flat, H, W)
            log.append(
                dict(
                    lam=lam,
                    min_T=float(T.min()),
                    L2=float(np.linalg.norm(phi_flat - phi_in_flat)),
                    nit=int(res.nit),
                    wall=time.time() - t0,
                    escalate=True,
                )
            )
            if lam > 1e16:
                break

    phi_out = np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)])
    info['final_min_T'] = float(_tri_areas_flat(phi_flat, H, W).min())
    info['final_L2'] = float(np.linalg.norm((phi_out - phi_in).ravel()))
    info['refine_steps'] = len(log)
    info['log_last3'] = log[-3:]
    return (phi_out, info) if record_history else phi_out
