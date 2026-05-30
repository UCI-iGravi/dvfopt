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

from dvfopt.core.tri_primitives import (
    tri_areas_flat as _tri_areas_flat,
)
from dvfopt.core.tri_primitives import (
    tri_grad_T_v as _tri_grad_T_v,
)


def _soft_pen_objective(phi_flat, phi_in_flat, H, W, threshold, lam, anchor, eps_l1):
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
    anchor: str = 'l2',
    lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8, 1e10),
    inner_maxiter: int = 2000,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    eps_l1: float = 1e-4,
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
            anchor=anchor,
            eps_l1=eps_l1,
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
            print(
                f'  refine lam={lam:.0e}  min_T={min_T:+.5f}  L2={L2:.1f}  '
                f'nit={res.nit}  ({time.time() - t0:.1f}s)',
                flush=True,
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
