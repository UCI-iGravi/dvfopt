"""Soft-quadratic-penalty L-BFGS-B refinement for the 6-tet constraint (3D).

3D analog of :mod:`dvfopt.core.wallbreakers._l2_refine` (the 2D m12
step). Replaces a log barrier (which freezes its active set in 1-2
iterations and can't reduce L2 further) with a one-sided quadratic
penalty that is exactly zero for cells with ``V_k > threshold``.

Non-active cells contribute no constraint gradient and L-BFGS-B can
move them freely toward ``phi_in`` (large L2 reduction possible);
active cells get a smooth quadratic kickback proportional to
``lambda`` that pins them at the boundary as ``lambda`` is annealed
up.

Used as stage 2 of the full m14-3D pipeline (m10 seed → l2 refine →
harmonic repair → barrier polish).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from dvfopt._logging import log_info
from dvfopt.jacobian.tetrahedron_sign import tet_grad_T_v, tet_volumes_flat
from dvfopt.objectives import L2Objective, Objective, _kind_eps


def _soft_pen_objective_3d(phi_flat, phi_anchor_flat, D, H, W, threshold, lam, anchor, eps_l1):
    """Quadratic-penalty objective: only violating tets contribute."""
    diff = phi_flat - phi_anchor_flat
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
    V = tet_volumes_flat(phi_flat, D, H, W)
    viol = np.maximum(0.0, threshold - V)
    if viol.any():
        val += lam * float((viol * viol).sum())
        grad = grad - 2.0 * lam * tet_grad_T_v(phi_flat, D, H, W, viol)
    return val, grad


def l2_refine_3d(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    seed: Optional[np.ndarray] = None,
    margin: float = 1e-3,
    objective: Objective | None = None,
    lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8, 1e10),
    inner_maxiter: int = 2000,
    time_budget_s: float = 600.0,
    require_feasibility: bool = True,
    verbose: int = 1,
    record_history: bool = False,
):
    """Anneal soft-penalty L-BFGS-B from ``seed`` while anchoring to ``phi_in``.

    Parameters
    ----------
    phi_in : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
        The anchor reference — distance is measured against this.
    seed : ndarray, shape ``(3, D, H, W)`` or None
        Starting iterate. If ``None``, falls back to ``phi_in`` (i.e.,
        no pre-seed: the soft penalty alone tries to pull initial folds
        out — may not reach feasibility on dense cases). The full m14-3D
        pipeline passes the m10-3D output here.
    threshold : float, optional
        Lower bound on per-tet volume. Default ``DEFAULT_PARAMS['threshold']``.
    margin : float
        Safety margin above ``threshold``.
    objective : Objective or None
        Data term; ``None`` (default) means
        :class:`~dvfopt.objectives.L2Objective`.
    lam_schedule : tuple of float
        Penalty parameter anneal schedule. Each step is an L-BFGS-B
        inner solve with a fixed ``lam``.
    inner_maxiter : int
        Per-step L-BFGS-B iteration cap.
    require_feasibility : bool
        If ``True``, escalate ``lam`` (one extra step) until every tet
        clears ``threshold``.
    time_budget_s, verbose, record_history : as elsewhere.

    Returns
    -------
    phi : ndarray, shape ``(3, D, H, W)``.
    info : dict, only if ``record_history=True``.
    """
    from dvfopt._defaults import DEFAULT_PARAMS

    objective = objective or L2Objective()
    anchor, eps_l1 = _kind_eps(objective)
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    _, D, H, W = phi_in.shape
    target = threshold + margin

    # Phi pack: [dx, dy, dz] (DX_FIRST).
    phi_anchor_flat = np.concatenate([phi_in[2].ravel(), phi_in[1].ravel(), phi_in[0].ravel()])
    if seed is None:
        seed = phi_in
    phi_flat = np.concatenate([seed[2].ravel(), seed[1].ravel(), seed[0].ravel()])

    t0 = time.time()
    log = []
    last_min = -np.inf

    for lam in lam_schedule:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(
            _soft_pen_objective_3d,
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            args=(phi_anchor_flat, D, H, W, threshold, lam, anchor, eps_l1),
            options=dict(maxiter=inner_maxiter, ftol=1e-12, gtol=1e-8),
        )
        phi_flat = res.x
        V = tet_volumes_flat(phi_flat, D, H, W)
        min_T = float(V.min())
        n_neg = int((V <= 0).sum())
        last_min = min_T
        log.append(
            dict(
                phase='l2_refine',
                lam=float(lam),
                inner_nit=int(res.nit),
                min_T=min_T,
                n_neg=n_neg,
                wall_s=time.time() - t0,
            )
        )
        if verbose:
            log_info(
                f'  l2_refine lam={lam:.0e}  inner={res.nit:4d}  '
                f'min_V={min_T:+.6f}  n_neg={n_neg}  ({time.time() - t0:.1f}s)',
            )
        if require_feasibility and min_T >= target:
            break

    # Unpack [dx, dy, dz] back to (3, D, H, W) [dz, dy, dx].
    n = D * H * W
    dx = phi_flat[:n].reshape(D, H, W)
    dy = phi_flat[n : 2 * n].reshape(D, H, W)
    dz = phi_flat[2 * n :].reshape(D, H, W)
    phi_out = np.stack([dz, dy, dx])

    info = dict(
        lam_used=len(log),
        min_T_final=last_min,
        feasible=last_min >= target,
        log=log,
    )
    if record_history:
        return phi_out, info
    return phi_out


__all__ = ['l2_refine_3d']
