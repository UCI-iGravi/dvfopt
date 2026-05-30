"""Augmented Lagrangian method (ALM) for the 2-triangle constraint.

PHR (Powell-Hestenes-Rockafellar) augmented Lagrangian with L-BFGS-B
inner. Unlike SLSQP, ALM does NOT solve a QP and is therefore not
subject to the active-set degeneracy at the constraint wall — the
inner problem is smooth and unconstrained.

For inequality constraints ``c_i(phi) >= tau`` rewrite as slacks
``c_i - tau >= 0`` and minimise::

    L_rho(phi, mu) = f(phi) + (1/(2 rho)) * sum (psi^2 - mu^2)
    psi_i = max(0, mu_i - rho (c_i - tau))

with ``f = 0.5 ||phi - phi_in||^2`` (or smoothed L1). Outer loop:
``mu_i <- max(0, mu_i - rho (c_i - tau))``; ``rho`` grows only when
constraint violation fails to halve (Birgin-Martinez safeguard).

Promoted from ``notebooks/experiments/wall_breakers/methods/m03_augmented_lagrangian.py``.
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


def _alm_objective(phi_flat, phi_init_flat, H, W, threshold, mu, rho, anchor='l2', eps_l1=1e-4):
    diff = phi_flat - phi_init_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff)
        grad = diff.copy()
    elif anchor == 'l1':
        s_anc = np.sqrt(diff * diff + eps_l1 * eps_l1)
        val = float((s_anc - eps_l1).sum())
        grad = diff / s_anc
    else:
        val = 0.0
        grad = np.zeros_like(diff)

    T = _tri_areas_flat(phi_flat, H, W)
    slack = T - threshold
    psi = np.maximum(0.0, mu - rho * slack)
    val += float((psi * psi - mu * mu).sum()) / (2.0 * rho)
    grad = grad - _tri_grad_T_v(phi_flat, H, W, psi)
    return val, grad


def augmented_lagrangian_2d(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    margin: float = 1e-3,
    anchor: str = 'l2',
    rho_init: float = 1.0,
    rho_growth: float = 5.0,
    rho_max: float = 1e8,
    outer_max: int = 60,
    inner_maxiter: int = 200,
    ftol_inner: float = 1e-10,
    gtol_inner: float = 1e-7,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    record_history: bool = False,
):
    """PHR augmented Lagrangian; stops when feasible at ``threshold + margin``.

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)``
        Starting field.
    anchor : {'l2', 'l1', 'none'}
        Data-term form. ``'none'`` skips the anchor entirely and just
        pushes for feasibility.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    info : dict, only if ``record_history=True``
    """
    from dvfopt._defaults import DEFAULT_PARAMS

    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    H, W = phi_in.shape[1], phi_in.shape[2]
    phi_init = phi_in.copy()
    phi_init_flat = np.concatenate([phi_init[0].ravel(), phi_init[1].ravel()])
    n_constr = 2 * (H - 1) * (W - 1)

    mu = np.zeros(n_constr)
    rho = rho_init
    phi_flat = phi_init_flat.copy()

    target = threshold + margin
    last_min = -np.inf
    log = []
    t0 = time.time()

    for outer in range(outer_max):
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(
            _alm_objective,
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            args=(phi_init_flat, H, W, threshold, mu, rho, anchor, 1e-4),
            options=dict(maxiter=inner_maxiter, ftol=ftol_inner, gtol=gtol_inner),
        )
        phi_flat = res.x

        T = _tri_areas_flat(phi_flat, H, W)
        slack = T - threshold
        min_T = float(T.min())
        mu = np.maximum(0.0, mu - rho * slack)
        viol_now = float(np.maximum(0.0, target - T).max())
        if outer > 0 and viol_now > 0.5 * log[-1]['viol']:
            rho = min(rho_max, rho * rho_growth)

        log.append(
            dict(
                outer=outer,
                inner_nit=int(res.nit),
                min_T=min_T,
                viol=viol_now,
                rho=rho,
                mu_max=float(mu.max()),
                wall=time.time() - t0,
            )
        )
        if verbose:
            print(
                f'  ALM out={outer:3d}  inner={res.nit:4d}  '
                f'min_T={min_T:+.5f}  viol={viol_now:.3e}  '
                f'rho={rho:.1e}  ({time.time() - t0:.1f}s)',
                flush=True,
            )
        last_min = min_T
        if min_T >= target:
            break

    phi_out = np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)])
    info = dict(
        outer_used=len(log),
        rho_final=rho,
        min_T_final=last_min,
        feasible=last_min >= target,
        log_first5=log[:5],
        log_last5=log[-5:],
    )
    return (phi_out, info) if record_history else phi_out
