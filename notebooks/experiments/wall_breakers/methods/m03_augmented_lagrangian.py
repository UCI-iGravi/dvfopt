"""Augmented Lagrangian method (ALM) for the 2-triangle constraint.

For inequality constraints :math:`c_i(\\phi) \\ge \\tau` rewrite as slacks
:math:`c_i - \\tau \\ge 0` and use the standard PHR (Powell-Hestenes-
Rockafellar) augmented Lagrangian:

.. math::

    L_\\rho(\\phi, \\mu) = f(\\phi) + \\frac{1}{2\\rho}
        \\sum_i [\\max(0,\\, \\mu_i - \\rho (c_i - \\tau))^2 - \\mu_i^2]

with ``f(phi) = 0.5 ||phi - phi_init||^2``. Outer loop updates the
multipliers ``mu_i <- max(0, mu_i - rho (c_i - tau))`` and tightens
``rho`` only on poor progress (Birgin-Martinez safeguard).

Why it should help where SLSQP fails
------------------------------------
SLSQP's `status 8` comes from an active-set degeneracy at the constraint
boundary -- many constraints simultaneously crowd ``c_i = 0``, making the
QP's KKT matrix near-singular and the line search fail. PHR-ALM
never solves a QP. The inner problem is a smooth unconstrained
minimisation (L-BFGS-B), which doesn't care about active-set transitions
because there *is* no active set: the max() in the penalty is smoothed
in gradient by the multipliers and stays differentiable almost
everywhere.

Crucially, ALM does NOT require strict feasibility of the iterates --
unlike a log-barrier -- so we can start from the input field as-is even
when many constraints are violated.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from dvfopt.core.iterative2d_tri_barrier import _tri_areas_flat, _tri_grad_T_v

NAME = 'aug_lagrangian'
DESCRIPTION = 'PHR augmented Lagrangian with L-BFGS-B inner; no active-set / interior-point fragility'


def _alm_objective(phi_flat, phi_init_flat, H, W, threshold, mu, rho,
                   anchor='l2', eps_l1=1e-4):
    """L_rho and its gradient at phi_flat.

    diff   = phi - phi_init
    f      = ||diff||^2 / 2          (or smooth L1, or zero)
    s_i    = c_i - threshold         (positive when feasible)
    psi_i  = max(0, mu_i - rho * s_i)
    L_rho  = f + (1/(2 rho)) * sum_i (psi_i^2 - mu_i^2)
    dL/dphi = df/dphi - sum_i psi_i  * dc_i/dphi
            = df/dphi - J^T psi
    """
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
    # ALM body
    val += float((psi * psi - mu * mu).sum()) / (2.0 * rho)
    # Gradient: -J^T psi (chain rule from c_i = T_i)
    grad = grad - _tri_grad_T_v(phi_flat, H, W, psi)
    return val, grad


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
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
          verbose: int = 0) -> dict:
    """PHR augmented Lagrangian; stops when feasible at threshold + margin."""
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
            _alm_objective, phi_flat, jac=True, method='L-BFGS-B',
            args=(phi_init_flat, H, W, threshold, mu, rho, anchor, 1e-4),
            options=dict(maxiter=inner_maxiter, ftol=ftol_inner, gtol=gtol_inner))
        phi_flat = res.x

        T = _tri_areas_flat(phi_flat, H, W)
        slack = T - threshold
        min_T = float(T.min())

        # Multiplier update.
        mu = np.maximum(0.0, mu - rho * slack)

        # Constraint-violation safeguard: tighten rho only if min progress.
        # We tighten when violation has NOT halved.
        viol_now = float(np.maximum(0.0, target - T).max())
        if outer > 0 and viol_now > 0.5 * log[-1]['viol']:
            rho = min(rho_max, rho * rho_growth)

        log.append(dict(outer=outer, inner_nit=int(res.nit),
                        min_T=min_T, viol=viol_now, rho=rho,
                        mu_max=float(mu.max()),
                        wall=time.time() - t0))
        if verbose:
            print(f'  ALM out={outer:3d}  inner={res.nit:4d}  '
                  f'min_T={min_T:+.5f}  viol={viol_now:.3e}  '
                  f'rho={rho:.1e}  mu_max={mu.max():.3e}  '
                  f'({time.time()-t0:.1f}s)', flush=True)
        last_min = min_T
        if min_T >= target:
            break

    phi_out = np.stack([phi_flat[:H * W].reshape(H, W),
                        phi_flat[H * W:].reshape(H, W)])
    return {'phi_out': phi_out,
            'info': dict(outer_used=len(log), rho_final=rho,
                         min_T_final=last_min,
                         feasible=last_min >= target,
                         log_first5=log[:5], log_last5=log[-5:])}
