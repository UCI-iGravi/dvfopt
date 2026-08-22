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

from dvfopt._logging import log_info
from dvfopt.core.primitives.tri import (
    tri_areas_flat as _tri_areas_flat,
)
from dvfopt.core.primitives.tri import (
    tri_grad_T_v as _tri_grad_T_v,
)

# Optional fused JIT path for the ALM inner objective — the hottest
# path of the m10/m14 seed stages (executed ~1e5 times per slice
# inside L-BFGS-B). The fused kernel folds the anchor, the forward
# T-area geometry, and the psi-weighted adjoint scatter into a single
# pass over cells (the legacy path walks the identical deformed-corner
# geometry twice: once in tri_areas_flat, once in tri_grad_T_v, plus
# ~0.7 ms of diff/copy/concat temporaries per call). Auto-detected at
# import, mirroring `_l2_refine._soft_pen_l1_fused_kernel`.
try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore

_ANCHOR_FLAGS = {'l2': 2, 'l1': 1}  # anything else -> 0 ('none')


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False)
    def _alm_fused_kernel(
        dy, dx, dy_in, dx_in, mu1, mu2, H, W, threshold, rho, anchor_flag, eps_l1
    ):
        """Fused anchor + T-areas + PHR co-vector + adjoint scatter.

        Returns ``(val, g_dy, g_dx)``. The PHR term is accumulated as
        ``psi_i^2 / (2 rho)`` over ACTIVE cells only (``psi_i > 0``);
        the caller must add the precomputed constant
        ``-sum(mu^2) / (2 rho)`` (over ALL cells) to recover the legacy
        value ``sum(psi^2 - mu^2) / (2 rho)`` exactly — for an inactive
        cell ``psi = 0``, so its legacy contribution is exactly
        ``-mu_i^2 / (2 rho)``, which the constant covers.

        Gradient: ``grad = anchor_grad - J^T psi``; the scatter uses
        coefficient ``-0.5 * psi`` with the same corner/sign pattern as
        ``tri_primitives._tri_grad_T_v_numba_kernel`` (whose J^T v
        coefficient is ``+0.5 * v``). Inactive cells (psi <= 0) scatter
        nothing, matching ``max(0, .)`` in the legacy path.
        ``anchor_flag``: 2 = l2, 1 = smoothed l1, 0 = none."""
        g_dy = np.zeros((H, W))
        g_dx = np.zeros((H, W))
        val = 0.0
        # Anchor on every grid vertex.
        if anchor_flag == 2:
            for i in range(H):
                for j in range(W):
                    diff_y = dy[i, j] - dy_in[i, j]
                    diff_x = dx[i, j] - dx_in[i, j]
                    val += 0.5 * (diff_y * diff_y + diff_x * diff_x)
                    g_dy[i, j] = diff_y
                    g_dx[i, j] = diff_x
        elif anchor_flag == 1:
            for i in range(H):
                for j in range(W):
                    diff_y = dy[i, j] - dy_in[i, j]
                    diff_x = dx[i, j] - dx_in[i, j]
                    s_y = np.sqrt(diff_y * diff_y + eps_l1 * eps_l1)
                    s_x = np.sqrt(diff_x * diff_x + eps_l1 * eps_l1)
                    val += (s_y - eps_l1) + (s_x - eps_l1)
                    g_dy[i, j] = diff_y / s_y
                    g_dx[i, j] = diff_x / s_x
        # PHR term + adjoint in one pass over cells.
        inv_2rho = 1.0 / (2.0 * rho)
        for i in range(H - 1):
            for j in range(W - 1):
                # Deformed positions of the four cell corners
                # (ref_y[i, j] = i, ref_x[i, j] = j — unit grid).
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
                psi1 = mu1[i, j] - rho * (T1 - threshold)
                psi2 = mu2[i, j] - rho * (T2 - threshold)
                if psi1 <= 0.0 and psi2 <= 0.0:
                    continue
                if psi1 > 0.0:
                    val += psi1 * psi1 * inv_2rho
                    c1 = -0.5 * psi1
                    g_dx[i, j + 1] += c1 * (y_br - y_bl)
                    g_dy[i, j + 1] += c1 * (x_bl - x_br)
                    g_dx[i + 1, j] += -c1 * (y_br - y_tr)
                    g_dy[i + 1, j] += c1 * (x_br - x_tr)
                    g_dx[i + 1, j + 1] += c1 * (y_bl - y_tr)
                    g_dy[i + 1, j + 1] += -c1 * (x_bl - x_tr)
                if psi2 > 0.0:
                    val += psi2 * psi2 * inv_2rho
                    c2 = -0.5 * psi2
                    g_dx[i, j] += c2 * (y_tr - y_bl)
                    g_dy[i, j] += c2 * (x_bl - x_tr)
                    g_dx[i + 1, j] += -c2 * (y_tr - y_tl)
                    g_dy[i + 1, j] += c2 * (x_tr - x_tl)
                    g_dx[i, j + 1] += c2 * (y_bl - y_tl)
                    g_dy[i, j + 1] += -c2 * (x_bl - x_tl)
        return val, g_dy, g_dx


def _alm_objective_ref(phi_flat, phi_init_flat, H, W, threshold, mu, rho, anchor='l2', eps_l1=1e-4):
    """Legacy two-pass numpy objective. Kept as the no-numba fallback
    and as the equivalence reference for the fused kernel tests."""
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


def _alm_objective(
    phi_flat, phi_init_flat, H, W, threshold, mu, rho, anchor='l2', eps_l1=1e-4, mu_sq_const=None
):
    """PHR-ALM inner objective ``(value, gradient)``.

    Dispatches to the fused Numba kernel when available. ``mu_sq_const``
    is the precomputed ``-sum(mu^2) / (2 rho)`` scalar — ``mu`` and
    ``rho`` are fixed within one inner L-BFGS-B solve, so
    :func:`augmented_lagrangian_2d` computes it once per outer
    iteration and passes it in; standalone callers may omit it."""
    if not _HAVE_NUMBA:
        return _alm_objective_ref(phi_flat, phi_init_flat, H, W, threshold, mu, rho, anchor, eps_l1)
    if mu_sq_const is None:
        mu_sq_const = -float(mu @ mu) / (2.0 * rho)
    HW = H * W
    n_cells = (H - 1) * (W - 1)
    dy = np.ascontiguousarray(phi_flat[:HW].reshape(H, W))
    dx = np.ascontiguousarray(phi_flat[HW:].reshape(H, W))
    dy_in = np.ascontiguousarray(phi_init_flat[:HW].reshape(H, W))
    dx_in = np.ascontiguousarray(phi_init_flat[HW:].reshape(H, W))
    mu1 = np.ascontiguousarray(mu[:n_cells].reshape(H - 1, W - 1))
    mu2 = np.ascontiguousarray(mu[n_cells:].reshape(H - 1, W - 1))
    val, g_dy, g_dx = _alm_fused_kernel(
        dy,
        dx,
        dy_in,
        dx_in,
        mu1,
        mu2,
        H,
        W,
        threshold,
        rho,
        _ANCHOR_FLAGS.get(anchor, 0),
        eps_l1,
    )
    return val + mu_sq_const, np.concatenate([g_dy.ravel(), g_dx.ravel()])


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
        # mu and rho are fixed within the inner solve, so the
        # -sum(mu^2)/(2 rho) part of the PHR value is a constant —
        # precompute it once here instead of per objective call (the
        # fused kernel only accumulates psi^2/(2 rho) over active cells).
        mu_sq_const = -float(mu @ mu) / (2.0 * rho)
        res = minimize(
            _alm_objective,
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            args=(phi_init_flat, H, W, threshold, mu, rho, anchor, 1e-4, mu_sq_const),
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
            log_info(
                f'  ALM out={outer:3d}  inner={res.nit:4d}  '
                f'min_T={min_T:+.5f}  viol={viol_now:.3e}  '
                f'rho={rho:.1e}  ({time.time() - t0:.1f}s)',
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
