"""TV-regularised correction (drop the L2-minimum principle).

The hypothesis underlying the wall is that no L2-bounded fold-free
correction exists for the dense slices. Replace the L2 anchor with a
TV (total-variation) anchor:

    R(phi - phi_init) = sum_i sqrt(|grad_x delta|^2 + |grad_y delta|^2 + eps^2)

TV tolerates jumps: a sharp local replacement in the fold core costs
only the length of the replacement boundary, not the volume integral.
So if the only way to satisfy the constraints is a large local change,
TV-regularised optimisation does it cheaply where L2 would resist.

Combined with the analytical 2-tri penalty/barrier, this gives the
same outer structure as ``iterative_2d_tri_barrier`` but a different
objective. Implementation: penalty -> barrier with L-BFGS-B; the TV
gradient is the analytical anisotropic-TV form with a smoothing eps.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from dvfopt.core.iterative2d_tri_barrier import _tri_areas_flat, _tri_grad_T_v

NAME = 'tv_anchor'
DESCRIPTION = 'TV anchor on (phi - phi_init) + 2-tri penalty/barrier; tolerates sharp local edits'


def _tv_value_grad(delta_flat, H, W, eps):
    """Smoothed anisotropic TV of (delta_y, delta_x) packed [dy_flat, dx_flat].
    Returns (value, gradient_same_shape)."""
    HW = H * W
    dy = delta_flat[:HW].reshape(H, W)
    dx = delta_flat[HW:].reshape(H, W)
    # Forward differences (with zero-flux at boundary).
    dy_x = np.zeros_like(dy); dy_x[:, :-1] = dy[:, 1:] - dy[:, :-1]
    dy_y = np.zeros_like(dy); dy_y[:-1, :] = dy[1:, :] - dy[:-1, :]
    dx_x = np.zeros_like(dx); dx_x[:, :-1] = dx[:, 1:] - dx[:, :-1]
    dx_y = np.zeros_like(dx); dx_y[:-1, :] = dx[1:, :] - dx[:-1, :]

    mag = np.sqrt(dy_x ** 2 + dy_y ** 2 + dx_x ** 2 + dx_y ** 2 + eps * eps)
    val = float((mag - eps).sum())

    # Gradient: TV is sum over pixels of mag_i; dmag_i/dphi requires chain
    # rule through forward diffs. Use the standard divergence formulation
    # (Chambolle): grad_TV = -div(grad / mag) for an isotropic 4-term TV.
    nyx_dy = dy_x / mag  # / forward x
    nyy_dy = dy_y / mag  # / forward y
    nxx_dx = dx_x / mag
    nxy_dx = dx_y / mag

    g_dy = np.zeros_like(dy)
    g_dx = np.zeros_like(dx)
    # negative divergence: for forward diff D_x, transpose is -D_x backward
    # d/dphi_{i,j} of sum mag = -(D_x^T n_x + D_y^T n_y)
    g_dy[:, :-1] -= -nyx_dy[:, :-1]  # +nyx[:, :-1]
    g_dy[:, 1:]  -= +nyx_dy[:, :-1]  # -nyx shifted
    g_dy[:-1, :] -= -nyy_dy[:-1, :]
    g_dy[1:, :]  -= +nyy_dy[:-1, :]
    g_dx[:, :-1] -= -nxx_dx[:, :-1]
    g_dx[:, 1:]  -= +nxx_dx[:, :-1]
    g_dx[:-1, :] -= -nxy_dx[:-1, :]
    g_dx[1:, :]  -= +nxy_dx[:-1, :]
    return val, np.concatenate([g_dy.ravel(), g_dx.ravel()])


def _objective(phi_flat, phi_init_flat, H, W, lam, threshold, margin,
               eps_tv, phase, mu=0.0):
    """phase in {'penalty', 'barrier'}."""
    delta = phi_flat - phi_init_flat
    tv_val, tv_grad = _tv_value_grad(delta, H, W, eps_tv)
    val = tv_val
    grad = tv_grad
    T = _tri_areas_flat(phi_flat, H, W)
    if phase == 'penalty':
        viol = np.maximum(0.0, threshold + margin - T)
        if viol.any():
            val += lam * float((viol * viol).sum())
            grad = grad - 2.0 * lam * _tri_grad_T_v(phi_flat, H, W, viol)
    else:  # 'barrier'
        s = T - threshold
        if (s <= 0).any():
            return np.inf, grad
        val += -mu * float(np.log(s).sum())
        grad = grad - mu * _tri_grad_T_v(phi_flat, H, W, 1.0 / s)
    return val, grad


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin: float = 1e-3,
          lam_schedule: tuple = (1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7),
          mu_schedule: tuple = (1e-1, 1e-2, 1e-3, 1e-4),
          eps_tv: float = 1e-3,
          inner_maxiter: int = 200,
          time_budget_s: float = 600.0,
          verbose: int = 0) -> dict:
    H, W = phi_in.shape[1], phi_in.shape[2]
    phi_init_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = phi_init_flat.copy()
    log = []
    t0 = time.time()

    for lam in lam_schedule:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(_objective, phi_flat, jac=True, method='L-BFGS-B',
                       args=(phi_init_flat, H, W, lam, threshold, margin,
                             eps_tv, 'penalty', 0.0),
                       options=dict(maxiter=inner_maxiter, ftol=1e-10,
                                    gtol=1e-7))
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        min_T = float(T.min())
        log.append(dict(phase='pen', lam=lam, min_T=min_T,
                        wall=time.time() - t0))
        if verbose:
            print(f'  TV pen lam={lam:.1e}  min_T={min_T:+.5f}  '
                  f'({time.time()-t0:.1f}s)', flush=True)
        if min_T > threshold + margin:
            break

    T = _tri_areas_flat(phi_flat, H, W)
    if T.min() > threshold:
        for mu in mu_schedule:
            if time.time() - t0 > time_budget_s:
                break
            res = minimize(_objective, phi_flat, jac=True, method='L-BFGS-B',
                           args=(phi_init_flat, H, W, lam_schedule[-1],
                                 threshold, margin, eps_tv, 'barrier', mu),
                           options=dict(maxiter=inner_maxiter, ftol=1e-10,
                                        gtol=1e-7))
            phi_flat = res.x
            T = _tri_areas_flat(phi_flat, H, W)
            log.append(dict(phase='bar', mu=mu, min_T=float(T.min()),
                            wall=time.time() - t0))
            if verbose:
                print(f'  TV bar mu={mu:.1e}  min_T={T.min():+.5f}  '
                      f'({time.time()-t0:.1f}s)', flush=True)

    phi_out = np.stack([phi_flat[:H * W].reshape(H, W),
                        phi_flat[H * W:].reshape(H, W)])
    return {'phi_out': phi_out,
            'info': {'log_last5': log[-5:],
                     'final_min_T': float(T.min())}}
