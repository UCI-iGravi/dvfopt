"""Quasi-conformal regularisation via Beltrami coefficient bounding.

For a 2D map :math:`f(z) = u(x,y) + i v(x,y)` the Beltrami coefficient is

.. math::

    \\mu(z) = \\frac{f_{\\bar z}}{f_z} = \\frac{(u_x - v_y) + i(v_x + u_y)}
                                              {(u_x + v_y) + i(v_x - u_y)}

and the map is orientation-preserving (i.e. Jdet > 0) iff
:math:`|\\mu(z)| < 1` everywhere. So instead of constraining the
Jacobian directly we constrain :math:`|\\mu(z)|^2 < 1`. The constraint
is smoother (no det involved) and explicitly tied to a conformality
quantity, which makes line searches cleaner.

Implementation: same penalty -> barrier L-BFGS structure as
``iterative_2d_tri_barrier`` but with the constraint replaced by the
shoelace Jdet (== ``|f_z|^2 - |f_{bar z}|^2``) reinterpreted through
the Beltrami inequality. We use forward differences to define the
Wirtinger derivatives consistently with the shoelace metric the field
itself is evaluated under.

Note: this is the central-diff Jdet form, not the 2-tri form. Useful as
a *reference field* the 2-tri solver can then polish from.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from dvfopt.jacobian.shoelace import _ref_grid
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

NAME = 'quasi_conformal'
DESCRIPTION = 'Bound the Beltrami coefficient |mu| < 1 (shoelace Jdet > 0 form), then 2-tri polish'


def _wirtinger_forward(phi_flat, H, W):
    """Return (fz, fzbar) -- complex Wirtinger derivatives via forward
    differences. fz, fzbar each have shape (H-1, W-1) (per-cell), complex.

    f = (x + dx) + i (y + dy)  in coordinate convention.
    f_x = 1 + dx_x + i dy_x
    f_y =     dx_y + i (1 + dy_y)
    f_z      = 0.5 (f_x - i f_y)
    f_zbar   = 0.5 (f_x + i f_y)
    """
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    # forward diffs on the (H-1, W-1) cell grid
    dx_x = dx[:-1, 1:] - dx[:-1, :-1]
    dy_x = dy[:-1, 1:] - dy[:-1, :-1]
    dx_y = dx[1:, :-1] - dx[:-1, :-1]
    dy_y = dy[1:, :-1] - dy[:-1, :-1]
    fx_re = 1.0 + dx_x; fx_im = dy_x
    fy_re = dx_y;       fy_im = 1.0 + dy_y
    # f_z = 0.5 (f_x - i f_y) = 0.5 ((fx_re + fy_im) + i (fx_im - fy_re))
    fz_re = 0.5 * (fx_re + fy_im); fz_im = 0.5 * (fx_im - fy_re)
    fzb_re = 0.5 * (fx_re - fy_im); fzb_im = 0.5 * (fx_im + fy_re)
    return fz_re, fz_im, fzb_re, fzb_im


def _mu_sq_minus_one(phi_flat, H, W):
    """|mu|^2 - 1  = (|f_zbar|^2 - |f_z|^2) / |f_z|^2. We use the *signed*
    form (|f_zbar|^2 - |f_z|^2) directly to avoid divide-by-zero at
    branch points; require this be < 0 (which is equivalent to |mu|<1
    when |f_z|^2 > 0). Returns (H-1, W-1)."""
    fz_re, fz_im, fzb_re, fzb_im = _wirtinger_forward(phi_flat, H, W)
    return (fzb_re ** 2 + fzb_im ** 2) - (fz_re ** 2 + fz_im ** 2)


def _qc_objective(phi_flat, phi_init_flat, H, W, lam, margin, anchor):
    """Phi -> 0.5||diff||^2 + lam * sum max(0, (|mu|^2 - 1) + margin)^2 ."""
    diff = phi_flat - phi_init_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff)
        grad = diff.copy()
    else:
        s = np.sqrt(diff * diff + 1e-8)
        val = float((s - 1e-4).sum())
        grad = diff / s
    g = _mu_sq_minus_one(phi_flat, H, W) + margin
    viol = np.maximum(0.0, g)
    if viol.any():
        val += lam * float((viol * viol).sum())
    # Gradient: numeric finite-difference fallback, vectorised by stencil
    # would be O(HW); for simplicity we let L-BFGS use the value-only call
    # with a tight tolerance and rely on a numerical gradient instead.
    # Actually, return value with finite-difference flag handled by caller.
    return val, grad


def _val_only(phi_flat, phi_init_flat, H, W, lam, margin, anchor):
    """Value-only callable so scipy uses its FD jac (avoid hand-deriving)."""
    diff = phi_flat - phi_init_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff)
    else:
        s = np.sqrt(diff * diff + 1e-8)
        val = float((s - 1e-4).sum())
    g = _mu_sq_minus_one(phi_flat, H, W) + margin
    viol = np.maximum(0.0, g)
    if viol.any():
        val += lam * float((viol * viol).sum())
    return val


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin_mu: float = 0.05,
          lam_schedule: tuple = (1.0, 1e2, 1e4, 1e6, 1e8),
          inner_maxiter: int = 150,
          do_2tri_polish: bool = True,
          time_budget_s: float = 600.0,
          verbose: int = 0) -> dict:
    """Bound |mu| < 1 - margin_mu. Optionally then run a short 2-tri barrier
    pass to convert the |mu| feasibility into 2-tri feasibility.

    The first phase uses scipy's FD gradient (no analytical mu-grad here;
    the FD column sweep is fine for moderate-size slices but is the
    speed bottleneck for large ones).
    """
    H, W = phi_in.shape[1], phi_in.shape[2]
    phi_init_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = phi_init_flat.copy()

    t0 = time.time()
    log = []
    for lam in lam_schedule:
        if time.time() - t0 > time_budget_s:
            break
        # Use jac=False -> scipy 2-point FD.
        res = minimize(_val_only, phi_flat, method='L-BFGS-B',
                       args=(phi_init_flat, H, W, lam, margin_mu, 'l2'),
                       jac=False,
                       options=dict(maxiter=inner_maxiter, ftol=1e-10, gtol=1e-7))
        phi_flat = res.x
        g = _mu_sq_minus_one(phi_flat, H, W) + margin_mu
        log.append(dict(lam=lam, max_g=float(g.max()),
                        wall=time.time() - t0))
        if verbose:
            print(f'  qc lam={lam:.1e}  max(|mu|^2-1+margin)={g.max():+.4f}'
                  f'  ({time.time()-t0:.1f}s)', flush=True)
        if g.max() <= 0:
            break

    phi_out = np.stack([phi_flat[:H * W].reshape(H, W),
                        phi_flat[H * W:].reshape(H, W)])

    info: dict = {'phase': 'qc-only', 'log': log[-5:]}
    if do_2tri_polish and time.time() - t0 < time_budget_s:
        # Hand off to torch 2-tri barrier as polish.
        try:
            from . import m05_torch_full_grid as torch_m
            polish = torch_m.solve(phi_out, threshold=threshold,
                                   margin=1e-3, anchor='l2',
                                   time_budget_s=max(60.0,
                                                     time_budget_s - (time.time() - t0)))
            phi_out = polish['phi_out']
            info = {'phase': 'qc+torch-polish',
                    'qc_log': log[-5:], 'polish': polish['info']}
        except Exception as exc:
            info['polish_error'] = str(exc)

    T1, T2 = _triangle_areas_2d(phi_out[0], phi_out[1])
    info['final_min_T'] = float(np.minimum(T1, T2).min())
    return {'phi_out': phi_out, 'info': info}
