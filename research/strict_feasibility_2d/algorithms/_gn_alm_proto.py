"""PROTOTYPE: Gauss-Newton inner step for the 2-tri PHR augmented Lagrangian.

The profiling verdict was that both pipelines are L-BFGS-B-iteration-bound
(`setulb` 35-48%). L-BFGS-B rebuilds curvature from scratch and takes
thousands of steps. This prototype replaces the ALM inner solve with a
sparse Gauss-Newton step that USES the known structure:

    inner subproblem (fixed mu, rho):
        min_phi  f(phi) + (1/2rho) * sum(psi_i^2 - mu_i^2),
        psi_i = max(0, mu_i - rho (T_i - tau))
    gradient:  g = grad_f - J^T psi
    GN Hessian (drop 2nd-order d^2T): H = grad^2_f + rho * Ja^T Ja
        with Ja = rows of J for ACTIVE constraints (psi_i > 0)
    step:      solve (I + rho Ja^T Ja) delta = -g   (sparse SPD)  + Armijo

Same PHR outer loop (mu/rho updates, Birgin-Martinez) as
``augmented_lagrangian_2d`` so it is a drop-in comparison: only the inner
optimiser changes (L-BFGS-B -> sparse GN). DY_FIRST pack throughout.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from dvfopt.core.tri_primitives import tri_areas_flat, tri_grad_T_v
from research.strict_feasibility_2d.algorithms.tri_linearize import (
    build_sparse_jacobian_T,
)


def _L_rho(phi, anchor, H, W, threshold, mu, rho):
    diff = phi - anchor
    val = 0.5 * float(diff @ diff)
    T = tri_areas_flat(phi, H, W)
    psi = np.maximum(0.0, mu - rho * (T - threshold))
    val += float((psi * psi - mu * mu).sum()) / (2.0 * rho)
    return val, T


def augmented_lagrangian_2d_gn(
    phi_in_2hw,
    *,
    threshold=0.01,
    margin=1e-3,
    rho_init=1.0,
    rho_growth=5.0,
    rho_max=1e8,
    outer_max=60,
    inner_max=25,
    gtol=1e-7,
    verbose=0,
):
    H, W = phi_in_2hw.shape[1:]
    anchor = np.concatenate([phi_in_2hw[0].ravel(), phi_in_2hw[1].ravel()])  # DY_FIRST
    phi = anchor.copy()
    n_var = 2 * H * W
    I = sp.eye(n_var, format='csc')
    mu = np.zeros(2 * (H - 1) * (W - 1))
    rho = rho_init
    target = threshold + margin
    last_viol = np.inf
    total_inner = 0

    for outer in range(outer_max):
        # ---- GN inner solve for fixed (mu, rho) ----
        for _ in range(inner_max):
            T = tri_areas_flat(phi, H, W)
            psi = np.maximum(0.0, mu - rho * (T - threshold))
            g = (phi - anchor) - tri_grad_T_v(phi, H, W, psi)
            if np.max(np.abs(g)) < gtol:
                break
            J = build_sparse_jacobian_T(phi, H, W).tocsr()
            active = psi > 0.0
            if active.any():
                Ja = J[active]
                Hgn = (I + rho * (Ja.T @ Ja)).tocsc()
            else:
                Hgn = I
            delta = spsolve(Hgn, -g)
            # Armijo backtracking on L_rho.
            f0, _ = _L_rho(phi, anchor, H, W, threshold, mu, rho)
            slope = float(g @ delta)
            alpha = 1.0
            while alpha > 1e-6:
                f1, _ = _L_rho(phi + alpha * delta, anchor, H, W, threshold, mu, rho)
                if f1 <= f0 + 1e-4 * alpha * slope:
                    break
                alpha *= 0.5
            phi = phi + alpha * delta
            total_inner += 1

        # ---- PHR outer update ----
        T = tri_areas_flat(phi, H, W)
        mu = np.maximum(0.0, mu - rho * (T - threshold))
        viol = float(np.maximum(0.0, target - T).max())
        if outer > 0 and viol > 0.5 * last_viol:
            rho = min(rho_max, rho * rho_growth)
        last_viol = viol
        if verbose:
            print(f'  GN-ALM out={outer:3d} inner_tot={total_inner:4d} '
                  f'min_T={float(T.min()):+.5f} viol={viol:.2e} rho={rho:.1e}',
                  flush=True)
        if float(T.min()) >= target:
            break

    dy = phi[:H * W].reshape(H, W)
    dx = phi[H * W:].reshape(H, W)
    out = np.stack([dy, dx])
    return out, {'total_inner': total_inner, 'outer_used': outer + 1,
                 'min_T': float(tri_areas_flat(phi, H, W).min())}


__all__ = ['augmented_lagrangian_2d_gn']
