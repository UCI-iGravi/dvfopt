"""Whole-slice GPU PHR augmented-Lagrangian 2D untangler.

A first-order, whole-slice untangler that resolves negative Jacobian
folds by minimizing a smooth PHR (Powell-Hestenes-Rockafellar)
augmented-Lagrangian energy on the GPU (or CPU) with Adam. Rather than
seed + SLP per cluster, the ENTIRE (H, W) field updates in parallel each
step; per-triangle multipliers escape the plain quadratic-penalty plateau.

Pack convention: DY_FIRST — ``phi (2, H, W) = [dy, dx]`` (matches the
2-triangle solver family).

The torch triangle-area kernel :func:`_areas_torch` matches
``dvfopt.jacobian.triangle_sign._triangle_areas_2d`` /
``dvfopt.core.tri_primitives.tri_areas_flat`` EXACTLY — including the
leading ``-0.5`` sign — with T1: A=TR, B=BL, C=BR and T2: A=TL, B=BL,
C=TR.

Intended as a low-L1 *seed* for the SLP solver on dense slices: it clears
the bulk of the folds fast but plateaus on coupled sliver-folds, so it is
not a standalone feasibility solver.

``torch`` is an OPTIONAL dependency (the ``benchmarks`` extra); it is
imported lazily INSIDE :func:`gpu_untangle_alm_2d`, never at module top
level.
"""

from __future__ import annotations

import numpy as np


def _areas_torch(dy, dx, torch):
    """Per-triangle signed areas (T1, T2) matching tri_areas_flat."""
    H, W = dy.shape
    ii = torch.arange(H, device=dy.device, dtype=dy.dtype)[:, None]
    jj = torch.arange(W, device=dy.device, dtype=dy.dtype)[None, :]
    Y = ii + dy
    X = jj + dx
    y_tl, x_tl = Y[:-1, :-1], X[:-1, :-1]
    y_tr, x_tr = Y[:-1, 1:], X[:-1, 1:]
    y_bl, x_bl = Y[1:, :-1], X[1:, :-1]
    y_br, x_br = Y[1:, 1:], X[1:, 1:]
    # Matches dvfopt.jacobian.triangle_sign._triangle_areas_2d exactly
    # (note the leading -0.5): T1 A=TR,B=BL,C=BR ; T2 A=TL,B=BL,C=TR.
    t1 = -0.5 * ((x_bl - x_tr) * (y_br - y_tr) - (x_br - x_tr) * (y_bl - y_tr))
    t2 = -0.5 * ((x_bl - x_tl) * (y_tr - y_tl) - (x_tr - x_tl) * (y_bl - y_tl))
    return t1, t2


def gpu_untangle_alm_2d(phi_in_2hw, *, threshold=0.01, margin=2e-3,
                        n_outer=40, n_inner=300, lr=5e-3, mu0=1e3, mu_max=1e8,
                        mu_grow=3.0, eps_l1=1e-3, device=None, verbose=0):
    """Whole-slice GPU untangler with a PHR augmented Lagrangian.

    Per-triangle multipliers escape the quadratic-penalty plateau: the
    constraint g_k = A_k - tgt >= 0 is enforced via
      psi(g, lam, mu) = -lam*g + (mu/2) g^2   if g <= lam/mu   else  -lam^2/(2mu)
    with multiplier update lam <- max(0, lam - mu*g) each outer step and mu
    grown only when the worst violation stalls. Data term = smooth-L1 to
    input. Returns feasible-or-closest phi (2, H, W).
    """
    phi_np = np.asarray(phi_in_2hw, np.float64)
    H, W = phi_np.shape[1], phi_np.shape[2]
    if H < 2 or W < 2:
        # Degenerate slice: no 2x2 cell exists, so there are ZERO triangles
        # and the field is trivially feasible. Return the input unchanged
        # before any torch work — otherwise _areas_torch yields empty
        # tensors and g.min() raises RuntimeError after burning a full
        # inner loop of Adam steps.
        return phi_np.copy()

    import torch

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    phi0 = torch.tensor(phi_np, device=dev, dtype=torch.float64)
    dy = phi0[0].clone().requires_grad_(True)
    dx = phi0[1].clone().requires_grad_(True)
    tgt = threshold + margin

    t1, t2 = _areas_torch(dy, dx, torch)
    # Upfront no-op check at plain `threshold` (NOT tgt = threshold+margin):
    # the downstream SLP only needs min area >= threshold, and returning the
    # input unchanged is exactly L1-optimal, so an already-feasible slice
    # must not burn n_inner Adam steps chasing the extra `margin`. The
    # margin only matters once the field has to MOVE (it buys the SLP's
    # linearisation some slack), which is why the in-loop exit below keeps
    # the stricter `worst >= 0` (i.e. min area >= tgt) condition: exiting
    # mid-run at plain threshold would hand SLP a moved field without the
    # margin the seed intends.
    if float(min(t1.min().item(), t2.min().item())) >= threshold:
        return phi_np.copy()
    lam1 = torch.zeros_like(t1)
    lam2 = torch.zeros_like(t2)
    mu = mu0
    prev_worst = None
    for outer in range(n_outer):
        opt = torch.optim.Adam([dy, dx], lr=lr)
        for _ in range(n_inner):
            opt.zero_grad()
            t1, t2 = _areas_torch(dy, dx, torch)
            g1, g2 = t1 - tgt, t2 - tgt
            p1 = torch.where(g1 <= lam1 / mu, -lam1 * g1 + 0.5 * mu * g1 * g1,
                             -0.5 * lam1 * lam1 / mu)
            p2 = torch.where(g2 <= lam2 / mu, -lam2 * g2 + 0.5 * mu * g2 * g2,
                             -0.5 * lam2 * lam2 / mu)
            data = torch.sqrt((dy - phi0[0]) ** 2 + eps_l1 ** 2).sum() \
                + torch.sqrt((dx - phi0[1]) ** 2 + eps_l1 ** 2).sum()
            (p1.sum() + p2.sum() + data).backward()
            opt.step()
        with torch.no_grad():
            t1, t2 = _areas_torch(dy, dx, torch)
            g1, g2 = t1 - tgt, t2 - tgt
            lam1 = torch.clamp(lam1 - mu * g1, min=0.0)
            lam2 = torch.clamp(lam2 - mu * g2, min=0.0)
            worst = float(min(g1.min().item(), g2.min().item()))
            if prev_worst is not None and worst < prev_worst + 1e-4:
                mu = min(mu * mu_grow, mu_max)
            prev_worst = worst
            if verbose:
                nneg = int((t1 < threshold).sum() + (t2 < threshold).sum())
                print(f'    [alm outer {outer + 1}] worst_g={worst:+.5f} '
                      f'folds~{nneg} mu={mu:.0e}', flush=True)
            if worst >= 0.0:
                break
    with torch.no_grad():
        out = torch.stack([dy, dx]).cpu().numpy()
    return out


__all__ = ['gpu_untangle_alm_2d']
