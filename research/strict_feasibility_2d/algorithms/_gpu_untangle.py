"""Candidate faster 2D METHOD: first-order GPU untangler (whole-slice).

Rather than seed (harmonic+ALM, L-BFGS-B, the profiled bottleneck) + SLP
per cluster, untangle the ENTIRE slice at once with a first-order solver
on the GPU: minimize a smooth penalty energy

    E(phi) = MU * sum_k relu(tau + margin - A_k)^2
           +      sum   smooth_L1(phi - phi_in)

with Adam, MU annealed up. No clustering, no process pool, no L-BFGS-B
host loop — the whole (H,W) field updates in parallel each step. This is
the GPU-parallel analog of the penalty seed; the question is whether it
reaches feasibility fast and lands a comparable-L1 basin.

Torch 2-tri areas match dvfopt.core.tri_primitives.tri_areas_flat (verified
in the bench). DY_FIRST pack: phi (2, H, W) = [dy, dx].
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


def gpu_untangle_2d(phi_in_2hw, *, threshold=0.01, margin=2e-3, iters=4000,
                    lr=5e-3, mu0=1e3, mu_max=1e6, eps_l1=1e-3, device=None,
                    verbose=0):
    """Plain quadratic-penalty first-order untangler (baseline; plateaus)."""
    import torch

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    phi0 = torch.tensor(np.asarray(phi_in_2hw, np.float64), device=dev,
                        dtype=torch.float64)
    dy = phi0[0].clone().requires_grad_(True)
    dx = phi0[1].clone().requires_grad_(True)
    opt = torch.optim.Adam([dy, dx], lr=lr)
    tgt = threshold + margin
    mu = mu0
    for it in range(iters):
        opt.zero_grad()
        t1, t2 = _areas_torch(dy, dx, torch)
        viol = (torch.relu(tgt - t1) ** 2).sum() + (torch.relu(tgt - t2) ** 2).sum()
        data = torch.sqrt((dy - phi0[0]) ** 2 + eps_l1 ** 2).sum() \
            + torch.sqrt((dx - phi0[1]) ** 2 + eps_l1 ** 2).sum()
        (mu * viol + data).backward()
        opt.step()
        if (it + 1) % 500 == 0:
            mu = min(mu * 3, mu_max)
            if verbose:
                with torch.no_grad():
                    mn = min(t1.min().item(), t2.min().item())
                print(f'    [gpu it {it + 1}] min_A={mn:+.5f} mu={mu:.0e}',
                      flush=True)
    with torch.no_grad():
        out = torch.stack([dy, dx]).cpu().numpy()
    return out


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
    import torch

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    phi0 = torch.tensor(np.asarray(phi_in_2hw, np.float64), device=dev,
                        dtype=torch.float64)
    dy = phi0[0].clone().requires_grad_(True)
    dx = phi0[1].clone().requires_grad_(True)
    tgt = threshold + margin

    t1, t2 = _areas_torch(dy, dx, torch)
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


def gpu_untangle_full_2d(phi_in_2hw, *, threshold=0.01, margin=2e-3,
                         adam_outer=8, adam_inner=300, adam_lr=5e-3,
                         lbfgs_outer=40, lbfgs_inner=20, mu0=1e3, mu_max=1e9,
                         mu_grow=3.0, eps_l1=1e-3, device=None, verbose=0):
    """Hybrid GPU untangler aiming for FULL feasibility with no SLP mop-up.

    Phase 1 (Adam-ALM): cheap bulk untangling — clears the vast majority of
    folds fast, but plateaus on sliver-folds (coupled, ill-conditioned
    penalty landscape where Adam's diagonal preconditioning fails).
    Phase 2 (L-BFGS-ALM): quasi-Newton curvature + strong-Wolfe line search
    resolves the coupled slivers Adam cannot, driving the worst triangle
    area strictly above `threshold`. PHR multipliers (per triangle) keep
    feasibility without mu -> inf; a small margin ramp adds strict buffer.

    Returns (phi_out (2,H,W), info) with info['n_neg'] against `threshold`.
    """
    import torch

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    phi0 = torch.tensor(np.asarray(phi_in_2hw, np.float64), device=dev,
                        dtype=torch.float64)
    dy = phi0[0].clone().requires_grad_(True)
    dx = phi0[1].clone().requires_grad_(True)
    tgt = threshold + margin

    t1, t2 = _areas_torch(dy, dx, torch)
    lam1 = torch.zeros_like(t1)
    lam2 = torch.zeros_like(t2)
    mu = mu0

    def _penalty():
        a1, a2 = _areas_torch(dy, dx, torch)
        g1, g2 = a1 - tgt, a2 - tgt
        p1 = torch.where(g1 <= lam1 / mu, -lam1 * g1 + 0.5 * mu * g1 * g1,
                         -0.5 * lam1 * lam1 / mu)
        p2 = torch.where(g2 <= lam2 / mu, -lam2 * g2 + 0.5 * mu * g2 * g2,
                         -0.5 * lam2 * lam2 / mu)
        data = torch.sqrt((dy - phi0[0]) ** 2 + eps_l1 ** 2).sum() \
            + torch.sqrt((dx - phi0[1]) ** 2 + eps_l1 ** 2).sum()
        return p1.sum() + p2.sum() + data, g1, g2

    def _dual_update():
        nonlocal lam1, lam2, mu, prev_worst
        with torch.no_grad():
            a1, a2 = _areas_torch(dy, dx, torch)
            g1, g2 = a1 - tgt, a2 - tgt
            lam1 = torch.clamp(lam1 - mu * g1, min=0.0)
            lam2 = torch.clamp(lam2 - mu * g2, min=0.0)
            worst = float(min(g1.min().item(), g2.min().item()))
            nneg = int((a1 < threshold).sum() + (a2 < threshold).sum())
            if prev_worst is not None and worst < prev_worst + 1e-4:
                mu = min(mu * mu_grow, mu_max)
            prev_worst = worst
            return worst, nneg

    prev_worst = None
    # ---- Phase 1: Adam-ALM (bulk) ----
    for outer in range(adam_outer):
        opt = torch.optim.Adam([dy, dx], lr=adam_lr)
        for _ in range(adam_inner):
            opt.zero_grad()
            loss, _, _ = _penalty()
            loss.backward()
            opt.step()
        worst, nneg = _dual_update()
        if verbose:
            print(f'    [adam {outer + 1}] worst_g={worst:+.5f} folds={nneg} '
                  f'mu={mu:.0e}', flush=True)
        if nneg == 0:
            break

    # ---- Phase 2: L-BFGS-ALM (sliver polish to full feasibility) ----
    for outer in range(lbfgs_outer):
        opt = torch.optim.LBFGS([dy, dx], max_iter=lbfgs_inner,
                                history_size=20, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-14, tolerance_change=1e-18)

        def closure():
            opt.zero_grad()
            loss, _, _ = _penalty()
            loss.backward()
            return loss

        opt.step(closure)
        worst, nneg = _dual_update()
        if verbose:
            print(f'    [lbfgs {outer + 1}] worst_g={worst:+.5f} folds={nneg} '
                  f'mu={mu:.0e}', flush=True)
        if nneg == 0:
            break

    with torch.no_grad():
        out = torch.stack([dy, dx]).cpu().numpy()
    a1, a2 = _areas_torch(dy, dx, torch)
    nneg = int((a1 < threshold).sum() + (a2 < threshold).sum())
    return out, {'n_neg': nneg}


def gpu_barrier_untangle_2d(phi_in_2hw, *, threshold=0.01, t0=1.0, t_max=1e7,
                            t_grow=2.5, inner=150, lr=0.2, eps_l1=1e-3,
                            w_data=1.0, device=None, verbose=0):
    """Interior-point (log-barrier homotopy) GPU untangler — feasible by
    construction at every iterate.

    Start from the identity displacement (phi=0 -> all triangle areas 0.5,
    strictly feasible). Minimize a homotopy

        F_t(phi) = w_data * smooth_L1(phi - phi_in)
                 - (1/t) * sum_k log(A_k - threshold)

    for increasing t. The barrier -log(A_k - threshold) diverges as any
    area approaches the fold boundary, so a feasibility-guarded backtracking
    line search keeps A_k > threshold for ALL k at every step. As t -> inf
    the barrier vanishes and phi converges to the L1-closest strictly
    feasible field to phi_in. Guaranteed 0 folds (no SLP mop-up needed).

    Returns (phi_out (2,H,W), info) with info['n_neg'], info['min_area'].
    """
    import torch

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    phi_in = torch.tensor(np.asarray(phi_in_2hw, np.float64), device=dev,
                          dtype=torch.float64)
    # Feasible start: identity map (zero displacement) -> areas == 0.5.
    dy = torch.zeros_like(phi_in[0]).requires_grad_(True)
    dx = torch.zeros_like(phi_in[1]).requires_grad_(True)

    def _obj(t):
        a1, a2 = _areas_torch(dy, dx, torch)
        amin = min(a1.min().item(), a2.min().item())
        if amin <= threshold:
            return None, amin  # infeasible -> caller backtracks
        bar = -(torch.log(a1 - threshold).sum() + torch.log(a2 - threshold).sum())
        data = torch.sqrt((dy - phi_in[0]) ** 2 + eps_l1 ** 2).sum() \
            + torch.sqrt((dx - phi_in[1]) ** 2 + eps_l1 ** 2).sum()
        return w_data * data + bar / t, amin

    t = t0
    while t <= t_max:
        for _ in range(inner):
            if dy.grad is not None:
                dy.grad = None
                dx.grad = None
            loss, amin = _obj(t)
            loss.backward()
            with torch.no_grad():
                gy, gx = dy.grad, dx.grad
                gn = torch.sqrt((gy * gy).sum() + (gx * gx).sum()).item()
                if gn < 1e-10:
                    break
                step = lr
                base_dy, base_dx = dy.clone(), dx.clone()
                base_loss = loss.item()
                # feasibility-guarded backtracking (Armijo-lite)
                for _bt in range(40):
                    dy.copy_(base_dy - step * gy)
                    dx.copy_(base_dx - step * gx)
                    a1, a2 = _areas_torch(dy, dx, torch)
                    if a1.min().item() > threshold and a2.min().item() > threshold:
                        nl, _ = _obj(t)
                        if nl is not None and nl.item() <= base_loss + 1e-9:
                            break
                    step *= 0.5
                else:
                    dy.copy_(base_dy)
                    dx.copy_(base_dx)
                    break
        if verbose:
            with torch.no_grad():
                a1, a2 = _areas_torch(dy, dx, torch)
                amn = min(a1.min().item(), a2.min().item())
                l1 = (torch.abs(dy - phi_in[0]).sum()
                      + torch.abs(dx - phi_in[1]).sum()).item()
            print(f'    [barrier t={t:.1e}] min_area={amn:+.5f} L1={l1:.1f}',
                  flush=True)
        t *= t_grow

    with torch.no_grad():
        out = torch.stack([dy, dx]).cpu().numpy()
        a1, a2 = _areas_torch(dy, dx, torch)
        amn = min(a1.min().item(), a2.min().item())
        nneg = int((a1 < threshold).sum() + (a2 < threshold).sum())
    return out, {'n_neg': nneg, 'min_area': amn}


def gpu_shifted_barrier_untangle_2d(phi_in_2hw, *, threshold=0.01, buffer=0.05,
                                    s_decay=0.6, t0=1.0, t_grow=2.0, inner=150,
                                    lr=0.2, eps_l1=1e-3, w_data=1.0,
                                    max_outer=60, device=None, verbose=0):
    """Shifted (relaxed) interior-point GPU untangler starting FROM phi_in.

    A plain log barrier must start feasible (identity), which lands in a
    high-L1 basin. This starts from phi_in (the low-L1 basin) using a
    per-triangle slack s_k so the shifted margin A_k - threshold + s_k > 0
    holds even at the folded start, then anneals s_k -> 0. As the slacks
    vanish the barrier enforces strict A_k >= threshold, while the data
    term keeps phi near phi_in -> feasible AND low L1.

        F(phi) = w_data * smooth_L1(phi - phi_in)
               - (1/t) * sum_k log(A_k - threshold + s_k)

    Feasibility-guarded backtracking keeps every shifted margin positive.
    Returns (phi_out (2,H,W), info) with info['n_neg'], info['min_area'].
    """
    import torch

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    phi_in = torch.tensor(np.asarray(phi_in_2hw, np.float64), device=dev,
                          dtype=torch.float64)
    dy = phi_in[0].clone().requires_grad_(True)
    dx = phi_in[1].clone().requires_grad_(True)

    with torch.no_grad():
        a1, a2 = _areas_torch(dy, dx, torch)
        # slack makes the shifted margin == buffer at the (folded) start.
        s1 = torch.clamp(threshold - a1, min=0.0) + buffer
        s2 = torch.clamp(threshold - a2, min=0.0) + buffer

    def _margins():
        a1, a2 = _areas_torch(dy, dx, torch)
        return a1 - threshold + s1, a2 - threshold + s2

    def _obj(t):
        m1, m2 = _margins()
        bar = -(torch.log(m1).sum() + torch.log(m2).sum())
        data = torch.sqrt((dy - phi_in[0]) ** 2 + eps_l1 ** 2).sum() \
            + torch.sqrt((dx - phi_in[1]) ** 2 + eps_l1 ** 2).sum()
        return w_data * data + bar / t

    # Manual Adam state — Adam's per-parameter normalization keeps the step
    # ~lr despite the stiff barrier gradient (plain GD collapses to
    # microscopic steps here). Feasibility-guarded step scaling keeps every
    # shifted margin positive.
    b1, b2, adam_eps = 0.9, 0.999, 1e-8
    t = t0
    for outer in range(max_outer):
        my = torch.zeros_like(dy)
        vy = torch.zeros_like(dy)
        mx = torch.zeros_like(dx)
        vx = torch.zeros_like(dx)
        for k in range(1, inner + 1):
            if dy.grad is not None:
                dy.grad = None
                dx.grad = None
            loss = _obj(t)
            loss.backward()
            with torch.no_grad():
                gy, gx = dy.grad, dx.grad
                my = b1 * my + (1 - b1) * gy
                vy = b2 * vy + (1 - b2) * gy * gy
                mx = b1 * mx + (1 - b1) * gx
                vx = b2 * vx + (1 - b2) * gx * gx
                mhy = my / (1 - b1 ** k)
                vhy = vy / (1 - b2 ** k)
                mhx = mx / (1 - b1 ** k)
                vhx = vx / (1 - b2 ** k)
                dely = lr * mhy / (torch.sqrt(vhy) + adam_eps)
                delx = lr * mhx / (torch.sqrt(vhx) + adam_eps)
                base_dy, base_dx = dy.clone(), dx.clone()
                alpha = 1.0
                for _bt in range(50):
                    dy.copy_(base_dy - alpha * dely)
                    dx.copy_(base_dx - alpha * delx)
                    m1, m2 = _margins()
                    if m1.min().item() > 0 and m2.min().item() > 0:
                        break
                    alpha *= 0.5
                else:
                    dy.copy_(base_dy)
                    dx.copy_(base_dx)
        # Anneal slacks toward 0, but floored so the shifted margin stays
        # positive at the CURRENT phi. A still-folded triangle keeps s_k ~
        # (threshold - A_k), so its margin stays tiny and the barrier keeps
        # pushing it feasible; s_k can only vanish once A_k >= threshold.
        # This couples slack decay to real untangling progress.
        with torch.no_grad():
            a1, a2 = _areas_torch(dy, dx, torch)
            floor1 = torch.clamp(threshold - a1, min=0.0) + 1e-6
            floor2 = torch.clamp(threshold - a2, min=0.0) + 1e-6
            s1 = torch.maximum(s1 * s_decay, floor1)
            s2 = torch.maximum(s2 * s_decay, floor2)
            amn = min(a1.min().item(), a2.min().item())
            smax = max(s1.max().item(), s2.max().item())
            nneg = int((a1 < threshold).sum() + (a2 < threshold).sum())
        if verbose:
            with torch.no_grad():
                l1 = (torch.abs(dy - phi_in[0]).sum()
                      + torch.abs(dx - phi_in[1]).sum()).item()
            print(f'    [shift {outer + 1} t={t:.1e}] min_area={amn:+.5f} '
                  f'folds={nneg} s_max={smax:.4f} L1={l1:.1f}', flush=True)
        t *= t_grow
        if smax < 1e-4 and nneg == 0:
            break

    with torch.no_grad():
        out = torch.stack([dy, dx]).cpu().numpy()
        a1, a2 = _areas_torch(dy, dx, torch)
        amn = min(a1.min().item(), a2.min().item())
        nneg = int((a1 < threshold).sum() + (a2 < threshold).sum())
    return out, {'n_neg': nneg, 'min_area': amn}


__all__ = ['gpu_untangle_2d', 'gpu_untangle_alm_2d', 'gpu_untangle_full_2d',
           'gpu_barrier_untangle_2d', 'gpu_shifted_barrier_untangle_2d']
