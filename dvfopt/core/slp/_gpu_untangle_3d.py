"""Whole-volume GPU PHR augmented-Lagrangian 3D (simplex (3D)) untangler.

3D analogue of ``dvfopt.core.slp._gpu_untangle.gpu_untangle_alm_2d`` — the
2D accuracy='max' GPU seed. A first-order, whole-volume untangler that
resolves negative Kuhn-tet volumes by minimizing a smooth PHR
(Powell-Hestenes-Rockafellar) augmented-Lagrangian energy with Adam on the
GPU (or CPU). Rather than seed + solve per cluster, the ENTIRE (D, H, W)
field updates in parallel each step; per-tet multipliers escape the plain
quadratic-penalty plateau.

Channel convention: ``phi (3, D, H, W) = [dz, dy, dx]`` (the package-wide
3D array convention). ALL THREE channels are free — this is true 3D, no
``dz == 0`` restriction.

The tet volumes come from the canonical torch kernel
:func:`dvfopt.jacobian.tetrahedron_sign_torch.six_tet_volumes_3d_torch`
(the single torch-side source of the Kuhn decomposition), whose parity
with the numpy kernel is pinned to 1e-13 by the torch-gated tests in
``tests/test_tetrahedron_sign.py``.

Intended as a low-L1 *seed* for the exact-feasibility 3D pipeline
(``correct_dvf_3d``) on dense chunks: like its 2D sibling it clears the
bulk of the folds fast but plateaus on coupled sliver-tets, so it is NOT a
standalone feasibility solver.

``torch`` is an OPTIONAL dependency; it is imported lazily INSIDE
:func:`gpu_untangle_alm_3d`, never at module top level.
"""

from __future__ import annotations

import numpy as np

# Single source of truth for the Kuhn decomposition: same tables the numpy
# kernel uses. NOT re-derived here.
from dvfopt._logging import log_info


def _tet_volumes_torch(dz, dy, dx, torch):
    """Per-cell six signed Kuhn tet volumes, matching six_tet_volumes_3d.

    Thin delegate to the canonical torch kernel
    :func:`dvfopt.jacobian.tetrahedron_sign_torch.six_tet_volumes_3d_torch`
    (single source of the Kuhn decomposition on the torch side — parity
    with the numpy kernel is pinned by the torch-gated tests in
    ``tests/test_tetrahedron_sign.py``). ``torch.stack`` preserves
    autograd through the channel leaves.
    """
    from dvfopt.jacobian.tetrahedron_sign_torch import six_tet_volumes_3d_torch

    return six_tet_volumes_3d_torch(torch.stack([dz, dy, dx]))


def gpu_untangle_alm_3d(
    phi_in_3dhw,
    *,
    threshold=0.01,
    margin=2e-3,
    n_outer=40,
    n_inner=300,
    lr=5e-3,
    mu0=1e3,
    mu_max=1e8,
    mu_grow=3.0,
    eps_l1=1e-3,
    device=None,
    verbose=0,
):
    """Whole-volume GPU simplex (3D) untangler with a PHR augmented Lagrangian.

    Mirrors :func:`dvfopt.core.slp._gpu_untangle.gpu_untangle_alm_2d`
    (same defaults, same schedule) with triangle areas replaced by the six
    Kuhn tet volumes and all three displacement channels free.

    Per-tet multipliers escape the quadratic-penalty plateau: the
    constraint g_k = V_k - tgt >= 0 is enforced via
      psi(g, lam, mu) = -lam*g + (mu/2) g^2   if g <= lam/mu   else  -lam^2/(2mu)
    with multiplier update lam <- max(0, lam - mu*g) each outer step and mu
    grown only when the worst violation stalls. Data term = smooth-L1 to
    input over all three channels. Returns feasible-or-closest phi
    (3, D, H, W) float64.
    """
    phi_np = np.asarray(phi_in_3dhw, np.float64)
    if phi_np.ndim != 4 or phi_np.shape[0] != 3:
        raise ValueError(f'phi must have shape (3, D, H, W), got {phi_np.shape}')
    D, H, W = phi_np.shape[1:]
    if D < 2 or H < 2 or W < 2:
        # Degenerate volume: no 2x2x2 cell exists, so there are ZERO tets
        # and the field is trivially feasible (same guard as the 2D port).
        return phi_np.copy()

    import torch

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    phi0 = torch.tensor(phi_np, device=dev, dtype=torch.float64)
    dz = phi0[0].clone().requires_grad_(True)
    dy = phi0[1].clone().requires_grad_(True)
    dx = phi0[2].clone().requires_grad_(True)
    tgt = threshold + margin

    V = _tet_volumes_torch(dz, dy, dx, torch)
    # Upfront no-op check at plain `threshold` (NOT tgt): the downstream
    # exact solver only needs min V >= threshold and returning the input
    # unchanged is exactly L1-optimal. The margin only matters once the
    # field has to MOVE (same reasoning as the 2D seed's early-out).
    if float(V.min().item()) >= threshold:
        return phi_np.copy()
    lam = torch.zeros_like(V)
    mu = mu0
    prev_worst = None
    for outer in range(n_outer):
        opt = torch.optim.Adam([dz, dy, dx], lr=lr)
        for _ in range(n_inner):
            opt.zero_grad()
            V = _tet_volumes_torch(dz, dy, dx, torch)
            g = V - tgt
            p = torch.where(g <= lam / mu, -lam * g + 0.5 * mu * g * g, -0.5 * lam * lam / mu)
            data = (
                torch.sqrt((dz - phi0[0]) ** 2 + eps_l1**2).sum()
                + torch.sqrt((dy - phi0[1]) ** 2 + eps_l1**2).sum()
                + torch.sqrt((dx - phi0[2]) ** 2 + eps_l1**2).sum()
            )
            (p.sum() + data).backward()
            opt.step()
        with torch.no_grad():
            V = _tet_volumes_torch(dz, dy, dx, torch)
            g = V - tgt
            lam = torch.clamp(lam - mu * g, min=0.0)
            worst = float(g.min().item())
            if prev_worst is not None and worst < prev_worst + 1e-4:
                mu = min(mu * mu_grow, mu_max)
            prev_worst = worst
            if verbose:
                nneg = int((threshold > V).sum().item())
                log_info(
                    f'    [alm3d outer {outer + 1}] worst_g={worst:+.5f} folds~{nneg} mu={mu:.0e}'
                )
            if worst >= 0.0:
                break
    with torch.no_grad():
        out = torch.stack([dz, dy, dx]).cpu().numpy()
    return out


__all__ = ['gpu_untangle_alm_3d']
