"""Whole-volume GPU PHR augmented-Lagrangian 3D (6-tet) untangler.

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

The torch tet-volume kernel :func:`_tet_volumes_torch` matches
``dvfopt.jacobian.tetrahedron_sign.six_tet_volumes_3d`` EXACTLY — same
``_TET_VERTICES`` / ``_TET_SIGN`` tables (imported from that module as the
single source of truth), same scalar-triple-product expansion in (z, y, x)
component order. Parity is checked to 1e-10 against the numpy kernel by the
torch-gated tests in ``tests/test_tetrahedron_sign.py``.

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
from dvfopt.jacobian.tetrahedron_sign import _TET_SIGN, _TET_VERTICES


def _tet_volumes_torch(dz, dy, dx, torch):
    """Per-cell six signed Kuhn tet volumes, matching six_tet_volumes_3d.

    Parameters
    ----------
    dz, dy, dx : torch.Tensor, shape ``(D, H, W)``
        Displacement channels (autograd-capable).

    Returns
    -------
    torch.Tensor, shape ``(6, D-1, H-1, W-1)``
        ``V[k]`` = signed volume of tet k of every cell. Identity field →
        every entry ``+1/6``.
    """
    D, H, W = dz.shape
    zz = torch.arange(D, device=dz.device, dtype=dz.dtype)[:, None, None]
    yy = torch.arange(H, device=dz.device, dtype=dz.dtype)[None, :, None]
    xx = torch.arange(W, device=dz.device, dtype=dz.dtype)[None, None, :]
    Z = zz + dz
    Y = yy + dy
    X = xx + dx

    # Warped positions of the 8 cube corners of every cell, as slice views
    # (no in-place writes — autograd-friendly and allocation-light).
    # Corner i has offsets (oz, oy, ox) = ((i>>2)&1, (i>>1)&1, i&1) — the
    # exact convention of _voxel_corner_positions in tetrahedron_sign.py.
    P = []
    for i in range(8):
        oz = (i >> 2) & 1
        oy = (i >> 1) & 1
        ox = i & 1
        sl = (slice(oz, D - 1 + oz), slice(oy, H - 1 + oy), slice(ox, W - 1 + ox))
        P.append((Z[sl], Y[sl], X[sl]))

    vols = []
    for k in range(6):
        i0, i1, i2, i3 = (int(v) for v in _TET_VERTICES[k])
        Az, Ay, Ax = P[i0]
        Bz, By, Bx = P[i1]
        Cz, Cy, Cx = P[i2]
        Dz_, Dy_, Dx_ = P[i3]
        ABz, ABy, ABx = Bz - Az, By - Ay, Bx - Ax
        ACz, ACy, ACx = Cz - Az, Cy - Ay, Cx - Ax
        ADz, ADy, ADx = Dz_ - Az, Dy_ - Ay, Dx_ - Ax
        # det of [AB, AC, AD] columns, expanded along the first row —
        # component order (z, y, x), identical to _tet_volume_from_vertices.
        det = (
            ABz * (ACy * ADx - ACx * ADy)
            - ABy * (ACz * ADx - ACx * ADz)
            + ABx * (ACz * ADy - ACy * ADz)
        )
        vols.append(float(_TET_SIGN[k]) * det / 6.0)
    return torch.stack(vols, dim=0)


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
    """Whole-volume GPU 6-tet untangler with a PHR augmented Lagrangian.

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
