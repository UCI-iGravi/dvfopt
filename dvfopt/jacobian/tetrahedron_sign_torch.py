"""PyTorch forward for the 6-tetrahedron signed-volume check (3D).

Torch equivalent of :func:`dvfopt.jacobian.tetrahedron_sign.six_tet_volumes_3d`
— the same math, on a single ``(3, D, H, W)`` torch tensor that supports
autograd. Use when you want to run a tet-based fold check (or build a
barrier/penalty objective on top of it) on GPU.

This is a building block: the full barrier-on-tet GPU path that mirrors
:mod:`dvfopt.core.iterative3d_barrier_torch` (windowed, active-mask,
L-BFGS) is **not yet wired** — that's a separate ~hundred-line port. The
forward here is enough to:

* compute tet volumes on a torch tensor in one pass,
* let autograd produce the gradient (no analytical adjoint needed for
  the torch path — autograd handles it),
* drop into a hand-rolled penalty/barrier loop for experimentation.

The numpy path remains canonical for the constraint system
(:class:`dvfopt.constraints.Tet6Constraint3D`); this module is for users
who want a torch tensor in / torch tensor out.
"""

from __future__ import annotations

import numpy as np

# Tet vertex tables. Same indices, signs, and identity-field convention as
# :mod:`dvfopt.jacobian.tetrahedron_sign` — see the module-level docstring
# there for the cube-corner layout.
_TET_VERTICES_NP = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ],
    dtype=np.int64,
)

_TET_SIGN_NP = np.array([-1, +1, +1, -1, -1, +1], dtype=np.float32)


def _voxel_corner_positions_torch(phi):
    """Warped ``(z, y, x)`` positions of the 8 corners of every voxel.

    Parameters
    ----------
    phi : torch.Tensor, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.

    Returns
    -------
    torch.Tensor, shape ``(8, 3, D-1, H-1, W-1)`` — same layout as the
    numpy version.
    """
    import torch

    dz, dy, dx = phi[0], phi[1], phi[2]
    D, H, W = dz.shape

    zz, yy, xx = torch.meshgrid(
        torch.arange(D, dtype=phi.dtype, device=phi.device),
        torch.arange(H, dtype=phi.dtype, device=phi.device),
        torch.arange(W, dtype=phi.dtype, device=phi.device),
        indexing='ij',
    )
    Wz = zz + dz
    Wy = yy + dy
    Wx = xx + dx

    out = torch.empty((8, 3, D - 1, H - 1, W - 1), dtype=phi.dtype, device=phi.device)
    for i in range(8):
        oz = (i >> 2) & 1
        oy = (i >> 1) & 1
        ox = i & 1
        out[i, 0] = Wz[oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox]
        out[i, 1] = Wy[oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox]
        out[i, 2] = Wx[oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox]
    return out


def six_tet_volumes_3d_torch(phi):
    """Signed volumes of all six tetrahedra in every voxel cell (torch).

    Parameters
    ----------
    phi : torch.Tensor, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
        May have ``requires_grad=True`` — autograd backpropagates through
        this function.

    Returns
    -------
    torch.Tensor, shape ``(6, D-1, H-1, W-1)``
        Per-tet signed volumes. Identity field → every entry is ``+1/6``.
        ``<= 0`` indicates a flipped tet.

    Notes
    -----
    Uses the same scalar-triple-product formulation as the numpy version:
    ``V_k = sgn_k * (1/6) * (B-A) · ((C-A) × (D-A))``. Numerically
    identical results on float64 inputs (within ~1e-15 of the numpy
    version).
    """
    import torch

    pos = _voxel_corner_positions_torch(phi)  # (8, 3, D-1, H-1, W-1)

    signs = torch.tensor(_TET_SIGN_NP, dtype=phi.dtype, device=phi.device)
    vols = []
    for k in range(6):
        i0, i1, i2, i3 = _TET_VERTICES_NP[k]
        A = pos[i0]
        B = pos[i1]
        C = pos[i2]
        Dv = pos[i3]
        AB = B - A
        AC = C - A
        AD = Dv - A
        # det of 3 stacked column vectors via expansion along the first row.
        det = (
            AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
            - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
            + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])
        )
        vols.append(signs[k] * det / 6.0)
    return torch.stack(vols, dim=0)


__all__ = ['six_tet_volumes_3d_torch']
