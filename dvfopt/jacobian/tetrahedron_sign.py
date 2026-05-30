"""Per-voxel six-tetrahedron sign-only Jacobian check (3D).

3D analogue of :mod:`dvfopt.jacobian.triangle_sign`. Each cubic voxel cell
is decomposed into six tetrahedra that share the cell's main diagonal
(``C0`` → ``C7``). A negative tet signed volume means that piece of the
trilinear interpolant has flipped — a stronger / more localized check
than the central-difference Jdet, and the natural building block for a
future 3D 6-tet constraint.

The cube convention (axis order ``(z, y, x)``)::

      4 ---- 5
     /|     /|        Vertex i = (i>>2, (i>>1)&1, i&1) ∈ {0,1}^3
    6 ---- 7 |
    | 0 ---|-1        e.g. C[0] = (0,0,0)  ← cell's TLB corner
    |/     |/              C[7] = (1,1,1)  ← cell's BRF corner
    2 ---- 3

The six tetrahedra (each sharing edge C0–C7)::

    T0: (C0, C1, C3, C7)    T3: (C0, C2, C6, C7)
    T1: (C0, C1, C5, C7)    T4: (C0, C4, C5, C7)
    T2: (C0, C2, C3, C7)    T5: (C0, C4, C6, C7)

Each is wound so that the identity mapping yields ``+1/6`` signed volume.

These helpers are visualization-grade — they're not (yet) wired into the
constraint system. See [[plot_fold_overview_3d]] for the consumer.
"""

from __future__ import annotations

import numpy as np

# Tet vertex tables. Each row = (i0, i1, i2, i3) where C[i] is the
# corresponding cube corner. Winding chosen so identity → +1/6 volume.
_TET_VERTICES = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ],
    dtype=np.int8,
)

# Sign flips so the identity mapping yields +1/6 for every tet. The base
# winding gives -1/6 on the identity under the +y-down image convention,
# so we negate every entry. Established once empirically against the
# identity field; covered by tests/test_tetrahedron_sign.py.
_TET_SIGN = np.array([-1, +1, +1, -1, -1, +1], dtype=np.int8)


def _voxel_corner_positions(dz, dy, dx):
    """Warped positions of the 8 corners of every voxel cell.

    Parameters
    ----------
    dz, dy, dx : ndarray, shape ``(D, H, W)``

    Returns
    -------
    pos : ndarray, shape ``(8, 3, D-1, H-1, W-1)``
        ``pos[i]`` = ``(z, y, x)`` warped positions of corner ``i`` of
        every ``(D-1, H-1, W-1)`` cell. Axis 1 indexes the spatial
        coordinate ``[z, y, x]``.
    """
    D, H, W = dz.shape
    zz, yy, xx = np.meshgrid(
        np.arange(D, dtype=np.float64),
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing='ij',
    )
    Wz = zz + dz
    Wy = yy + dy
    Wx = xx + dx

    out = np.empty((8, 3, D - 1, H - 1, W - 1), dtype=np.float64)
    # Corner i has offsets (oz, oy, ox) = ((i>>2)&1, (i>>1)&1, i&1).
    for i in range(8):
        oz = (i >> 2) & 1
        oy = (i >> 1) & 1
        ox = i & 1
        out[i, 0] = Wz[oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox]
        out[i, 1] = Wy[oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox]
        out[i, 2] = Wx[oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox]
    return out


def _tet_volume_from_vertices(A, B, C, D):
    """Signed tet volume = ``(1/6) * det([B-A, C-A, D-A])``.

    Each input is shape ``(3, ...)``; returns scalar-shaped ``(...)``.
    """
    AB = B - A
    AC = C - A
    AD = D - A
    # det of 3 stacked column vectors
    det = (
        AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
        - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
        + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])
    )
    return det / 6.0


def six_tet_volumes_3d(phi: np.ndarray) -> np.ndarray:
    """Signed volumes of all six tetrahedra in every voxel cell.

    Parameters
    ----------
    phi : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.

    Returns
    -------
    ndarray, shape ``(6, D-1, H-1, W-1)``
        Per-tet signed volumes. Positive = valid; ``<=0`` = flipped.
        The identity field yields ``+1/6`` for every tet.
    """
    dz, dy, dx = phi[0], phi[1], phi[2]
    pos = _voxel_corner_positions(dz, dy, dx)  # (8, 3, D-1, H-1, W-1)
    out = np.empty((6, *pos.shape[2:]), dtype=np.float64)
    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        A, B, C, Dv = pos[i0], pos[i1], pos[i2], pos[i3]
        out[k] = _TET_SIGN[k] * _tet_volume_from_vertices(A, B, C, Dv)
    return out


def six_tet_fold_classification(phi: np.ndarray) -> np.ndarray:
    """Per-voxel classification of how many tets have flipped.

    Convenience wrapper over :func:`six_tet_volumes_3d`. Useful for
    coloring a 3D voxel volume by fold severity.

    Parameters
    ----------
    phi : ndarray, shape ``(3, D, H, W)``.

    Returns
    -------
    n_flipped : ndarray, shape ``(D-1, H-1, W-1)``, dtype int8
        Number of tetrahedra with signed volume ``<= 0`` per voxel
        cell (range 0–6).
    """
    V = six_tet_volumes_3d(phi)
    return (V <= 0).sum(axis=0).astype(np.int8)


# ---------------------------------------------------------------------------
# Flat-pack constraint primitives (for Tet6Constraint3D / barrier path)
# ---------------------------------------------------------------------------
#
# Phi pack convention: ``[dx.ravel(), dy.ravel(), dz.ravel()]`` (DX_FIRST,
# matches the existing 3D Jdet barrier path). The output is
# ``[V0.ravel(), V1.ravel(), ..., V5.ravel()]`` — six tet-volume arrays
# stacked, length ``6 * (D-1) * (H-1) * (W-1)``.
#
# Identity field → every output equals ``+1/6`` (verified by tests).
# A feasible field has every output > 0; a folded cell has at least one
# negative entry.


def _phi_flat_to_dz_dy_dx(phi_flat, D, H, W):
    """Unpack ``[dx, dy, dz]`` flat into ``(dz, dy, dx)`` arrays."""
    n = D * H * W
    dx = phi_flat[:n].reshape(D, H, W)
    dy = phi_flat[n : 2 * n].reshape(D, H, W)
    dz = phi_flat[2 * n :].reshape(D, H, W)
    return dz, dy, dx


def tet_volumes_flat(phi_flat: np.ndarray, D: int, H: int, W: int) -> np.ndarray:
    """Concatenated six per-cell tet volumes from a flat phi.

    Parameters
    ----------
    phi_flat : ndarray, shape ``(3 * D * H * W,)``
        Packed as ``[dx, dy, dz]`` (DX_FIRST).
    D, H, W : int

    Returns
    -------
    ndarray, shape ``(6 * (D-1) * (H-1) * (W-1),)``
        Stacked ``[V0.ravel(), V1.ravel(), ..., V5.ravel()]``.
        Each Vk is the signed volume of the k-th tet of every cell.
    """
    dz, dy, dx = _phi_flat_to_dz_dy_dx(phi_flat, D, H, W)
    phi = np.stack([dz, dy, dx])  # (3, D, H, W) [dz, dy, dx]
    V = six_tet_volumes_3d(phi)  # (6, D-1, H-1, W-1)
    return V.ravel()  # ravels by tet-then-cell


def _cross(a, b):
    """3-vector cross product, axis 0 = component."""
    return np.stack(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def tet_grad_T_v(phi_flat: np.ndarray, D: int, H: int, W: int, v: np.ndarray) -> np.ndarray:
    """``J^T @ v`` for the 6-tet constraint Jacobian, analytically.

    Uses the cross-product form ``V_k = sgn_k * (1/6) * (B-A) · ((C-A) × (D-A))``,
    which yields a clean per-tet gradient::

        ∂V/∂B = sgn * (1/6) * (C-A) × (D-A)
        ∂V/∂C = sgn * (1/6) * (D-A) × (B-A)
        ∂V/∂D = sgn * (1/6) * (B-A) × (C-A)
        ∂V/∂A = -(∂V/∂B + ∂V/∂C + ∂V/∂D)

    Then each tet's 4 vertex-gradients are scatter-added to the
    corresponding 4 cube corners of every cell.

    Parameters
    ----------
    phi_flat : ndarray, shape ``(3*D*H*W,)``, pack ``[dx, dy, dz]``.
    D, H, W : int
    v : ndarray, shape ``(6 * (D-1) * (H-1) * (W-1),)``
        Co-vector. Layout matches :func:`tet_volumes_flat`.

    Returns
    -------
    ndarray, shape ``(3*D*H*W,)``, pack ``[dx, dy, dz]``.
    """
    dz, dy, dx = _phi_flat_to_dz_dy_dx(phi_flat, D, H, W)
    # Corner positions: pos[i] has shape (3, D-1, H-1, W-1) for (z, y, x).
    pos = _voxel_corner_positions(dz, dy, dx)

    v_per_tet = v.reshape(6, D - 1, H - 1, W - 1)

    # Accumulators in (z, y, x) component space; flat-out at the end.
    g_dz = np.zeros((D, H, W))
    g_dy = np.zeros((D, H, W))
    g_dx = np.zeros((D, H, W))
    accumulators = (g_dz, g_dy, g_dx)

    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        sgn = float(_TET_SIGN[k])
        A = pos[i0]
        B = pos[i1]
        C = pos[i2]
        Dv = pos[i3]
        AB = B - A
        AC = C - A
        AD = Dv - A
        # Per-vertex gradients (component, D-1, H-1, W-1).
        gB = sgn * (1.0 / 6.0) * _cross(AC, AD)
        gC = sgn * (1.0 / 6.0) * _cross(AD, AB)
        gD_ = sgn * (1.0 / 6.0) * _cross(AB, AC)
        gA = -(gB + gC + gD_)
        vk = v_per_tet[k]  # (D-1, H-1, W-1)

        # Scatter to the 4 corners of every cell. Corner i has offset
        # (oz, oy, ox) = ((i>>2)&1, (i>>1)&1, i&1) within each cell.
        for corner_idx, grad in zip((i0, i1, i2, i3), (gA, gB, gC, gD_)):
            oz = (corner_idx >> 2) & 1
            oy = (corner_idx >> 1) & 1
            ox = corner_idx & 1
            # grad: shape (3, D-1, H-1, W-1), axes = (z-comp, y-comp, x-comp)
            for comp_idx, acc in enumerate(accumulators):
                acc[oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox] += grad[comp_idx] * vk

    # Pack out as [dx, dy, dz] to match the input layout.
    return np.concatenate([g_dx.ravel(), g_dy.ravel(), g_dz.ravel()])


__all__ = [
    'six_tet_fold_classification',
    'six_tet_volumes_3d',
    'tet_grad_T_v',
    'tet_volumes_flat',
]
