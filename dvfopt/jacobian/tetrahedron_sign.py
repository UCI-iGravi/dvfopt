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

# Optional Numba JIT for the per-cell-per-tet kernels. Same pattern as
# the 2D triangle_sign.py: walk every cell once, evaluate the six tets,
# scatter-add their gradient contributions to corner vertices. Avoids
# the 8-corner array allocation and per-corner stride math of the
# pure-numpy reference path.
try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore

# Hoist the tet table into a regular ndarray for JIT (numba prefers int64
# over int8 inside loop indexing).
_TET_VERTICES_INT64 = _TET_VERTICES.astype(np.int64)
_TET_SIGN_F64 = _TET_SIGN.astype(np.float64)

# Per-corner (oz, oy, ox) offsets, packed for JIT consumption.
_CORNER_OFFSETS = np.array(
    [[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)],
    dtype=np.int64,
)


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


def _six_tet_volumes_3d_numpy(phi: np.ndarray) -> np.ndarray:
    """Pure-numpy reference path; kept as a fallback when Numba is not
    installed."""
    dz, dy, dx = phi[0], phi[1], phi[2]
    pos = _voxel_corner_positions(dz, dy, dx)
    out = np.empty((6, *pos.shape[2:]), dtype=np.float64)
    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        A, B, C, Dv = pos[i0], pos[i1], pos[i2], pos[i3]
        out[k] = _TET_SIGN[k] * _tet_volume_from_vertices(A, B, C, Dv)
    return out


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False)
    def _six_tet_volumes_kernel(dz, dy, dx, D, H, W):
        """Cell-walking JIT kernel for the 6 per-cell tet volumes.

        For each (z, y, x) cell, looks up the 8 corner deformed positions
        inline (no temporary 8-corner array) and evaluates all 6 tets'
        signed volumes via the 3x3 determinant. Folds 8 corner slices +
        6 broadcast multiplies + 6 broadcast subtracts into one fused
        loop with no intermediate allocations."""
        out = np.empty((6, D - 1, H - 1, W - 1))
        for cz in range(D - 1):
            for cy in range(H - 1):
                for cx in range(W - 1):
                    # All 8 corner deformed positions for this cell.
                    # Corner i has offsets (oz, oy, ox) = ((i>>2)&1, (i>>1)&1, i&1).
                    # Reference position adds (cz+oz, cy+oy, cx+ox).
                    # Indexed [corner][component] where component = (z, y, x).
                    Pz = np.empty(8)
                    Py = np.empty(8)
                    Px = np.empty(8)
                    for i in range(8):
                        oz = (i >> 2) & 1
                        oy = (i >> 1) & 1
                        ox = i & 1
                        Pz[i] = (cz + oz) + dz[cz + oz, cy + oy, cx + ox]
                        Py[i] = (cy + oy) + dy[cz + oz, cy + oy, cx + ox]
                        Px[i] = (cx + ox) + dx[cz + oz, cy + oy, cx + ox]
                    for k in range(6):
                        i0 = _TET_VERTICES_INT64[k, 0]
                        i1 = _TET_VERTICES_INT64[k, 1]
                        i2 = _TET_VERTICES_INT64[k, 2]
                        i3 = _TET_VERTICES_INT64[k, 3]
                        # AB = P[i1] - P[i0], AC = P[i2] - P[i0], AD = P[i3] - P[i0]
                        ABz = Pz[i1] - Pz[i0]
                        ABy = Py[i1] - Py[i0]
                        ABx = Px[i1] - Px[i0]
                        ACz = Pz[i2] - Pz[i0]
                        ACy = Py[i2] - Py[i0]
                        ACx = Px[i2] - Px[i0]
                        ADz = Pz[i3] - Pz[i0]
                        ADy = Py[i3] - Py[i0]
                        ADx = Px[i3] - Px[i0]
                        # det of [AB, AC, AD] as columns.
                        det = (
                            ABz * (ACy * ADx - ACx * ADy)
                            - ABy * (ACz * ADx - ACx * ADz)
                            + ABx * (ACz * ADy - ACy * ADz)
                        )
                        out[k, cz, cy, cx] = _TET_SIGN_F64[k] * det / 6.0
        return out


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

    Uses the Numba JIT kernel when available; falls back to the
    pure-numpy implementation otherwise. The two paths are
    numerically equivalent to ~1e-12 absolute on representative
    inputs.
    """
    if not _HAVE_NUMBA:
        return _six_tet_volumes_3d_numpy(phi)
    D, H, W = phi.shape[1:]
    dz = np.ascontiguousarray(phi[0])
    dy = np.ascontiguousarray(phi[1])
    dx = np.ascontiguousarray(phi[2])
    return _six_tet_volumes_kernel(dz, dy, dx, D, H, W)


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


def _tet_grad_T_v_numpy(phi_flat, D, H, W, v):
    """Pure-numpy reference path (kept as fallback when Numba absent)."""
    dz, dy, dx = _phi_flat_to_dz_dy_dx(phi_flat, D, H, W)
    pos = _voxel_corner_positions(dz, dy, dx)

    v_per_tet = v.reshape(6, D - 1, H - 1, W - 1)

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
        gB = sgn * (1.0 / 6.0) * _cross(AC, AD)
        gC = sgn * (1.0 / 6.0) * _cross(AD, AB)
        gD_ = sgn * (1.0 / 6.0) * _cross(AB, AC)
        gA = -(gB + gC + gD_)
        vk = v_per_tet[k]

        for corner_idx, grad in zip((i0, i1, i2, i3), (gA, gB, gC, gD_)):
            oz = (corner_idx >> 2) & 1
            oy = (corner_idx >> 1) & 1
            ox = corner_idx & 1
            for comp_idx, acc in enumerate(accumulators):
                acc[oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox] += grad[comp_idx] * vk

    return np.concatenate([g_dx.ravel(), g_dy.ravel(), g_dz.ravel()])


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False)
    def _tet_grad_T_v_kernel(dz, dy, dx, v, D, H, W):
        """Single-pass JIT kernel for J^T @ v on the 6-tet constraint.

        For each (cz, cy, cx) cell:
          - Compute 8 corner deformed positions inline (no 8-corner array).
          - For each of 6 tets: compute (B-A), (C-A), (D-A); evaluate the
            three cross products; scale by sgn * v_k * (1/6); scatter-add
            the 4 vertex gradients (gA = -(gB+gC+gD), gB, gC, gD) to the
            corresponding 4 corners of the cell.
          - Skip cells where every v_per_tet[k, cz, cy, cx] == 0 (sparse
            active set during late lambda annealing).

        Replaces the pure-numpy version's 6 outer iterations + 4 inner
        per-corner broadcasts (24 scatter-add ops) with one fused triple-
        loop and no intermediate per-tet (3, D-1, H-1, W-1) arrays."""
        g_dz = np.zeros((D, H, W))
        g_dy = np.zeros((D, H, W))
        g_dx = np.zeros((D, H, W))
        n_cells = (D - 1) * (H - 1) * (W - 1)
        for cz in range(D - 1):
            for cy in range(H - 1):
                for cx in range(W - 1):
                    # Sparsity early-exit.
                    any_nz = False
                    for k in range(6):
                        vk = v[k * n_cells + cz * (H - 1) * (W - 1) + cy * (W - 1) + cx]
                        if vk != 0.0:
                            any_nz = True
                            break
                    if not any_nz:
                        continue
                    # Inline 8 corner deformed positions.
                    Pz = np.empty(8)
                    Py = np.empty(8)
                    Px = np.empty(8)
                    for i in range(8):
                        oz = (i >> 2) & 1
                        oy = (i >> 1) & 1
                        ox = i & 1
                        Pz[i] = (cz + oz) + dz[cz + oz, cy + oy, cx + ox]
                        Py[i] = (cy + oy) + dy[cz + oz, cy + oy, cx + ox]
                        Px[i] = (cx + ox) + dx[cz + oz, cy + oy, cx + ox]
                    for k in range(6):
                        vk = v[k * n_cells + cz * (H - 1) * (W - 1) + cy * (W - 1) + cx]
                        if vk == 0.0:
                            continue
                        i0 = _TET_VERTICES_INT64[k, 0]
                        i1 = _TET_VERTICES_INT64[k, 1]
                        i2 = _TET_VERTICES_INT64[k, 2]
                        i3 = _TET_VERTICES_INT64[k, 3]
                        coef = _TET_SIGN_F64[k] * (1.0 / 6.0) * vk
                        # AB = P[i1] - P[i0]
                        ABz = Pz[i1] - Pz[i0]
                        ABy = Py[i1] - Py[i0]
                        ABx = Px[i1] - Px[i0]
                        ACz = Pz[i2] - Pz[i0]
                        ACy = Py[i2] - Py[i0]
                        ACx = Px[i2] - Px[i0]
                        ADz = Pz[i3] - Pz[i0]
                        ADy = Py[i3] - Py[i0]
                        ADx = Px[i3] - Px[i0]
                        # gB = (AC x AD)
                        gBz = coef * (ACy * ADx - ACx * ADy)
                        gBy = coef * (ACx * ADz - ACz * ADx)
                        gBx = coef * (ACz * ADy - ACy * ADz)
                        # gC = (AD x AB)
                        gCz = coef * (ADy * ABx - ADx * ABy)
                        gCy = coef * (ADx * ABz - ADz * ABx)
                        gCx = coef * (ADz * ABy - ADy * ABz)
                        # gD = (AB x AC)
                        gDz = coef * (ABy * ACx - ABx * ACy)
                        gDy = coef * (ABx * ACz - ABz * ACx)
                        gDx = coef * (ABz * ACy - ABy * ACz)
                        # gA = -(gB + gC + gD)
                        gAz = -(gBz + gCz + gDz)
                        gAy = -(gBy + gCy + gDy)
                        gAx = -(gBx + gCx + gDx)
                        # Scatter-add to 4 corners. corner_idx, (gA, gB, gC, gD).
                        # Corner i has offset ((i>>2)&1, (i>>1)&1, i&1).
                        for slot in range(4):
                            if slot == 0:
                                ci = i0
                                gz_v = gAz; gy_v = gAy; gx_v = gAx
                            elif slot == 1:
                                ci = i1
                                gz_v = gBz; gy_v = gBy; gx_v = gBx
                            elif slot == 2:
                                ci = i2
                                gz_v = gCz; gy_v = gCy; gx_v = gCx
                            else:
                                ci = i3
                                gz_v = gDz; gy_v = gDy; gx_v = gDx
                            oz2 = (ci >> 2) & 1
                            oy2 = (ci >> 1) & 1
                            ox2 = ci & 1
                            g_dz[cz + oz2, cy + oy2, cx + ox2] += gz_v
                            g_dy[cz + oz2, cy + oy2, cx + ox2] += gy_v
                            g_dx[cz + oz2, cy + oy2, cx + ox2] += gx_v
        return g_dz, g_dy, g_dx


def tet_grad_T_v(phi_flat: np.ndarray, D: int, H: int, W: int, v: np.ndarray) -> np.ndarray:
    """``J^T @ v`` for the 6-tet constraint Jacobian, analytically.

    Uses a Numba @njit kernel when available (5-10x speedup on the hot
    L-BFGS-B gradient path); falls back to the pure-numpy implementation
    when Numba is not installed.

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
    if not _HAVE_NUMBA:
        return _tet_grad_T_v_numpy(phi_flat, D, H, W, v)
    dz, dy, dx = _phi_flat_to_dz_dy_dx(phi_flat, D, H, W)
    dz = np.ascontiguousarray(dz)
    dy = np.ascontiguousarray(dy)
    dx = np.ascontiguousarray(dx)
    v_c = np.ascontiguousarray(v)
    g_dz, g_dy, g_dx = _tet_grad_T_v_kernel(dz, dy, dx, v_c, D, H, W)
    return np.concatenate([g_dx.ravel(), g_dy.ravel(), g_dz.ravel()])


# ---------------------------------------------------------------------------
# Sparse forward Jacobian (SLSQP path)
# ---------------------------------------------------------------------------
#
# 3D analogue of ``dvfopt.core.iterative2d_tri_slsqp._build_full_grid_tri_jac``.
# SLSQP's interior solver wants a sparse CSR Jacobian ``J`` of the constraint
# vector w.r.t. the flat decision vector — same shape conventions as the
# barrier-side primitives:
#
#   variables (phi pack)   : [dx.ravel(), dy.ravel(), dz.ravel()]  length 3*D*H*W
#   constraints (V pack)   : [V0.ravel(), V1.ravel(), ..., V5.ravel()]
#                                                       length 6*(D-1)*(H-1)*(W-1)
#
# Each tet has 4 vertices × 3 displacement components = 12 partials. The
# sparsity pattern is constant for a given (D, H, W) — only the values
# change per iterate — so we precompute (rows, cols) once and assemble a
# fresh CSR each call.


def _tet_corner_offsets():
    """``(8, 3)`` offsets (oz, oy, ox) of each cube corner."""
    return np.array(
        [[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)],
        dtype=np.int64,
    )


def build_tet_sparse_jac(D: int, H: int, W: int):
    """Build a callable ``jac(phi_flat) -> csr_matrix`` for the 6-tet constraint.

    Variable layout: ``[dx, dy, dz]`` (length ``3*D*H*W``).
    Constraint layout: ``[V_0, V_1, ..., V_5]`` (length
    ``6 * (D-1) * (H-1) * (W-1)``).

    The (rows, cols) pattern is precomputed once at build time so each call
    is dominated by the gradient arithmetic, not index bookkeeping.

    Returns
    -------
    jac : callable
        ``jac(phi_flat)`` → :class:`scipy.sparse.csr_matrix` of shape
        ``(n_constraints, n_variables)``.
    """
    import scipy.sparse as sp

    Dc, Hc, Wc = D - 1, H - 1, W - 1
    n_cells = Dc * Hc * Wc
    n_constr = 6 * n_cells
    n_vars = 3 * D * H * W
    DHW = D * H * W

    # Cell-grid index arrays (broadcastable to (Dc, Hc, Wc)).
    cz = np.arange(Dc, dtype=np.int64)[:, None, None]
    cy = np.arange(Hc, dtype=np.int64)[None, :, None]
    cx = np.arange(Wc, dtype=np.int64)[None, None, :]
    # Flat cell index for each (cz, cy, cx).
    cell_flat = (cz * Hc * Wc + cy * Wc + cx).ravel()  # shape (n_cells,)

    # Per-corner (oz, oy, ox) offsets.
    offsets = _tet_corner_offsets()  # (8, 3)

    # Flat vertex (grid-pixel) index for each corner of each cell:
    # shape (8, n_cells).
    vertex_flat = np.empty((8, n_cells), dtype=np.int64)
    for i in range(8):
        oz, oy, ox = offsets[i]
        vertex_flat[i] = ((cz + oz) * H * W + (cy + oy) * W + (cx + ox)).ravel()

    # Build (rows, cols) one tet at a time. For tet k with vertex indices
    # (i0, i1, i2, i3), each cell contributes 4 vertices × 3 components = 12
    # entries. Row = k*n_cells + cell_flat; col = component*DHW + vertex_flat.
    rows_chunks = []
    cols_chunks = []
    # key_order tracks how the value arrays from the jac() callable line up
    # with the row/col slots. Layout per tet k:
    #   for v in (A, B, C, D):
    #       for c in (x, y, z):           # 3 components per vertex
    #           entry (row = k*n_cells + cell, col = c*DHW + vertex_flat[v_corner])
    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        rows_for_tet = (k * n_cells + cell_flat).astype(np.int64)
        for corner_idx in (i0, i1, i2, i3):
            vcols = vertex_flat[corner_idx]  # (n_cells,)
            for comp in range(3):  # 0=x, 1=y, 2=z (matches phi_flat layout)
                rows_chunks.append(rows_for_tet)
                cols_chunks.append(comp * DHW + vcols)

    rows_flat = np.concatenate(rows_chunks)
    cols_flat = np.concatenate(cols_chunks)
    # nnz = 6 tets * 4 corners * 3 comps * n_cells = 72 * n_cells.
    assert rows_flat.size == 72 * n_cells

    def jac(phi_flat):
        # Re-use the analytical adjoint's vertex-gradient computation, but
        # instead of scatter-adding into a per-pixel buffer we keep the
        # per-tet per-cell per-corner per-component values and dump them
        # into the sparse-matrix data array.
        dz, dy, dx = _phi_flat_to_dz_dy_dx(phi_flat, D, H, W)
        pos = _voxel_corner_positions(dz, dy, dx)  # (8, 3, Dc, Hc, Wc)

        data_chunks = []
        for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
            sgn = float(_TET_SIGN[k])
            A = pos[i0]
            B = pos[i1]
            C = pos[i2]
            Dv = pos[i3]
            AB = B - A
            AC = C - A
            AD = Dv - A
            inv6 = sgn / 6.0
            # Per-vertex gradients in (z, y, x) component order — same as
            # tet_grad_T_v.
            gB = inv6 * _cross(AC, AD)
            gC = inv6 * _cross(AD, AB)
            gD_ = inv6 * _cross(AB, AC)
            gA = -(gB + gC + gD_)
            # For each of the 4 corners (A, B, C, D), push the 3
            # components (x, y, z). Components are in axis-0 of g* as
            # (z-comp, y-comp, x-comp), so we permute to (x, y, z).
            for grad in (gA, gB, gC, gD_):
                data_chunks.append(grad[2].ravel())  # ∂V/∂(corner.x)
                data_chunks.append(grad[1].ravel())  # ∂V/∂(corner.y)
                data_chunks.append(grad[0].ravel())  # ∂V/∂(corner.z)

        data_flat = np.concatenate(data_chunks)
        assert data_flat.size == rows_flat.size
        return sp.csr_matrix((data_flat, (rows_flat, cols_flat)), shape=(n_constr, n_vars))

    return jac


__all__ = [
    'build_tet_sparse_jac',
    'six_tet_fold_classification',
    'six_tet_volumes_3d',
    'tet_grad_T_v',
    'tet_volumes_flat',
]
