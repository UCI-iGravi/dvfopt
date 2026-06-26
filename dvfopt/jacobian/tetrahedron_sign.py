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
    from numba import njit, prange  # type: ignore

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore
    prange = range  # type: ignore

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

    @njit(cache=True, fastmath=True, boundscheck=False, parallel=True)
    def _six_tet_volumes_kernel(dz, dy, dx, D, H, W):
        """Cell-walking JIT kernel for the 6 per-cell tet volumes.

        For each (z, y, x) cell, looks up the 8 corner deformed positions
        inline (no temporary 8-corner array) and evaluates all 6 tets'
        signed volumes via the 3x3 determinant. Folds 8 corner slices +
        6 broadcast multiplies + 6 broadcast subtracts into one fused
        loop with no intermediate allocations.

        Parallelised over the outer ``cz`` loop via ``prange``: each cz
        layer writes a disjoint output slice ``out[:, cz, :, :]`` so the
        parallelisation is race-free and bit-identical to the serial
        path. Measured ~12x on 24 cores."""
        out = np.empty((6, D - 1, H - 1, W - 1))
        for cz in prange(D - 1):
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


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False, parallel=True)
    def _six_tet_min_volume_kernel(dz, dy, dx, D, H, W):
        """Fused per-cube min-of-6-tet-volumes kernel (parallel).

        Computes ``min_k V_k`` for every cube directly, without
        materialising the full ``(6, D-1, H-1, W-1)`` volume array — the
        common case for fold counting, ``min_T``, and accept/reject.
        Race-free ``prange`` over ``cz`` (disjoint output slices);
        bit-identical to ``six_tet_volumes_3d(phi).min(axis=0)``.
        Measured ~32x vs the materialise-then-reduce path."""
        out = np.empty((D - 1, H - 1, W - 1))
        for cz in prange(D - 1):
            for cy in range(H - 1):
                for cx in range(W - 1):
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
                    mv = 1.0e300
                    for k in range(6):
                        i0 = _TET_VERTICES_INT64[k, 0]
                        i1 = _TET_VERTICES_INT64[k, 1]
                        i2 = _TET_VERTICES_INT64[k, 2]
                        i3 = _TET_VERTICES_INT64[k, 3]
                        ABz = Pz[i1] - Pz[i0]
                        ABy = Py[i1] - Py[i0]
                        ABx = Px[i1] - Px[i0]
                        ACz = Pz[i2] - Pz[i0]
                        ACy = Py[i2] - Py[i0]
                        ACx = Px[i2] - Px[i0]
                        ADz = Pz[i3] - Pz[i0]
                        ADy = Py[i3] - Py[i0]
                        ADx = Px[i3] - Px[i0]
                        det = (
                            ABz * (ACy * ADx - ACx * ADy)
                            - ABy * (ACz * ADx - ACx * ADz)
                            + ABx * (ACz * ADy - ACy * ADz)
                        )
                        vk = _TET_SIGN_F64[k] * det / 6.0
                        if vk < mv:
                            mv = vk
                    out[cz, cy, cx] = mv
        return out


def six_tet_min_volume_3d(phi: np.ndarray) -> np.ndarray:
    """Per-cube minimum signed tet volume — the fold-test scalar field.

    Equivalent to ``six_tet_volumes_3d(phi).min(axis=0)`` but computed in
    a single fused parallel kernel without materialising the full
    ``(6, ...)`` array. Use this for fold counting (``(min_V <= 0).sum()``),
    ``min_T`` (``min_V.min()``), and accept/reject checks — the hot paths
    that previously paid for a full volume array plus a reduction.

    Parameters
    ----------
    phi : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.

    Returns
    -------
    ndarray, shape ``(D-1, H-1, W-1)`` — per-cube worst (minimum) tet vol.
    """
    if not _HAVE_NUMBA:
        return _six_tet_volumes_3d_numpy(phi).min(axis=0)
    D, H, W = phi.shape[1:]
    dz = np.ascontiguousarray(phi[0])
    dy = np.ascontiguousarray(phi[1])
    dx = np.ascontiguousarray(phi[2])
    return _six_tet_min_volume_kernel(dz, dy, dx, D, H, W)


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
# Variable-diagonal (Kuhn-triangulation choice) feasibility
# ---------------------------------------------------------------------------
#
# The default 6-tet test fixes ONE cube diagonal (corner 0 -> corner 7) for
# the Kuhn decomposition. But a cube has 4 main diagonals, each giving a
# different valid 6-tet decomposition. The discrete-bijectivity literature's
# correct predicate is "there EXISTS a positive triangulation," i.e. the
# cube is acceptable if ANY of its 4 diagonals yields all-positive tets —
# the natural test for a HEX lattice (we triangulate only to check). Allowing
# the per-cell diagonal to vary recovers a large fraction of cells that are
# "folded" only under the arbitrary fixed split. See REPORT Part III/IX and
# the framing corrections in REPORT Part XVI.

_MAIN_DIAGONALS = ((0, 7), (1, 6), (2, 5), (3, 4))


def _tets_for_diagonal(start, end):
    """The 6 tets of the Kuhn fan around the cube diagonal (start, end).

    The two diagonal endpoints are shared by all six tets; the other two
    vertices of each tet are the endpoints of a cube edge on the
    "perimeter" path between start and end (edges not touching either
    endpoint). Mirrors the construction in the research runners.
    """
    cube_edges = [
        (v, w) for v in range(8) for w in range(v + 1, 8) if (v ^ w) in (1, 2, 4)
    ]
    perimeter = [e for e in cube_edges if start not in e and end not in e]
    return [(start, a, b, end) for (a, b) in perimeter]


def _build_all_diagonal_tables():
    """Precompute, once, the (4, 6, 4) tet-vertex table and (4, 6)
    sign table for all four main cube diagonals. Signs are normalised so
    the identity field gives positive volumes for every tet."""
    id_pos = np.array(
        [[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)],
        dtype=np.float64,
    )  # (8, 3)

    def _vol(a, b, c, d):
        ab = b - a; ac = c - a; ad = d - a
        return (
            ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
            - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
            + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
        ) / 6.0

    tets = np.zeros((4, 6, 4), dtype=np.int64)
    signs = np.zeros((4, 6), dtype=np.float64)
    for d in range(4):
        if d == 0:
            dtets = [tuple(int(x) for x in row) for row in _TET_VERTICES]
            dsigns = [float(s) for s in _TET_SIGN]
        else:
            s, e = _MAIN_DIAGONALS[d]
            dtets = _tets_for_diagonal(s, e)
            dsigns = [
                1.0 if _vol(id_pos[i0], id_pos[i1], id_pos[i2], id_pos[i3]) > 0
                else -1.0
                for (i0, i1, i2, i3) in dtets
            ]
        for k, (i0, i1, i2, i3) in enumerate(dtets):
            tets[d, k] = (i0, i1, i2, i3)
            signs[d, k] = dsigns[k]
    return tets, signs


_ALL_DIAG_TETS, _ALL_DIAG_SIGNS = _build_all_diagonal_tables()


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False, parallel=True)
    def _all_diag_min_kernel(dz, dy, dx, D, H, W, tets, signs):
        """Per-cube min-of-6-tets for each of the 4 diagonals (parallel).

        Returns ``(4, D-1, H-1, W-1)``. Race-free ``prange`` over cz.
        Replaces the old non-JIT numpy path for diagonals 1-3 (~600x)."""
        out = np.empty((4, D - 1, H - 1, W - 1))
        for cz in prange(D - 1):
            for cy in range(H - 1):
                for cx in range(W - 1):
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
                    for d in range(4):
                        mv = 1.0e300
                        for k in range(6):
                            i0 = tets[d, k, 0]
                            i1 = tets[d, k, 1]
                            i2 = tets[d, k, 2]
                            i3 = tets[d, k, 3]
                            ABz = Pz[i1] - Pz[i0]
                            ABy = Py[i1] - Py[i0]
                            ABx = Px[i1] - Px[i0]
                            ACz = Pz[i2] - Pz[i0]
                            ACy = Py[i2] - Py[i0]
                            ACx = Px[i2] - Px[i0]
                            ADz = Pz[i3] - Pz[i0]
                            ADy = Py[i3] - Py[i0]
                            ADx = Px[i3] - Px[i0]
                            det = (
                                ABz * (ACy * ADx - ACx * ADy)
                                - ABy * (ACz * ADx - ACx * ADz)
                                + ABx * (ACz * ADy - ACy * ADz)
                            )
                            vk = signs[d, k] * det / 6.0
                            if vk < mv:
                                mv = vk
                        out[d, cz, cy, cx] = mv
        return out


def six_tet_volumes_all_diagonals(phi: np.ndarray) -> np.ndarray:
    """Per-cube min tet volume under each of the 4 main cube diagonals.

    Parameters
    ----------
    phi : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.

    Returns
    -------
    ndarray, shape ``(4, D-1, H-1, W-1)``
        ``out[d]`` is the minimum of the 6 signed tet volumes of every
        cube under diagonal ``_MAIN_DIAGONALS[d]``. ``out[0]`` matches
        ``six_tet_volumes_3d(phi).min(axis=0)`` (the default diagonal).

    Each diagonal's tet signs are normalised against the identity field
    so a valid cube yields positive volumes for that diagonal. Computed
    in one fused parallel kernel (~600x faster than the old per-diagonal
    numpy path).
    """
    if not _HAVE_NUMBA:
        # Pure-numpy fallback (kept for the no-Numba path).
        dz, dy, dx = phi[0], phi[1], phi[2]
        pos = _voxel_corner_positions(dz, dy, dx)
        spatial = pos.shape[2:]
        out = np.empty((4, *spatial), dtype=np.float64)
        out[0] = _six_tet_volumes_3d_numpy(phi).min(axis=0)
        id_pos = _voxel_corner_positions(
            np.zeros_like(dz), np.zeros_like(dz), np.zeros_like(dz)
        )
        for d in range(1, 4):
            s, e = _MAIN_DIAGONALS[d]
            tets = _tets_for_diagonal(s, e)
            V_d = np.empty((6, *spatial), dtype=np.float64)
            for k, (i0, i1, i2, i3) in enumerate(tets):
                v_id = float(_tet_volume_from_vertices(
                    id_pos[i0], id_pos[i1], id_pos[i2], id_pos[i3]
                )[(0,) * len(spatial)])
                sgn = 1.0 if v_id > 0 else -1.0
                V_d[k] = sgn * _tet_volume_from_vertices(
                    pos[i0], pos[i1], pos[i2], pos[i3])
            out[d] = V_d.min(axis=0)
        return out
    D, H, W = phi.shape[1:]
    dz = np.ascontiguousarray(phi[0])
    dy = np.ascontiguousarray(phi[1])
    dx = np.ascontiguousarray(phi[2])
    return _all_diag_min_kernel(
        dz, dy, dx, D, H, W, _ALL_DIAG_TETS, _ALL_DIAG_SIGNS
    )


def best_diagonal_min_volume(phi: np.ndarray):
    """Per-cube best achievable min tet volume over the 4 diagonals.

    Returns
    -------
    best_min : ndarray, shape ``(D-1, H-1, W-1)``
        For each cube, ``max_d min_k V[d, k]`` — the most positive
        worst-tet over the 4 diagonal choices. This is the
        "exists-a-positive-triangulation" feasibility value.
    best_diag : ndarray, shape ``(D-1, H-1, W-1)``, dtype int8
        Which diagonal (0..3) achieves it per cube.
    """
    all_diag = six_tet_volumes_all_diagonals(phi)
    best_diag = np.argmax(all_diag, axis=0).astype(np.int8)
    best_min = np.max(all_diag, axis=0)
    return best_min, best_diag


def n_neg_best_diagonal(phi: np.ndarray, threshold: float = 0.0) -> int:
    """Fold count under the per-cell best-diagonal (variable-triangulation)
    feasibility test.

    A cube counts as folded only if NO diagonal makes its worst tet
    exceed ``threshold``. Compare to the fixed-diagonal count
    ``int((six_tet_volumes_3d(phi).min(axis=0) <= threshold).sum())`` to
    quantify how many "folds" are artifacts of the arbitrary fixed split.
    """
    best_min, _ = best_diagonal_min_volume(phi)
    return int((best_min <= threshold).sum())


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
    def _tet_grad_T_v_cz_layer(cz, dz, dy, dx, v, g_dz, g_dy, g_dx, D, H, W, n_cells):
        """Scatter the J^T@v contribution of one cz cube-layer into the
        shared gradient arrays. Writes only corner planes cz and cz+1, so
        cz-layers two apart touch disjoint memory (used by the 2-colour
        parallel driver). Keeps the sparsity early-exit per cell."""
        HW1 = (H - 1) * (W - 1)
        W1 = W - 1
        for cy in range(H - 1):
            for cx in range(W - 1):
                # Sparsity early-exit.
                any_nz = False
                for k in range(6):
                    vk = v[k * n_cells + cz * HW1 + cy * W1 + cx]
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
                    vk = v[k * n_cells + cz * HW1 + cy * W1 + cx]
                    if vk == 0.0:
                        continue
                    i0 = _TET_VERTICES_INT64[k, 0]
                    i1 = _TET_VERTICES_INT64[k, 1]
                    i2 = _TET_VERTICES_INT64[k, 2]
                    i3 = _TET_VERTICES_INT64[k, 3]
                    coef = _TET_SIGN_F64[k] * (1.0 / 6.0) * vk
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

    @njit(cache=True, fastmath=True, boundscheck=False, parallel=True)
    def _tet_grad_T_v_kernel(dz, dy, dx, v, D, H, W):
        """J^T @ v on the 6-tet constraint, parallelised by 2-colour cz
        sweep. Cube layer cz writes corner planes {cz, cz+1}; layers two
        apart are disjoint, so each colour's ``prange`` is race-free. The
        two colours run serially (a barrier between them) so the shared
        boundary plane is never written concurrently — bit-identical to
        the serial scatter. Measured ~8x on 24 cores."""
        g_dz = np.zeros((D, H, W))
        g_dy = np.zeros((D, H, W))
        g_dx = np.zeros((D, H, W))
        n_cells = (D - 1) * (H - 1) * (W - 1)
        n_cz = D - 1
        for color in range(2):
            n_layers = (n_cz - color + 1) // 2
            for li in prange(n_layers):
                cz = color + 2 * li
                _tet_grad_T_v_cz_layer(
                    cz, dz, dy, dx, v, g_dz, g_dy, g_dx, D, H, W, n_cells
                )
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
