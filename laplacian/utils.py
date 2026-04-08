"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Atchuth Naveen
Code developed at UC Irvine.

Optimized Laplacian matrix construction for 1D, 2D, and 3D spaces
with Dirichlet boundary conditions.

Uses boolean masking instead of np.delete, arithmetic index computation
instead of meshgrid, and scipy.sparse.diags where possible for
significantly reduced memory usage and faster construction.
"""

import numpy as np
import scipy
import gc


def laplacianA1D(n, boundaryIndices):
    """
    Build a 1D Laplacian matrix with Dirichlet boundary conditions.

    Parameters
    ----------
    n : int
        Number of grid points.
    boundaryIndices : array-like
        Flat indices of Dirichlet boundary points.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse Laplacian matrix of shape (n, n).
    """
    boundaryIndices = np.asarray(boundaryIndices, dtype=int)

    # Build boolean mask for boundary points (much faster than np.delete)
    is_boundary = np.zeros(n, dtype=bool)
    is_boundary[boundaryIndices] = True

    # Diagonal: 2 for interior, 1 for boundary; subtract 1 per missing neighbour
    # at volume edges (x==0 or x==n-1)
    data = np.full(n, 2.0)
    # x==0 boundary: missing left neighbour
    data[0] -= 1
    # x==n-1 boundary: missing right neighbour
    data[n - 1] -= 1
    # Dirichlet points: diagonal = 1, no off-diagonal connections
    data[boundaryIndices] = 1.0

    # Off-diagonal: connect (i) <-> (i±1), skip boundary rows and volume edges
    # Left neighbour: row i, col i-1 for i in [1, n), excluding boundary rows
    rows_left = np.arange(1, n)
    valid_left = ~is_boundary[rows_left]
    r_left = rows_left[valid_left]
    c_left = r_left - 1

    # Right neighbour: row i, col i+1 for i in [0, n-1), excluding boundary rows
    rows_right = np.arange(0, n - 1)
    valid_right = ~is_boundary[rows_right]
    r_right = rows_right[valid_right]
    c_right = r_right + 1

    row = np.concatenate([np.arange(n), r_left, r_right])
    col = np.concatenate([np.arange(n), c_left, c_right])
    val = np.concatenate([data, -np.ones(len(r_left) + len(r_right))])

    A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
    del row, col, val, data
    gc.collect()
    return A


def laplacianA2D(shape, boundaryIndices):
    """
    Build a 2D Laplacian matrix with Dirichlet boundary conditions.

    Uses arithmetic index computation (no meshgrid) and boolean masking
    (no np.delete) for reduced memory and faster construction.

    Parameters
    ----------
    shape : tuple of int
        (n0, n1) shape of the 2D grid (rows, columns).
    boundaryIndices : array-like
        Flat indices of Dirichlet boundary points.
        Flat index for pixel (r, c) = r * shape[1] + c.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse Laplacian matrix of shape (N, N) where N = shape[0]*shape[1].
    """
    n0, n1 = shape
    N = n0 * n1
    boundaryIndices = np.asarray(boundaryIndices, dtype=int)

    # Boolean mask for boundary points
    is_boundary = np.zeros(N, dtype=bool)
    is_boundary[boundaryIndices] = True

    # Compute row/col grid coordinates arithmetically (no meshgrid)
    ids = np.arange(N, dtype=np.int64)
    R = ids // n1        # row coordinate
    C = ids % n1         # col coordinate

    # Diagonal: start at 2*k = 4 for interior points
    data = np.full(N, 4.0)
    # Subtract 1 for each missing neighbour at volume edges
    data[R == 0] -= 1       # no (r-1) neighbour
    data[R == n0 - 1] -= 1  # no (r+1) neighbour
    data[C == 0] -= 1       # no (c-1) neighbour
    data[C == n1 - 1] -= 1  # no (c+1) neighbour
    # Dirichlet BCs: diagonal = 1
    data[boundaryIndices] = 1.0

    # Off-diagonal entries: connect to 4-connected neighbours, skipping
    # boundary rows and volume-edge rows

    # (r-1, c): valid when R > 0 and not boundary
    mask = (R > 0) & ~is_boundary
    r_up = ids[mask]
    c_up = r_up - n1

    # (r+1, c): valid when R < n0-1 and not boundary
    mask = (R < n0 - 1) & ~is_boundary
    r_dn = ids[mask]
    c_dn = r_dn + n1

    # (r, c-1): valid when C > 0 and not boundary
    mask = (C > 0) & ~is_boundary
    r_lt = ids[mask]
    c_lt = r_lt - 1

    # (r, c+1): valid when C < n1-1 and not boundary
    mask = (C < n1 - 1) & ~is_boundary
    r_rt = ids[mask]
    c_rt = r_rt + 1

    n_offdiag = len(r_up) + len(r_dn) + len(r_lt) + len(r_rt)
    row = np.concatenate([ids, r_up, r_dn, r_lt, r_rt])
    col = np.concatenate([ids, c_up, c_dn, c_lt, c_rt])
    val = np.concatenate([data, -np.ones(n_offdiag)])

    A = scipy.sparse.csr_matrix((val, (row, col)), shape=(N, N))
    del row, col, val, data, ids, R, C, is_boundary
    gc.collect()
    return A


def laplacianA3D(shape, boundaryIndices, spacing=None, dtype=None, log_fn=None):
    """
    Build a 3D Laplacian matrix with Dirichlet boundary conditions.

    Optimized to use arithmetic index computation instead of np.meshgrid
    (saves ~1.8 GB for a 456x320x528 volume) and boolean masking instead
    of np.delete (avoids 6 O(N) copies).

    When *spacing* is provided, the finite-difference stencil weights each
    axis by ``1/h²`` so that the Laplacian is correct for anisotropic
    voxels.  When spacing is ``None`` or isotropic, the classical uniform
    stencil (weights = -1) is used.

    Parameters
    ----------
    shape : tuple of int
        ``(n0, n1, n2)`` shape of the 3D volume (axis-0, axis-1, axis-2).
    boundaryIndices : array-like
        Flat indices of Dirichlet boundary points.
        Flat index for voxel (i, j, k) = i*n1*n2 + j*n2 + k.
    spacing : tuple of float, optional
        Physical voxel size per axis (e.g. ``(0.025, 0.025, 0.050)`` for
        25 µm in-plane and 50 µm through-plane).  Units are arbitrary but
        must be consistent.  ``None`` (default) uses unit spacing.
    dtype : numpy dtype, optional
        Floating-point type for the matrix values (e.g. ``np.float32`` or
        ``np.float64``).  ``None`` (default) uses float64.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse Laplacian matrix of shape (N, N) where N = n0*n1*n2.
    """
    n0, n1, n2 = shape
    N = n0 * n1 * n2
    stride_0 = n1 * n2
    stride_1 = n2

    boundaryIndices = np.asarray(boundaryIndices, dtype=int)

    if dtype is None:
        dtype = np.float64

    # Compute axis weights from spacing: w_d = 1/h_d^2
    # For isotropic spacing this reduces to the uniform stencil.
    if spacing is not None:
        h0, h1, h2 = float(spacing[0]), float(spacing[1]), float(spacing[2])
    else:
        h0 = h1 = h2 = 1.0
    w0 = 1.0 / (h0 * h0)
    w1 = 1.0 / (h1 * h1)
    w2 = 1.0 / (h2 * h2)

    # Boolean mask for boundary points (O(N) instead of repeated np.delete)
    is_boundary = np.zeros(N, dtype=bool)
    is_boundary[boundaryIndices] = True

    # Compute grid coordinates arithmetically (no meshgrid — saves
    # 3 arrays of N int64 = ~1.8 GB for a typical brain volume)
    ids = np.arange(N, dtype=np.int64)
    I0 = ids // stride_0
    I1 = (ids % stride_0) // stride_1
    I2 = ids % stride_1

    _log = log_fn if log_fn is not None else (lambda msg, level='info': print(msg))
    _log("Building data for Laplacian Sparse Matrix A (optimized)", 'info')

    # Diagonal: sum of axis weights for present neighbours, set to 1 at
    # Dirichlet points
    data = np.full(N, 2.0 * (w0 + w1 + w2), dtype=dtype)
    data[I0 == 0] -= w0
    data[I0 == n0 - 1] -= w0
    data[I1 == 0] -= w1
    data[I1 == n1 - 1] -= w1
    data[I2 == 0] -= w2
    data[I2 == n2 - 1] -= w2
    data[boundaryIndices] = 1.0

    # Off-diagonal entries: connect to 6-connected neighbours.
    # For each direction: include only voxels where BOTH source and target
    # are non-boundary.  Zeroing boundary *columns* (not just rows) makes
    # the matrix symmetric, which is required for the CG solver.
    # The removed A[i,b]=-w entries are compensated in the RHS via
    # propagate_dirichlet_rhs() in the calling code.

    # (i-1, j, k)
    mask = (I0 > 0) & ~is_boundary
    r_0m = ids[mask];  c_0m = r_0m - stride_0
    keep = ~is_boundary[c_0m]; r_0m = r_0m[keep]; c_0m = c_0m[keep]

    # (i+1, j, k)
    mask = (I0 < n0 - 1) & ~is_boundary
    r_0p = ids[mask];  c_0p = r_0p + stride_0
    keep = ~is_boundary[c_0p]; r_0p = r_0p[keep]; c_0p = c_0p[keep]

    # (i, j-1, k)
    mask = (I1 > 0) & ~is_boundary
    r_1m = ids[mask];  c_1m = r_1m - stride_1
    keep = ~is_boundary[c_1m]; r_1m = r_1m[keep]; c_1m = c_1m[keep]

    # (i, j+1, k)
    mask = (I1 < n1 - 1) & ~is_boundary
    r_1p = ids[mask];  c_1p = r_1p + stride_1
    keep = ~is_boundary[c_1p]; r_1p = r_1p[keep]; c_1p = c_1p[keep]

    # (i, j, k-1)
    mask = (I2 > 0) & ~is_boundary
    r_2m = ids[mask];  c_2m = r_2m - 1
    keep = ~is_boundary[c_2m]; r_2m = r_2m[keep]; c_2m = c_2m[keep]

    # (i, j, k+1)
    mask = (I2 < n2 - 1) & ~is_boundary
    r_2p = ids[mask];  c_2p = r_2p + 1
    keep = ~is_boundary[c_2p]; r_2p = r_2p[keep]; c_2p = c_2p[keep]

    del I0, I1, I2, is_boundary, mask; gc.collect()

    # Build off-diagonal weight arrays per axis direction
    w_0m = np.full(len(r_0m), -w0, dtype=dtype)
    w_0p = np.full(len(r_0p), -w0, dtype=dtype)
    w_1m = np.full(len(r_1m), -w1, dtype=dtype)
    w_1p = np.full(len(r_1p), -w1, dtype=dtype)
    w_2m = np.full(len(r_2m), -w2, dtype=dtype)
    w_2p = np.full(len(r_2p), -w2, dtype=dtype)

    row = np.concatenate([ids, r_0m, r_0p, r_1m, r_1p, r_2m, r_2p])
    col = np.concatenate([ids, c_0m, c_0p, c_1m, c_1p, c_2m, c_2p])
    val = np.concatenate([data, w_0m, w_0p, w_1m, w_1p, w_2m, w_2p])

    del ids, data
    del r_0m, c_0m, r_0p, c_0p, r_1m, c_1m, r_1p, c_1p, r_2m, c_2m, r_2p, c_2p
    del w_0m, w_0p, w_1m, w_1p, w_2m, w_2p
    gc.collect()

    _log("Creating Laplacian Sparse Matrix A", 'info')
    A = scipy.sparse.csr_matrix((val, (row, col)), shape=(N, N))

    del row, col, val
    gc.collect()
    return A


def propagate_dirichlet_rhs(shape, boundary_indices, *rhs_arrays, spacing=None):
    """
    Adjust RHS vectors for the symmetric Dirichlet-BC Laplacian.

    When boundary columns are zeroed out for symmetry (see laplacianA3D),
    the removed ``A[i,b] = -w`` contributions must be moved to the RHS:
    for each non-boundary point *i* adjacent to boundary point *b*,
    ``rhs[i] += w * displacement[b]``, where ``w = 1/h²`` for the
    corresponding axis direction.

    Parameters
    ----------
    shape : tuple of int
        ``(n0, n1, n2)`` volume shape.
    boundary_indices : array-like
        Flat indices of Dirichlet boundary points (need not be unique).
    *rhs_arrays : np.ndarray
        One or more flat RHS vectors to adjust *in-place*.
    spacing : tuple of float, optional
        Physical voxel size per axis. ``None`` uses unit spacing (w=1).
    """
    n0, n1, n2 = shape
    N = n0 * n1 * n2
    stride_0 = n1 * n2
    stride_1 = n2

    # Compute axis weights (must match laplacianA3D)
    if spacing is not None:
        h0, h1, h2 = float(spacing[0]), float(spacing[1]), float(spacing[2])
    else:
        h0 = h1 = h2 = 1.0
    w0 = 1.0 / (h0 * h0)
    w1 = 1.0 / (h1 * h1)
    w2 = 1.0 / (h2 * h2)

    boundary_indices = np.unique(np.asarray(boundary_indices, dtype=int))

    is_bnd = np.zeros(N, dtype=bool)
    is_bnd[boundary_indices] = True

    # Grid coordinates of boundary points (for bounds checking)
    bI0 = boundary_indices // stride_0
    bI1 = (boundary_indices % stride_0) // stride_1
    bI2 = boundary_indices % stride_1

    directions = [
        (stride_0,  bI0 < n0 - 1, w0),   # axis-0 forward
        (-stride_0, bI0 > 0,      w0),   # axis-0 backward
        (stride_1,  bI1 < n1 - 1, w1),   # axis-1 forward
        (-stride_1, bI1 > 0,      w1),   # axis-1 backward
        (1,         bI2 < n2 - 1, w2),   # axis-2 forward
        (-1,        bI2 > 0,      w2),   # axis-2 backward
    ]

    for offset, valid_mask, weight in directions:
        src = boundary_indices[valid_mask]
        nb = src + offset
        # Only propagate to non-boundary neighbours
        not_bnd = ~is_bnd[nb]
        nb_sel = nb[not_bnd]
        src_sel = src[not_bnd]
        for rhs in rhs_arrays:
            rhs[nb_sel] += weight * rhs[src_sel]
