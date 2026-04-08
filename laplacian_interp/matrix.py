"""Sparse Laplacian matrix construction for 3D volumes."""

import gc

import numpy as np
import scipy.sparse


def get_laplacian_index(z: int, y: int, x: int, shape: tuple):
    """Get the flattened index of a voxel at (z, y, x) in a 3D volume of shape *shape*."""
    return z * shape[1] * shape[2] + y * shape[2] + x


def get_adjacent_indices(z: int, y: int, x: int, shape: tuple):
    adjacent_indices = [None, None, None, None]  # Left, right, up, down
    if x > 0:
        adjacent_indices[0] = get_laplacian_index(z, y, x - 1, shape)
    if x < shape[2] - 1:
        adjacent_indices[1] = get_laplacian_index(z, y, x + 1, shape)
    if y > 0:
        adjacent_indices[2] = get_laplacian_index(z, y - 1, x, shape)
    if y < shape[1] - 1:
        adjacent_indices[3] = get_laplacian_index(z, y + 1, x, shape)
    return adjacent_indices


def laplacian_a_3d(shape, boundary_indices, use_correspondences=True):
    """Build the sparse Laplacian matrix *A* for a 3D volume.

    Parameters
    ----------
    shape : tuple
        3D shape ``(nz, ny, nx)``.
    boundary_indices : array-like
        Flattened indices of Dirichlet boundary points.
    use_correspondences : bool
        When ``True`` (default), apply full Dirichlet BCs at boundary
        indices (augment diagonal, use float64).  When ``False``,
        override boundary to ``[0]`` and use int8 dtype — suitable for
        interpolation without correspondence constraints.
    """
    if not use_correspondences:
        boundary_indices = np.array([0])
    else:
        boundary_indices = np.asarray(boundary_indices).astype(int)

    k = len(shape)
    X, Y, Z = np.meshgrid(range(shape[0]), range(shape[1]), range(shape[2]), indexing='ij')

    X = X.flatten().astype(int)
    Y = Y.flatten().astype(int)
    Z = Z.flatten().astype(int)

    n = len(X)
    ids_0 = X * shape[1] * shape[2] + Y * shape[2] + Z
    data = np.ones(n) * 2 * k

    # Pre-compute boundary mask once
    bnd = np.zeros(n, dtype=bool)
    bnd[boundary_indices] = True

    # x-1
    cids_x1 = (X - 1) * shape[1] * shape[2] + Y * shape[2] + Z
    valid = (X > 0) & ~bnd
    rids_x1 = ids_0[valid]
    cids_x1 = cids_x1[valid]
    data[~valid] -= 1

    # x+1
    cids_x2 = (X + 1) * shape[1] * shape[2] + Y * shape[2] + Z
    valid = (X < shape[0] - 1) & ~bnd
    rids_x2 = ids_0[valid]
    cids_x2 = cids_x2[valid]
    data[~valid] -= 1

    # y-1
    cids_y1 = X * shape[1] * shape[2] + (Y - 1) * shape[2] + Z
    valid = (Y > 0) & ~bnd
    rids_y1 = ids_0[valid]
    cids_y1 = cids_y1[valid]
    data[~valid] -= 1

    # y+1
    cids_y2 = X * shape[1] * shape[2] + (Y + 1) * shape[2] + Z
    valid = (Y < shape[1] - 1) & ~bnd
    rids_y2 = ids_0[valid]
    cids_y2 = cids_y2[valid]
    data[~valid] -= 1

    # z-1
    cids_z1 = X * shape[1] * shape[2] + Y * shape[2] + Z - 1
    valid = (Z > 0) & ~bnd
    rids_z1 = ids_0[valid]
    cids_z1 = cids_z1[valid]
    data[~valid] -= 1

    # z+1
    cids_z2 = X * shape[1] * shape[2] + Y * shape[2] + Z + 1
    valid = (Z < shape[2] - 1) & ~bnd
    rids_z2 = ids_0[valid]
    cids_z2 = cids_z2[valid]
    data[~valid] -= 1

    if use_correspondences:
        data[boundary_indices] += 1

    rowx = np.hstack([ids_0, rids_x1, rids_x2, rids_y1, rids_y2, rids_z1, rids_z2])
    rowy = np.hstack([ids_0, cids_x1, cids_x2, cids_y1, cids_y2, cids_z1, cids_z2])
    rowv = np.hstack([data, -1 * np.ones(rowx.shape[0] - ids_0.shape[0])])

    dtype = None if use_correspondences else np.int8
    A = scipy.sparse.csr_matrix((rowv, (rowx, rowy)), shape=(X.shape[0], X.shape[0]), dtype=dtype)
    del rowx, rowy, rowv, X, Y, Z, data
    gc.collect()

    return A
