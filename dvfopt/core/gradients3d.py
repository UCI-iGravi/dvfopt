"""Analytical gradient (Jacobian matrix) of the 3D Jdet constraint.

The 3D Jacobian determinant is the determinant of the 3x3 deformation
gradient tensor F, where F[i,j] = delta[i,j] + d(displacement_i)/d(x_j).

Using np.gradient for all 9 partial derivatives:

    | a11  a12  a13 |     | 1+ddx_dx   ddx_dy   ddx_dz |
F = | a21  a22  a23 |  =  |   ddy_dx 1+ddy_dy   ddy_dz |
    | a31  a32  a33 |     |   ddz_dx   ddz_dy 1+ddz_dz |

J = det(F) expanded by the first row:
  = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)

dJ/d(dx) = C11 * d(ddx_dx)/d(dx) + C12 * d(ddx_dy)/d(dx) + C13 * d(ddx_dz)/d(dx)
dJ/d(dy) = C21 * d(ddy_dx)/d(dy) + C22 * d(ddy_dy)/d(dy) + C23 * d(ddy_dz)/d(dy)
dJ/d(dz) = C31 * d(ddz_dx)/d(dz) + C32 * d(ddz_dy)/d(dz) + C33 * d(ddz_dz)/d(dz)

where C_ij are cofactors of F.
"""

import numpy as np
import scipy.sparse


def _gradient_stencil(idx, n):
    """Return (indices, coefficients) for np.gradient at position idx in dim of size n."""
    if n == 1:
        return [idx], [0.0]
    if idx == 0:
        return [0, 1], [-1.0, 1.0]
    if idx == n - 1:
        return [n - 2, n - 1], [-1.0, 1.0]
    return [idx - 1, idx + 1], [-0.5, 0.5]


def jdet_constraint_jacobian_3d(phi_flat, subvolume_size, freeze_mask=None):
    """Sparse Jacobian matrix of the 3D Jdet constraint w.r.t. phi_flat.

    Parameters
    ----------
    phi_flat : 1-D array
        Packed as ``[dx_flat, dy_flat, dz_flat]``.
    subvolume_size : tuple
        ``(sz, sy, sx)`` sub-volume size.
    freeze_mask : ndarray or None
        Boolean mask, shape ``(sz, sy, sx)``.  When given, only non-frozen
        voxels are included in the constraint.

    Returns
    -------
    scipy.sparse.csr_matrix, shape ``(n_constraints, len(phi_flat))``
    """
    sz, sy, sx = subvolume_size
    voxels = sz * sy * sx

    dx = phi_flat[:voxels].reshape(sz, sy, sx)
    dy = phi_flat[voxels:2 * voxels].reshape(sz, sy, sx)
    dz = phi_flat[2 * voxels:].reshape(sz, sy, sx)

    # Compute all 9 gradient components
    ddx_dx = np.gradient(dx, axis=2)
    ddx_dy = np.gradient(dx, axis=1)
    ddx_dz = np.gradient(dx, axis=0)
    ddy_dx = np.gradient(dy, axis=2)
    ddy_dy = np.gradient(dy, axis=1)
    ddy_dz = np.gradient(dy, axis=0)
    ddz_dx = np.gradient(dz, axis=2)
    ddz_dy = np.gradient(dz, axis=1)
    ddz_dz = np.gradient(dz, axis=0)

    a11 = 1 + ddx_dx;  a12 = ddx_dy;      a13 = ddx_dz
    a21 = ddy_dx;       a22 = 1 + ddy_dy;  a23 = ddy_dz
    a31 = ddz_dx;       a32 = ddz_dy;      a33 = 1 + ddz_dz

    # Determine which voxels to include
    if freeze_mask is not None:
        voxel_coords = list(zip(*np.where(~freeze_mask)))
    else:
        voxel_coords = [(k, i, j) for k in range(sz)
                        for i in range(sy) for j in range(sx)]

    n_rows = len(voxel_coords)
    rows = []
    cols = []
    vals = []

    for row_idx, (k, i, j) in enumerate(voxel_coords):
        # Cofactors at this voxel
        C11 = a22[k,i,j] * a33[k,i,j] - a23[k,i,j] * a32[k,i,j]
        C12 = -(a21[k,i,j] * a33[k,i,j] - a23[k,i,j] * a31[k,i,j])
        C13 = a21[k,i,j] * a32[k,i,j] - a22[k,i,j] * a31[k,i,j]

        C21 = -(a12[k,i,j] * a33[k,i,j] - a13[k,i,j] * a32[k,i,j])
        C22 = a11[k,i,j] * a33[k,i,j] - a13[k,i,j] * a31[k,i,j]
        C23 = -(a11[k,i,j] * a32[k,i,j] - a12[k,i,j] * a31[k,i,j])

        C31 = a12[k,i,j] * a23[k,i,j] - a13[k,i,j] * a22[k,i,j]
        C32 = -(a11[k,i,j] * a23[k,i,j] - a13[k,i,j] * a21[k,i,j])
        C33 = a11[k,i,j] * a22[k,i,j] - a12[k,i,j] * a21[k,i,j]

        # --- dx block (columns 0..voxels-1) ---
        # ddx_dx: gradient along axis=2 (x) at column j
        stencil_j, coeff_j = _gradient_stencil(j, sx)
        for jj, c in zip(stencil_j, coeff_j):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(k * sy * sx + i * sx + jj)
                vals.append(C11 * c)

        # ddx_dy: gradient along axis=1 (y) at row i
        stencil_i, coeff_i = _gradient_stencil(i, sy)
        for ii, c in zip(stencil_i, coeff_i):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(k * sy * sx + ii * sx + j)
                vals.append(C12 * c)

        # ddx_dz: gradient along axis=0 (z) at slice k
        stencil_k, coeff_k = _gradient_stencil(k, sz)
        for kk, c in zip(stencil_k, coeff_k):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(kk * sy * sx + i * sx + j)
                vals.append(C13 * c)

        # --- dy block (columns voxels..2*voxels-1) ---
        # ddy_dx: gradient along axis=2 (x) at column j
        for jj, c in zip(stencil_j, coeff_j):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(voxels + k * sy * sx + i * sx + jj)
                vals.append(C21 * c)

        # ddy_dy: gradient along axis=1 (y) at row i
        for ii, c in zip(stencil_i, coeff_i):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(voxels + k * sy * sx + ii * sx + j)
                vals.append(C22 * c)

        # ddy_dz: gradient along axis=0 (z) at slice k
        for kk, c in zip(stencil_k, coeff_k):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(voxels + kk * sy * sx + i * sx + j)
                vals.append(C23 * c)

        # --- dz block (columns 2*voxels..3*voxels-1) ---
        # ddz_dx: gradient along axis=2 (x) at column j
        for jj, c in zip(stencil_j, coeff_j):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(2 * voxels + k * sy * sx + i * sx + jj)
                vals.append(C31 * c)

        # ddz_dy: gradient along axis=1 (y) at row i
        for ii, c in zip(stencil_i, coeff_i):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(2 * voxels + k * sy * sx + ii * sx + j)
                vals.append(C32 * c)

        # ddz_dz: gradient along axis=0 (z) at slice k
        for kk, c in zip(stencil_k, coeff_k):
            if c != 0.0:
                rows.append(row_idx)
                cols.append(2 * voxels + kk * sy * sx + i * sx + j)
                vals.append(C33 * c)

    n_cols = 3 * voxels
    return scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_rows, n_cols))
