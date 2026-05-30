"""Analytical gradient (Jacobian matrix) of the 3D Jdet constraint.

The 3D Jacobian determinant is the determinant of the 3x3 deformation
gradient tensor F, where F[i,j] = delta[i,j] + d(displacement_i)/d(x_j).

Using np.gradient for all 9 partial derivatives:

    | a11  a12  a13 |     | 1+ddx_dx   ddx_dy   ddx_dz |
F = | a21  a22  a23 |  =  |   ddy_dx 1+ddy_dy   ddy_dz |
    | a31  a32  a33 |     |   ddz_dx   ddz_dy 1+ddz_dz |

J = det(F).  For each voxel ``p``:

    dJ(p)/dF[i,j] = cofactor C_{i,j}(p)

and ``F[i,j]`` is itself a linear functional of the displacement field via
the np.gradient stencil along axis ``j``.  Letting ``G_axis`` be the
sparse ``(N, N)`` matrix encoding that gradient operator, the full
``(N, 3N)`` constraint Jacobian is the horizontal stack

    [ diag(C11) G_x + diag(C12) G_y + diag(C13) G_z |
      diag(C21) G_x + diag(C22) G_y + diag(C23) G_z |
      diag(C31) G_x + diag(C32) G_y + diag(C33) G_z ]

which is what this module assembles.  This replaces a per-voxel Python
loop with three cached sparse operators plus diagonal scaling -- the
inner SLSQP loop is now dominated by scipy native code.
"""

import numpy as np
import scipy.sparse

from dvfopt.core.slsqp._grad_op import gradient_operator, scale_rows


def _gradient_stencil(idx, n):
    """Return ``(indices, coefficients)`` for ``np.gradient`` at position ``idx`` in a dimension of size ``n``.

    Kept as a public helper because the test suite verifies stencil
    coefficients independently of the assembled sparse matrix.
    """
    if n == 1:
        return [idx], [0.0]
    if idx == 0:
        return [0, 1], [-1.0, 1.0]
    if idx == n - 1:
        return [n - 2, n - 1], [-1.0, 1.0]
    return [idx - 1, idx + 1], [-0.5, 0.5]


def jdet_constraint_jacobian_3d(phi_flat, subvolume_size, freeze_mask=None):
    """Sparse Jacobian matrix of the 3D Jdet constraint w.r.t. ``phi_flat``.

    Parameters
    ----------
    phi_flat : 1-D array
        Packed as ``[dx_flat, dy_flat, dz_flat]``.
    subvolume_size : tuple
        ``(sz, sy, sx)`` sub-volume size.
    freeze_mask : ndarray or None
        Boolean mask, shape ``(sz, sy, sx)``.  When given, only non-frozen
        voxels are included in the constraint (matches the row ordering
        produced by ``np.where(~freeze_mask)`` in C-order).

    Returns
    -------
    scipy.sparse.csr_matrix, shape ``(n_constraints, len(phi_flat))``
    """
    sz, sy, sx = subvolume_size
    voxels = sz * sy * sx
    shape = (sz, sy, sx)

    dx = phi_flat[:voxels].reshape(shape)
    dy = phi_flat[voxels : 2 * voxels].reshape(shape)
    dz = phi_flat[2 * voxels :].reshape(shape)

    a11 = 1 + np.gradient(dx, axis=2)
    a12 = np.gradient(dx, axis=1)
    a13 = np.gradient(dx, axis=0)
    a21 = np.gradient(dy, axis=2)
    a22 = 1 + np.gradient(dy, axis=1)
    a23 = np.gradient(dy, axis=0)
    a31 = np.gradient(dz, axis=2)
    a32 = np.gradient(dz, axis=1)
    a33 = 1 + np.gradient(dz, axis=0)

    # Cofactors of F at every voxel (full grids, not just one voxel).
    C11 = a22 * a33 - a23 * a32
    C12 = -(a21 * a33 - a23 * a31)
    C13 = a21 * a32 - a22 * a31
    C21 = -(a12 * a33 - a13 * a32)
    C22 = a11 * a33 - a13 * a31
    C23 = -(a11 * a32 - a12 * a31)
    C31 = a12 * a23 - a13 * a22
    C32 = -(a11 * a23 - a13 * a21)
    C33 = a11 * a22 - a12 * a21

    # Cached sparse gradient operators -- one per axis, geometry only.
    G_x = gradient_operator(shape, axis=2)
    G_y = gradient_operator(shape, axis=1)
    G_z = gradient_operator(shape, axis=0)

    # Build each (N, N) channel block, then horizontally stack to (N, 3N).
    M_dx = scale_rows(C11, G_x) + scale_rows(C12, G_y) + scale_rows(C13, G_z)
    M_dy = scale_rows(C21, G_x) + scale_rows(C22, G_y) + scale_rows(C23, G_z)
    M_dz = scale_rows(C31, G_x) + scale_rows(C32, G_y) + scale_rows(C33, G_z)

    J = scipy.sparse.hstack([M_dx, M_dy, M_dz], format="csr")

    if freeze_mask is not None:
        keep = ~np.asarray(freeze_mask).ravel()
        J = J[keep, :]

    return J
