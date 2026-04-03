"""3D constraint builders for SLSQP optimisation."""

import numpy as np
import scipy.sparse
from scipy.optimize import LinearConstraint, NonlinearConstraint

from dvfopt._defaults import _unpack_size_3d
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_3d
from dvfopt.core.gradients3d import jdet_constraint_jacobian_3d


def jacobian_constraint_3d(phi_flat, subvolume_size, freeze_mask=None):
    """Return flattened Jacobian determinant values for optimiser constraints.

    phi_flat packing: ``[dx_flat, dy_flat, dz_flat]``.

    When *freeze_mask* is given, only non-frozen voxels are returned.
    """
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    voxels = sz * sy * sx
    dx = phi_flat[:voxels].reshape((sz, sy, sx))
    dy = phi_flat[voxels:2 * voxels].reshape((sz, sy, sx))
    dz = phi_flat[2 * voxels:].reshape((sz, sy, sx))
    jdet = _numpy_jdet_3d(dz, dy, dx)
    if freeze_mask is not None:
        return jdet[~freeze_mask].flatten()
    return jdet.flatten()


def _build_constraints_3d(phi_sub_flat, subvolume_size, freeze_mask, threshold):
    """Build SLSQP constraints for a 3D sub-volume optimisation.

    The Jacobian constraint excludes only frozen boundary voxels.
    Grid-edge boundary voxels are NOT frozen and ARE constrained.
    """
    fm = freeze_mask
    nlc = NonlinearConstraint(
        lambda phi1: jacobian_constraint_3d(phi1, subvolume_size, fm),
        threshold, np.inf,
        jac=lambda phi1: jdet_constraint_jacobian_3d(phi1, subvolume_size, fm),
    )
    constraints = [nlc]

    if freeze_mask.any():
        sz, sy, sx = _unpack_size_3d(subvolume_size)
        voxels = sz * sy * sx
        edge_indices = np.argwhere(freeze_mask)
        fixed_indices = []
        for z, y, x in edge_indices:
            idx = z * sy * sx + y * sx + x
            fixed_indices.extend([idx, idx + voxels, idx + 2 * voxels])

        fixed_indices = np.array(fixed_indices)
        fixed_values = phi_sub_flat[fixed_indices]
        n_fixed = len(fixed_indices)
        A_eq = scipy.sparse.csr_matrix(
            (np.ones(n_fixed), (np.arange(n_fixed), fixed_indices)),
            shape=(n_fixed, phi_sub_flat.size))

        constraints.append(LinearConstraint(A_eq, fixed_values, fixed_values))

    return constraints
