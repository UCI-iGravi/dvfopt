"""Constraint builders and quality-map helpers for SLSQP optimisation."""

import numpy as np
import scipy.sparse
from scipy.optimize import LinearConstraint, NonlinearConstraint

from dvfopt._defaults import _unpack_size
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d, jacobian_det2D
from dvfopt.jacobian.shoelace import (
    _shoelace_areas_2d,
    shoelace_det2D,
    shoelace_constraint,
)
from dvfopt.jacobian.monotonicity import (
    _monotonicity_diffs_2d,
    _diagonal_monotonicity_diffs_2d,
    injectivity_constraint,
)
from dvfopt.core.gradients import (
    jdet_constraint_jacobian_2d,
    shoelace_constraint_jacobian_2d,
    injectivity_constraint_jacobian_2d,
)


def jacobian_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened Jacobian determinant values for optimiser constraints."""
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    jdet = _numpy_jdet_2d(dy, dx)
    if exclude_boundaries:
        return jdet[1:-1, 1:-1].flatten()
    else:
        return jdet.flatten()


def _quality_map(phi, enforce_shoelace, enforce_injectivity=False,
                 jacobian_matrix=None):
    """Per-pixel quality metric combining gradient-Jdet and optional extras.

    When both *enforce_shoelace* and *enforce_injectivity* are ``False``,
    returns ``jacobian_det2D(phi)``.  Otherwise, each active metric is
    spread to per-pixel values and the element-wise minimum is returned
    so that the worst violation drives pixel selection and convergence.

    Parameters
    ----------
    jacobian_matrix : ndarray or None
        Pre-computed Jacobian; avoids redundant recomputation.

    Returns shape ``(1, H, W)`` — same as ``jacobian_det2D``.
    """
    jdet = jacobian_matrix if jacobian_matrix is not None else jacobian_det2D(phi)
    if not enforce_shoelace and not enforce_injectivity:
        return jdet
    result = jdet.copy()
    H, W = jdet.shape[1:]

    if enforce_shoelace:
        areas = shoelace_det2D(phi)           # (1, H-1, W-1)
        shoe = np.full((1, H, W), np.inf)
        a = areas[0]
        shoe[0, :-1, :-1] = np.minimum(shoe[0, :-1, :-1], a)
        shoe[0, :-1, 1:]  = np.minimum(shoe[0, :-1, 1:],  a)
        shoe[0, 1:,  :-1] = np.minimum(shoe[0, 1:,  :-1], a)
        shoe[0, 1:,  1:]  = np.minimum(shoe[0, 1:,  1:],  a)
        result = np.minimum(result, shoe)

    if enforce_injectivity:
        dy = phi[0]
        dx = phi[1]
        h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
        d1, d2 = _diagonal_monotonicity_diffs_2d(dy, dx)
        mono = np.full((1, H, W), np.inf)
        # h_mono is (H, W-1): gap between col j and col j+1
        mono[0, :, :-1] = np.minimum(mono[0, :, :-1], h_mono)
        mono[0, :, 1:]  = np.minimum(mono[0, :, 1:],  h_mono)
        # v_mono is (H-1, W): gap between row i and row i+1
        mono[0, :-1, :] = np.minimum(mono[0, :-1, :], v_mono)
        mono[0, 1:,  :] = np.minimum(mono[0, 1:,  :], v_mono)
        # d1 is (H-1, W-1): involves (r, c+1) and (r+1, c)
        mono[0, :-1, 1:] = np.minimum(mono[0, :-1, 1:], d1)
        mono[0, 1:, :-1] = np.minimum(mono[0, 1:, :-1], d1)
        # d2 is (H-1, W-1): involves (r+1, c) and (r, c+1)
        mono[0, 1:, :-1] = np.minimum(mono[0, 1:, :-1], d2)
        mono[0, :-1, 1:] = np.minimum(mono[0, :-1, 1:], d2)
        result = np.minimum(result, mono)

    return result


def _build_constraints(phi_sub_flat, submatrix_size, is_at_edge,
                       window_reached_max, threshold, enforce_shoelace=False,
                       enforce_injectivity=False, injectivity_threshold=None):
    """Build SLSQP constraints for a sub-window optimisation.

    Returns a list of constraint objects suitable for
    ``scipy.optimize.minimize``.

    When *enforce_shoelace* is ``True``, an additional
    ``NonlinearConstraint`` requires all shoelace quad-cell areas to
    exceed *threshold* as well.

    When *enforce_injectivity* is ``True``, an additional
    ``NonlinearConstraint`` enforces monotonicity of deformed coordinates
    (h, v, and anti-diagonal), which together guarantee each deformed quad
    cell is convex — a sufficient condition for preventing self-intersections.

    Parameters
    ----------
    injectivity_threshold : float or None
        Lower bound used for the injectivity constraint.  When ``None``
        (default), falls back to *threshold*.  Setting a value larger than
        *threshold* (e.g. ``0.3``) forces more vertex separation in deformed
        space, preventing distant cells from overlapping under large shear.
    """
    exclude_bounds = not is_at_edge and not window_reached_max
    inj_lb = threshold if injectivity_threshold is None else injectivity_threshold

    nlc = NonlinearConstraint(
        lambda phi1: jacobian_constraint(phi1, submatrix_size, exclude_bounds),
        threshold, np.inf,
        jac=lambda phi1: jdet_constraint_jacobian_2d(phi1, submatrix_size, exclude_bounds),
    )
    constraints = [nlc]

    if enforce_shoelace:
        constraints.append(NonlinearConstraint(
            lambda phi1: shoelace_constraint(phi1, submatrix_size, exclude_bounds),
            threshold, np.inf,
            jac=lambda phi1: shoelace_constraint_jacobian_2d(phi1, submatrix_size, exclude_bounds),
        ))

    if enforce_injectivity:
        constraints.append(NonlinearConstraint(
            lambda phi1: injectivity_constraint(phi1, submatrix_size, exclude_bounds),
            inj_lb, np.inf,
            jac=lambda phi1: injectivity_constraint_jacobian_2d(phi1, submatrix_size, exclude_bounds),
        ))

    if exclude_bounds:
        sy, sx = _unpack_size(submatrix_size)
        edge_mask = np.zeros((sy, sx), dtype=bool)
        edge_mask[[0, -1], :] = True
        edge_mask[:, [0, -1]] = True

        edge_indices = np.argwhere(edge_mask)
        fixed_indices = []
        y_offset_sub = sy * sx
        for y, x in edge_indices:
            idx = y * sx + x
            fixed_indices.extend([idx, idx + y_offset_sub])

        fixed_indices = np.array(fixed_indices)
        fixed_values = phi_sub_flat[fixed_indices]
        n_fixed = len(fixed_indices)
        A_eq = scipy.sparse.csr_matrix(
            (np.ones(n_fixed), (np.arange(n_fixed), fixed_indices)),
            shape=(n_fixed, phi_sub_flat.size))

        constraints.append(LinearConstraint(A_eq, fixed_values, fixed_values))

    return constraints
