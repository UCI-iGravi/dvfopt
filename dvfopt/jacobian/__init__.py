"""Jacobian determinant computation — unified 2D/3D entry point."""

from dvfopt.jacobian.numpy_jdet import (
    _numpy_jdet_2d,
    jacobian_det2D,
    _numpy_jdet_3d,
    jacobian_det3D,
)
from dvfopt.jacobian.sitk_jdet import (
    sitk_jacobian_determinant,
)
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
from dvfopt.jacobian.intersection import has_quad_self_intersections

__all__ = [
    "jacobian_det2D",
    "jacobian_det3D",
    "sitk_jacobian_determinant",
    "shoelace_det2D",
    "shoelace_constraint",
    "injectivity_constraint",
    "has_quad_self_intersections",
]
