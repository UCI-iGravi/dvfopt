"""Jacobian determinant computation — unified 2D/3D entry point."""

from dvfopt.jacobian.injectivity_radius import (
    cell_min_jdet_2d,
    cell_to_pixel_min,
    ift_radius_2d,
)
from dvfopt.jacobian.intersection import has_quad_self_intersections
from dvfopt.jacobian.monotonicity import (
    _diagonal_monotonicity_diffs_2d,
    _monotonicity_diffs_2d,
    injectivity_constraint,
)
from dvfopt.jacobian.numpy_jdet import (
    _numpy_jdet_2d,
    _numpy_jdet_3d,
    jacobian_det2D,
    jacobian_det3D,
)
from dvfopt.jacobian.shoelace import (
    _all_triangle_areas_2d,
    _shoelace_areas_2d,
    _triangulated_shoelace_areas_2d,
    shoelace_constraint,
    shoelace_det2D,
    triangle_constraint,
    triangle_det2D,
    triangulated_shoelace_constraint,
    triangulated_shoelace_det2D,
)
from dvfopt.jacobian.sitk_jdet import (
    sitk_jacobian_determinant,
)
from dvfopt.jacobian.tetrahedron_sign import (
    six_tet_fold_classification,
    six_tet_volumes_3d,
    tet_grad_T_v,
    tet_volumes_flat,
)
from dvfopt.jacobian.triangle_sign import (
    _triangle_areas_2d,
    _triangle_signs_2d,
    triangle_sign_areas2D,
    triangle_sign_constraint,
    triangle_sign_count_negatives,
    triangle_sign_det2D,
)

__all__ = [
    # Private re-exports (internal access for solvers / strategies).
    "_all_triangle_areas_2d",
    "_diagonal_monotonicity_diffs_2d",
    "_monotonicity_diffs_2d",
    "_numpy_jdet_2d",
    "_numpy_jdet_3d",
    "_shoelace_areas_2d",
    "_triangle_areas_2d",
    "_triangle_signs_2d",
    "_triangulated_shoelace_areas_2d",
    # Public surface.
    "cell_min_jdet_2d",
    "cell_to_pixel_min",
    "has_quad_self_intersections",
    "ift_radius_2d",
    "injectivity_constraint",
    "jacobian_det2D",
    "jacobian_det3D",
    "shoelace_constraint",
    "shoelace_det2D",
    "sitk_jacobian_determinant",
    "six_tet_fold_classification",
    "six_tet_volumes_3d",
    "tet_grad_T_v",
    "tet_volumes_flat",
    "triangle_constraint",
    "triangle_det2D",
    "triangle_sign_areas2D",
    "triangle_sign_constraint",
    "triangle_sign_count_negatives",
    "triangle_sign_det2D",
    "triangulated_shoelace_constraint",
    "triangulated_shoelace_det2D",
]
