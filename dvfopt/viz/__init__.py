"""Visualisation sub-package for deformation field correction.

Re-exports the main public functions so callers can write::

    from dvfopt.viz import plot_deformations, plot_grid_before_after
"""

from dvfopt.viz._style import (
    CMAP,
    INTERP,
    NEG_CONTOUR_COLOR,
    QUIVER_COLOR,
)
from dvfopt.viz.closeups import (
    plot_checkerboard_before_after,
    plot_neg_jdet_neighborhoods,
)
from dvfopt.viz.debug import DebugTracer
from dvfopt.viz.fields import (
    plot_deformation_field,
    plot_deformations,
    plot_initial_deformation,
    plot_jacobians_iteratively,
)
from dvfopt.viz.fields3d import (
    plot_deformation_grid_3d,
    plot_grid_before_after_3d,
    plot_jdet_3d,
    plot_jdet_3d_before_after,
    plot_jdet_slices,
    plot_neg_voxels_before_after,
)
from dvfopt.viz.grids import (
    plot_2d_deformation_grid,
    plot_deformed_quads,
    plot_deformed_quads_colored,
    plot_grid,
    plot_grid_before_after,
)
from dvfopt.viz.overview import (
    plot_before_after,
    plot_before_after_3d,
    plot_fold_overview,
    plot_fold_overview_3d,
    plot_solver_comparison,
)
from dvfopt.viz.snapshots import plot_step_snapshot
from dvfopt.viz.solveinfo import plot_solve_info
from dvfopt.viz.theme import (
    PALETTE,
    Palette,
    apply_theme,
    jdet_norm,
    reset_theme,
)
from dvfopt.viz.triangle_debug import (
    find_problematic_pixels,
    plot_problematic_triangles,
    plot_triangle_debug,
)

__all__ = [
    "CMAP",
    "INTERP",
    "NEG_CONTOUR_COLOR",
    "PALETTE",
    "QUIVER_COLOR",
    "DebugTracer",
    "Palette",
    "apply_theme",
    "find_problematic_pixels",
    "jdet_norm",
    "plot_2d_deformation_grid",
    "plot_before_after",
    "plot_before_after_3d",
    "plot_checkerboard_before_after",
    "plot_deformation_field",
    "plot_deformation_grid_3d",
    "plot_deformations",
    "plot_deformed_quads",
    "plot_deformed_quads_colored",
    "plot_fold_overview",
    "plot_fold_overview_3d",
    "plot_grid",
    "plot_grid_before_after",
    "plot_grid_before_after_3d",
    "plot_initial_deformation",
    "plot_jacobians_iteratively",
    "plot_jdet_3d",
    "plot_jdet_3d_before_after",
    "plot_jdet_slices",
    "plot_neg_jdet_neighborhoods",
    "plot_neg_voxels_before_after",
    "plot_problematic_triangles",
    "plot_solve_info",
    "plot_solver_comparison",
    "plot_step_snapshot",
    "plot_triangle_debug",
    "reset_theme",
]
