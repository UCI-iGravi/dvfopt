"""Visualisation sub-package for deformation field correction.

Re-exports the main public functions so callers can write::

    from dvfopt.viz import plot_deformations, plot_grid_before_after
"""

from dvfopt.viz._style import (
    CMAP,
    INTERP,
    QUIVER_COLOR,
    NEG_CONTOUR_COLOR,
)
from dvfopt.viz.snapshots import plot_step_snapshot
from dvfopt.viz.fields import (
    plot_initial_deformation,
    plot_deformations,
    plot_jacobians_iteratively,
    plot_deformation_field,
)
from dvfopt.viz.grids import (
    plot_2d_deformation_grid,
    plot_deformed_quads,
    plot_deformed_quads_colored,
    plot_grid_before_after,
)
from dvfopt.viz.closeups import (
    plot_checkerboard_before_after,
    plot_neg_jdet_neighborhoods,
)
from dvfopt.viz.pipeline import run_lapl_and_correction
from dvfopt.viz.fields3d import (
    plot_jdet_slices,
    plot_jdet_3d,
    plot_jdet_3d_before_after,
    plot_neg_voxels_before_after,
    plot_deformation_grid_3d,
    plot_grid_before_after_3d,
)

__all__ = [
    "CMAP",
    "INTERP",
    "QUIVER_COLOR",
    "NEG_CONTOUR_COLOR",
    "plot_step_snapshot",
    "plot_initial_deformation",
    "plot_deformations",
    "plot_jacobians_iteratively",
    "plot_deformation_field",
    "plot_2d_deformation_grid",
    "plot_deformed_quads",
    "plot_deformed_quads_colored",
    "plot_grid_before_after",
    "plot_checkerboard_before_after",
    "plot_neg_jdet_neighborhoods",
    "run_lapl_and_correction",
    "plot_jdet_slices",
    "plot_jdet_3d",
    "plot_jdet_3d_before_after",
    "plot_neg_voxels_before_after",
    "plot_deformation_grid_3d",
    "plot_grid_before_after_3d",
]
