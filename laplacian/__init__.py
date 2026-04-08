"""
Laplacian refinement subpackage.

Provides slice-to-slice Laplacian registration (contour correspondence
matching + PDE-based interpolation) and the standalone PDE solver.
"""
from .correspondence import (
    getDataContours,
    getTemplateContours,
    getContours,
    estimate_normal,
    orient_normals_nd,
    orient2Dnormals,
    estimate2Dnormals,
    get2DCorrespondences_batch,
    get2DCorrespondences,
    sliceToSlice3DLaplacian,
)
from .solver import solveLaplacianFromCorrespondences
from .utils import (
    laplacianA1D,
    laplacianA2D,
    laplacianA3D,
    propagate_dirichlet_rhs,
)
