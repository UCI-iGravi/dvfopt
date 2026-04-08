"""
Laplacian refinement subpackage.

Provides slice-to-slice Laplacian registration (contour correspondence
matching + PDE-based interpolation) and the standalone PDE solver.

Lightweight core (utils + solver) is imported eagerly.
The correspondence pipeline (requires scikit-image, joblib, tqdm) is
imported lazily on first access so that callers who only need the PDE
solver or matrix builders do not pay the cost of those optional
dependencies at import time.
"""
from .solver import solveLaplacianFromCorrespondences
from .utils import (
    laplacianA1D,
    laplacianA2D,
    laplacianA3D,
    propagate_dirichlet_rhs,
)

# Names provided by the lazily-loaded correspondence module.
_CORRESPONDENCE_NAMES = frozenset({
    "getDataContours",
    "getTemplateContours",
    "getContours",
    "estimate_normal",
    "orient_normals_nd",
    "orient2Dnormals",
    "estimate2Dnormals",
    "get2DCorrespondences_batch",
    "get2DCorrespondences",
    "sliceToSlice3DLaplacian",
})

_correspondence_module = None


def __getattr__(name):
    """Lazily import correspondence functions when first accessed."""
    if name in _CORRESPONDENCE_NAMES:
        global _correspondence_module
        if _correspondence_module is None:
            from . import correspondence as _correspondence_module  # noqa: F401
        return getattr(_correspondence_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
