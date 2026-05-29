"""dvfopt — Deformation Vector Field Optimizer.

Correction of negative Jacobian determinants in 2D (and 3D) deformation
(displacement) fields via SLSQP-based optimisation.

Public API
----------
SLSQP-based correctors (Jdet constraint, windowed)::

    from dvfopt import iterative_serial, iterative_parallel, iterative_3d

2D 2-triangle correctors (in increasing capability on dense folds)::

    iterative_2d_tri_slsqp         # full-grid SLSQP + L1/L2 + warm-restart (notebook 14)
    iterative_2d_tri_barrier       # penalty -> log-barrier L-BFGS-B
    iterative_2d_tri_schwarz       # hybrid overlapping-tile Schwarz + per-cluster SLSQP
    iterative_2d_tri_harmonic_polished   # m10: harmonic seed + ALM + barrier polish
                                          # ("always-feasibility baseline")
    iterative_2d_tri_refine_repair       # m14: m10 seed + soft-penalty pull
                                          # + repair + polish. anchor='l1' = m14_l1.
    iterative_2d_tri_refine_repair_schwarz  # m14-Schwarz: cluster-localized m14
                                             # for large slices with sparse folds

2-triangle building blocks (use directly if assembling your own pipeline)::

    solve_cluster_2tri_2d          # per-cluster SLSQP with frozen-edge interior mask
    harmonic_extension_2d          # m02 — Laplacian extension over fold cores
    augmented_lagrangian_2d        # m03 — PHR-ALM with L-BFGS-B
    l2_refine_2d                   # m12 — soft-quadratic penalty refinement

Unified high-level API (lazy-imports torch)::

    from dvfopt import DVFopt, DVFoptConfig

Jacobian computation::

    from dvfopt import jacobian_det2D, jacobian_det3D, sitk_jacobian_determinant

DVF utilities::

    from dvfopt import generate_random_dvf, scale_dvf

Laplacian interpolation (separate ``laplacian`` package)::

    from laplacian import solveLaplacianFromCorrespondences, sliceToSlice3DLaplacian, laplacianA3D

Visualisation (imports matplotlib)::

    from dvfopt.viz import plot_deformations, plot_grid_before_after

Which 2D 2-triangle corrector to pick?
--------------------------------------
* Mild folds, full-grid problem fits in memory: ``iterative_2d_tri_slsqp``
  with ``anchor='l1'`` — simplest, smallest L1 deviation.
* Need the strict 100%-feasibility guarantee even on dense slices
  (e.g. B0039 z=12): ``iterative_2d_tri_harmonic_polished`` (fast,
  larger L2) or ``iterative_2d_tri_refine_repair(anchor='l1')`` (slower,
  ~half the L2, ~80% less L1).
* Many small fold clusters across a large slice: ``iterative_2d_tri_schwarz``.
* Large slice (e.g. full 320x456) with sparse-to-moderate folds:
  ``iterative_2d_tri_refine_repair_schwarz`` — m14 with cluster-localized
  domain decomposition. ~5x faster than global m14 on the full B0039
  z=12 slice with ~11% lower L1.
"""

# -- Package metadata -------------------------------------------------------
__version__ = "0.1.0"

# -- Core solvers ------------------------------------------------------------
from dvfopt.core import (
    iterative_serial,
    iterative_parallel,
    iterative_3d,
)
from dvfopt.core.iterative2d_tri_barrier import iterative_2d_tri_barrier
from dvfopt.core.iterative2d_tri_slsqp import iterative_2d_tri_slsqp
from dvfopt.core.iterative2d_tri_schwarz import iterative_2d_tri_schwarz
from dvfopt.core._cluster_2tri import solve_cluster_2tri_2d

# Wall-breaker methods promoted from notebooks/experiments/wall_breakers
# are imported lazily — they're large modules used by a minority of
# callers, and pulling them at package-load adds a noticeable import-time
# cost to the SLSQP-only path. The lazy-attribute hook below routes
# ``dvfopt.iterative_2d_tri_harmonic_polished`` and friends to the
# subpackage on first access.

# -- Jacobian computation ---------------------------------------------------
from dvfopt.jacobian import (
    jacobian_det2D,
    jacobian_det3D,
    sitk_jacobian_determinant,
    shoelace_det2D,
    shoelace_constraint,
    triangle_det2D,
    triangle_constraint,
    injectivity_constraint,
)

# -- DVF generation / scaling ------------------------------------------------
from dvfopt.dvf import (
    generate_random_dvf,
    generate_random_dvf_3d,
    scale_dvf,
    scale_dvf_3d,
)

# -- I/O ---------------------------------------------------------------------
from dvfopt.io import load_nii_images

# -- Defaults ----------------------------------------------------------------
from dvfopt._defaults import DEFAULT_PARAMS

# -- Lazy attributes ---------------------------------------------------------
# ``dvfopt.unified`` pulls in the barrier solver, which imports torch at
# module load. Defer that cost so ``import dvfopt`` is cheap for callers
# that only need the SLSQP path. The wall-breaker subpackage is similarly
# heavy (scipy.sparse, scipy.ndimage, scipy.optimize) and only a minority
# of callers need it.
_LAZY_UNIFIED = {'DVFopt', 'DVFoptConfig', 'Result', 'SliceResult'}
_LAZY_WALLBREAKERS = {
    'iterative_2d_tri_harmonic_polished',         # m10
    'iterative_2d_tri_refine_repair',             # m14 (anchor='l1' = m14_l1)
    'iterative_2d_tri_refine_repair_schwarz',     # m14-Schwarz (cluster-localized)
    'harmonic_extension_2d',                      # m02 building block
    'augmented_lagrangian_2d',                    # m03 building block
    'l2_refine_2d',                               # m12 building block
}


def __getattr__(name):
    if name in _LAZY_UNIFIED:
        from dvfopt import unified
        return getattr(unified, name)
    if name in _LAZY_WALLBREAKERS:
        from dvfopt.core import wallbreakers
        return getattr(wallbreakers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
