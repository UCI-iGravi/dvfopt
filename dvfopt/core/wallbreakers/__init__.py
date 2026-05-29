"""Wall-breaker correction methods promoted from ``notebooks/experiments/wall_breakers/``.

These are the methods that proved out as the best-performing on the
dense, hard B0039 slices (including z=12 and similar) — the ones the
SLSQP-only pipeline can't crack because the active-set line search
degenerates at the constraint wall.

Public API
----------
* :func:`harmonic_extension_2d` (m02): drop-in feasible-seed via Laplacian
  extension over fold cores.
* :func:`augmented_lagrangian_2d` (m03): PHR-ALM with L-BFGS-B inner.
* :func:`l2_refine_2d` (m12): soft-quadratic penalty refinement of a
  feasible seed.
* :func:`iterative_2d_tri_harmonic_polished` (m10): 3-stage —
  harmonic → ALM → log-barrier L2 polish. The "always-feasibility"
  baseline. 100% feasibility on the original B0039 DVF.
* :func:`iterative_2d_tri_refine_repair` (m14): 4-stage — m10 seed →
  soft-penalty pull → harmonic repair → barrier polish. The L2/L1
  winner. Run with ``anchor='l1'`` for the smallest deviation from input.
"""
from dvfopt.core.wallbreakers._harmonic import harmonic_extension_2d
from dvfopt.core.wallbreakers._alm import augmented_lagrangian_2d
from dvfopt.core.wallbreakers._l2_refine import l2_refine_2d
from dvfopt.core.wallbreakers._harmonic_polished import (
    iterative_2d_tri_harmonic_polished,
)
from dvfopt.core.wallbreakers._refine_repair import (
    iterative_2d_tri_refine_repair,
)

__all__ = [
    'harmonic_extension_2d',
    'augmented_lagrangian_2d',
    'l2_refine_2d',
    'iterative_2d_tri_harmonic_polished',
    'iterative_2d_tri_refine_repair',
]
