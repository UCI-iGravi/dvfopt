"""dvfopt — Deformation Vector Field Optimizer.

Correction of negative Jacobian determinants in 2D (and 3D) deformation
(displacement) fields. The package is organized around three orthogonal
axes:

* :class:`Constraint`  — what makes a configuration "feasible"
  (2-triangle areas, Jacobian determinant, 6-tet, ...).
* :class:`Objective`   — what to minimize subject to feasibility
  (L1, L2, or none = feasibility-only).
* :class:`Strategy`    — how to actually optimize
  (barrier, SLSQP, m10/m14 wallbreakers, ...).

A :class:`Solver` composes one of each. Most users want the high-level
facade::

    from dvfopt import Solver, correct_dvf

    # one-shot
    result = correct_dvf(phi_in, constraint='2tri', objective='l1',
                          strategy='auto')

    # explicit composition
    solver = Solver.from_spec(constraint='2tri', objective='l1',
                               strategy='m14_schwarz',
                               shape=(320, 456))
    result = solver.fit(phi_in)
    print(f'feasible={result.feasible}  L1={result.info.get("L1")}')

For per-slice processing across a 3D volume + tabular reports + plots,
use the higher-level facade :class:`DVFopt` / :class:`DVFoptConfig`.

Public API
----------

**Core composables**::

    from dvfopt import (
        Solver, correct_dvf, auto_strategy,
        Constraint, TriConstraint2D, TriConstraint2DFullCoverage,
        JdetConstraint2D, JdetConstraint3D, make_constraint,
        Objective, L1Objective, L2Objective, NoneObjective, make_objective,
        Strategy, BarrierStrategy, SLSQPFullGridStrategy,
        SLSQPWindowedStrategy, SchwarzStrategy, M10Strategy, M14Strategy,
        M14SchwarzStrategy, make_strategy,
    )

**High-level facade (3D volume, tabular reports)**::

    from dvfopt import DVFopt, DVFoptConfig

**Constraint primitives**::

    from dvfopt import (
        jacobian_det2D, jacobian_det3D, sitk_jacobian_determinant,
        shoelace_det2D, triangle_det2D,
    )

**DVF utilities + I/O**::

    from dvfopt import generate_random_dvf, scale_dvf, load_nii_images

Strategy selection guide
------------------------

Inside the 2-triangle constraint family (the canonical case), pick the
strategy by initial fold density:

* **Mild folds** (``n_neg <= 100``): :class:`SLSQPFullGridStrategy`
  (KKT semantics, smallest L1 with ``L1Objective``).
* **Moderate-to-dense** (100 < n_neg < 5000): :class:`BarrierStrategy`
  — dominates SLSQP by 100x at this density.
* **Many small fold clusters across a big slice**: :class:`SchwarzStrategy`.
* **Extreme density** (n_neg > 5000, e.g. full B0039 z=12 slice with
  8978 folds): the wallbreakers reach feasibility where barrier
  doesn't:
  - :class:`M10Strategy` — L2-optimal, fast, larger L1
  - :class:`M14Strategy` — m10 seed + refinement, smallest L1
  - :class:`M14SchwarzStrategy` — m14 with cluster-localized domain
    decomposition; ~5x faster than m14 on large slices

For Jdet (no wallbreakers): :class:`BarrierStrategy` for dense,
:class:`SLSQPWindowedStrategy` for mild. 3D Jdet supported by both.

The :func:`auto_strategy` helper encodes this routing as a function.
"""

# -- Package metadata -------------------------------------------------------
__version__ = "0.3.0"  # 6-tet 3D constraint + visualization theme + 2tri default flip

# -- New API: constraints, objectives, strategies, solver -------------------
# -- Logging ----------------------------------------------------------------
# A package-level logger ``dvfopt`` is set up with a NullHandler. Callers
# enable output via ``dvfopt.enable_default_handler()`` (simple stderr) or
# attach their own handlers to the ``dvfopt`` logger.
# -- Defaults ---------------------------------------------------------------
from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt._logging import enable_default_handler, logger
from dvfopt.constraints import (
    Constraint,
    JdetConstraint2D,
    JdetConstraint3D,
    PhiPack,
    Tet6Constraint3D,
    TriConstraint2D,
    TriConstraint2DFullCoverage,
    make_constraint,
    register_constraint,
)

# -- DVF generation / scaling ----------------------------------------------
from dvfopt.dvf import (
    generate_random_dvf,
    generate_random_dvf_3d,
    scale_dvf,
    scale_dvf_3d,
)

# -- Exceptions --------------------------------------------------------------
from dvfopt.exceptions import (
    BudgetExhaustedError,
    DVFoptError,
    FeasibilityError,
    IncompatibleConstraintError,
    SolverConfigError,
)

# -- I/O --------------------------------------------------------------------
from dvfopt.io import load_nii_images

# -- Jacobian primitives ----------------------------------------------------
from dvfopt.jacobian import (
    injectivity_constraint,
    jacobian_det2D,
    jacobian_det3D,
    shoelace_constraint,
    shoelace_det2D,
    sitk_jacobian_determinant,
    triangle_constraint,
    triangle_det2D,
)
from dvfopt.objectives import (
    L1Objective,
    L2Objective,
    NoneObjective,
    Objective,
    ScaledObjective,
    SumObjective,
    make_objective,
)
from dvfopt.solver import (
    PhaseInfo,
    SolveInfo,
    Solver,
    SolveResult,
    auto_strategy,
    correct_dvf,
)
from dvfopt.strategies import (
    BarrierStrategy,
    Harmonic3DStrategy,
    M10Strategy,
    M14SchwarzStrategy,
    M14Strategy,
    SchwarzStrategy,
    SLSQPFullGrid3DStrategy,
    SLSQPFullGridStrategy,
    SLSQPWindowedStrategy,
    Strategy,
    make_strategy,
    register_strategy,
)

# -- Validation helpers ------------------------------------------------------
# All package entry points route input through these. Surfaced publicly
# so callers can use them on their own data before constructing a Solver.
from dvfopt.validation import (
    coerce_to_ndarray,
    validate_dvf,
    validate_finite,
    validate_spatial_min_size,
)

# -- Lazy attributes: high-level facade --------------------------------------
# ``dvfopt.unified`` pulls in torch at import time. Defer that cost so
# ``import dvfopt`` is cheap for callers that only need the SLSQP path.
_LAZY_UNIFIED = {'DVFopt', 'DVFoptConfig', 'Result', 'SliceResult'}


def __getattr__(name):
    if name in _LAZY_UNIFIED:
        from dvfopt import unified

        return getattr(unified, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'DEFAULT_PARAMS',
    'BarrierStrategy',
    # Defaults
    'BudgetExhaustedError',
    # Composables
    'Constraint',
    # High-level facade (lazy)
    'DVFopt',
    'DVFoptConfig',
    'DVFoptError',
    'FeasibilityError',
    'Harmonic3DStrategy',
    'IncompatibleConstraintError',
    'JdetConstraint2D',
    'JdetConstraint3D',
    'L1Objective',
    'L2Objective',
    'M10Strategy',
    'M14SchwarzStrategy',
    'M14Strategy',
    'NoneObjective',
    'Objective',
    'PhaseInfo',
    'PhiPack',
    'Result',
    'SLSQPFullGrid3DStrategy',
    'SLSQPFullGridStrategy',
    'SLSQPWindowedStrategy',
    'ScaledObjective',
    'SchwarzStrategy',
    'SliceResult',
    'SolveInfo',
    'SolveResult',
    'Solver',
    'SolverConfigError',
    'Strategy',
    'SumObjective',
    'Tet6Constraint3D',
    'TriConstraint2D',
    'TriConstraint2DFullCoverage',
    'auto_strategy',
    'coerce_to_ndarray',
    'correct_dvf',
    'enable_default_handler',
    # DVF utilities
    'generate_random_dvf',
    'generate_random_dvf_3d',
    'injectivity_constraint',
    # Jacobian primitives
    'jacobian_det2D',
    'jacobian_det3D',
    # I/O
    'load_nii_images',
    'logger',
    'make_constraint',
    'make_objective',
    'make_strategy',
    'register_constraint',
    'register_strategy',
    'scale_dvf',
    'scale_dvf_3d',
    'shoelace_constraint',
    'shoelace_det2D',
    'sitk_jacobian_determinant',
    'triangle_constraint',
    'triangle_det2D',
    'validate_dvf',
    'validate_finite',
    'validate_spatial_min_size',
]
