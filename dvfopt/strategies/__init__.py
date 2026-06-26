"""Optimization strategies for DVFopt.

A :class:`Strategy` is "how to actually optimize given a constraint and
an objective." Strategies are stateless callables; they accept

    ``phi_in`` (the input field, ``(C, *shape)`` ndarray)
    ``constraint`` (a :class:`Constraint` instance)
    ``objective`` (an :class:`Objective` instance)
    ``threshold`` (the feasibility lower bound)

plus per-strategy options. They return ``(phi_out, SolveInfo)``.

Strategies declare what they support via class attributes:

* ``supports_3d``         — does the strategy handle 3D fields?
* ``accepts_constraints`` — tuple of accepted Constraint subclasses, or
                             ``None`` to accept anything. Used by
                             :class:`Solver.__init__` to surface bad
                             compositions at construction time.

The strategy hierarchy:

    Strategy                                            (abstract base)
    ├── NMVFStrategy                                     neighborhood-mean heuristic (legacy)
    ├── BarrierStrategy                                  penalty -> log-barrier L-BFGS-B
    ├── SLSQPFullGridStrategy                            full-grid SLSQP + warm-restart
    ├── SLSQPWindowedStrategy                            windowed SLSQP (Jdet + 2-tri)
    ├── SchwarzStrategy                                  overlapping-tile SLSQP/Schwarz
    ├── SchwarzWrapperStrategy                           cluster-Schwarz wrapper around any inner Strategy (2D + 3D)
    ├── HarmonicALMBarrierStrategy                       (m10) harmonic -> PHR-ALM -> log-barrier polish
    ├── HarmonicALMRefineRepairStrategy                  (m14) m10 seed -> soft-penalty L2 refine -> harmonic repair -> log-barrier polish
    └── SchwarzHarmonicALMRefineRepairStrategy           (m14-schwarz) cluster-localized m14 + global polish (= SchwarzWrapperStrategy(inner=HarmonicALMRefineRepairStrategy()))

The "m10/m14/..." names are the original research tags and remain
exported as back-compat aliases (e.g. ``M10Strategy is
HarmonicALMBarrierStrategy``).

Each strategy lives in its own file under :mod:`dvfopt.strategies`;
this ``__init__`` re-exports the public surface so existing
``from dvfopt.strategies import BarrierStrategy`` calls work unchanged.
"""

from __future__ import annotations

# Import concrete strategies — each calls @register_strategy at module
# import time, so importing this package gives you a populated registry.
from dvfopt.strategies.barrier import (
    BarrierStrategy,
    BarrierTet3DTorchStrategy,
)

# Base + registry (no side effects beyond class registration in deps below).
# Internal: solver.py imports this. Kept exported for backward compat
# with any external code that referenced `dvfopt.strategies._STRATEGY_REGISTRY`.
from dvfopt.strategies.base import (
    _STRATEGY_REGISTRY,  # noqa: F401
    Strategy,
    _build_solve_info,
    make_strategy,
    register_strategy,
)
from dvfopt.strategies.nmvf import NMVFStrategy
from dvfopt.strategies.slp import SLPStrategy
from dvfopt.strategies.schwarz import SchwarzStrategy
from dvfopt.strategies.schwarz_wrapper import SchwarzWrapperStrategy
from dvfopt.strategies.slsqp import (
    SLSQPFullGrid3DStrategy,
    SLSQPFullGridStrategy,
    SLSQPWindowedStrategy,
)
from dvfopt.strategies.wallbreakers import (
    ALM3DStrategy,
    ActiveBandALM3DStrategy,
    CoupledKRing3DStrategy,
    Harmonic3DStrategy,
    HarmonicALMBarrier3DStrategy,
    HarmonicALMBarrierStrategy,
    HarmonicALMRefineRepair3DStrategy,
    HarmonicALMRefineRepairStrategy,
    M10Strategy,
    M10TetStrategy,
    M14Schwarz3DStrategy,
    M14SchwarzStrategy,
    M14Strategy,
    M14TetStrategy,
    SchwarzHarmonicALMRefineRepair3DStrategy,
    SchwarzHarmonicALMRefineRepairStrategy,
)

# Names in ``__all__`` are sorted alphabetically (ruff RUF022).
# Descriptive names live alongside the original ``M*Strategy`` aliases
# the package still exports — they are simply class identities pointing
# at the same dataclass.
__all__ = [
    'ALM3DStrategy',
    'BarrierStrategy',
    'BarrierTet3DTorchStrategy',
    'ActiveBandALM3DStrategy',
    'CoupledKRing3DStrategy',
    'Harmonic3DStrategy',
    'HarmonicALMBarrier3DStrategy',
    'HarmonicALMBarrierStrategy',
    'HarmonicALMRefineRepair3DStrategy',
    'HarmonicALMRefineRepairStrategy',
    'M10Strategy',
    'M10TetStrategy',
    'M14Schwarz3DStrategy',
    'M14SchwarzStrategy',
    'M14Strategy',
    'M14TetStrategy',
    'NMVFStrategy',
    'SLPStrategy',
    'SLSQPFullGrid3DStrategy',
    'SLSQPFullGridStrategy',
    'SLSQPWindowedStrategy',
    'SchwarzHarmonicALMRefineRepair3DStrategy',
    'SchwarzHarmonicALMRefineRepairStrategy',
    'SchwarzStrategy',
    'SchwarzWrapperStrategy',
    'Strategy',
    '_build_solve_info',
    'make_strategy',
    'register_strategy',
]
