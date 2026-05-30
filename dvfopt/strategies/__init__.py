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

    Strategy                         (abstract base)
    ├── BarrierStrategy              penalty -> log-barrier L-BFGS-B
    ├── SLSQPFullGridStrategy        full-grid SLSQP + warm-restart
    ├── SLSQPWindowedStrategy        windowed SLSQP (Jdet + 2-tri)
    ├── SchwarzStrategy              overlapping-tile SLSQP/Schwarz
    ├── M10Strategy                  harmonic -> ALM -> polish
    ├── M14Strategy                  m10 -> soft-penalty -> repair -> polish
    └── M14SchwarzStrategy           cluster-localized m14 + global polish

Each strategy lives in its own file under :mod:`dvfopt.strategies`;
this ``__init__`` re-exports the public surface so existing
``from dvfopt.strategies import BarrierStrategy`` calls work unchanged.
"""

from __future__ import annotations

# Import concrete strategies — each calls @register_strategy at module
# import time, so importing this package gives you a populated registry.
from dvfopt.strategies.barrier import BarrierStrategy

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
from dvfopt.strategies.schwarz import SchwarzStrategy
from dvfopt.strategies.slsqp import SLSQPFullGridStrategy, SLSQPWindowedStrategy
from dvfopt.strategies.wallbreakers import (
    M10Strategy,
    M14SchwarzStrategy,
    M14Strategy,
)

__all__ = [
    'BarrierStrategy',
    'M10Strategy',
    'M14SchwarzStrategy',
    'M14Strategy',
    'SLSQPFullGridStrategy',
    'SLSQPWindowedStrategy',
    'SchwarzStrategy',
    'Strategy',
    '_build_solve_info',
    'make_strategy',
    'register_strategy',
]
