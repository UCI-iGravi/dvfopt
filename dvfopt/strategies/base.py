"""Abstract :class:`Strategy` base class + shared helpers.

This is the contract every solver strategy implements. The concrete
strategies live in sibling files (`barrier.py`, `slsqp.py`, etc.) and
register themselves via :func:`register_strategy` at import time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from dvfopt.constraints import Constraint
from dvfopt.objectives import Objective


def _build_solve_info(strategy_name: str, info, threshold: float = 0.01):
    """Normalize a strategy's return into a populated :class:`SolveInfo`.

    Strategies should call this on their underlying implementation's
    output. The helper handles all three legacy shapes:

    * ``list[dict]`` — barrier-core / SLSQP history.
    * ``dict`` with ``'history': list`` — barrier-core info wrapper.
    * stage-keyed ``dict`` — m10/m14 per-stage stats.

    Returning a uniform :class:`SolveInfo` lets visualization code rely
    on ``.phases``, ``.total_iter``, ``.feasible_after_phase`` instead
    of branching on strategy.

    ``SolveInfo`` is imported lazily to avoid a circular dependency
    (the solver module imports :class:`Strategy`).
    """
    from dvfopt.solver import PhaseInfo, SolveInfo

    if isinstance(info, SolveInfo):
        return info
    if not info:
        return SolveInfo(strategy_name=strategy_name)
    if isinstance(info, list):
        return SolveInfo.from_legacy_history(strategy_name, info, threshold)
    if isinstance(info, dict):
        history = info.get('history')
        if isinstance(history, list):
            out = SolveInfo.from_legacy_history(strategy_name, history, threshold)
            out.extras.update({k: v for k, v in info.items() if k != 'history'})
            return out
        # Stage-keyed dicts (m10/m14): each top-level key becomes a phase.
        phases = [
            PhaseInfo(
                name=k,
                wall_s=float(v.get('wall', 0.0)) if isinstance(v, dict) else 0.0,
                n_neg=int(v.get('n_neg', -1)) if isinstance(v, dict) else -1,
                min_T=(
                    float(v.get('min_T', float('nan'))) if isinstance(v, dict) else float('nan')
                ),
                extras=v if isinstance(v, dict) else {'value': v},
            )
            for k, v in info.items()
            if k != 'extras'
        ]
        return SolveInfo(
            strategy_name=strategy_name,
            phases=phases,
            total_iter=sum(p.n_iter for p in phases),
            extras=info.get('extras', {}),
        )
    return SolveInfo(strategy_name=strategy_name, extras={'raw': info})


class Strategy(ABC):
    """Abstract base class for solver strategies.

    Subclasses declare what they accept via class attributes:

    * ``accepts_constraints`` — a tuple of accepted :class:`Constraint`
      subclasses, or ``None`` to accept anything. Used in
      :meth:`_check_constraint` to surface incompatible compositions
      at :class:`Solver` construction time rather than mid-run.
    * ``supports_3d`` — whether the strategy handles 3D constraints
      (i.e. ``constraint.dim == 3``). Most 2-tri-specific strategies
      are 2D-only by construction.
    """

    # ``None`` = accept any Constraint subclass. A tuple narrows it.
    accepts_constraints: Optional[tuple[type, ...]] = None
    supports_3d: bool = False

    @abstractmethod
    def solve(
        self,
        phi_in: np.ndarray,
        *,
        constraint: Constraint,
        objective: Objective,
        threshold: float,
        **kwargs,
    ):
        """Run the strategy.

        Returns
        -------
        phi_out : ndarray, same shape as ``phi_in``
        info : SolveInfo
        """

    # ---- validation helper used by subclasses ------------------------
    def _check_constraint(self, constraint: Constraint) -> None:
        from dvfopt.exceptions import IncompatibleConstraintError

        if self.accepts_constraints is not None and not isinstance(
            constraint, self.accepts_constraints
        ):
            accepted = ', '.join(t.__name__ for t in self.accepts_constraints)
            raise IncompatibleConstraintError(
                f'{type(self).__name__} requires one of '
                f'({accepted}); got {type(constraint).__name__}'
            )
        if not self.supports_3d and constraint.dim == 3:
            raise IncompatibleConstraintError(
                f'{type(self).__name__} does not support 3D constraints'
            )

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, type] = {}


def register_strategy(label: str):
    """Decorator that registers a Strategy subclass under ``label``.

    Used at class-definition time::

        @register_strategy('my_solver')
        class MyStrategy(Strategy): ...

    External packages can register their own strategies the same way;
    after import they become available via
    :func:`make_strategy('my_solver')`.
    """

    def deco(cls: type) -> type:
        if not issubclass(cls, Strategy):
            raise TypeError(f'{cls.__name__} is not a Strategy subclass')
        _STRATEGY_REGISTRY[label] = cls
        return cls

    return deco


def make_strategy(spec, **kwargs) -> Strategy:
    """Construct a Strategy from a string label or an instance."""
    if isinstance(spec, Strategy):
        return spec
    try:
        cls = _STRATEGY_REGISTRY[spec]
    except KeyError as exc:
        raise ValueError(
            f'unknown strategy: {spec!r}; valid: {sorted(_STRATEGY_REGISTRY)}'
        ) from exc
    return cls(**kwargs)


__all__ = [
    'Strategy',
    '_build_solve_info',
    'make_strategy',
    'register_strategy',
]
