"""Schwarz strategy — overlapping-tile per-cluster SLSQP."""

from __future__ import annotations

from dataclasses import dataclass

from dvfopt.constraints import TriConstraint2D, TriConstraint2DFullCoverage
from dvfopt.strategies.base import Strategy, register_strategy


@register_strategy('schwarz')
@dataclass
class SchwarzStrategy(Strategy):
    """Overlapping-tile Schwarz + per-cluster SLSQP. 2-tri only —
    wraps :func:`dvfopt.core.iterative2d_tri_schwarz.iterative_2d_tri_schwarz`."""

    max_outer: int = 6

    supports_3d: bool = False
    accepts_constraints = (TriConstraint2D, TriConstraint2DFullCoverage)

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        from dvfopt.core.iterative2d_tri_schwarz import iterative_2d_tri_schwarz

        self._check_constraint(constraint)
        out = iterative_2d_tri_schwarz(
            phi_in,
            threshold=threshold,
            max_outer=self.max_outer,
            verbose=verbose,
            record_history=record_history,
        )
        return self._finish(out, record_history, threshold, wrap_history=True)


__all__ = ['SchwarzStrategy']
