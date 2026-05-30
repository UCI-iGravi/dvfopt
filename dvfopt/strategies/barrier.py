"""Penalty -> log-barrier L-BFGS-B strategy.

The "workhorse" for moderate-density problems. Works for every concrete
:class:`Constraint` subclass — 2D 2-tri, 2D Jdet, 3D Jdet — because the
underlying :func:`dvfopt.core._barrier_core.run_penalty_barrier_lbfgs`
only requires ``constraint.values`` + ``constraint.adjoint``.
"""

from __future__ import annotations

from dataclasses import dataclass

from dvfopt.strategies.base import Strategy, _build_solve_info, register_strategy


@register_strategy('barrier')
@dataclass
class BarrierStrategy(Strategy):
    """Penalty -> log-barrier L-BFGS-B."""

    margin: float = 1e-3
    lam_schedule: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
    mu_schedule: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)
    max_iter: int = 300

    supports_3d: bool = True

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        from dvfopt.core._barrier_core import run_penalty_barrier_lbfgs

        self._check_constraint(constraint)
        phi_flat = constraint.flatten(phi_in)
        phi_anchor = phi_flat.copy()
        anchor_kind = objective.label or 'l2'
        anchor_eps = getattr(objective, 'eps', 1e-4)

        out_flat, info = run_penalty_barrier_lbfgs(
            phi_flat,
            phi_anchor,
            constraint_values=constraint.values,
            constraint_adjoint=constraint.adjoint,
            threshold=threshold,
            margin=self.margin,
            lam_schedule=self.lam_schedule,
            mu_schedule=self.mu_schedule,
            max_iter=self.max_iter,
            anchor=anchor_kind,
            eps_l1=anchor_eps,
            verbose=verbose,
            record_history=record_history,
        )
        return constraint.unflatten(out_flat), _build_solve_info('BarrierStrategy', info, threshold)


__all__ = ['BarrierStrategy']
