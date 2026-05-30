"""Wallbreaker strategies — m10, m14, m14-Schwarz.

All three thin-wrap the corresponding pipelines in
:mod:`dvfopt.core.wallbreakers`. 2-tri constraints only by design —
each pipeline embeds triangle-area-specific reasoning (harmonic
extension assumes PL bijectivity via triangle areas, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from dvfopt.constraints import TriConstraint2D, TriConstraint2DFullCoverage
from dvfopt.strategies.base import Strategy, _build_solve_info, register_strategy


@register_strategy('m10')
@dataclass
class M10Strategy(Strategy):
    """Harmonic seed -> ALM -> log-barrier polish.

    The "always-feasibility" wallbreaker — reaches feasibility on cases
    where the barrier strategy stalls (e.g. extreme density >5000
    folds).
    """

    margin: float = 1e-3
    ring_pad: int = 2
    max_grow_iters: int = 8
    mu_schedule: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    inner_maxiter: int = 300
    time_budget_s: float = 600.0

    supports_3d: bool = False
    accepts_constraints = (TriConstraint2D, TriConstraint2DFullCoverage)

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        from dvfopt.core.wallbreakers import iterative_2d_tri_harmonic_polished

        self._check_constraint(constraint)
        out = iterative_2d_tri_harmonic_polished(
            phi_in,
            threshold=threshold,
            margin=self.margin,
            ring_pad=self.ring_pad,
            max_grow_iters=self.max_grow_iters,
            mu_schedule=self.mu_schedule,
            inner_maxiter=self.inner_maxiter,
            anchor=objective.label or 'l2',
            eps_l1=getattr(objective, 'eps', 1e-4),
            time_budget_s=self.time_budget_s,
            verbose=verbose,
            record_history=record_history,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info('M10Strategy', info, threshold)
        return out, _build_solve_info('M10Strategy', {}, threshold)


@register_strategy('m14')
@dataclass
class M14Strategy(Strategy):
    """m10 seed -> soft-penalty pull -> harmonic repair -> barrier polish."""

    margin: float = 1e-3
    lam_schedule: tuple[float, ...] = (1e2, 1e4, 1e6, 1e8)
    inner_maxiter: int = 300
    ring_pad: int = 2
    max_grow_iters: int = 8
    polish_mu: tuple[float, ...] = (1e-2, 1e-4, 1e-6)
    polish_maxiter: int = 200
    time_budget_s: float = 600.0

    supports_3d: bool = False
    accepts_constraints = (TriConstraint2D, TriConstraint2DFullCoverage)

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        from dvfopt.core.wallbreakers import iterative_2d_tri_refine_repair

        self._check_constraint(constraint)
        out = iterative_2d_tri_refine_repair(
            phi_in,
            threshold=threshold,
            margin=self.margin,
            anchor=objective.label or 'l2',
            lam_schedule=self.lam_schedule,
            inner_maxiter=self.inner_maxiter,
            ring_pad=self.ring_pad,
            max_grow_iters=self.max_grow_iters,
            polish_mu=self.polish_mu,
            polish_maxiter=self.polish_maxiter,
            time_budget_s=self.time_budget_s,
            verbose=verbose,
            eps_l1=getattr(objective, 'eps', 1e-4),
            record_history=record_history,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info('M14Strategy', info, threshold)
        return out, _build_solve_info('M14Strategy', {}, threshold)


@register_strategy('m14_schwarz')
@dataclass
class M14SchwarzStrategy(Strategy):
    """Cluster-localized m14 + final global barrier polish.

    ~5x faster than global m14 on large slices (>20K corners) with
    ~11% lower L1 on the B0039 z=12 full slice.
    """

    margin: float = 1e-3
    pad: int = 4
    merge_dilation: int = 2
    max_outer_iters: int = 3
    fallback_size_ratio: float = 0.7
    max_grow_iters: int = 8  # forwarded to per-cluster m14
    time_budget_s: float = 600.0
    final_polish: bool = True
    final_polish_max_iter: int = 200

    supports_3d: bool = False
    accepts_constraints = (TriConstraint2D, TriConstraint2DFullCoverage)

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        from dvfopt.core.wallbreakers import iterative_2d_tri_refine_repair_schwarz

        self._check_constraint(constraint)
        out = iterative_2d_tri_refine_repair_schwarz(
            phi_in,
            threshold=threshold,
            margin=self.margin,
            anchor=objective.label or 'l2',
            eps_l1=getattr(objective, 'eps', 1e-4),
            pad=self.pad,
            merge_dilation=self.merge_dilation,
            max_outer_iters=self.max_outer_iters,
            fallback_size_ratio=self.fallback_size_ratio,
            time_budget_s=self.time_budget_s,
            final_polish=self.final_polish,
            final_polish_max_iter=self.final_polish_max_iter,
            max_grow_iters=self.max_grow_iters,
            verbose=verbose,
            record_history=record_history,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info('M14SchwarzStrategy', info, threshold)
        return out, _build_solve_info('M14SchwarzStrategy', {}, threshold)


__all__ = ['M10Strategy', 'M14SchwarzStrategy', 'M14Strategy']
