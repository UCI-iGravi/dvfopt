"""Wallbreaker strategies — m10, m14, m14-Schwarz (2D) + Harmonic3DStrategy.

The 2D wallbreakers thin-wrap the corresponding pipelines in
:mod:`dvfopt.core.wallbreakers`. They're triangle-area-specific by
design (harmonic extension assumes PL bijectivity via triangle areas,
etc.) and don't generalize directly to 3D.

The 3D path so far has just the **harmonic** primitive
(:class:`Harmonic3DStrategy`) — the 3D analog of m02 / step 1 of m10.
The full 3D m10 pipeline (harmonic → ALM → polish) is deferred; the
strategy here pairs the harmonic seed with an optional barrier polish
when a 6-tet constraint is composed in.
"""

from __future__ import annotations

from dataclasses import dataclass

from dvfopt.constraints import (
    Tet6Constraint3D,
    TriConstraint2D,
    TriConstraint2DFullCoverage,
)
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


@register_strategy('harmonic_3d')
@dataclass
class Harmonic3DStrategy(Strategy):
    """3D harmonic-extension wallbreaker for the 6-tet constraint.

    3D analog of the harmonic step in :class:`M10Strategy`. Finds
    fold cores, dilates each by a ring of feasible boundary, and
    solves a 7-point Laplacian on each displacement channel to
    Dirichlet-fill the interior with the smoothest possible map.

    Useful when :class:`BarrierStrategy` stalls (the dense-fold
    "wall" — barrier's penalty phase fails to find a feasible step
    when many tets are crowded against zero simultaneously).

    Optional ``polish`` runs barrier from the harmonic seed to
    minimise L2/L1 distance from the input — the harmonic patch is
    *globally feasible* but not *minimum-displacement*; barrier
    initialised from a feasible point converges fast and produces
    a much better L2.

    .. note::
        Compared to the 2D :class:`M10Strategy`, only the harmonic
        step is ported. The full 3D m10 pipeline (harmonic → ALM →
        polish, then refinement / m14 / m14-Schwarz) is deferred —
        this is the foundation other 3D wallbreakers would build on.
    """

    margin: float = 1e-3
    ring_pad: int = 2
    max_grow_iters: int = 6
    merge_dilation: int = 2
    polish: bool = True  # run BarrierStrategy from the harmonic seed
    polish_max_iter: int = 200

    supports_3d: bool = True
    accepts_constraints = (Tet6Constraint3D,)

    def solve(
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        **_,
    ):
        import time

        from dvfopt.core.wallbreakers._harmonic_3d import harmonic_extension_3d
        from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

        self._check_constraint(constraint)

        t0 = time.time()
        out = harmonic_extension_3d(
            phi_in,
            threshold=threshold,
            ring_pad=self.ring_pad,
            max_grow_iters=self.max_grow_iters,
            merge_dilation=self.merge_dilation,
            margin=self.margin,
            record_history=record_history,
        )
        if record_history:
            phi_harmonic, info = out
        else:
            phi_harmonic = out
            info = {}
        harmonic_wall = time.time() - t0

        # Build a proper history-list phase entry for the harmonic step
        # using the canonical schema (phase, n_neg, min_T, wall_s, …).
        # The shared _build_solve_info adapter recognises this shape via
        # SolveInfo.from_legacy_history.
        V_h = six_tet_volumes_3d(phi_harmonic)
        harmonic_phase = {
            'phase': 'harmonic',
            'n_neg': int((V_h <= 0).sum()),
            'min_T': float(V_h.min()),
            'wall_s': harmonic_wall,
            'patches': info.get('patches', 0),
            'records': info.get('records', []),
        }

        if not self.polish:
            return phi_harmonic, _build_solve_info(
                'Harmonic3DStrategy', [harmonic_phase], threshold
            )

        # Polish via barrier-on-tet from the harmonic seed. We import
        # BarrierStrategy lazily to avoid an import cycle.
        from dataclasses import asdict

        from dvfopt.strategies.barrier import BarrierStrategy

        barrier = BarrierStrategy(max_iter=self.polish_max_iter)
        phi_out, polish_info = barrier.solve(
            phi_harmonic,
            constraint=constraint,
            objective=objective,
            threshold=threshold,
            verbose=verbose,
            record_history=record_history,
        )
        # ``polish_info`` is a SolveInfo dataclass. Flatten each phase
        # into the legacy history schema so they flow through the
        # adapter alongside the harmonic phase (rather than being lost
        # to a SolveInfo-shaped blob inside a stage-keyed dict, which
        # was the original PR #13 bug the reviewer caught).
        polish_phases = []
        for p in polish_info.phases:
            entry = {
                'phase': f'polish_{p.name}',
                'n_iter': p.n_iter,
                'n_neg': p.n_neg,
                'min_T': p.min_T,
                'wall_s': p.wall_s,
                **asdict(p).get('extras', {}),
            }
            polish_phases.append(entry)

        return phi_out, _build_solve_info(
            'Harmonic3DStrategy',
            [harmonic_phase, *polish_phases],
            threshold,
        )


__all__ = ['Harmonic3DStrategy', 'M10Strategy', 'M14SchwarzStrategy', 'M14Strategy']
