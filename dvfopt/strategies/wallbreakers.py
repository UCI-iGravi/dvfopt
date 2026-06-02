"""Wallbreaker strategies — phase-stack-explicit 2D + 3D pipelines.

Each Strategy spells out the optimisation phases it chains. The
research tags they were known by during development (m10, m14,
m14-Schwarz, ...) remain as ``M*Strategy`` back-compat aliases at the
bottom of each section.

2D (2-triangle constraint)
--------------------------
* :class:`HarmonicALMBarrierStrategy` — alias :class:`M10Strategy`.
  Harmonic Laplacian extension → PHR augmented Lagrangian → log-barrier
  L-BFGS-B polish.
* :class:`HarmonicALMRefineRepairStrategy` — alias :class:`M14Strategy`.
  HarmonicALMBarrier seed → soft-penalty L2 refine → harmonic repair →
  log-barrier polish.
* :class:`SchwarzHarmonicALMRefineRepairStrategy` — alias
  :class:`M14SchwarzStrategy`. Cluster-localized version of the above.

3D (6-tetrahedron constraint)
-----------------------------
* :class:`Harmonic3DStrategy` — the 3D harmonic primitive on its own
  (m02 analogue).
* :class:`ALM3DStrategy` — the 3D PHR-ALM phase on its own.
* :class:`HarmonicALMBarrier3DStrategy` — alias :class:`M10TetStrategy`.
* :class:`HarmonicALMRefineRepair3DStrategy` — alias
  :class:`M14TetStrategy`.
* :class:`SchwarzHarmonicALMRefineRepair3DStrategy` — alias
  :class:`M14Schwarz3DStrategy`.

The 2D wallbreakers thin-wrap the corresponding pipelines in
:mod:`dvfopt.core.wallbreakers`. They're triangle-area-specific by
design (harmonic extension assumes PL bijectivity via triangle areas,
etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from dvfopt.constraints import (
    Tet6Constraint3D,
    TriConstraint2D,
    TriConstraint2DFullCoverage,
)
from dvfopt.strategies.base import Strategy, _build_solve_info, register_strategy


@register_strategy('harmonic_alm_barrier')
@register_strategy('m10')  # back-compat alias for the original "m10" tag
@dataclass
class HarmonicALMBarrierStrategy(Strategy):
    """Harmonic seed → ALM → log-barrier polish (2D 2-triangle).

    Three-stage "always-feasibility" wallbreaker:
    harmonic-extension into fold cores → augmented-Lagrangian tighten
    → log-barrier polish anchored to the input.

    Reaches feasibility on cases where :class:`BarrierStrategy` stalls
    (e.g. extreme density >5000 folds). Larger L1 deviation than the
    refine-repair pipeline; faster though.

    Historically tagged as "m10" in the wallbreaker experiments; the
    old ``M10Strategy`` name + ``'m10'`` registry string remain as
    back-compat aliases.
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
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        step_callback=None,
        **_,
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
            step_callback=step_callback,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info('HarmonicALMBarrierStrategy', info, threshold)
        return out, _build_solve_info('HarmonicALMBarrierStrategy', {}, threshold)


# Back-compat alias for code that imports ``M10Strategy`` directly.
M10Strategy = HarmonicALMBarrierStrategy


@register_strategy('harmonic_alm_refine_repair')
@register_strategy('m14')  # back-compat alias
@dataclass
class HarmonicALMRefineRepairStrategy(Strategy):
    """Full 4-stage refine-repair pipeline (2D 2-triangle).

    Pipeline:
    HarmonicALMBarrier seed (the full m10 pipeline) → soft-penalty
    L2 pull → harmonic repair of
    residual folds → barrier polish anchored to ``phi_in``.

    The L2 / L1 winner on dense folds: typically ~50% lower L2 than
    HarmonicALMBarrierStrategy alone (and ~80% lower L1 with the L1
    anchor) by trading the harmonic seed's smooth-but-far solution for
    a refined one that stays closer to the input.

    Historically tagged as "m14" in the wallbreaker experiments; the
    old ``M14Strategy`` name + ``'m14'`` registry string remain as
    back-compat aliases.
    """

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
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        step_callback=None,
        **_,
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
            step_callback=step_callback,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info('HarmonicALMRefineRepairStrategy', info, threshold)
        return out, _build_solve_info('HarmonicALMRefineRepairStrategy', {}, threshold)


# Back-compat alias.
M14Strategy = HarmonicALMRefineRepairStrategy


@register_strategy('schwarz_harmonic_alm_refine_repair')
@register_strategy('m14_schwarz')  # back-compat alias
@dataclass
class SchwarzHarmonicALMRefineRepairStrategy(Strategy):
    """Cluster-localized HarmonicALMRefineRepair + final global barrier polish (2D).

    Detects connected fold clusters, runs the full
    HarmonicALMRefineRepair pipeline on each cluster's padded crop
    independently, and finishes with a global log-barrier polish.
    ~5x faster than the global pipeline on large
    slices (>20K corners) with ~11% lower L1 on the B0039 z=12 full
    slice.

    Historically tagged as "m14_schwarz"; the old ``M14SchwarzStrategy``
    name + ``'m14_schwarz'`` registry string remain as back-compat
    aliases.
    """

    margin: float = 1e-3
    pad: int = 4
    merge_dilation: int = 2
    max_outer_iters: int = 3
    fallback_size_ratio: float = 0.7
    max_grow_iters: int = 8  # forwarded to per-cluster refine-repair
    time_budget_s: float = 600.0
    final_polish: bool = True
    final_polish_max_iter: int = 200

    supports_3d: bool = False
    accepts_constraints = (TriConstraint2D, TriConstraint2DFullCoverage)

    def solve(
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        step_callback=None,
        **_,
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
            step_callback=step_callback,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info(
                'SchwarzHarmonicALMRefineRepairStrategy', info, threshold
            )
        return out, _build_solve_info('SchwarzHarmonicALMRefineRepairStrategy', {}, threshold)


# Back-compat alias.
M14SchwarzStrategy = SchwarzHarmonicALMRefineRepairStrategy


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


@register_strategy('alm_3d')
@dataclass
class ALM3DStrategy(Strategy):
    """PHR augmented Lagrangian for the 3D 6-tet constraint.

    3D analog of the standalone 2D m03 ALM wallbreaker. Avoids the
    barrier's penalty-phase stall on dense folds — the inner problem
    is unconstrained L-BFGS-B over a smooth augmented Lagrangian.
    """

    margin: float = 1e-3
    rho_init: float = 1.0
    rho_growth: float = 5.0
    rho_max: float = 1e8
    outer_max: int = 60
    inner_maxiter: int = 200
    ftol_inner: float = 1e-10
    gtol_inner: float = 1e-7
    time_budget_s: float = 600.0

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
        from dvfopt.core.wallbreakers._alm_3d import augmented_lagrangian_3d

        self._check_constraint(constraint)
        out = augmented_lagrangian_3d(
            phi_in,
            threshold=threshold,
            margin=self.margin,
            anchor=objective.label or 'l2',
            eps_l1=getattr(objective, 'eps', 1e-4),
            rho_init=self.rho_init,
            rho_growth=self.rho_growth,
            rho_max=self.rho_max,
            outer_max=self.outer_max,
            inner_maxiter=self.inner_maxiter,
            ftol_inner=self.ftol_inner,
            gtol_inner=self.gtol_inner,
            time_budget_s=self.time_budget_s,
            verbose=verbose,
            record_history=record_history,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info(
                'ALM3DStrategy',
                [
                    dict(
                        phase=f'alm_{h["outer"]}',
                        n_iter=h['inner_nit'],
                        min_T=h['min_T'],
                        wall_s=h['wall'],
                        **{
                            k: v
                            for k, v in h.items()
                            if k not in {'outer', 'inner_nit', 'min_T', 'wall'}
                        },
                    )
                    for h in info.get('log_last5', [])
                ],
                threshold,
            )
        return out, _build_solve_info('ALM3DStrategy', {}, threshold)


@register_strategy('harmonic_alm_barrier_3d')
@register_strategy('m10_3d')  # back-compat alias
@dataclass
class HarmonicALMBarrier3DStrategy(Strategy):
    """Full m10-3D pipeline: harmonic seed → ALM → barrier polish.

    3D analog of :class:`HarmonicALMBarrierStrategy`. Three stages:

    1. **Harmonic seed.** Find fold cores, Dirichlet-fill via 7-point
       Laplacian. Guaranteed feasible but L2 = O(1) (smoothest possible
       reconstruction, not minimum-displacement).
    2. **ALM.** Tighten the feasible iterate via PHR augmented
       Lagrangian. Smooth inner problem; no active-set degeneracy.
    3. **Barrier polish.** Optional log-barrier interior-point step
       from the ALM iterate to minimise L2/L1 distance from the input.

    Use this when :class:`BarrierStrategy` stalls (dense 3D folds where
    the penalty phase can't find a feasible step) — the harmonic seed
    guarantees a feasible start.
    """

    margin: float = 1e-3
    # Harmonic stage
    ring_pad: int = 2
    max_grow_iters: int = 6
    merge_dilation: int = 2
    # ALM stage
    rho_init: float = 1.0
    rho_growth: float = 5.0
    rho_max: float = 1e8
    outer_max: int = 60
    alm_inner_maxiter: int = 200
    # Polish stage (barrier from feasible)
    polish: bool = True
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
        from dataclasses import asdict

        import numpy as np

        from dvfopt.core.wallbreakers._alm_3d import augmented_lagrangian_3d
        from dvfopt.core.wallbreakers._harmonic_3d import harmonic_extension_3d
        from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

        self._check_constraint(constraint)
        anchor_label = objective.label or 'l2'
        eps_l1 = getattr(objective, 'eps', 1e-4)

        # ---- Stage 1: harmonic seed ----
        t0 = time.time()
        h_out = harmonic_extension_3d(
            phi_in,
            threshold=threshold,
            ring_pad=self.ring_pad,
            max_grow_iters=self.max_grow_iters,
            merge_dilation=self.merge_dilation,
            margin=self.margin,
            record_history=record_history,
        )
        if record_history:
            phi_h, h_info = h_out
        else:
            phi_h = h_out
            h_info = {}
        wall_h = time.time() - t0
        V_h = six_tet_volumes_3d(phi_h)
        harmonic_phase = {
            'phase': 'harmonic',
            'n_neg': int((V_h <= 0).sum()),
            'min_T': float(V_h.min()),
            'wall_s': wall_h,
            'patches': h_info.get('patches', 0),
        }

        # ---- Stage 2: ALM (start from harmonic seed, anchor to original input) ----
        t1 = time.time()
        # Coerce phi_in into the canonical (3, D, H, W) [dz, dy, dx] shape
        # for the anchor — harmonic_extension_3d already produces it.
        phi_anchor = np.asarray(phi_in, dtype=np.float64)
        if phi_anchor.shape != phi_h.shape:
            raise RuntimeError(
                f'HarmonicALMBarrier3DStrategy: phi_in shape {phi_anchor.shape} '
                f'differs from harmonic output {phi_h.shape}'
            )
        alm_out = augmented_lagrangian_3d(
            phi_h,
            threshold=threshold,
            margin=self.margin,
            anchor=anchor_label,
            eps_l1=eps_l1,
            phi_anchor=phi_anchor,
            rho_init=self.rho_init,
            rho_growth=self.rho_growth,
            rho_max=self.rho_max,
            outer_max=self.outer_max,
            inner_maxiter=self.alm_inner_maxiter,
            verbose=verbose,
            record_history=record_history,
        )
        if record_history:
            phi_alm, alm_info = alm_out
        else:
            phi_alm = alm_out
            alm_info = {}
        wall_alm = time.time() - t1
        V_alm = six_tet_volumes_3d(phi_alm)
        alm_phase = {
            'phase': 'alm',
            'n_neg': int((V_alm <= 0).sum()),
            'min_T': float(V_alm.min()),
            'wall_s': wall_alm,
            'outer_used': alm_info.get('outer_used', -1),
            'rho_final': alm_info.get('rho_final', 0.0),
        }

        if not self.polish:
            return phi_alm, _build_solve_info(
                'HarmonicALMBarrier3DStrategy', [harmonic_phase, alm_phase], threshold
            )

        # ---- Stage 3: barrier polish ----
        from dvfopt.strategies.barrier import BarrierStrategy

        barrier = BarrierStrategy(max_iter=self.polish_max_iter)
        phi_out, polish_info = barrier.solve(
            phi_alm,
            constraint=constraint,
            objective=objective,
            threshold=threshold,
            verbose=verbose,
            record_history=record_history,
        )
        polish_phases = [
            {
                'phase': f'polish_{p.name}',
                'n_iter': p.n_iter,
                'n_neg': p.n_neg,
                'min_T': p.min_T,
                'wall_s': p.wall_s,
                **asdict(p).get('extras', {}),
            }
            for p in polish_info.phases
        ]
        return phi_out, _build_solve_info(
            'HarmonicALMBarrier3DStrategy',
            [harmonic_phase, alm_phase, *polish_phases],
            threshold,
        )


# Back-compat alias.
M10TetStrategy = HarmonicALMBarrier3DStrategy


@register_strategy('harmonic_alm_refine_repair_3d')
@register_strategy('m14_3d')  # back-compat alias
@dataclass
class HarmonicALMRefineRepair3DStrategy(Strategy):
    """Full m14-3D refine-repair pipeline.

    3D analog of :class:`M14Strategy` (2D). Four stages:

    1. **m10-3D seed**: harmonic patch + ALM tightening. Feasible by
       construction (Radó-Kneser-Choquet for harmonic; ALM smooth pull).
    2. **Soft-penalty L2 pull** (``l2_refine_3d``): one-sided quadratic
       penalty annealed up; large L2 reduction on non-active cells.
    3. **Harmonic repair** of any residual folds created by stage 2.
    4. **Barrier polish** anchored to ``phi_in`` for the strict
       L2-optimum of the central path.

    Use when :class:`BarrierStrategy` stalls on dense folds AND you
    want the smallest L2 / L1 deviation possible — m14 typically
    dominates m10 by ~50% L2 reduction on dense 2D cases. The 3D
    behavior follows the same pattern.
    """

    margin: float = 1e-3
    # Stage 1 (seed) — m10-3D-style
    ring_pad: int = 2
    max_grow_iters: int = 6
    merge_dilation: int = 2
    alm_outer_max: int = 30
    alm_inner_maxiter: int = 200
    # Stage 2 (l2 refine)
    lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8)
    inner_maxiter: int = 300
    # Stage 4 (polish)
    polish_mu: tuple = (1e-2, 1e-4, 1e-6)
    polish_maxiter: int = 200
    time_budget_s: float = 600.0

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
        from dvfopt.core.wallbreakers._refine_repair_3d import iterative_3d_tet_refine_repair

        self._check_constraint(constraint)
        out = iterative_3d_tet_refine_repair(
            phi_in,
            threshold=threshold,
            margin=self.margin,
            anchor=objective.label or 'l2',
            eps_l1=getattr(objective, 'eps', 1e-4),
            ring_pad=self.ring_pad,
            max_grow_iters=self.max_grow_iters,
            merge_dilation=self.merge_dilation,
            alm_outer_max=self.alm_outer_max,
            alm_inner_maxiter=self.alm_inner_maxiter,
            lam_schedule=self.lam_schedule,
            inner_maxiter=self.inner_maxiter,
            polish_mu=self.polish_mu,
            polish_maxiter=self.polish_maxiter,
            time_budget_s=self.time_budget_s,
            verbose=verbose,
            record_history=record_history,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info('HarmonicALMRefineRepair3DStrategy', info, threshold)
        return out, _build_solve_info('HarmonicALMRefineRepair3DStrategy', {}, threshold)


# Back-compat alias.
M14TetStrategy = HarmonicALMRefineRepair3DStrategy


@register_strategy('schwarz_harmonic_alm_refine_repair_3d')
@register_strategy('m14_schwarz_3d')  # back-compat alias
@dataclass
class SchwarzHarmonicALMRefineRepair3DStrategy(Strategy):
    """Cluster-localized RefineRepair-3D (Schwarz for 6-tet).

    3D analog of :class:`SchwarzHarmonicALMRefineRepairStrategy` (2D). Detects
    connected fold components via 26-connectivity CCL, runs global
    RefineRepair3D on each padded crop independently, splices back,
    and (if necessary) repeats to clear Schwarz-overlap artifacts at
    crop boundaries.

    Falls back to global RefineRepair3D when any single cluster spans
    more than ``fallback_size_ratio`` of any axis, or when outer
    iterations fail to reduce ``n_neg`` for two consecutive rounds.

    Use when fold clusters cover a small fraction of a large 3D volume.
    On dense single-cluster volumes the wrapper falls back to global
    RefineRepair3D and behaves identically to
    :class:`HarmonicALMRefineRepair3DStrategy`.
    """

    margin: float = 1e-3
    pad: int = 4
    merge_dilation: int = 2
    max_outer_iters: int = 3
    fallback_size_ratio: float = 0.7
    time_budget_s: float = 600.0

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
        from dvfopt.core.wallbreakers._m14_schwarz_3d import (
            iterative_3d_tet_refine_repair_schwarz,
        )

        self._check_constraint(constraint)
        out = iterative_3d_tet_refine_repair_schwarz(
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
            verbose=verbose,
            record_history=record_history,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info(
                'SchwarzHarmonicALMRefineRepair3DStrategy', info, threshold
            )
        return out, _build_solve_info('SchwarzHarmonicALMRefineRepair3DStrategy', {}, threshold)


# Back-compat alias.
M14Schwarz3DStrategy = SchwarzHarmonicALMRefineRepair3DStrategy


# Names in ``__all__`` are sorted alphabetically (ruff RUF022).
# The ``M*Strategy`` entries are back-compat aliases for the original
# "m10/m14/..." research tags — they refer to the same classes as the
# descriptive names alongside them.
__all__ = [
    'ALM3DStrategy',
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
    'SchwarzHarmonicALMRefineRepair3DStrategy',
    'SchwarzHarmonicALMRefineRepairStrategy',
]
