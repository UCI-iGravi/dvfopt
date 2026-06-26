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

import numpy as np

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
        step_callback=None,
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
        if step_callback is not None:
            step_callback({'phi': phi_h, 'stage': 'harmonic'})

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
        if step_callback is not None:
            step_callback({'phi': phi_alm, 'stage': 'alm'})

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
        step_callback=None,
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
            step_callback=step_callback,
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
        step_callback=None,
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
            step_callback=step_callback,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info(
                'SchwarzHarmonicALMRefineRepair3DStrategy', info, threshold
            )
        return out, _build_solve_info('SchwarzHarmonicALMRefineRepair3DStrategy', {}, threshold)


# Back-compat alias.
M14Schwarz3DStrategy = SchwarzHarmonicALMRefineRepair3DStrategy


@register_strategy('coupled_kring_3d')
@dataclass
class CoupledKRing3DStrategy(Strategy):
    """Coupled k-ring SLSQP for breaking the 3D fold-attractor.

    Use this AFTER an :class:`HarmonicALMBarrier3DStrategy` pass that
    has converged to ~1-3 residual fold cubes. The barrier solver hits
    a Nash-equilibrium local minimum where each residual cube's
    neighbours pin its corners (see ``research/strict_feasibility_3d/
    REPORT.md`` Parts XI-XIV for the analysis). This strategy escapes
    that attractor by jointly optimising every corner in a k-ring halo
    around the worst-fold cube, subject to every cube in the halo
    having its six tet volumes above ``feasibility_thr``.

    Two reasons it works where the standalone barrier doesn't:

    1. **Constraint-aware coordinated motion**: SLSQP shifts ~64
       corners by ~0.09 voxels each in one move; the barrier can only
       move one corner at a time and stalls when each move costs more
       in neighbour-barrier than it saves in the L1 anchor.
    2. **Lagrangian relaxation**: setting ``feasibility_thr`` below
       ``threshold`` (e.g. 1e-3 vs the 0.01 strict threshold) lets
       the SLSQP land in a state whose subsequent barrier polish
       resolves the residual ~5 leaked boundary folds trivially.

    Typically chained as ``M10TetStrategy(...) -> CoupledKRing3DStrategy
    -> M10TetStrategy(threshold=0.012)`` for a complete pipeline.
    Empirical wall on B0039 z=0..15: ~5 s for the SLSQP step itself.

    Parameters
    ----------
    k_ring : int, default 2
        Halo radius in cubes around each centre. ``k_ring=2`` gives
        ~5×5×5 cubes, ~500 DOF, ~750 constraints — solves in seconds.
        ``k_ring=3`` is ~10× slower; ``k_ring=4`` is hours.
    feasibility_thr : float, default 1e-3
        Lower bound the SLSQP enforces on every tet volume in the halo.
        Set below ``threshold`` for Lagrangian relaxation (recommended).
        With ``feasibility_thr == threshold`` the SLSQP is too strict
        and often fails to converge from near-feasible starts.
    target_cube : tuple[int, int, int] | None, default None
        Cube to centre the halo on. If ``None``, automatically picks
        the worst fold cube (lowest min tet volume). Ignored when
        ``mode='cluster'``.
    mode : {'worst', 'cluster'}, default 'worst'
        ``'worst'`` runs one SLSQP around the single worst fold cube
        — fastest for near-feasible inputs (~5 s on 1-fold states).
        ``'cluster'`` identifies every fold cluster (via Chebyshev
        proximity), then runs one SLSQP per cluster centroid. Use
        ``'cluster'`` after M10Tet has converged but residual folds
        are spread across multiple regions of the field.
    n_workers : int | None, default None
        Workers for parallel cluster SLSQP. Only used when
        ``mode='cluster'``. ``None`` uses ``os.cpu_count()``. Set to
        1 for serial execution. Each worker processes one cluster's
        SLSQP independently; halos that geographically overlap are
        scheduled in separate sequential batches.
    cluster_radius : int, default 2
        Chebyshev radius used to group fold cubes into clusters when
        ``mode='cluster'``. Independent of ``k_ring``.
    recover : bool, default False
        After the SLSQP escape, run a LOCAL M10Tet recovery
        (:func:`dvfopt.core.wallbreakers._coupled_kring_3d.local_alm_recovery_3d`)
        restricted to a crop around the perturbed halo, instead of a
        global recovery. The escape leaves the field strictly fold-free
        (n_neg=0) but with cells just below the strict threshold
        (``n<threshold``); the local recovery tightens them. Validated
        ~430x faster than a global M10Tet recovery on B0039 (8 s vs
        57 min) for the same final result. With ``recover=True`` this
        strategy is a self-contained escape+tighten step.
    recover_pad : int, default 4
        Padding ring (cubes) around the perturbed region for the local
        recovery crop. Larger absorbs more boundary effect at higher
        cost.
    recover_threshold : float | None, default None
        Feasibility threshold for the local recovery's inner M10Tet.
        ``None`` uses ``1.2 * threshold`` (so a strict check at 0.01
        recovers to a 0.012 margin), matching the validated pipeline.
    maxiter : int, default 200
    ftol : float, default 1e-9
    use_analytical_jacobian : bool, default False
        Pass ``True`` to use the closed-form constraint Jacobian
        implemented in :func:`dvfopt.core.wallbreakers._coupled_kring_3d`.
        Verified correct to 1e-11 against finite differences, but
        empirically scipy SLSQP's QP subproblem rejects the resulting
        directions ("Positive directional derivative for linesearch")
        on the canonical test case. scipy's built-in FD (the default)
        is more robust here; the analytical Jacobian is kept for use
        with non-SLSQP solvers (trust-constr, Ipopt) and as a
        documented research artefact.
    """

    k_ring: int = 2
    feasibility_thr: float = 1e-3
    target_cube: tuple[int, int, int] | None = None
    mode: str = 'worst'
    n_workers: int | None = None
    cluster_radius: int = 2
    recover: bool = False
    recover_pad: int = 4
    recover_threshold: float | None = None
    maxiter: int = 200
    ftol: float = 1e-9
    use_analytical_jacobian: bool = False

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
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            cluster_fold_cubes,
            coupled_kring_slsqp_3d,
            coupled_kring_slsqp_3d_parallel,
            find_worst_fold_cube,
            local_alm_recovery_3d,
        )
        from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

        self._check_constraint(constraint)

        if self.mode not in ('worst', 'cluster'):
            raise ValueError(
                f"mode must be 'worst' or 'cluster', got {self.mode!r}"
            )

        rec_thr = (
            self.recover_threshold
            if self.recover_threshold is not None
            else 1.2 * threshold
        )

        if self.mode == 'worst':
            if self.target_cube is None:
                fold = find_worst_fold_cube(phi_in)
                if fold is None:
                    return phi_in, _build_solve_info(
                        'CoupledKRing3DStrategy', {}, threshold
                    )
                cz, cy, cx = fold
            else:
                cz, cy, cx = self.target_cube

            phi_out, info = coupled_kring_slsqp_3d(
                phi_in,
                cz,
                cy,
                cx,
                k_ring=self.k_ring,
                feasibility_thr=self.feasibility_thr,
                maxiter=self.maxiter,
                ftol=self.ftol,
                use_analytical_jacobian=self.use_analytical_jacobian,
            )
            V = six_tet_volumes_3d(phi_out)
            phases = [{
                'phase': 'coupled_kring_slsqp',
                'n_neg': int((V <= 0).sum()),
                'min_T': float(V.min()),
                'wall_s': info['wall_s'],
                **info,
            }]
            if self.recover:
                phi_out, rinfo = local_alm_recovery_3d(
                    phi_out,
                    center=(cz, cy, cx),
                    k_ring=self.k_ring,
                    pad=self.recover_pad,
                    threshold=rec_thr,
                    verbose=verbose,
                )
                Vr = six_tet_volumes_3d(phi_out)
                phases.append({
                    'phase': 'local_alm_recovery',
                    'n_neg': int((Vr <= 0).sum()),
                    'min_T': float(Vr.min()),
                    'wall_s': rinfo['wall_s'],
                    **rinfo,
                })
            return phi_out, _build_solve_info(
                'CoupledKRing3DStrategy', phases, threshold
            )

        # mode == 'cluster': identify clusters, run SLSQP per centroid
        # (in parallel where halos don't overlap).
        V_in = six_tet_volumes_3d(phi_in)
        fold_mask = V_in.min(axis=0) <= 0
        fold_cells = [
            tuple(int(c) for c in p) for p in zip(*np.where(fold_mask))
        ]
        if not fold_cells:
            return phi_in, _build_solve_info(
                'CoupledKRing3DStrategy', {}, threshold
            )
        centroids, _, _ = cluster_fold_cubes(
            fold_cells, radius=self.cluster_radius
        )
        phi_out, infos = coupled_kring_slsqp_3d_parallel(
            phi_in,
            centroids,
            k_ring=self.k_ring,
            feasibility_thr=self.feasibility_thr,
            maxiter=self.maxiter,
            ftol=self.ftol,
            use_analytical_jacobian=self.use_analytical_jacobian,
            n_workers=self.n_workers,
        )
        V_out = six_tet_volumes_3d(phi_out)
        wall_total = sum(i['wall_s'] for i in infos)
        phases = [{
            'phase': 'coupled_kring_slsqp_cluster',
            'n_neg': int((V_out <= 0).sum()),
            'min_T': float(V_out.min()),
            'wall_s': wall_total,
            'n_clusters': len(centroids),
            'cluster_infos': infos,
        }]
        if self.recover:
            # Local recovery around each cluster centroid (sequential;
            # each verifies globally before accepting).
            rec_wall = 0.0
            for c in centroids:
                phi_out, rinfo = local_alm_recovery_3d(
                    phi_out,
                    center=c,
                    k_ring=self.k_ring,
                    pad=self.recover_pad,
                    threshold=rec_thr,
                    verbose=verbose,
                )
                rec_wall += rinfo['wall_s']
            Vr = six_tet_volumes_3d(phi_out)
            phases.append({
                'phase': 'local_alm_recovery_cluster',
                'n_neg': int((Vr <= 0).sum()),
                'min_T': float(Vr.min()),
                'wall_s': rec_wall,
                'n_clusters': len(centroids),
            })
        return phi_out, _build_solve_info(
            'CoupledKRing3DStrategy', phases, threshold
        )


@register_strategy('active_band_alm_3d')
@dataclass
class ActiveBandALM3DStrategy(Strategy):
    """Active-band M10Tet: bulk barrier/ALM restricted to fold-cluster crops.

    A drop-in faster replacement for ``HarmonicALMBarrier3DStrategy`` on
    bulk passes whose folds are SPARSE/scattered (the common state after
    the 2D per-slice pass). Finds the connected fold clusters, runs
    M10Tet on a padded crop around each (not the whole field), pastes
    back, and accepts only if the global fold count does not increase —
    so the strict 6-tet guarantee is preserved and re-verified globally.

    Measured ~70x vs global M10Tet on a scattered-fold field (273 s →
    3.9 s, identical n_neg=0). On a pathological dense band where folds
    form one large cluster spanning the region there is no locality to
    exploit, so it degenerates to ~global (a no-op, not a loss — it
    rejects regressing crop solves); that band still needs the full
    multi-scale/iterated pipeline.

    Delegates to
    :func:`dvfopt.core.wallbreakers._coupled_kring_3d.active_band_alm_recovery_3d`.

    Parameters
    ----------
    pad : int, default 4
        Padding ring (cubes) around each fold cluster.
    merge_dilation : int, default 2
        Cluster-merge dilation for connected-component labelling.
    max_widen : int, default 1
        Pad-widen retries if a crop paste regresses the global count.
    max_band_fraction : float, default 0.7
        Clusters spanning more than this fraction of an axis fall back to
        a global solve (no locality to exploit).
    n_workers : int | None, default 1
        Process-pool workers for non-overlapping cluster crops. ``1`` =
        sequential. Parallelism only pays off with many large clusters
        (Windows process-spawn + Numba recompile tax); see the function
        docstring.
    band_threshold : float | None, default None
        Inner M10Tet feasibility threshold. ``None`` uses ``1.2 *
        threshold`` (recover to a margin above the strict check).
    """

    pad: int = 4
    merge_dilation: int = 2
    max_widen: int = 1
    max_band_fraction: float = 0.7
    n_workers: int | None = 1
    band_threshold: float | None = None

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
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            active_band_alm_recovery_3d,
        )

        self._check_constraint(constraint)
        band_thr = (
            self.band_threshold
            if self.band_threshold is not None
            else 1.2 * threshold
        )
        phi_out, info = active_band_alm_recovery_3d(
            phi_in,
            threshold=band_thr,
            pad=self.pad,
            merge_dilation=self.merge_dilation,
            max_widen=self.max_widen,
            max_band_fraction=self.max_band_fraction,
            n_workers=self.n_workers,
            verbose=verbose,
        )
        phase = {
            'phase': 'active_band_alm',
            'n_neg': info['n_neg_after'],
            'min_T': float('nan'),
            'wall_s': info['wall_s'],
            **{k: v for k, v in info.items() if k != 'per_cluster'},
        }
        return phi_out, _build_solve_info(
            'ActiveBandALM3DStrategy', [phase], threshold
        )


# Names in ``__all__`` are sorted alphabetically (ruff RUF022).
# The ``M*Strategy`` entries are back-compat aliases for the original
# "m10/m14/..." research tags — they refer to the same classes as the
# descriptive names alongside them.
__all__ = [
    'ALM3DStrategy',
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
    'SchwarzHarmonicALMRefineRepair3DStrategy',
    'SchwarzHarmonicALMRefineRepairStrategy',
]
