"""Smoke tests for the DVFopt high-level facade.

DVFopt is a thin per-slice orchestrator on top of the parameterized
:class:`dvfopt.solver.Solver`. These tests cover:

* the facade's input-shape dispatch (``(2, H, W)``, ``(3, H, W)``,
  ``(3, D, H, W)``);
* ``record_history`` plumbing (regression for the dx/dy unpack bug);
* the strategy auto-resolver thresholds (now driven by
  :func:`dvfopt.solver.auto_strategy`);
* invalid-config rejection at construction time;
* end-to-end barrier reducing folds.
"""

import importlib.util

import numpy as np
import pytest

from dvfopt import DVFopt, DVFoptConfig
from dvfopt.constraints import (
    JdetConstraint2D,
    SimplexConstraint2D,
    SimplexConstraint2DBilinear,
    SimplexConstraint2DFullCoverage,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.solver import auto_strategy


def _planted_fold_2d(H=12, W=12, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, 0.3, (H, W)), rng.normal(0, 0.3, (H, W))])


def _jdet_mild_label():
    """The 2D Jdet mild tier's expected label — install-dependent:
    ``isqp_windowed`` (windowed isqp engine) when osqp is importable,
    else the legacy ``slsqp_windowed``."""
    import importlib.util

    return 'isqp_windowed' if importlib.util.find_spec('osqp') is not None else 'slsqp_windowed'


class TestRecordHistoryRoundTrip:
    """Regression: ``record_history=False`` must not corrupt dx."""

    @pytest.mark.parametrize("record_history", [True, False])
    def test_2tri_barrier_preserves_two_channels(self, record_history):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(
            solver="barrier", constraint="simplex", record_history=record_history, verbose=0
        )
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == phi.shape
        # If the unpack-bug regresses, dx would equal dy.
        assert not np.allclose(res.corrected[0], res.corrected[1])

    def test_history_present_when_recorded(self):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=True, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert len(res.slice_results[0].history) > 0

    def test_history_empty_when_disabled(self):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.slice_results[0].history == []


class TestAutoStrategy:
    """Auto-strategy heuristic — now in :func:`dvfopt.solver.auto_strategy`.

    The DVFopt facade calls ``auto_strategy`` when ``solver='auto'`` and
    forwards the resolved label into ``Strategy`` construction.
    """

    def test_2tri_uses_smaller_severe_threshold(self):
        # Legacy tiering now applies only to non-l1 objectives (l1 routes
        # straight to the SLP champion, see TestAutoStrategySLPRouting).
        c = SimplexConstraint2D((12, 12))
        # init_min between -0.25 and -1.0 is severe for 2tri.
        assert auto_strategy(c, init_n_neg=5, init_min=-0.5, objective_label='l2') == "barrier"
        # init_min above -0.25 with few folds is mild.
        assert auto_strategy(c, init_n_neg=5, init_min=-0.1, objective_label='l2') == "slsqp"

    def test_jdet_uses_minus_one_threshold(self):
        c = JdetConstraint2D((12, 12))
        # init_min between -0.25 and -1.0 is still mild for Jdet. The mild
        # tier prefers the windowed isqp engine when osqp is installed
        # (2D), and keeps the legacy windowed SLSQP otherwise.
        assert auto_strategy(c, init_n_neg=5, init_min=-0.5) == _jdet_mild_label()
        assert auto_strategy(c, init_n_neg=5, init_min=-1.5) == "barrier"

    def test_count_overrides_min(self):
        c = SimplexConstraint2D((12, 12))
        # n_neg > 100 routes to barrier for 2tri (non-l1) regardless of min_T.
        assert auto_strategy(c, init_n_neg=600, init_min=-0.01, objective_label='l2') == "barrier"

    def test_extreme_picks_wallbreaker(self):
        # Wallbreaker routing in the extreme tier now applies to non-l1,
        # non-l2 objectives (l1 routes to 'slp'; l2 to 'm10' below) on a
        # constraint the windowed engine cannot serve — the full-coverage
        # family has no locality entry, so 'none' does NOT route to
        # isqp_windowed there (unlike 'simplex_standard', see
        # TestAutoStrategyWindowedRouting).
        c_small = SimplexConstraint2DFullCoverage((20, 20))
        # Smaller slice (<20K corners) still picks m14 instead of m14_schwarz.
        assert (
            auto_strategy(c_small, init_n_neg=6000, init_min=-15, objective_label='none') == "m14"
        )
        # Large slice (>20K corners) routes to m14_schwarz.
        c_big = SimplexConstraint2DFullCoverage((320, 456))
        assert (
            auto_strategy(c_big, init_n_neg=6000, init_min=-15, objective_label='none')
            == "m14_schwarz"
        )
        # L2 objective routes to m10 (L2-optimal in ALM phase).
        assert auto_strategy(c_big, init_n_neg=6000, init_min=-15, objective_label='l2') == "m10"

    def test_explicit_solver_passthrough(self):
        """When ``solver != 'auto'``, DVFopt honors the user's choice
        even if auto_strategy would have picked something else."""
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(
            solver="slsqp", constraint="simplex", verbose=0, strategy_kwargs={"max_iter": 40}
        )
        res = DVFopt(cfg).fit(phi)
        # Facade should have used slsqp even on a planted-fold input.
        assert res.slice_results[0].solver_used == "slsqp"

    def test_solver_accepts_strategy_instance(self):
        """DVFoptConfig.solver also accepts a Strategy instance — used
        when callers need non-default strategy knobs."""
        from dvfopt import BarrierStrategy

        phi = _planted_fold_2d()
        strat = BarrierStrategy(max_iter=200, margin=1e-3)
        cfg = DVFoptConfig(solver=strat, constraint="simplex", verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.feasible
        # solver_used should reflect the class name when an instance was
        # passed (rather than a string label).
        assert res.slice_results[0].solver_used == "BarrierStrategy"


class TestAutoStrategySLPRouting:
    """simplex (2D) + L1 auto-routes to the SLP champion at EVERY fold tier.

    SLPStrategy handles small/large slices internally
    (``cluster_pixel_threshold``) and reaches strict feasibility on every
    benchmarked slice, Pareto-dominating m14/m10 in the L1 regime — so the
    fold-density tiering only applies to non-l1 objectives.
    """

    # (n_neg, min_T) representative of each legacy tier: mild, moderate,
    # extreme-by-count, extreme-by-depth.
    TIERS = [(5, -0.1), (5, -0.5), (600, -0.01), (6000, -15.0)]

    @pytest.mark.parametrize('n_neg,init_min', TIERS)
    def test_2tri_l1_routes_to_slp_every_tier(self, n_neg, init_min):
        for c in (SimplexConstraint2D((20, 20)), SimplexConstraint2D((320, 456))):
            assert (
                auto_strategy(c, init_n_neg=n_neg, init_min=init_min, objective_label='l1') == 'slp'
            )

    def test_2tri_l1_default_label_routes_to_slp(self):
        """objective_label defaults to 'l1' → 'slp'."""
        c = SimplexConstraint2D((12, 12))
        assert auto_strategy(c, init_n_neg=5, init_min=-0.1) == 'slp'

    def test_2tri_fullcoverage_l1_routes_to_slp(self):
        from dvfopt.constraints import SimplexConstraint2DFullCoverage

        c = SimplexConstraint2DFullCoverage((20, 20))
        assert auto_strategy(c, init_n_neg=600, init_min=-0.5, objective_label='l1') == 'slp'

    @pytest.mark.parametrize(
        'n_neg,init_min,expected',
        [(5, -0.1, 'slsqp'), (5, -0.5, 'barrier'), (600, -0.01, 'barrier'), (6000, -15.0, 'm10')],
    )
    def test_2tri_l2_keeps_legacy_routing(self, n_neg, init_min, expected):
        c = SimplexConstraint2D((20, 20))
        assert (
            auto_strategy(c, init_n_neg=n_neg, init_min=init_min, objective_label='l2') == expected
        )

    def test_jdet_l1_never_routes_to_slp(self):
        """SLP is simplex (2D)-only; the Jdet family keeps its own routing
        even for l1."""
        c = JdetConstraint2D((12, 12))
        assert (
            auto_strategy(c, init_n_neg=5, init_min=-0.5, objective_label='l1')
            == _jdet_mild_label()
        )
        assert auto_strategy(c, init_n_neg=5000, init_min=-1.5, objective_label='l1') == 'barrier'


@pytest.mark.skipif(
    importlib.util.find_spec('osqp') is None, reason='windowed isqp routing needs osqp'
)
class TestAutoStrategyWindowedRouting:
    """The measured robust 2D recipe: bilinear rows on the windowed engine.

    ``constraint='bilinear'`` + ``strategy='isqp_windowed'`` +
    ``objective='none'`` clears raw B0039 slices to 0 simplex folds where
    the 2-triangle-row methods stall on twisted cells, so ``auto`` routes
    there — for bilinear at any objective, and for the standard simplex
    rows only under ``'none'`` (an L1/L2 anchor is a different fidelity
    request and keeps its own route).
    """

    TIERS = TestAutoStrategySLPRouting.TIERS

    @pytest.mark.parametrize('n_neg,init_min', TIERS)
    @pytest.mark.parametrize('objective', ['l1', 'l2', 'none'])
    def test_bilinear_routes_to_isqp_windowed_every_tier(self, n_neg, init_min, objective):
        c = SimplexConstraint2DBilinear((20, 20))
        assert (
            auto_strategy(c, init_n_neg=n_neg, init_min=init_min, objective_label=objective)
            == 'isqp_windowed'
        )

    @pytest.mark.parametrize('n_neg,init_min', TIERS)
    def test_simplex_standard_none_routes_to_isqp_windowed_every_tier(self, n_neg, init_min):
        c = SimplexConstraint2D((20, 20))
        assert (
            auto_strategy(c, init_n_neg=n_neg, init_min=init_min, objective_label='none')
            == 'isqp_windowed'
        )

    @pytest.mark.parametrize('n_neg,init_min', TIERS)
    def test_full_coverage_none_keeps_legacy_tiering(self, n_neg, init_min):
        """The full-coverage family has no windowed-engine locality entry."""
        c = SimplexConstraint2DFullCoverage((20, 20))
        assert (
            auto_strategy(c, init_n_neg=n_neg, init_min=init_min, objective_label='none')
            != 'isqp_windowed'
        )

    def test_l1_keeps_slp_and_logs_the_recipe_hint(self):
        """L1 fidelity semantics are never swapped out — but the recipe is hinted."""
        import io
        import logging

        from dvfopt._logging import logger
        from dvfopt.solver import _log_bilinear_recipe_hint

        _log_bilinear_recipe_hint.cache_clear()
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        prev_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            c = SimplexConstraint2D((20, 20))
            assert auto_strategy(c, init_n_neg=6000, init_min=-15, objective_label='l1') == 'slp'
            assert 'bilinear' in buf.getvalue() and 'isqp_windowed' in buf.getvalue()
            # Once per process, not once per slice.
            buf.truncate(0), buf.seek(0)
            auto_strategy(c, init_n_neg=6000, init_min=-15, objective_label='l1')
            assert buf.getvalue() == ''
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)


class TestAutoStrategyWindowedRoutingWithoutOSQP:
    """Without ``osqp`` every windowed route falls back to the tier heuristic."""

    @pytest.fixture
    def no_osqp(self, monkeypatch):
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, 'find_spec', lambda n, *a: None if n == 'osqp' else real(n, *a)
        )

    @pytest.mark.parametrize('label', ['bilinear', 'simplex_standard'])
    @pytest.mark.parametrize('objective', ['l1', 'l2', 'none'])
    def test_falls_back_to_an_accepting_strategy(self, no_osqp, label, objective):
        from dvfopt import Solver
        from dvfopt.constraints import make_constraint
        from dvfopt.objectives import make_objective
        from dvfopt.strategies import make_strategy

        c = make_constraint(label, (20, 20))
        picked = auto_strategy(c, init_n_neg=6000, init_min=-15, objective_label=objective)
        assert picked != 'isqp_windowed'
        # Whatever it picks must still compose.
        Solver(constraint=c, objective=make_objective(objective), strategy=make_strategy(picked))


class TestConfigValidation:
    """DVFopt rejects bad config keys at construction."""

    def test_invalid_constraint(self):
        with pytest.raises(ValueError):
            DVFopt(DVFoptConfig(constraint='bogus'))

    def test_invalid_solver(self):
        with pytest.raises(ValueError):
            DVFopt(DVFoptConfig(solver='bogus'))

    def test_invalid_objective(self):
        with pytest.raises(ValueError):
            DVFopt(DVFoptConfig(objective='bogus'))

    def test_invalid_accuracy(self):
        """Bad accuracy fails fast at config construction (post_init),
        regardless of which solver path would eventually consume it."""
        with pytest.raises(ValueError):
            DVFoptConfig(accuracy='nonsense')

    @pytest.mark.parametrize('constraint', ['jdet_3d', 'simplex_3d', '6tet'])
    def test_3d_constraints_rejected_at_construction(self, constraint):
        """The facade is per-slice 2D. It used to accept 3D constraint
        labels at validation, only to fail later inside
        ``_build_constraint`` ('simplex_3d') or run jdet_3d mis-shaped
        per-slice. Now they're rejected at construction with a pointer
        to the true-3D APIs."""
        with pytest.raises(ValueError, match='per-slice 2D'):
            DVFopt(DVFoptConfig(constraint=constraint))


class TestAccuracyPlumbing:
    """DVFoptConfig.accuracy → strategy plumbing."""

    def test_strategy_kwargs_accuracy_takes_precedence(self, monkeypatch):
        """A user-supplied strategy_kwargs['accuracy'] must win over the
        config-level c.accuracy shorthand (setdefault, not overwrite)."""
        import dvfopt.unified as unified_mod

        captured = {}
        real_make = unified_mod.make_strategy

        def spy(label, **kw):
            captured['label'] = label
            captured['kwargs'] = dict(kw)
            return real_make(label, **kw)

        monkeypatch.setattr(unified_mod, 'make_strategy', spy)
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(
            solver='slp',
            accuracy='max',
            strategy_kwargs={'accuracy': 'fast', 'n_workers': 1},
            verbose=0,
            record_history=False,
        )
        DVFopt(cfg).fit(phi)
        assert captured['label'] == 'slp'
        # 'fast' from strategy_kwargs wins; c.accuracy='max' must not clobber it.
        assert captured['kwargs']['accuracy'] == 'fast'

    def test_instance_strategy_accuracy_warns(self):
        """accuracy != 'fast' with a Strategy INSTANCE cannot be applied
        (instances are used as-is) — warn instead of silently dropping."""
        from dvfopt import BarrierStrategy

        phi = _planted_fold_2d()
        cfg = DVFoptConfig(
            solver=BarrierStrategy(),
            accuracy='max',
            constraint='simplex',
            verbose=0,
            record_history=False,
        )
        with pytest.warns(UserWarning, match='ignored'):
            DVFopt(cfg).fit(phi)

    def test_auto_solver_max_accuracy_warns_when_not_slp(self):
        """accuracy='max' with solver='auto' warns when auto resolves to a
        non-SLP label. With 2tri + an explicit objective='l2' (the facade
        default is now 'l1'), auto keeps the legacy (non-slp) routing, so
        accuracy would silently do nothing — the facade warns and names
        the label auto actually selected. (2tri + objective='l1' — now
        also the default — auto-resolves to 'slp' and must NOT warn — see
        the companion test below.)"""
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(
            solver='auto',
            accuracy='max',
            constraint='simplex',
            objective='l2',
            verbose=0,
            record_history=False,
        )
        with pytest.warns(UserWarning, match="applies only to.*solver='slp'"):
            DVFopt(cfg).fit(phi)

    def test_auto_solver_l1_max_accuracy_injects_into_slp(self, monkeypatch, recwarn):
        """2tri + objective='l1' + solver='auto' resolves to 'slp', so
        accuracy='max' is forwarded into SLPStrategy instead of warning.

        The make_strategy spy downgrades the actual construction to
        accuracy='fast' so the test never needs torch/GPU — the assertion
        is about the label and kwargs the facade *requested*."""
        import dvfopt.unified as unified_mod

        captured = {}
        real_make = unified_mod.make_strategy

        def spy(label, **kw):
            captured['label'] = label
            captured['kwargs'] = dict(kw)
            if label == 'slp':
                kw = {**kw, 'accuracy': 'fast'}
            return real_make(label, **kw)

        monkeypatch.setattr(unified_mod, 'make_strategy', spy)
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(
            solver='auto',
            accuracy='max',
            constraint='simplex',
            objective='l1',
            verbose=0,
            record_history=False,
        )
        DVFopt(cfg).fit(phi)
        assert captured['label'] == 'slp'
        assert captured['kwargs']['accuracy'] == 'max'
        # No "applies only to solver='slp'" warning in the auto+l1 path.
        assert not [w for w in recwarn.list if 'applies only' in str(w.message)]

    def test_auto_solver_fast_accuracy_no_warning(self, recwarn):
        """The default accuracy='fast' with solver='auto' stays silent."""
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(
            solver='auto',
            constraint='simplex',
            verbose=0,
            record_history=False,
        )
        DVFopt(cfg).fit(phi)
        assert not [w for w in recwarn.list if 'slp' in str(w.message)]

    def test_instance_strategy_fast_accuracy_no_warning(self, recwarn):
        """The default accuracy='fast' with an instance stays silent."""
        from dvfopt import BarrierStrategy

        phi = _planted_fold_2d()
        cfg = DVFoptConfig(
            solver=BarrierStrategy(),
            constraint='simplex',
            verbose=0,
            record_history=False,
        )
        DVFopt(cfg).fit(phi)
        assert not [w for w in recwarn.list if 'ignored' in str(w.message)]


class TestInputShapeDispatch:
    def test_2hw_input_returns_2hw_output(self):
        phi = _planted_fold_2d(H=10, W=10)
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == (2, 10, 10)

    def test_3hw_input_returns_3hw_output(self):
        phi2 = _planted_fold_2d(H=10, W=10)
        phi = np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == (3, 10, 10)
        # dz should remain zero throughout (only dy, dx are touched).
        np.testing.assert_array_equal(res.corrected[0], 0.0)

    def test_already_feasible_short_circuits(self):
        phi = np.zeros((2, 8, 8))
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.feasible
        assert res.slice_results[0].solver_used == "none"


class TestSliceResultMemory:
    """Finding: SliceResult retained a full per-slice copy of the
    corrected field plus the raw legacy history a second time in
    ``SolveInfo.extras['_legacy_history']`` (~1.2 GB redundant on a
    528-slice volume)."""

    def test_slice_corrected_is_readonly_view_matching_volume(self):
        phi = _planted_fold_2d(H=10, W=10)
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=True, verbose=0)
        res = DVFopt(cfg).fit(phi)
        sr = res.slice_results[0]
        # Values match the assembled volume slice ([dy, dx] channels).
        np.testing.assert_array_equal(sr.corrected, res.corrected)
        # It's a read-only view, not an independent writable copy.
        assert not sr.corrected.flags.writeable
        assert sr.corrected.base is not None

    def test_3d_volume_slice_views_match_assembled(self):
        rng = np.random.default_rng(1)
        phi = rng.normal(0, 0.3, (3, 2, 10, 10))
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == (3, 2, 10, 10)
        for sr in res.slice_results:
            np.testing.assert_array_equal(sr.corrected, res.corrected[1:3, sr.z])
            assert not sr.corrected.flags.writeable

    def test_no_duplicate_legacy_history_in_extras(self):
        from dvfopt.solver import SolveInfo

        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=True, verbose=0)
        res = DVFopt(cfg).fit(phi)
        sr = res.slice_results[0]
        # History is exposed once (SliceResult.history) ...
        assert len(sr.history) > 0
        # ... and NOT stashed a second time inside SolveInfo.extras.
        if isinstance(sr.info, SolveInfo):
            assert '_legacy_history' not in sr.info.extras


class TestPerSliceParallelism:
    """``DVFoptConfig.n_workers`` — per-slice process pool (spawn-safe)."""

    def test_n_workers_matches_serial(self):
        rng = np.random.default_rng(2)
        phi = rng.normal(0, 0.3, (3, 3, 10, 10))
        kw = dict(solver="barrier", constraint="simplex", record_history=False, verbose=0)
        serial = DVFopt(DVFoptConfig(**kw)).fit(phi)
        parallel = DVFopt(DVFoptConfig(**kw, n_workers=2)).fit(phi)
        np.testing.assert_array_equal(parallel.corrected, serial.corrected)
        assert [s.z for s in parallel.slice_results] == [0, 1, 2]  # slice order preserved
        assert [s.final_n_neg for s in parallel.slice_results] == [
            s.final_n_neg for s in serial.slice_results
        ]

    def test_single_slice_stays_serial(self, monkeypatch):
        import concurrent.futures

        def _no_pool(*a, **kw):
            raise AssertionError("a single slice must not spawn a process pool")

        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _no_pool)
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="simplex", verbose=0, n_workers=4)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == phi.shape


class TestBarrierReducesFolds:
    """End-to-end: barrier should reduce or eliminate folds."""

    def test_2tri_barrier_reduces_neg_triangles(self):
        phi = _planted_fold_2d(H=14, W=14, seed=3)
        T1_init, T2_init = _triangle_areas_2d(phi[0], phi[1])
        init_neg = int(((T1_init <= 0) | (T2_init <= 0)).sum())
        assert init_neg > 0, "test setup needs an initial fold"

        cfg = DVFoptConfig(solver="barrier", constraint="simplex", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        T1_final, T2_final = _triangle_areas_2d(res.corrected[0], res.corrected[1])
        final_neg = int(((T1_final <= 0) | (T2_final <= 0)).sum())
        assert final_neg == 0, f"expected barrier to clear all folds, got {final_neg} remaining"
