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

import numpy as np
import pytest

from dvfopt import DVFopt, DVFoptConfig
from dvfopt.constraints import JdetConstraint2D, TriConstraint2D
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
            solver="barrier", constraint="2tri", record_history=record_history, verbose=0
        )
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == phi.shape
        # If the unpack-bug regresses, dx would equal dy.
        assert not np.allclose(res.corrected[0], res.corrected[1])

    def test_history_present_when_recorded(self):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=True, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert len(res.slice_results[0].history) > 0

    def test_history_empty_when_disabled(self):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=False, verbose=0)
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
        c = TriConstraint2D((12, 12))
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
        c = TriConstraint2D((12, 12))
        # n_neg > 100 routes to barrier for 2tri (non-l1) regardless of min_T.
        assert auto_strategy(c, init_n_neg=600, init_min=-0.01, objective_label='l2') == "barrier"

    def test_extreme_picks_wallbreaker(self):
        # Wallbreaker routing in the extreme tier now applies to non-l1,
        # non-l2 objectives (l1 routes to 'slp'; l2 to 'm10' below).
        c_small = TriConstraint2D((20, 20))
        # Smaller slice (<20K corners) still picks m14 instead of m14_schwarz.
        assert (
            auto_strategy(c_small, init_n_neg=6000, init_min=-15, objective_label='none') == "m14"
        )
        # Large slice (>20K corners) routes to m14_schwarz.
        c_big = TriConstraint2D((320, 456))
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
            solver="slsqp", constraint="2tri", verbose=0, strategy_kwargs={"max_iter": 40}
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
        cfg = DVFoptConfig(solver=strat, constraint="2tri", verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.feasible
        # solver_used should reflect the class name when an instance was
        # passed (rather than a string label).
        assert res.slice_results[0].solver_used == "BarrierStrategy"


class TestAutoStrategySLPRouting:
    """2-tri + L1 auto-routes to the SLP champion at EVERY fold tier.

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
        for c in (TriConstraint2D((20, 20)), TriConstraint2D((320, 456))):
            assert (
                auto_strategy(c, init_n_neg=n_neg, init_min=init_min, objective_label='l1') == 'slp'
            )

    def test_2tri_l1_default_label_routes_to_slp(self):
        """objective_label defaults to 'l1' → 'slp'."""
        c = TriConstraint2D((12, 12))
        assert auto_strategy(c, init_n_neg=5, init_min=-0.1) == 'slp'

    def test_2tri_fullcoverage_l1_routes_to_slp(self):
        from dvfopt.constraints import TriConstraint2DFullCoverage

        c = TriConstraint2DFullCoverage((20, 20))
        assert auto_strategy(c, init_n_neg=600, init_min=-0.5, objective_label='l1') == 'slp'

    @pytest.mark.parametrize(
        'n_neg,init_min,expected',
        [(5, -0.1, 'slsqp'), (5, -0.5, 'barrier'), (600, -0.01, 'barrier'), (6000, -15.0, 'm10')],
    )
    def test_2tri_l2_keeps_legacy_routing(self, n_neg, init_min, expected):
        c = TriConstraint2D((20, 20))
        assert (
            auto_strategy(c, init_n_neg=n_neg, init_min=init_min, objective_label='l2') == expected
        )

    def test_jdet_l1_never_routes_to_slp(self):
        """SLP is 2-tri-only; the Jdet family keeps its own routing
        even for l1."""
        c = JdetConstraint2D((12, 12))
        assert (
            auto_strategy(c, init_n_neg=5, init_min=-0.5, objective_label='l1')
            == _jdet_mild_label()
        )
        assert auto_strategy(c, init_n_neg=5000, init_min=-1.5, objective_label='l1') == 'barrier'


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

    @pytest.mark.parametrize('constraint', ['jdet_3d', '6tet', '6tet_3d'])
    def test_3d_constraints_rejected_at_construction(self, constraint):
        """The facade is per-slice 2D. It used to accept 3D constraint
        labels at validation, only to fail later inside
        ``_build_constraint`` ('6tet') or run jdet_3d mis-shaped
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
            constraint='2tri',
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
            constraint='2tri',
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
            constraint='2tri',
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
            constraint='2tri',
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
            constraint='2tri',
            verbose=0,
            record_history=False,
        )
        DVFopt(cfg).fit(phi)
        assert not [w for w in recwarn.list if 'ignored' in str(w.message)]


class TestInputShapeDispatch:
    def test_2hw_input_returns_2hw_output(self):
        phi = _planted_fold_2d(H=10, W=10)
        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == (2, 10, 10)

    def test_3hw_input_returns_3hw_output(self):
        phi2 = _planted_fold_2d(H=10, W=10)
        phi = np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])
        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == (3, 10, 10)
        # dz should remain zero throughout (only dy, dx are touched).
        np.testing.assert_array_equal(res.corrected[0], 0.0)

    def test_already_feasible_short_circuits(self):
        phi = np.zeros((2, 8, 8))
        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=False, verbose=0)
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
        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=True, verbose=0)
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
        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == (3, 2, 10, 10)
        for sr in res.slice_results:
            np.testing.assert_array_equal(sr.corrected, res.corrected[1:3, sr.z])
            assert not sr.corrected.flags.writeable

    def test_no_duplicate_legacy_history_in_extras(self):
        from dvfopt.solver import SolveInfo

        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=True, verbose=0)
        res = DVFopt(cfg).fit(phi)
        sr = res.slice_results[0]
        # History is exposed once (SliceResult.history) ...
        assert len(sr.history) > 0
        # ... and NOT stashed a second time inside SolveInfo.extras.
        if isinstance(sr.info, SolveInfo):
            assert '_legacy_history' not in sr.info.extras


class TestBarrierReducesFolds:
    """End-to-end: barrier should reduce or eliminate folds."""

    def test_2tri_barrier_reduces_neg_triangles(self):
        phi = _planted_fold_2d(H=14, W=14, seed=3)
        T1_init, T2_init = _triangle_areas_2d(phi[0], phi[1])
        init_neg = int(((T1_init <= 0) | (T2_init <= 0)).sum())
        assert init_neg > 0, "test setup needs an initial fold"

        cfg = DVFoptConfig(solver="barrier", constraint="2tri", record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        T1_final, T2_final = _triangle_areas_2d(res.corrected[0], res.corrected[1])
        final_neg = int(((T1_final <= 0) | (T2_final <= 0)).sum())
        assert final_neg == 0, f"expected barrier to clear all folds, got {final_neg} remaining"
