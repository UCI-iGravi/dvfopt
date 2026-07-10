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
        c = TriConstraint2D((12, 12))
        # init_min between -0.25 and -1.0 is severe for 2tri.
        assert auto_strategy(c, init_n_neg=5, init_min=-0.5) == "barrier"
        # init_min above -0.25 with few folds is mild.
        assert auto_strategy(c, init_n_neg=5, init_min=-0.1) == "slsqp"

    def test_jdet_uses_minus_one_threshold(self):
        c = JdetConstraint2D((12, 12))
        # init_min between -0.25 and -1.0 is still mild for Jdet.
        assert auto_strategy(c, init_n_neg=5, init_min=-0.5) == "slsqp_windowed"
        assert auto_strategy(c, init_n_neg=5, init_min=-1.5) == "barrier"

    def test_count_overrides_min(self):
        c = TriConstraint2D((12, 12))
        # n_neg > 100 routes to barrier for 2tri regardless of min_T.
        assert auto_strategy(c, init_n_neg=600, init_min=-0.01) == "barrier"

    def test_extreme_picks_wallbreaker(self):
        c_small = TriConstraint2D((20, 20))
        # Smaller slice (<20K corners) still picks m14 instead of m14_schwarz.
        assert auto_strategy(c_small, init_n_neg=6000, init_min=-15, objective_label='l1') == "m14"
        # Large slice (>20K corners) routes to m14_schwarz.
        c_big = TriConstraint2D((320, 456))
        assert (
            auto_strategy(c_big, init_n_neg=6000, init_min=-15, objective_label='l1')
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
