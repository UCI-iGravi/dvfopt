"""Smoke tests for the DVFopt unified high-level API.

These guard against the bugs identified in the unified.py code review:
- ``record_history=False`` previously unpacked an ndarray into (phi_new, hist),
  silently overwriting dx with dy.
- ``_run_trust_constr`` previously raised UnboundLocalError when
  ``max_outer_iters=0`` produced an empty range.
- ``_resolve_solver`` previously used a Jdet-scaled cutoff for 2tri.
- ``_run_slsqp`` silently ignored ``mode``, ``objective``, ``use_continuation``,
  ``record_history``; it should warn instead.
"""

import warnings

import numpy as np
import pytest

from dvfopt import DVFopt, DVFoptConfig
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _planted_fold_2d(H=12, W=12, seed=0):
    """Build a (2, H, W) field with a few negative triangles."""
    rng = np.random.default_rng(seed)
    phi = np.stack([rng.normal(0, 0.3, (H, W)),
                    rng.normal(0, 0.3, (H, W))])
    return phi


class TestRecordHistoryRoundTrip:
    """Bug regression: `record_history=False` must not corrupt dx."""

    @pytest.mark.parametrize("record_history", [True, False])
    def test_2tri_barrier_preserves_two_channels(self, record_history):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="2tri",
                           record_history=record_history, verbose=0)
        res = DVFopt(cfg).fit(phi)

        assert res.corrected.shape == phi.shape
        # dx must remain a real displacement, not a copy of dy. If the bug
        # is back, dx (channel 1) ends up equal to dy (channel 0).
        assert not np.allclose(res.corrected[0], res.corrected[1])

    def test_history_present_when_recorded(self):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="2tri",
                           record_history=True, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert len(res.slice_results[0].history) > 0

    def test_history_empty_when_disabled(self):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="barrier", constraint="2tri",
                           record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.slice_results[0].history == []


class TestTrustConstrEdgeCases:
    def test_zero_outer_iters_no_unbound_local(self):
        """Regression: max_outer_iters=0 previously raised UnboundLocalError."""
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="trust-constr", constraint="2tri",
                           max_outer_iters=0, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.slice_results[0].n_outer_iters == 0


class TestResolveSolverHeuristic:
    def test_2tri_uses_smaller_severe_threshold(self):
        """The 2tri severe-fold cutoff is -0.25 (not the Jdet -1.0)."""
        opt = DVFopt(DVFoptConfig(solver="auto", constraint="2tri", verbose=0))
        # init_min between -0.25 and -1.0 is severe for 2tri but would be
        # "mild" under the old Jdet heuristic.
        assert opt._resolve_solver(init_n_neg=5, init_min=-0.5) == "barrier"
        assert opt._resolve_solver(init_n_neg=5, init_min=-0.1) == "slsqp"

    def test_jdet_keeps_minus_one_threshold(self):
        opt = DVFopt(DVFoptConfig(solver="auto", constraint="jdet", verbose=0))
        assert opt._resolve_solver(init_n_neg=5, init_min=-0.5) == "slsqp"
        assert opt._resolve_solver(init_n_neg=5, init_min=-1.5) == "barrier"

    def test_count_overrides_min(self):
        opt = DVFopt(DVFoptConfig(solver="auto", constraint="2tri", verbose=0))
        assert opt._resolve_solver(init_n_neg=600, init_min=-0.01) == "barrier"

    def test_explicit_solver_passthrough(self):
        opt = DVFopt(DVFoptConfig(solver="slsqp", constraint="2tri", verbose=0))
        assert opt._resolve_solver(init_n_neg=99999, init_min=-99.0) == "slsqp"


class TestSlsqpWarnings:
    def test_full_grid_emits_warning(self):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="slsqp", mode="full-grid", verbose=0,
                           constraint="2tri")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DVFopt(cfg).fit(phi)
        msgs = [str(w.message) for w in caught]
        assert any("only supports mode='windowed'" in m for m in msgs)

    def test_non_l2_objective_emits_warning(self):
        phi = _planted_fold_2d()
        cfg = DVFoptConfig(solver="slsqp", objective="l1", verbose=0,
                           constraint="2tri")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DVFopt(cfg).fit(phi)
        msgs = [str(w.message) for w in caught]
        assert any("objective='l2'" in m for m in msgs)


class TestInputShapeDispatch:
    def test_2hw_input_returns_2hw_output(self):
        phi = _planted_fold_2d(H=10, W=10)
        cfg = DVFoptConfig(solver="barrier", constraint="2tri",
                           record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == (2, 10, 10)

    def test_3hw_input_returns_3hw_output(self):
        phi2 = _planted_fold_2d(H=10, W=10)
        phi = np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])
        cfg = DVFoptConfig(solver="barrier", constraint="2tri",
                           record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        assert res.corrected.shape == (3, 10, 10)
        # dz should remain zero throughout (only dy, dx are touched).
        np.testing.assert_array_equal(res.corrected[0], 0.0)

    def test_already_feasible_short_circuits(self):
        # An identity field has no folds.
        phi = np.zeros((2, 8, 8))
        cfg = DVFoptConfig(solver="barrier", constraint="2tri",
                           record_history=False, verbose=0)
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

        cfg = DVFoptConfig(solver="barrier", constraint="2tri",
                           record_history=False, verbose=0)
        res = DVFopt(cfg).fit(phi)
        T1_final, T2_final = _triangle_areas_2d(res.corrected[0],
                                                res.corrected[1])
        final_neg = int(((T1_final <= 0) | (T2_final <= 0)).sum())
        assert final_neg <= init_neg
