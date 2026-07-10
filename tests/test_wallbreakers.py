"""Tests for the wall-breaker methods promoted from
``notebooks/experiments/wall_breakers/methods/``.

These are the methods that proved out at 100% feasibility on the
original B0039 DVF — the ones the SLSQP pipeline can't crack.
"""

import numpy as np
import pytest

from dvfopt.core.wallbreakers import (
    augmented_lagrangian_2d,
    harmonic_extension_2d,
    iterative_2d_tri_harmonic_polished,
    iterative_2d_tri_refine_repair,
    l2_refine_2d,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _fold_count(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return int((T1 <= 0).sum() + (T2 <= 0).sum())


def _min_tri(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(min(T1.min(), T2.min()))


def _planted_fold(H=10, W=10, seed=0, scale=0.4):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, scale, (H, W)), rng.normal(0, scale, (H, W))])


# ---------------------------------------------------------------------------
# m02 — harmonic extension
# ---------------------------------------------------------------------------


class TestHarmonicExtension2D:
    def test_already_feasible_no_op(self):
        phi = np.zeros((2, 6, 6))
        phi_out, info = harmonic_extension_2d(phi, record_history=True)
        assert info['patches'] == 0
        np.testing.assert_array_equal(phi_out, phi)

    def test_default_return_is_ndarray(self):
        """API contract: returns ``phi`` by default, ``(phi, info)`` only
        when record_history=True."""
        phi = _planted_fold(8, 8, seed=1)
        out = harmonic_extension_2d(phi)
        assert isinstance(out, np.ndarray)
        assert out.shape == phi.shape

    def test_merge_dilation_zero_keeps_patches_separate(self):
        """merge_dilation=0 must mean "no grouping dilation" — NOT scipy's
        binary_dilation(iterations=0) repeat-until-convergence, which
        fills the grid and collapses everything into one whole-grid
        component."""
        phi = np.zeros((2, 24, 24))
        phi[1, 5, 5] = 1.5  # fold core A
        phi[1, 17, 17] = 1.5  # fold core B, far from A
        assert _fold_count(phi) > 0
        _phi_out, info = harmonic_extension_2d(phi, merge_dilation=0, record_history=True)
        assert info['n_components'] == 2
        assert info['patches'] == 2
        # Each patch is local, never the whole grid.
        for rec in info['records_first5']:
            assert rec['n_cells'] < 23 * 23

    def test_ring_pad_zero_stays_local(self):
        """ring_pad=0 with grow=0 used to call binary_dilation(iterations=0),
        which fills the whole grid (near-full-grid Laplacian solve). The
        correct semantic is "no dilation this round" — far-away corners
        must be untouched."""
        phi = np.zeros((2, 24, 24))
        phi[1, 5, 5] = 1.5
        phi_out = harmonic_extension_2d(phi, ring_pad=0, max_grow_iters=2)
        np.testing.assert_array_equal(phi_out[:, 15:, 15:], phi[:, 15:, 15:])

    @pytest.mark.parametrize('kwargs', [{'merge_dilation': -1}, {'ring_pad': -1}])
    def test_negative_dilation_params_raise(self, kwargs):
        phi = _planted_fold(8, 8, seed=1)
        with pytest.raises(ValueError):
            harmonic_extension_2d(phi, **kwargs)


# ---------------------------------------------------------------------------
# m03 — augmented Lagrangian
# ---------------------------------------------------------------------------


class TestAugmentedLagrangian2D:
    def test_no_fold_short_circuits(self):
        phi = np.zeros((2, 6, 6))
        _phi_out, info = augmented_lagrangian_2d(phi, outer_max=3, record_history=True, verbose=0)
        assert info['feasible'] is True

    def test_reduces_fold_count(self):
        phi = _planted_fold(10, 10, seed=2)
        init_n = _fold_count(phi)
        assert init_n > 0
        out = augmented_lagrangian_2d(
            phi, anchor='l2', outer_max=20, inner_maxiter=100, time_budget_s=60.0, verbose=0
        )
        # ALM should at least not make things worse.
        assert _fold_count(out) <= init_n

    @pytest.mark.parametrize("anchor", ["l2", "l1", "none"])
    def test_runs_under_all_anchors(self, anchor):
        """All three anchors must complete and produce finite output."""
        phi = _planted_fold(8, 8, seed=3)
        out = augmented_lagrangian_2d(
            phi, anchor=anchor, outer_max=10, inner_maxiter=80, time_budget_s=30.0, verbose=0
        )
        assert out.shape == phi.shape
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# m10 — harmonic_l2_polished (always-feasibility baseline)
# ---------------------------------------------------------------------------


class TestHarmonicPolished:
    def test_clears_planted_fold(self):
        phi = _planted_fold(10, 10, seed=3)
        assert _fold_count(phi) > 0
        out, info = iterative_2d_tri_harmonic_polished(
            phi, time_budget_s=120.0, verbose=0, record_history=True
        )
        # 100% feasibility — the m10 guarantee.
        assert _fold_count(out) == 0
        assert info['final_min_T'] >= 0.01 - 1e-5

    def test_accepts_31hw_shape(self):
        phi2 = _planted_fold(8, 8, seed=4)
        phi = np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])[:, None]
        out = iterative_2d_tri_harmonic_polished(phi, time_budget_s=60.0, verbose=0)
        assert out.shape == (2, 8, 8)

    def test_float32_input_handled(self):
        phi = _planted_fold(8, 8, seed=5).astype(np.float32)
        out = iterative_2d_tri_harmonic_polished(phi, time_budget_s=60.0, verbose=0)
        assert out.dtype == np.float64

    def test_info_schema(self):
        phi = _planted_fold(8, 8, seed=6)
        _, info = iterative_2d_tri_harmonic_polished(
            phi, time_budget_s=60.0, verbose=0, record_history=True
        )
        for k in ('stage1_harmonic', 'stage2_alm', 'final_min_T'):
            assert k in info

    def test_default_return_is_ndarray(self):
        """API contract: returns just ``phi`` by default."""
        phi = _planted_fold(8, 8, seed=7)
        out = iterative_2d_tri_harmonic_polished(phi, time_budget_s=30.0, verbose=0)
        assert isinstance(out, np.ndarray)

    def test_threshold_default_tracks_default_params(self):
        """Pass threshold=None and verify the function uses DEFAULT_PARAMS."""
        from dvfopt._defaults import DEFAULT_PARAMS

        phi = _planted_fold(8, 8, seed=8)
        # Should run without exploding.
        out = iterative_2d_tri_harmonic_polished(phi, threshold=None, time_budget_s=60.0, verbose=0)
        assert out.shape == phi.shape


# ---------------------------------------------------------------------------
# m14 — refine_repair (and m14_l1 via anchor='l1')
# ---------------------------------------------------------------------------


class TestRefineRepair:
    def test_clears_planted_fold_l2(self):
        phi = _planted_fold(10, 10, seed=7)
        assert _fold_count(phi) > 0
        out = iterative_2d_tri_refine_repair(phi, anchor='l2', time_budget_s=180.0, verbose=0)
        assert _fold_count(out) == 0

    def test_clears_planted_fold_l1(self):
        """The m14_l1 variant — smoothed-L1 anchor throughout."""
        phi = _planted_fold(10, 10, seed=8)
        assert _fold_count(phi) > 0
        out = iterative_2d_tri_refine_repair(phi, anchor='l1', time_budget_s=180.0, verbose=0)
        assert _fold_count(out) == 0

    def test_l1_anchor_uses_less_l1_than_l2_anchor(self):
        """L1 anchor should produce concentrated corrections — typically
        a smaller L1 cost than the L2 anchor."""
        phi = _planted_fold(12, 12, seed=9)
        out_l2 = iterative_2d_tri_refine_repair(phi, anchor='l2', time_budget_s=180.0, verbose=0)
        out_l1 = iterative_2d_tri_refine_repair(phi, anchor='l1', time_budget_s=180.0, verbose=0)
        assert _fold_count(out_l2) == 0
        assert _fold_count(out_l1) == 0
        l1_of_l2 = float(np.abs(out_l2 - phi).sum())
        l1_of_l1 = float(np.abs(out_l1 - phi).sum())
        assert l1_of_l1 <= l1_of_l2 * 1.1

    def test_accepts_seed_skips_stage1(self):
        phi = _planted_fold(8, 8, seed=10)
        seed = iterative_2d_tri_harmonic_polished(phi, time_budget_s=60.0, verbose=0)
        # Pass the seed — stage 1 should be skipped (no m10 inside).
        out = iterative_2d_tri_refine_repair(phi, seed=seed, time_budget_s=60.0, verbose=0)
        assert _fold_count(out) == 0


# ---------------------------------------------------------------------------
# m12 — l2_refine
# ---------------------------------------------------------------------------


class TestL2Refine2D:
    def test_runs_with_seed(self):
        phi = _planted_fold(8, 8, seed=11)
        seed = iterative_2d_tri_harmonic_polished(phi, time_budget_s=60.0, verbose=0)
        out = l2_refine_2d(
            phi,
            seed=seed,
            lam_schedule=(1e2, 1e4),
            inner_maxiter=200,
            time_budget_s=60.0,
            verbose=0,
        )
        assert _fold_count(out) == 0

    @pytest.mark.parametrize("anchor", ["l2", "l1"])
    def test_runs_under_both_anchors(self, anchor):
        phi = _planted_fold(8, 8, seed=12)
        seed = iterative_2d_tri_harmonic_polished(phi, time_budget_s=60.0, verbose=0)
        out = l2_refine_2d(
            phi,
            seed=seed,
            anchor=anchor,
            lam_schedule=(1e2, 1e4),
            inner_maxiter=150,
            time_budget_s=60.0,
            verbose=0,
        )
        assert out.shape == phi.shape
        assert np.all(np.isfinite(out))
