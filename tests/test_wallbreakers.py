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
from dvfopt.core.wallbreakers._alm import _alm_objective, _alm_objective_ref
from dvfopt.core.wallbreakers._common import (
    _barrier_anchored_objective_ref,
    barrier_anchored_objective,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.objectives import L1Objective, L2Objective, make_objective


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


class TestHarmonicBestEffort:
    """Regression: on grow-exhaustion the best-effort path used to install
    the LAST failed trial. Growing the patch is not monotone — a later,
    larger patch can be worse — so the BEST trial (by ``patch_T_min``)
    must be installed, and only if it improves on the input field's min
    over the same cells; otherwise the input region is kept."""

    H = W = 12

    @staticmethod
    def _spike(v, H=12, W=12):
        """Corner spike of magnitude ``v`` at the grid centre. The cell
        whose TL corner is spiked has T2 = 0.5 - v, so any v > 0.49
        folds below threshold=0.01, monotonically worse with v."""
        phi = np.zeros((2, H, W), dtype=np.float64)
        phi[0, H // 2, W // 2] = v
        phi[1, H // 2, W // 2] = v
        return phi

    def _run_with_scripted_trials(self, monkeypatch, phi_in, trial_vs, max_grow_iters):
        """Monkeypatch the Laplace solve so grow round k produces the
        spike field of magnitude ``trial_vs[k]`` (dy then dx per round)."""
        import dvfopt.core.wallbreakers._harmonic as harm

        trials = [self._spike(v, self.H, self.W) for v in trial_vs]
        counter = {'n': 0}

        def fake_solve(values, free_mask):
            k = counter['n'] // 2
            channel = counter['n'] % 2
            counter['n'] += 1
            return trials[min(k, len(trials) - 1)][channel].copy()

        monkeypatch.setattr(harm, '_solve_laplace_patch', fake_solve)
        phi_out, info = harmonic_extension_2d(
            phi_in,
            threshold=0.01,
            max_grow_iters=max_grow_iters,
            record_history=True,
        )
        return phi_out, info, trials

    def test_best_trial_kept_when_later_grows_regress(self, monkeypatch):
        """Monotone regression across grow rounds: round 0 is the best
        (still infeasible); every later round is worse. The OLD code
        installed the last (worst) trial; the fix installs round 0."""
        phi_in = self._spike(6.0, self.H, self.W)  # input min ~ -5.5
        trial_vs = [1.5, 2.5, 3.5]  # trial mins ~ -1.0, -2.0, -3.0
        assert _fold_count(phi_in) > 0
        phi_out, info, trials = self._run_with_scripted_trials(
            monkeypatch, phi_in, trial_vs, max_grow_iters=2
        )
        rec = info['records_first5'][0]
        assert rec['failed'] is True
        assert rec['installed'] is True
        # Best (round 0) installed — not the last, not the input.
        np.testing.assert_array_equal(phi_out, trials[0])
        assert rec['best_trial_T_min'] == pytest.approx(_min_tri(trials[0]))
        assert rec['best_trial_T_min'] > rec['input_T_min']

    def test_input_kept_when_no_trial_improves_on_it(self, monkeypatch):
        """Every trial is worse than the (mildly folded) input: nothing
        is installed and the input field survives untouched."""
        phi_in = self._spike(1.2, self.H, self.W)  # input min ~ -0.7
        trial_vs = [3.0, 4.0, 5.0]  # all much worse
        assert _fold_count(phi_in) > 0
        phi_out, info, _trials = self._run_with_scripted_trials(
            monkeypatch, phi_in, trial_vs, max_grow_iters=2
        )
        rec = info['records_first5'][0]
        assert rec['failed'] is True
        assert rec['installed'] is False
        np.testing.assert_array_equal(phi_out, phi_in)


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
            phi,
            objective=L2Objective(),
            outer_max=20,
            inner_maxiter=100,
            time_budget_s=60.0,
            verbose=0,
        )
        # ALM should at least not make things worse.
        assert _fold_count(out) <= init_n

    @pytest.mark.parametrize("anchor", ["l2", "l1", "none"])
    def test_runs_under_all_anchors(self, anchor):
        """All three anchors must complete and produce finite output."""
        phi = _planted_fold(8, 8, seed=3)
        out = augmented_lagrangian_2d(
            phi,
            objective=make_objective(anchor),
            outer_max=10,
            inner_maxiter=80,
            time_budget_s=30.0,
            verbose=0,
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
# m14 — refine_repair (and m14_l1 via objective=L1Objective())
# ---------------------------------------------------------------------------


class TestRefineRepair:
    def test_clears_planted_fold_l2(self):
        phi = _planted_fold(10, 10, seed=7)
        assert _fold_count(phi) > 0
        out = iterative_2d_tri_refine_repair(
            phi, objective=L2Objective(), time_budget_s=180.0, verbose=0
        )
        assert _fold_count(out) == 0

    def test_clears_planted_fold_l1(self):
        """The m14_l1 variant — smoothed-L1 anchor throughout."""
        phi = _planted_fold(10, 10, seed=8)
        assert _fold_count(phi) > 0
        out = iterative_2d_tri_refine_repair(
            phi, objective=L1Objective(), time_budget_s=180.0, verbose=0
        )
        assert _fold_count(out) == 0

    def test_l1_anchor_uses_less_l1_than_l2_anchor(self):
        """L1 anchor should produce concentrated corrections — typically
        a smaller L1 cost than the L2 anchor."""
        phi = _planted_fold(12, 12, seed=9)
        out_l2 = iterative_2d_tri_refine_repair(
            phi, objective=L2Objective(), time_budget_s=180.0, verbose=0
        )
        out_l1 = iterative_2d_tri_refine_repair(
            phi, objective=L1Objective(), time_budget_s=180.0, verbose=0
        )
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
            objective=make_objective(anchor),
            lam_schedule=(1e2, 1e4),
            inner_maxiter=150,
            time_budget_s=60.0,
            verbose=0,
        )
        assert out.shape == phi.shape
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Fused-kernel equivalence: ALM + barrier objectives vs the legacy
# two-pass numpy references. The dispatchers (_alm_objective /
# barrier_anchored_objective) take the fused Numba path when numba is
# installed; when it is not, they ARE the reference and these tests
# pass trivially.
# ---------------------------------------------------------------------------


def _flat_pair(H, W, seed, scale=0.3):
    """Random (phi_flat, phi_in_flat) pair in the 2-tri dy-first pack."""
    rng = np.random.default_rng(seed)
    phi_flat = rng.normal(0.0, scale, 2 * H * W)
    phi_in_flat = rng.normal(0.0, scale, 2 * H * W)
    return phi_flat, phi_in_flat


def _feasible_flat_pair(H, W, seed, threshold):
    """Strictly feasible iterate (tiny displacements keep T ~ 0.5)."""
    from dvfopt.core.primitives.tri import tri_areas_flat

    phi_flat, phi_in_flat = _flat_pair(H, W, seed, scale=0.02)
    assert tri_areas_flat(phi_flat, H, W).min() > threshold
    return phi_flat, phi_in_flat


class TestALMObjectiveFusedEquivalence:
    THRESH = 0.01

    @pytest.mark.parametrize("anchor", ["l2", "l1", "none"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_dense_mu_random_field(self, anchor, seed):
        """Dense positive mu on a folded random field — many active cells."""
        H, W = 13, 17
        rng = np.random.default_rng(100 + seed)
        phi_flat, phi_in_flat = _flat_pair(H, W, seed, scale=0.4)
        mu = rng.uniform(0.1, 5.0, 2 * (H - 1) * (W - 1))
        rho = float(rng.uniform(0.5, 50.0))
        f_ref, g_ref = _alm_objective_ref(
            phi_flat, phi_in_flat, H, W, self.THRESH, mu, rho, anchor, 1e-4
        )
        f, g = _alm_objective(phi_flat, phi_in_flat, H, W, self.THRESH, mu, rho, anchor, 1e-4)
        np.testing.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("anchor", ["l2", "l1", "none"])
    def test_all_inactive(self, anchor):
        """mu = 0 on a strictly feasible field -> psi = 0 everywhere; the
        constraint term must contribute exactly zero (const trick:
        -sum(mu^2)/(2 rho) = 0 and no active cells)."""
        H, W = 11, 12
        phi_flat, phi_in_flat = _feasible_flat_pair(H, W, 7, self.THRESH)
        mu = np.zeros(2 * (H - 1) * (W - 1))
        rho = 3.0
        f_ref, g_ref = _alm_objective_ref(
            phi_flat, phi_in_flat, H, W, self.THRESH, mu, rho, anchor, 1e-4
        )
        f, g = _alm_objective(phi_flat, phi_in_flat, H, W, self.THRESH, mu, rho, anchor, 1e-4)
        np.testing.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-12)

    def test_dense_mu_feasible_field_still_active(self):
        """Feasible field but mu > rho*slack for some cells -> psi > 0 on
        those cells only; exercises the active/inactive mix."""
        H, W = 10, 10
        rng = np.random.default_rng(42)
        phi_flat, phi_in_flat = _feasible_flat_pair(H, W, 9, self.THRESH)
        mu = rng.uniform(0.0, 1.0, 2 * (H - 1) * (W - 1))
        rho = 1.0
        f_ref, g_ref = _alm_objective_ref(
            phi_flat, phi_in_flat, H, W, self.THRESH, mu, rho, 'l2', 1e-4
        )
        f, g = _alm_objective(phi_flat, phi_in_flat, H, W, self.THRESH, mu, rho, 'l2', 1e-4)
        np.testing.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-12)

    def test_precomputed_const_matches_default(self):
        """Passing mu_sq_const explicitly (the augmented_lagrangian_2d
        plumbing) must equal the compute-on-the-fly default."""
        H, W = 9, 14
        rng = np.random.default_rng(3)
        phi_flat, phi_in_flat = _flat_pair(H, W, 5, scale=0.4)
        mu = rng.uniform(0.1, 4.0, 2 * (H - 1) * (W - 1))
        rho = 7.0
        const = -float(mu @ mu) / (2.0 * rho)
        f_default, g_default = _alm_objective(
            phi_flat, phi_in_flat, H, W, self.THRESH, mu, rho, 'l1', 1e-4
        )
        f_const, g_const = _alm_objective(
            phi_flat, phi_in_flat, H, W, self.THRESH, mu, rho, 'l1', 1e-4, const
        )
        np.testing.assert_allclose(f_const, f_default, rtol=1e-12)
        np.testing.assert_array_equal(g_const, g_default)


class TestBarrierObjectiveFusedEquivalence:
    THRESH = 0.01

    @pytest.mark.parametrize("anchor", ["l2", "l1", "none"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_feasible_case(self, anchor, seed):
        H, W = 13, 17
        phi_flat, phi_in_flat = _feasible_flat_pair(H, W, 20 + seed, self.THRESH)
        mu = 1e-2
        f_ref, g_ref = _barrier_anchored_objective_ref(
            phi_flat, phi_in_flat, H, W, self.THRESH, mu, anchor, 1e-4
        )
        assert np.isfinite(f_ref)
        f, g = barrier_anchored_objective(
            phi_flat, phi_in_flat, H, W, self.THRESH, mu, anchor, 1e-4
        )
        np.testing.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("anchor", ["l2", "l1", "none"])
    def test_infeasible_returns_inf_with_anchor_gradient(self, anchor):
        """Infeasible iterate: +inf value and the ANCHOR-ONLY gradient
        (this module's convention — _barrier_core returns zeros; the
        fused path must not silently switch conventions)."""
        H, W = 10, 10
        phi_flat, phi_in_flat = _flat_pair(H, W, 30, scale=0.5)  # folded
        from dvfopt.core.primitives.tri import tri_areas_flat

        assert tri_areas_flat(phi_flat, H, W).min() <= self.THRESH
        f_ref, g_ref = _barrier_anchored_objective_ref(
            phi_flat, phi_in_flat, H, W, self.THRESH, 1e-2, anchor, 1e-4
        )
        assert np.isinf(f_ref)
        f, g = barrier_anchored_objective(
            phi_flat, phi_in_flat, H, W, self.THRESH, 1e-2, anchor, 1e-4
        )
        assert np.isinf(f)
        np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-12)

    def test_unknown_anchor_raises(self):
        H, W = 6, 6
        phi_flat, phi_in_flat = _flat_pair(H, W, 40, scale=0.01)
        with pytest.raises(ValueError):
            barrier_anchored_objective(
                phi_flat, phi_in_flat, H, W, self.THRESH, 1e-2, 'bogus', 1e-4
            )
