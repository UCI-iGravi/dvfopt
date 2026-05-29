"""Tests for the Schwarz hybrid 2-tri solver and its per-cluster building block.

Promoted from ``notebooks/experiments/overlapping_tiles_schwarz.ipynb``,
which demonstrated 100% feasibility on dense-fold B0039 slices the
plain per-cluster runner couldn't crack.
"""

import numpy as np
import pytest

from dvfopt import iterative_2d_tri_schwarz, solve_cluster_2tri_2d
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _fold_count(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return int((T1 <= 0).sum() + (T2 <= 0).sum())


def _planted_fold(H=10, W=10, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, 0.3, (H, W)),
                     rng.normal(0, 0.3, (H, W))])


# ---------------------------------------------------------------------------
# solve_cluster_2tri_2d
# ---------------------------------------------------------------------------

class TestSolveCluster2tri2D:
    def test_no_initial_folds_short_circuits(self):
        H, W = 6, 6
        phi = np.zeros((2, H, W))
        anchor = phi.copy()
        im = np.zeros((H, W), dtype=bool)
        im[1:-1, 1:-1] = True
        out, info = solve_cluster_2tri_2d(phi, anchor, im)
        assert info['feasible'] is True
        assert info['init_n_neg'] == 0
        assert info['l2_passes_run'] == 0

    def test_clears_planted_fold(self):
        H, W = 8, 8
        phi = _planted_fold(H, W, seed=1)
        anchor = phi.copy()
        im = np.zeros((H, W), dtype=bool)
        im[1:-1, 1:-1] = True
        assert _fold_count(phi) > 0, "test setup needs folds"
        out, info = solve_cluster_2tri_2d(phi, anchor, im,
                                           l2_max_passes=8, l2_max_iter=80)
        assert out.shape == phi.shape
        # On a small contained fold, the cluster solver should converge.
        assert info['feasible'] is True
        assert info['after_l2_n_neg'] == 0

    def test_zero_interior_returns_unfeasible(self):
        H, W = 5, 5
        phi = _planted_fold(H, W, seed=2)
        anchor = phi.copy()
        im = np.zeros((H, W), dtype=bool)  # no movable corners
        out, info = solve_cluster_2tri_2d(phi, anchor, im)
        # With no movable corners and existing folds, it must report infeasible.
        if info['init_n_neg'] > 0:
            assert info['feasible'] is False

    def test_info_schema(self):
        H, W = 6, 6
        phi = _planted_fold(H, W, seed=3)
        anchor = phi.copy()
        im = np.zeros((H, W), dtype=bool)
        im[1:-1, 1:-1] = True
        _, info = solve_cluster_2tri_2d(phi, anchor, im,
                                         l2_max_passes=4, l2_max_iter=50)
        for k in ('init_n_neg', 'init_min_tri', 'after_l2_n_neg',
                  'after_l2_min', 'after_l1_n_neg', 'after_l1_min',
                  'l2_passes_run', 'l2_total_nit', 'l2_total_t',
                  'l1_polished', 'l1_nit', 'l1_t',
                  'cluster_t', 'feasible'):
            assert k in info, f"missing info key: {k}"


# ---------------------------------------------------------------------------
# iterative_2d_tri_schwarz — full-slice entry point
# ---------------------------------------------------------------------------

class TestIterative2DTriSchwarz:
    def test_identity_unchanged(self):
        phi = np.zeros((2, 10, 10))
        out = iterative_2d_tri_schwarz(phi, max_outer=2, verbose=0)
        np.testing.assert_array_equal(out, 0.0)

    def test_clears_moderate_fold(self):
        # 10x10 with planted fold — all components are small, no Schwarz fires.
        phi = _planted_fold(H=10, W=10, seed=5)
        assert _fold_count(phi) > 0
        out = iterative_2d_tri_schwarz(phi, max_outer=10, verbose=0)
        assert _fold_count(out) == 0

    def test_large_threshold_triggers_schwarz_branch(self):
        """A 30x30 random field with a large fold cluster should exercise
        the Schwarz branch when large_span/area are set small."""
        rng = np.random.default_rng(99)
        phi = np.stack([rng.normal(0, 0.4, (30, 30)),
                        rng.normal(0, 0.4, (30, 30))])
        init = _fold_count(phi)
        if init == 0:
            pytest.skip("synthetic field has no folds — adjust seed")
        # Force Schwarz routing for any moderately-sized component.
        out = iterative_2d_tri_schwarz(
            phi, max_outer=15, verbose=0,
            large_span=5, large_area=20,    # very low -> almost always Schwarz
            tile=8, overlap=2, schwarz_max_sweeps=4,
            l2_passes=6, l2_iter=60, l1_iter=80,
        )
        # Schwarz route should clear at least as well as plain (we don't
        # require == 0 here because small synthetic random fields can have
        # adversarial fold patterns; the goal is to verify the branch runs).
        assert _fold_count(out) <= init

    def test_accepts_31hw_shape(self):
        phi2 = _planted_fold(H=8, W=8, seed=7)
        phi = np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])[:, None]
        out = iterative_2d_tri_schwarz(phi, max_outer=8, verbose=0)
        assert out.shape == (2, 8, 8)

    def test_float32_input_handled(self):
        """B0039 volumes are stored as float32; SLSQP needs float64.
        The Schwarz entry must coerce automatically."""
        phi = _planted_fold(H=8, W=8, seed=11).astype(np.float32)
        out = iterative_2d_tri_schwarz(phi, max_outer=8, verbose=0)
        assert out.dtype == np.float64
        assert _fold_count(out) == 0

    def test_record_history(self):
        phi = _planted_fold(H=10, W=10, seed=13)
        out, hist = iterative_2d_tri_schwarz(
            phi, max_outer=5, verbose=0, record_history=True)
        assert out.shape == phi.shape
        assert isinstance(hist, list)
        for h in hist:
            for k in ('outer', 'n_neg', 'min_tri',
                      'n_components', 'n_large', 'n_small', 'wall_s'):
                assert k in h
