"""Tests for iterative_2d_tri_refine_repair_schwarz (m14-Schwarz)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dvfopt.core.wallbreakers import iterative_2d_tri_refine_repair_schwarz
from dvfopt.core.wallbreakers._schwarz_common import (
    _fold_clusters_2d as _fold_clusters,
)
from dvfopt.core.wallbreakers._schwarz_common import (
    _stats_2d as _stats,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _plant_fold(arr, cy, cx, amp=0.8):
    arr[cy, cx] += amp
    arr[cy + 1, cx] -= amp
    arr[cy, cx + 1] -= amp
    arr[cy + 1, cx + 1] += amp


def _synth_sparse(seed=0, H=30, W=30):
    rng = np.random.default_rng(seed)
    dy = rng.normal(0, 0.05, (H, W))
    dx = rng.normal(0, 0.05, (H, W))
    _plant_fold(dx, 5, 5)
    _plant_fold(dx, 5, 20)
    _plant_fold(dy, 22, 12)
    return np.stack([dy, dx])


class TestFoldClusters:
    def test_no_folds_returns_empty(self):
        H, W = 6, 6
        phi = np.zeros((2, H, W))
        bboxes, fold_mask = _fold_clusters(phi, merge_dilation=2)
        assert bboxes == []
        assert not fold_mask.any()

    def test_sparse_three_clusters(self):
        phi = _synth_sparse(0)
        bboxes, _fold_mask = _fold_clusters(phi, merge_dilation=2)
        assert len(bboxes) == 3
        for b in bboxes:
            assert b['n_folds'] >= 1
            assert b['cy0'] <= b['cy1']
            assert b['cx0'] <= b['cx1']

    def test_close_clusters_merge_under_high_dilation(self):
        H, W = 20, 20
        dy = np.zeros((H, W))
        dx = np.zeros((H, W))
        _plant_fold(dx, 5, 5)
        _plant_fold(dx, 8, 8)
        phi = np.stack([dy, dx])
        bboxes_lo, _ = _fold_clusters(phi, merge_dilation=1)
        bboxes_hi, _ = _fold_clusters(phi, merge_dilation=4)
        assert len(bboxes_lo) >= len(bboxes_hi)

    def test_merge_dilation_zero_keeps_clusters_separate(self):
        """merge_dilation=0 means "no grouping dilation" — NOT scipy's
        binary_dilation(iterations=0) repeat-until-convergence, which
        would fill the grid and collapse everything into one whole-grid
        cluster."""
        H, W = 30, 30
        dy = np.zeros((H, W))
        dx = np.zeros((H, W))
        _plant_fold(dx, 5, 5)
        _plant_fold(dx, 22, 22)
        phi = np.stack([dy, dx])
        bboxes, _fold_mask = _fold_clusters(phi, merge_dilation=0)
        assert len(bboxes) == 2
        # Each cluster bbox stays local to its planted core.
        for b in bboxes:
            assert b['cy1'] - b['cy0'] < 10
            assert b['cx1'] - b['cx0'] < 10

    def test_negative_merge_dilation_raises(self):
        phi = np.zeros((2, 8, 8))
        with pytest.raises(ValueError, match='merge_dilation'):
            _fold_clusters(phi, merge_dilation=-1)


class TestFoldClusters3D:
    """The 3D twin in _schwarz_common gets the same dilation guards."""

    def test_merge_dilation_zero_keeps_clusters_separate(self):
        from dvfopt.core.wallbreakers._schwarz_common import _fold_clusters_3d

        phi = np.zeros((3, 5, 20, 20))
        phi[1, 2, 4, 4] = 1.5
        phi[2, 2, 4, 4] = 1.5
        phi[1, 2, 15, 15] = 1.5
        phi[2, 2, 15, 15] = 1.5
        bboxes, fold_cells = _fold_clusters_3d(phi, threshold=0.0, merge_dilation=0)
        assert fold_cells.any(), 'test setup planted no folds'
        assert len(bboxes) == 2

    def test_negative_merge_dilation_raises(self):
        from dvfopt.core.wallbreakers._schwarz_common import _fold_clusters_3d

        with pytest.raises(ValueError, match='merge_dilation'):
            _fold_clusters_3d(np.zeros((3, 4, 4, 4)), threshold=0.0, merge_dilation=-1)


class TestSmoke:
    def test_identity_field_passes_through(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phi = np.zeros((2, 8, 8))
            out = iterative_2d_tri_refine_repair_schwarz(phi, threshold=0.01, verbose=0)
        np.testing.assert_allclose(out, phi, atol=1e-9)

    def test_sparse_synthetic_reaches_feasibility(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phi = _synth_sparse(0)
            out = iterative_2d_tri_refine_repair_schwarz(
                phi.copy(), threshold=0.01, anchor='l1', verbose=0, max_outer_iters=2
            )
        T1, T2 = _triangle_areas_2d(out[0], out[1])
        n_neg = int((np.minimum(T1, T2) <= 0).sum())
        min_T = float(min(T1.min(), T2.min()))
        assert n_neg == 0, f'left {n_neg} folds'
        assert min_T >= 0.01 - 1e-5, f'min_T={min_T:+.5f} below threshold'

    def test_history_recorded(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phi = _synth_sparse(0)
            _out, info = iterative_2d_tri_refine_repair_schwarz(
                phi.copy(),
                threshold=0.01,
                anchor='l1',
                verbose=0,
                max_outer_iters=2,
                record_history=True,
            )
        # Three planted clusters at this seed.
        assert info['init']['n_neg'] > 0
        assert 'cluster_runs' in info
        assert 'outer_rounds' in info
        assert 'final' in info
        assert info['final']['n_neg'] == 0


class TestFallback:
    def test_single_large_cluster_falls_back(self):
        """A near-saturated 8x8 should trigger the size-ratio fallback."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rng = np.random.default_rng(3)
            phi = np.stack([rng.normal(0, 0.6, (8, 8)), rng.normal(0, 0.6, (8, 8))])
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            n_neg_init = int((np.minimum(T1, T2) <= 0).sum())
            if n_neg_init == 0:
                pytest.skip('seed produced no folds')
            out, info = iterative_2d_tri_refine_repair_schwarz(
                phi.copy(),
                threshold=0.01,
                anchor='l1',
                fallback_size_ratio=0.5,
                verbose=0,
                record_history=True,
            )
        T1, T2 = _triangle_areas_2d(out[0], out[1])
        assert int((np.minimum(T1, T2) <= 0).sum()) == 0
        assert info['fallback_to_global']


class TestUnifiedAPI:
    def test_solver_m14_schwarz_routes(self):
        from dvfopt import DVFopt, DVFoptConfig

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phi = _synth_sparse(0)
            cfg = DVFoptConfig(solver='m14_schwarz', constraint='2tri', objective='l1', verbose=0)
            res = DVFopt(cfg).fit(phi)
        assert res.feasible
        assert res.slice_results[0].solver_used == 'm14_schwarz'

    def test_auto_routes_large_extreme_to_schwarz(self):
        """The auto resolver picks m14_schwarz for large slices in the
        extreme-density tier when objective != 'l2'."""
        from dvfopt.constraints import TriConstraint2D
        from dvfopt.solver import auto_strategy

        c_big = TriConstraint2D((320, 456))
        c_small = TriConstraint2D((60, 60))
        assert auto_strategy(c_big, 6000, -15.0, objective_label='l1') == 'm14_schwarz'
        # Small extreme dense — falls back to plain m14.
        assert auto_strategy(c_small, 6000, -15.0, objective_label='l1') == 'm14'

    def test_auto_l2_still_picks_m10_on_extreme(self):
        from dvfopt.constraints import TriConstraint2D
        from dvfopt.solver import auto_strategy

        c_big = TriConstraint2D((320, 456))
        assert auto_strategy(c_big, 6000, -15.0, objective_label='l2') == 'm10'
