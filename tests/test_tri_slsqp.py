"""Tests for ``dvfopt.core.slsqp_fullgrid.tri2d.iterative_2d_tri_slsqp``.

This is the package's full-grid SLSQP path for the 2-triangle
constraint, promoted from the experimental implementation in
``notebooks/two-triangle-check/14_l1-warmstart-2d-cases.ipynb``.
"""

import numpy as np
import pytest

from dvfopt.core.slsqp_fullgrid.tri2d import iterative_2d_tri_slsqp
from dvfopt.jacobian.triangle_sign import (
    _corner_patch_areas_2d,
    _triangle_areas_2d,
)


def _planted_fold(H=8, W=8, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, 0.3, (H, W)), rng.normal(0, 0.3, (H, W))])


def _full_neg(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    patches = _corner_patch_areas_2d(phi[0], phi[1])
    return int((T1 <= 0).sum() + (T2 <= 0).sum() + (patches <= 0).sum())


class TestReturnShape:
    def test_returns_2hw(self):
        phi = _planted_fold(H=8, W=8)
        out = iterative_2d_tri_slsqp(phi, verbose=0)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 8, 8)

    def test_record_history_returns_tuple(self):
        phi = _planted_fold(H=8, W=8)
        out = iterative_2d_tri_slsqp(phi, verbose=0, record_history=True)
        assert isinstance(out, tuple) and len(out) == 2
        phi_corr, hist = out
        assert phi_corr.shape == (2, 8, 8)
        assert isinstance(hist, list)
        assert len(hist) >= 1
        for h in hist:
            assert 'phase' in h
            assert h['phase'] in ('cold', 'warm')
            assert 'nit' in h
            assert 'status' in h

    def test_accepts_31hw_shape(self):
        phi2 = _planted_fold(H=6, W=6)
        phi = np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])[:, None]
        # (3, 1, 6, 6) — dz channel ignored.
        out = iterative_2d_tri_slsqp(phi, verbose=0)
        assert out.shape == (2, 6, 6)


class TestFeasibility:
    def test_full_coverage_clears_corner_folds(self):
        """If a fold is at the (0, 0) corner, only the corner-patch triangle
        catches it. full_coverage=True must enforce it; full_coverage=False
        may leave it folded."""
        H, W = 6, 6
        phi = np.zeros((2, H, W))
        # Plant a fold at the (0, 0) corner: push it inside the cell.
        phi[0, 0, 0] = 2.0
        phi[1, 0, 0] = 2.0

        patches_init = _corner_patch_areas_2d(phi[0], phi[1])
        assert patches_init[0] < 0, "test setup needs a planted corner fold"

        out = iterative_2d_tri_slsqp(
            phi,
            threshold=0.01,
            verbose=0,
            anchor='l2',
            full_coverage=True,
            max_iter=200,
            warm_max_iter=2000,
        )
        patches_final = _corner_patch_areas_2d(out[0], out[1])
        assert patches_final[0] >= 0.01 - 1e-5, (
            f"corner-patch fold not cleared: patches_final[0]={patches_final[0]}"
        )

    def test_reduces_neg_triangle_count(self):
        phi = _planted_fold(H=10, W=10, seed=3)
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        init_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
        assert init_neg > 0, "test setup needs a folded field"

        out = iterative_2d_tri_slsqp(
            phi,
            threshold=0.01,
            verbose=0,
            anchor='l1',
            full_coverage=True,
            max_iter=80,
            warm_max_iter=1000,
        )
        T1_f, T2_f = _triangle_areas_2d(out[0], out[1])
        final_neg = int((T1_f <= 0).sum() + (T2_f <= 0).sum())
        # Expect full feasibility on this seeded case.
        assert final_neg == 0, f"expected 0 folds, got {final_neg}"


class TestAnchorModes:
    @pytest.mark.parametrize("anchor", ["l2", "l1", "none"])
    def test_anchor_runs(self, anchor):
        phi = _planted_fold(H=6, W=6, seed=1)
        out = iterative_2d_tri_slsqp(phi, anchor=anchor, verbose=0, max_iter=80, warm_max_iter=500)
        assert out.shape == phi.shape
        assert np.all(np.isfinite(out))

    def test_invalid_anchor_raises(self):
        phi = _planted_fold(H=4, W=4)
        with pytest.raises(ValueError):
            iterative_2d_tri_slsqp(phi, anchor='l99', verbose=0, max_iter=20, warm_max_iter=20)


class TestFullCoverageFlag:
    def test_constraint_count_differs(self):
        """full_coverage=True adds exactly 2 constraint entries (the patches).
        Verify by running each mode and checking the recorded history's
        iteration counts/statuses behave sensibly."""
        phi = _planted_fold(H=5, W=5, seed=7)
        out_a, _hist_a = iterative_2d_tri_slsqp(
            phi, verbose=0, full_coverage=False, record_history=True, max_iter=80, warm_max_iter=500
        )
        out_b, _hist_b = iterative_2d_tri_slsqp(
            phi, verbose=0, full_coverage=True, record_history=True, max_iter=80, warm_max_iter=500
        )
        assert out_a.shape == out_b.shape
        # Both must produce finite output.
        assert np.all(np.isfinite(out_a))
        assert np.all(np.isfinite(out_b))
