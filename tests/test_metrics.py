"""Tests for dvfopt.metrics — the canonical fold-statistics helpers."""

import numpy as np
import pytest

from dvfopt import correct_dvf
from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.metrics import constraint_fold_stats, field_change_stats, fold_stats
from tests.conftest import planted_fold, planted_fold_3d


def test_fold_stats_counts_and_severity():
    v = np.array([-1.0, 0.0, 0.005, 0.02, 1.0])
    st = fold_stats(v, threshold=0.01)
    assert st.n_neg == 2  # <= 0: -1.0 and 0.0
    assert st.n_below == 3  # < 0.01 - 1e-5: -1.0, 0.0, 0.005
    assert st.min_val == -1.0
    assert np.isclose(st.neg_volume, (0.01 + 1.0) + 0.01 + 0.005)
    assert not st.feasible


def test_fold_stats_default_threshold_feasible():
    st = fold_stats(np.array([0.5, 1.0]))  # default threshold 0.01
    assert st.feasible and st.n_neg == 0 and st.n_below == 0


def test_constraint_fold_stats_auto_2d():
    phi = planted_fold(10, 10, seed=0, scale=0.4)
    name, st = constraint_fold_stats(phi)
    assert name == 'simplex'
    assert st.n_neg > 0


def test_constraint_fold_stats_auto_3d():
    phi = planted_fold_3d()
    name, st = constraint_fold_stats(phi)
    assert name == 'simplex_3d'
    assert st.n_neg > 0


def test_constraint_fold_stats_matches_solver_init_stats():
    # The metrics module and Solver.fit must agree on what "folded" means.
    phi = planted_fold(10, 10, seed=0, scale=0.4)
    _, st = constraint_fold_stats(phi, constraint='simplex')
    res = correct_dvf(phi, constraint='simplex', objective='l1', strategy='auto')
    assert res.init_n_neg == st.n_neg


def test_field_change_stats_identity_is_all_zero():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(2, 6, 6))
    st = field_change_stats(phi, phi.copy())
    assert st == dict(
        move_med_px=0.0,
        move_p95_px=0.0,
        move_max_px=0.0,
        move_med_moved_px=0.0,
        move_p95_moved_px=0.0,
        moved_frac=0.0,
        moved_frac_0p5px=0.0,
        jdet_change_med=0.0,
        jdet_change_p95=0.0,
        jdet_change_max=0.0,
    )


def test_field_change_stats_single_voxel_move():
    phi_in = np.zeros((2, 4, 4))
    phi_out = phi_in.copy()
    phi_out[0, 2, 2] = 1.0  # one voxel, 1 px in dy
    st = field_change_stats(phi_in, phi_out)
    n = 4 * 4
    assert st['move_max_px'] == 1.0
    assert st['moved_frac'] == 1 / n
    assert st['moved_frac_0p5px'] == 1 / n  # 1.0 px > moved_px (0.5)


def test_field_change_stats_jdet_3d_matches_direct_call():
    rng = np.random.default_rng(1)
    phi_in = rng.normal(scale=0.1, size=(3, 3, 5, 5))
    phi_out = phi_in.copy()
    phi_out[2, 1, 2, 2] += 0.3
    st = field_change_stats(phi_in, phi_out)
    expected = np.abs(jacobian_det3D(phi_out) - jacobian_det3D(phi_in))
    assert st['jdet_change_max'] == pytest.approx(expected.max())
    assert st['jdet_change_med'] == pytest.approx(np.median(expected))


def test_field_change_stats_jdet_per_slice_2d_matches_direct_call():
    rng = np.random.default_rng(2)
    phi_in = rng.normal(scale=0.1, size=(3, 1, 5, 5))  # (3, 1, H, W): dz dropped
    phi_out = phi_in.copy()
    phi_out[2, 0, 2, 2] += 0.3
    st = field_change_stats(phi_in, phi_out)
    ji = jacobian_det2D(np.stack([phi_in[-2, 0], phi_in[-1, 0]]))
    jo = jacobian_det2D(np.stack([phi_out[-2, 0], phi_out[-1, 0]]))
    expected = np.abs(jo - ji)
    assert st['jdet_change_max'] == pytest.approx(expected.max())
    assert st['jdet_change_med'] == pytest.approx(np.median(expected))
