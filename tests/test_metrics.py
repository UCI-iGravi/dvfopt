"""Tests for dvfopt.metrics — the canonical fold-statistics helpers."""

import numpy as np

from dvfopt import correct_dvf
from dvfopt.metrics import constraint_fold_stats, fold_stats
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
    assert name == '2tri'
    assert st.n_neg > 0


def test_constraint_fold_stats_auto_3d():
    phi = planted_fold_3d()
    name, st = constraint_fold_stats(phi)
    assert name == '6tet'
    assert st.n_neg > 0


def test_constraint_fold_stats_matches_solver_init_stats():
    # The metrics module and Solver.fit must agree on what "folded" means.
    phi = planted_fold(10, 10, seed=0, scale=0.4)
    _, st = constraint_fold_stats(phi, constraint='2tri')
    res = correct_dvf(phi, constraint='2tri', objective='l1', strategy='auto')
    assert res.init_n_neg == st.n_neg
