"""Tests for the cluster_slp_iter continuous (as-completed) scheduler.

The continuous scheduler is an opt-in alternative to the default barrier
sub-round scheduler. It must (a) admit only non-conflicting clusters
(correctness of the splice-race avoidance) and (b) reach the same
feasibility as the subround scheduler with essentially identical L1.
"""
import numpy as np

from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
    _boxes_conflict,
    cluster_slp_iter,
)
from dvfopt.core.tri_primitives import tri_areas_flat


def _n_neg(phi):
    H, W = phi.shape[1:]
    a = tri_areas_flat(np.concatenate([phi[0].ravel(), phi[1].ravel()]), H, W)
    return int((a <= 0).sum())


def test_boxes_conflict_predicate():
    a = {'y0': 0, 'y1': 4, 'x0': 0, 'x1': 4}
    overlap = {'y0': 3, 'y1': 6, 'x0': 3, 'x1': 6}    # bboxes overlap
    touching = {'y0': 0, 'y1': 4, 'x0': 4, 'x1': 8}   # share index x=4
    separated = {'y0': 0, 'y1': 4, 'x0': 5, 'x1': 8}  # a.x1=4 < b.x0=5: a gap
    assert _boxes_conflict(a, overlap) is True
    assert _boxes_conflict(a, touching) is True        # shared corner -> conflict
    assert _boxes_conflict(a, separated) is False      # disjoint frozen rings
    assert _boxes_conflict(a, a) is True


def _two_fold_field():
    """24x24 field with two separated fold spots -> two clusters."""
    phi = np.zeros((2, 24, 24), dtype=np.float64)
    for (r, c) in [(5, 5), (17, 17)]:
        phi[1, r, c] = +1.2
        phi[1, r, c + 1] = -1.2
    return phi


def test_continuous_matches_subround_feasibility_and_l1():
    phi = _two_fold_field()
    if _n_neg(phi) == 0:
        import pytest
        pytest.skip('no fold planted')
    out_sub, _ = cluster_slp_iter(phi, threshold=0.01, n_workers=2,
                                  scheduler='subround')
    out_con, _ = cluster_slp_iter(phi, threshold=0.01, n_workers=2,
                                  scheduler='continuous')
    # Both reach strict feasibility.
    assert _n_neg(out_sub) == 0
    assert _n_neg(out_con) == 0
    # The frozen-ring decomposition is deterministic in the math, so L1
    # should match closely (scheduling only changes order, not the result).
    l1_sub = float(np.abs(out_sub - phi).sum())
    l1_con = float(np.abs(out_con - phi).sum())
    assert abs(l1_con - l1_sub) <= 0.02 * max(l1_sub, 1e-9)
