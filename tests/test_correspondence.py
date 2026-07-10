"""Tests for laplacian.correspondence — normal estimation and matching.

Guards two review findings:
- ``estimate2Dnormals`` mutated the caller's points array in place and left
  fake ``(0, 0)`` entries for points without a valid normal; it must now
  operate on a copy and filter invalid points out of the returned arrays.
- ``get2DCorrespondences_batch`` used a strict ``<`` against the per-row
  90th-percentile distance, which rejects ties: with a single moving point
  (k=1) the percentile equals the only distance, so no match could ever be
  produced.
"""

import numpy as np
import pytest

pytest.importorskip("skimage")
from laplacian.correspondence import (
    estimate2Dnormals,
    get2DCorrespondences_batch,
)


def _circle_points(n=80, r=10.0, center=(20.0, 20.0)):
    # n=80 keeps adjacent spacing ~0.79 px so every point has >= 4
    # neighbours within the default KD-tree radius of 3.
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)], axis=1)


class TestEstimate2DNormals:
    def test_does_not_mutate_caller_array(self):
        pts = _circle_points()
        # Add an isolated point (fewer than 4 neighbours within radius 3):
        # previously its row was overwritten with (0, 0) in the caller's array.
        pts = np.vstack([pts, [500.0, 500.0]])
        pts_before = pts.copy()
        estimate2Dnormals(pts)
        np.testing.assert_array_equal(pts, pts_before)

    def test_invalid_points_filtered_not_zeroed(self):
        pts = _circle_points()
        pts = np.vstack([pts, [500.0, 500.0]])
        out_pts, out_normals = estimate2Dnormals(pts)
        assert len(out_pts) == len(out_normals)
        # The isolated point must be dropped, not replaced by a fake origin.
        assert not np.any(np.all(out_pts == 0, axis=1))
        assert [500.0, 500.0] not in out_pts.tolist()
        # Dense circle points all have >= 4 neighbours, so they survive.
        assert len(out_pts) == len(pts) - 1

    def test_normals_are_unit_length(self):
        pts = _circle_points()
        out_pts, out_normals = estimate2Dnormals(pts)
        norms = np.linalg.norm(out_normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_points_and_normals_stay_paired(self):
        """Each returned point must be one of the inputs (no synthetic points)."""
        pts = _circle_points()
        out_pts, _ = estimate2Dnormals(pts)
        in_set = {tuple(np.round(p, 6)) for p in pts}
        for p in out_pts:
            assert tuple(np.round(p, 6)) in in_set


class TestGet2DCorrespondencesBatch:
    def test_single_moving_point_matches(self):
        """Regression: k=1 => p90 equals the only distance; strict < rejected
        every candidate. A 1px-offset pair with identical normals must match."""
        fpoints = np.array([[5.0, 5.0]])
        fnormals = np.array([[0.0, 1.0]])
        mpoints = np.array([[5.0, 6.0]])
        mnormals = np.array([[0.0, 1.0]])
        corr = get2DCorrespondences_batch(fpoints, fnormals, mpoints, mnormals)
        assert corr[0] == 0

    def test_equidistant_perfect_candidates_match(self):
        """Four equidistant perfect candidates: all distances tie at the
        percentile boundary; a match must still be produced."""
        fpoints = np.array([[5.0, 5.0]])
        fnormals = np.array([[0.0, 1.0]])
        mpoints = np.array([[5.0, 7.0], [5.0, 3.0], [7.0, 5.0], [3.0, 5.0]])
        mnormals = np.tile([0.0, 1.0], (4, 1))
        corr = get2DCorrespondences_batch(fpoints, fnormals, mpoints, mnormals)
        assert corr[0] != -1
        assert corr[0] in (0, 1, 2, 3)

    def test_dense_cloud_prefers_nearest_valid(self):
        """Inclusive <= only admits boundary ties — in a dense cloud the
        nearest normal-compatible candidate must still win."""
        rng = np.random.default_rng(0)
        mpoints = rng.uniform(0, 50, size=(100, 2))
        mnormals = np.tile([0.0, 1.0], (100, 1))
        # Plant an exact-match target right next to the query point.
        mpoints = np.vstack([mpoints, [25.1, 25.0]])
        mnormals = np.vstack([mnormals, [0.0, 1.0]])
        fpoints = np.array([[25.0, 25.0]])
        fnormals = np.array([[0.0, 1.0]])
        corr = get2DCorrespondences_batch(fpoints, fnormals, mpoints, mnormals)
        assert corr[0] == 100, "nearest compatible candidate should be chosen"

    def test_normal_mismatch_still_rejected(self):
        """<= must not admit candidates whose normals disagree."""
        fpoints = np.array([[5.0, 5.0]])
        fnormals = np.array([[0.0, 1.0]])
        mpoints = np.array([[5.0, 6.0]])
        mnormals = np.array([[1.0, 0.0]])  # 90 degrees off
        corr = get2DCorrespondences_batch(fpoints, fnormals, mpoints, mnormals)
        assert corr[0] == -1

    def test_empty_inputs(self):
        corr = get2DCorrespondences_batch(
            np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
        )
        assert len(corr) == 0
