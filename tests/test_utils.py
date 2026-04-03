"""Tests for dvfopt.utils — checkerboard, correspondences, transform."""

import numpy as np
import pytest

from dvfopt.utils.checkerboard import create_checkerboard
from dvfopt.utils.correspondences import (
    remove_duplicates,
    orientation,
    on_segment,
    do_lines_intersect,
    swap_correspondences,
    detect_intersecting_segments,
)
from dvfopt.utils.transform import affineTransformPointCloud


# ── Checkerboard ─────────────────────────────────────────────────────────

class TestCheckerboard:
    def test_output_shape(self):
        board = create_checkerboard(num_squares=(4, 4), resolution=(100, 100))
        assert board.shape == (100, 100)

    def test_binary_values(self):
        board = create_checkerboard()
        assert set(np.unique(board)) == {0.0, 1.0}

    def test_alternating_pattern(self):
        board = create_checkerboard(num_squares=(2, 2), resolution=(100, 100))
        # Top-left and bottom-right quadrants should be same
        assert board[0, 0] == board[50, 50]
        # Top-left and top-right should differ
        assert board[0, 0] != board[0, 50]

    def test_custom_resolution(self):
        """Non-divisible resolution is truncated to nearest multiple of num_squares."""
        board = create_checkerboard(num_squares=(8, 8), resolution=(200, 300))
        # 300 // 8 = 37, 37 * 8 = 296; height: 200 // 8 = 25, 25 * 8 = 200
        assert board.shape == (200, 296)

    def test_small_board(self):
        board = create_checkerboard(num_squares=(2, 2), resolution=(4, 4))
        assert board.shape == (4, 4)


# ── Correspondences ──────────────────────────────────────────────────────

class TestRemoveDuplicates:
    def test_no_duplicates(self):
        l1 = np.array([[1, 2], [3, 4], [5, 6]])
        l2 = np.array([[7, 8], [9, 10], [11, 12]])
        r1, r2 = remove_duplicates(l1, l2)
        np.testing.assert_array_equal(r1, l1)
        np.testing.assert_array_equal(r2, l2)

    def test_with_duplicates(self):
        l1 = np.array([[1, 2], [3, 4], [1, 2]])
        l2 = np.array([[7, 8], [9, 10], [11, 12]])
        r1, r2 = remove_duplicates(l1, l2)
        assert len(r1) == 2
        assert len(r2) == 2

    def test_all_same(self):
        l1 = np.array([[1, 2], [1, 2], [1, 2]])
        l2 = np.array([[7, 8], [9, 10], [11, 12]])
        r1, r2 = remove_duplicates(l1, l2)
        assert len(r1) == 1

    def test_empty(self):
        l1 = np.array([]).reshape(0, 2)
        l2 = np.array([]).reshape(0, 2)
        r1, r2 = remove_duplicates(l1, l2)
        assert len(r1) == 0


class TestOrientation:
    def test_counterclockwise(self):
        assert orientation([0, 0], [1, 0], [0, 1]) == 2

    def test_clockwise(self):
        assert orientation([0, 0], [0, 1], [1, 0]) == 1

    def test_collinear(self):
        assert orientation([0, 0], [1, 1], [2, 2]) == 0


class TestOnSegment:
    def test_on_segment(self):
        assert on_segment([0, 0], [1, 1], [2, 2]) is True

    def test_not_on_segment(self):
        assert on_segment([0, 0], [3, 3], [2, 2]) is False

    def test_endpoint(self):
        assert on_segment([0, 0], [0, 0], [2, 2]) is True


class TestDoLinesIntersect:
    def test_crossing(self):
        assert do_lines_intersect([0, 0], [2, 2], [0, 2], [2, 0]) is True

    def test_parallel(self):
        assert do_lines_intersect([0, 0], [2, 0], [0, 1], [2, 1]) is False

    def test_shared_endpoint(self):
        # Shared endpoint counts as intersection in this implementation
        assert do_lines_intersect([0, 0], [1, 1], [1, 1], [2, 0]) is True

    def test_non_intersecting(self):
        assert do_lines_intersect([0, 0], [1, 0], [0, 1], [1, 1]) is False

    def test_collinear_overlap(self):
        assert do_lines_intersect([0, 0], [2, 0], [1, 0], [3, 0]) is True


class TestSwapCorrespondences:
    def test_no_intersections(self):
        mpts = np.array([[0, 0], [0, 5]])
        fpts = np.array([[0, 1], [0, 6]])
        swapped, ipts = swap_correspondences(mpts, fpts)
        assert len(ipts) == 0
        np.testing.assert_array_equal(swapped, fpts)

    def test_crossing_swapped(self):
        mpts = np.array([[0, 0], [0, 4]])
        fpts = np.array([[0, 4], [0, 0]])
        swapped, ipts = swap_correspondences(mpts, fpts)
        assert len(ipts) > 0


class TestDetectIntersectingSegments:
    def test_no_intersections(self):
        mpts = np.array([[0, 0], [0, 5]])
        fpts = np.array([[0, 1], [0, 6]])
        indices, segs, swapped = detect_intersecting_segments(mpts, fpts)
        assert len(indices) == 0

    def test_with_intersections(self):
        mpts = np.array([[0, 0], [0, 4]])
        fpts = np.array([[0, 4], [0, 0]])
        indices, segs, swapped = detect_intersecting_segments(mpts, fpts)
        assert len(indices) > 0
        assert len(segs) == len(indices)
        assert len(swapped) == len(indices)


# ── Transform ────────────────────────────────────────────────────────────

class TestAffineTransformPointCloud:
    def test_identity(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        A = np.eye(3, 4)
        result = affineTransformPointCloud(pts, A)
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_translation(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        A = np.array([[1, 0, 0, 1],
                       [0, 1, 0, 2],
                       [0, 0, 1, 3]], dtype=float)
        result = affineTransformPointCloud(pts, A)
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]])

    def test_scaling(self):
        pts = np.array([[1.0, 1.0, 1.0]])
        A = np.array([[2, 0, 0, 0],
                       [0, 3, 0, 0],
                       [0, 0, 4, 0]], dtype=float)
        result = affineTransformPointCloud(pts, A)
        np.testing.assert_allclose(result, [[2.0, 3.0, 4.0]])

    def test_preserves_shape(self):
        pts = np.random.randn(10, 3)
        A = np.eye(3, 4)
        result = affineTransformPointCloud(pts, A)
        assert result.shape == (10, 3)
