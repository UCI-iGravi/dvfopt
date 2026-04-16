"""Tests for dvfopt.core.parallel — internal helper functions."""

import numpy as np
import pytest

from dvfopt.core.slsqp.parallel import _find_negative_pixels


class TestFindNegativePixels:
    def test_no_negatives(self):
        jac = np.ones((1, 10, 10))
        result = _find_negative_pixels(jac, 0.01, 1e-5)
        assert result == []

    def test_finds_all_negatives(self):
        jac = np.ones((1, 10, 10))
        jac[0, 3, 4] = -0.5
        jac[0, 7, 8] = -0.2
        result = _find_negative_pixels(jac, 0.01, 1e-5)
        assert len(result) == 2

    def test_sorted_worst_first(self):
        jac = np.ones((1, 10, 10))
        jac[0, 3, 4] = -0.2  # worse
        jac[0, 7, 8] = -0.1  # less bad
        result = _find_negative_pixels(jac, 0.01, 1e-5)
        assert result[0] == (3, 4)  # worst first
        assert result[1] == (7, 8)

    def test_threshold_boundary(self):
        """Pixels exactly at threshold - err_tol should be included."""
        jac = np.ones((1, 10, 10))
        jac[0, 5, 5] = 0.01 - 1e-5  # exactly at boundary
        result = _find_negative_pixels(jac, 0.01, 1e-5)
        assert len(result) == 1

    def test_above_threshold_excluded(self):
        jac = np.ones((1, 10, 10))
        jac[0, 5, 5] = 0.02  # above threshold
        result = _find_negative_pixels(jac, 0.01, 1e-5)
        assert len(result) == 0

    def test_returns_tuples(self):
        jac = np.ones((1, 5, 5))
        jac[0, 2, 3] = -1.0
        result = _find_negative_pixels(jac, 0.01, 1e-5)
        assert isinstance(result[0], tuple)
        assert result[0] == (2, 3)
