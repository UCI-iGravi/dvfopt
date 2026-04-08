"""Tests for dvfopt.utils — checkerboard."""

import numpy as np
import pytest

from dvfopt.utils.checkerboard import create_checkerboard


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
