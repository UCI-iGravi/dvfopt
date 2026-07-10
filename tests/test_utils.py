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
        """Non-divisible resolutions must still return exactly the requested shape."""
        board = create_checkerboard(num_squares=(8, 8), resolution=(200, 300))
        assert board.shape == (200, 300)

    def test_small_board(self):
        board = create_checkerboard(num_squares=(2, 2), resolution=(4, 4))
        assert board.shape == (4, 4)

    @pytest.mark.parametrize(
        "num_squares,resolution",
        [
            ((7, 7), (400, 400)),
            ((8, 8), (100, 90)),
        ],
    )
    def test_exact_requested_shape(self, num_squares, resolution):
        """Regression: floor-sized squares silently truncated the board."""
        board = create_checkerboard(num_squares=num_squares, resolution=resolution)
        assert board.shape == resolution
        assert set(np.unique(board)) <= {0.0, 1.0}

    def test_cropped_board_still_alternates(self):
        board = create_checkerboard(num_squares=(7, 7), resolution=(400, 400))
        # First square is ceil(400/7)=58 px wide; adjacent squares differ.
        assert board[0, 0] != board[0, 58]
        assert board[0, 0] != board[58, 0]
