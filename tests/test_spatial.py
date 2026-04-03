"""Tests for dvfopt.core.spatial — sub-window positioning, pixel selection."""

import numpy as np
import pytest

from dvfopt.core.spatial import (
    nearest_center,
    get_nearest_center,
    argmin_quality,
    neg_jdet_bounding_window,
    _frozen_edges_clean,
    get_phi_sub_flat,
    _edge_flags,
    _select_non_overlapping,
)
from dvfopt._defaults import _unpack_size


class TestNearestCenter:
    def test_center_pixel_maps_to_self(self):
        shape = (1, 10, 10)
        nc = nearest_center(shape, 3)
        # (0, 5, 5) is well inside → should map to itself
        assert nc[(0, 5, 5)] == [0, 5, 5]

    def test_corner_clamps(self):
        shape = (1, 10, 10)
        nc = nearest_center(shape, 3)
        # (0, 0, 0) should clamp to (0, 1, 1) — the nearest valid center for 3x3
        assert nc[(0, 0, 0)] == [0, 1, 1]

    def test_bottom_right_clamps(self):
        shape = (1, 10, 10)
        nc = nearest_center(shape, 3)
        # (0, 9, 9) should clamp to (0, 8, 8)
        assert nc[(0, 9, 9)] == [0, 8, 8]

    def test_rectangular_window(self):
        shape = (1, 8, 12)
        nc = nearest_center(shape, (3, 5))
        # Top-left corner: hy=1, hx=2
        assert nc[(0, 0, 0)] == [0, 1, 2]


class TestGetNearestCenter:
    def test_cached_lookup(self):
        shape = (1, 10, 10)
        cache = {}
        result = get_nearest_center((5, 5), shape, 3, cache)
        assert result == [0, 5, 5]
        # Cache should now contain the key
        assert (3, 3) in cache

    def test_second_call_uses_cache(self):
        shape = (1, 10, 10)
        cache = {}
        get_nearest_center((3, 3), shape, 5, cache)
        result = get_nearest_center((7, 7), shape, 5, cache)
        assert result == [0, 7, 7]


class TestArgminQuality:
    def test_finds_minimum(self):
        jac = np.ones((1, 5, 5))
        jac[0, 2, 3] = -0.5
        y, x = argmin_quality(jac)
        assert (y, x) == (2, 3)

    def test_tie_returns_first_flat(self):
        jac = np.ones((1, 4, 4))
        jac[0, 0, 1] = -1.0
        jac[0, 3, 2] = -1.0
        y, x = argmin_quality(jac)
        # argmin returns first occurrence in flattened order
        assert (y, x) == (0, 1)


class TestNegJdetBoundingWindow:
    def test_single_negative_pixel(self):
        jac = np.ones((1, 10, 10))
        jac[0, 5, 5] = -0.1
        (h, w), (cy, cx) = neg_jdet_bounding_window(jac, (5, 5), 0.01, 1e-5)
        assert h >= 3 and w >= 3
        # Window should be centered near the negative pixel
        assert abs(cy - 5) <= 1
        assert abs(cx - 5) <= 1

    def test_larger_negative_region(self):
        jac = np.ones((1, 12, 12))
        jac[0, 4:7, 4:7] = -0.5  # 3x3 block
        (h, w), (cy, cx) = neg_jdet_bounding_window(jac, (5, 5), 0.01, 1e-5)
        # Should be larger than 3x3 to include the 1px positive border
        assert h >= 5
        assert w >= 5

    def test_non_negative_pixel_returns_3x3(self):
        """If the center pixel is somehow not negative, return default 3x3."""
        jac = np.ones((1, 10, 10))
        (h, w), center = neg_jdet_bounding_window(jac, (5, 5), 0.01, 1e-5)
        assert (h, w) == (3, 3)


class TestFrozenEdgesClean:
    def test_clean_edges(self):
        """All-positive Jdet → frozen edges are clean."""
        jac = np.ones((1, 10, 10))
        assert _frozen_edges_clean(jac, 5, 5, 3, 0.01, 1e-5) == True

    def test_dirty_edges(self):
        """Negative Jdet on the edge ring → not clean."""
        jac = np.ones((1, 10, 10))
        # Place negative value on the edge of a 3x3 window centered at (5,5)
        jac[0, 4, 4] = -0.5  # top-left corner of window
        assert _frozen_edges_clean(jac, 5, 5, 3, 0.01, 1e-5) == False


class TestGetPhiSubFlat:
    def test_identity_extraction(self):
        phi = np.zeros((2, 10, 10))
        flat = get_phi_sub_flat(phi, 0, 5, 5, (1, 10, 10), 3)
        # 3x3 window → 9 dx + 9 dy = 18
        assert flat.shape == (18,)
        np.testing.assert_array_equal(flat, 0.0)

    def test_values_match(self):
        """Extracted sub-window values should match phi slices."""
        rng = np.random.default_rng(42)
        phi = rng.standard_normal((2, 10, 10))
        flat = get_phi_sub_flat(phi, 0, 5, 5, (1, 10, 10), (3, 5))
        sy, sx = 3, 5
        expected_dx = phi[1, 4:7, 3:8].flatten()  # cy=5, hy=1, hx=2
        expected_dy = phi[0, 4:7, 3:8].flatten()
        np.testing.assert_array_equal(flat, np.concatenate([expected_dx, expected_dy]))

    def test_rectangular_window(self):
        phi = np.zeros((2, 10, 10))
        flat = get_phi_sub_flat(phi, 0, 5, 5, (1, 10, 10), (3, 5))
        assert flat.shape == (2 * 3 * 5,)


class TestEdgeFlags:
    def test_center_not_at_edge(self):
        is_at_edge, reached_max = _edge_flags(5, 5, 3, (1, 10, 10), 10)
        assert is_at_edge is False
        assert reached_max is False

    def test_top_edge(self):
        is_at_edge, _ = _edge_flags(1, 5, 3, (1, 10, 10), 10)
        assert is_at_edge is True

    def test_window_reached_max(self):
        _, reached_max = _edge_flags(5, 5, 10, (1, 10, 10), 10)
        assert reached_max is True


class TestSelectNonOverlapping:
    def test_single_pixel(self):
        neg_pixels = [(3, 3)]
        pixel_ws = {(3, 3): (3, 3)}
        shape = (1, 10, 10)
        cache = {}
        selected = _select_non_overlapping(neg_pixels, pixel_ws, shape, cache)
        assert len(selected) == 1

    def test_overlapping_rejected(self):
        """Two adjacent pixels with overlapping windows → only one selected."""
        neg_pixels = [(5, 5), (5, 6)]
        pixel_ws = {(5, 5): (5, 5), (5, 6): (5, 5)}
        shape = (1, 12, 12)
        cache = {}
        selected = _select_non_overlapping(neg_pixels, pixel_ws, shape, cache)
        assert len(selected) == 1

    def test_distant_both_selected(self):
        """Two distant pixels → both selected."""
        neg_pixels = [(2, 2), (8, 8)]
        pixel_ws = {(2, 2): (3, 3), (8, 8): (3, 3)}
        shape = (1, 12, 12)
        cache = {}
        selected = _select_non_overlapping(neg_pixels, pixel_ws, shape, cache)
        assert len(selected) == 2
