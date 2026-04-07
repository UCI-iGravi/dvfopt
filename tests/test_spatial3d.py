"""Tests for dvfopt.core.spatial3d — 3D sub-volume positioning and helpers."""

import numpy as np
import pytest

from dvfopt.core.spatial3d import (
    get_nearest_center_3d,
    argmin_worst_voxel,
    neg_jdet_bounding_window_3d,
    _frozen_boundary_mask_3d,
    _frozen_edges_clean_3d,
    get_phi_sub_flat_3d,
    _edge_flags_3d,
)


class TestGetNearestCenter3D:
    def test_center_maps_to_self(self):
        assert get_nearest_center_3d((3, 4, 5), (8, 10, 12), 3) == (3, 4, 5)

    def test_corner_clamps(self):
        cz, cy, cx = get_nearest_center_3d((0, 0, 0), (8, 10, 12), 3)
        assert cz >= 1 and cy >= 1 and cx >= 1

    def test_far_corner_clamps(self):
        cz, cy, cx = get_nearest_center_3d((7, 9, 11), (8, 10, 12), 3)
        assert cz <= 6 and cy <= 8 and cx <= 10

    def test_rectangular_window(self):
        cz, cy, cx = get_nearest_center_3d((0, 0, 0), (6, 8, 10), (3, 5, 7))
        assert cz == 1 and cy == 2 and cx == 3


class TestArgminWorstVoxel:
    def test_finds_minimum(self):
        jac = np.ones((4, 5, 6))
        jac[2, 3, 4] = -0.5
        assert argmin_worst_voxel(jac) == (2, 3, 4)

    def test_tie_returns_first(self):
        jac = np.ones((4, 5, 6))
        jac[0, 0, 1] = -1.0
        jac[3, 4, 5] = -1.0
        assert argmin_worst_voxel(jac) == (0, 0, 1)


class TestNegJdetBoundingWindow3D:
    def test_single_negative_voxel(self):
        jac = np.ones((8, 8, 8))
        jac[4, 4, 4] = -0.1
        (sz, sy, sx), (cz, cy, cx) = neg_jdet_bounding_window_3d(
            jac, (4, 4, 4), 0.01, 1e-5)
        assert sz >= 3 and sy >= 3 and sx >= 3

    def test_not_negative_returns_default(self):
        jac = np.ones((8, 8, 8))
        (sz, sy, sx), center = neg_jdet_bounding_window_3d(
            jac, (4, 4, 4), 0.01, 1e-5)
        assert (sz, sy, sx) == (3, 3, 3)

    def test_larger_region(self):
        jac = np.ones((10, 10, 10))
        jac[3:6, 3:6, 3:6] = -0.5
        (sz, sy, sx), _ = neg_jdet_bounding_window_3d(
            jac, (4, 4, 4), 0.01, 1e-5)
        assert sz >= 5 and sy >= 5 and sx >= 5


class TestFrozenBoundaryMask3D:
    def test_interior_window(self):
        """Interior window: all 6 faces should be frozen."""
        mask = _frozen_boundary_mask_3d(3, 4, 5, (3, 3, 3), (8, 10, 12))
        # All boundary voxels should be True
        assert mask[0, :, :].all()
        assert mask[-1, :, :].all()
        assert mask[:, 0, :].all()
        assert mask[:, -1, :].all()
        assert mask[:, :, 0].all()
        assert mask[:, :, -1].all()
        # Interior should be False
        assert not mask[1, 1, 1]

    def test_corner_window(self):
        """Window at grid corner: faces touching grid edge should NOT be frozen."""
        mask = _frozen_boundary_mask_3d(1, 1, 1, (3, 3, 3), (8, 10, 12))
        # z=0 face touches grid edge → not frozen
        assert not mask[0, :, :].all()
        # But z=-1 face doesn't touch grid edge → frozen
        assert mask[-1, :, :].all()

    def test_grid_edge_touching(self):
        """Window at z=0: the first z-slice should not be frozen."""
        mask = _frozen_boundary_mask_3d(1, 4, 5, (3, 3, 3), (8, 10, 12))
        # cz=1, hz=1 → start_z=0 → on grid edge
        assert not mask[0, 1, 1]  # z=0 interior pixel not frozen


class TestFrozenEdgesClean3D:
    def test_clean(self):
        jac = np.ones((8, 8, 8))
        mask = _frozen_boundary_mask_3d(4, 4, 4, (3, 3, 3), (8, 8, 8))
        assert _frozen_edges_clean_3d(jac, 4, 4, 4, (3, 3, 3), 0.01, 1e-5, mask)

    def test_dirty(self):
        jac = np.ones((8, 8, 8))
        jac[3, 3, 3] = -0.5  # on boundary of 3x3x3 window centered at (4,4,4)
        mask = _frozen_boundary_mask_3d(4, 4, 4, (3, 3, 3), (8, 8, 8))
        assert not _frozen_edges_clean_3d(jac, 4, 4, 4, (3, 3, 3), 0.01, 1e-5, mask)

    def test_empty_mask(self):
        jac = np.ones((8, 8, 8))
        mask = np.zeros((3, 3, 3), dtype=bool)
        assert _frozen_edges_clean_3d(jac, 4, 4, 4, (3, 3, 3), 0.01, 1e-5, mask)


class TestGetPhiSubFlat3D:
    def test_identity(self):
        phi = np.zeros((3, 8, 8, 8))
        flat = get_phi_sub_flat_3d(phi, 4, 4, 4, (3, 3, 3))
        assert flat.shape == (3 * 27,)
        np.testing.assert_array_equal(flat, 0.0)

    def test_packing_order(self):
        """Packing should be [dx_flat, dy_flat, dz_flat]."""
        phi = np.zeros((3, 8, 8, 8))
        phi[0] = 1.0  # dz
        phi[1] = 2.0  # dy
        phi[2] = 3.0  # dx
        flat = get_phi_sub_flat_3d(phi, 4, 4, 4, (3, 3, 3))
        voxels = 27
        np.testing.assert_array_equal(flat[:voxels], 3.0)      # dx
        np.testing.assert_array_equal(flat[voxels:2*voxels], 2.0)  # dy
        np.testing.assert_array_equal(flat[2*voxels:], 1.0)    # dz

    def test_values_match(self):
        rng = np.random.default_rng(42)
        phi = rng.standard_normal((3, 8, 8, 8))
        flat = get_phi_sub_flat_3d(phi, 4, 4, 4, (3, 3, 3))
        expected_dx = phi[2, 3:6, 3:6, 3:6].flatten()
        expected_dy = phi[1, 3:6, 3:6, 3:6].flatten()
        expected_dz = phi[0, 3:6, 3:6, 3:6].flatten()
        np.testing.assert_array_equal(
            flat, np.concatenate([expected_dx, expected_dy, expected_dz]))


class TestEdgeFlags3D:
    def test_interior(self):
        is_edge, reached_max = _edge_flags_3d(4, 4, 4, (3, 3, 3), (10, 10, 10), (10, 10, 10))
        assert not is_edge
        assert not reached_max

    def test_at_edge(self):
        is_edge, _ = _edge_flags_3d(1, 4, 4, (3, 3, 3), (10, 10, 10), (10, 10, 10))
        assert is_edge

    def test_reached_max(self):
        _, reached_max = _edge_flags_3d(4, 4, 4, (10, 10, 10), (10, 10, 10), (10, 10, 10))
        assert reached_max
