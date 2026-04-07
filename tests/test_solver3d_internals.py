"""Tests for dvfopt.core.solver3d — 3D solver internal helper functions."""

import numpy as np
import pytest

from dvfopt.core.solver3d import (
    _init_phi_3d,
    _apply_result_3d,
    _patch_jacobian_3d,
    _update_metrics_3d,
)
from dvfopt.jacobian.numpy_jdet import jacobian_det3D


class TestInitPhi3D:
    def test_shape(self):
        d = np.zeros((3, 4, 6, 8), dtype=np.float64)
        phi, phi_init, D, H, W = _init_phi_3d(d)
        assert phi.shape == (3, 4, 6, 8)
        assert phi_init.shape == (3, 4, 6, 8)
        assert (D, H, W) == (4, 6, 8)

    def test_is_float64(self):
        d = np.zeros((3, 4, 5, 6), dtype=np.float32)
        phi, _, _, _, _ = _init_phi_3d(d)
        assert phi.dtype == np.float64

    def test_independent_copy(self):
        d = np.ones((3, 4, 5, 6), dtype=np.float64)
        phi, phi_init, _, _, _ = _init_phi_3d(d)
        phi[0, 0, 0, 0] = 999.0
        assert phi_init[0, 0, 0, 0] != 999.0

    def test_preserves_values(self):
        rng = np.random.default_rng(42)
        d = rng.standard_normal((3, 4, 5, 6))
        phi, _, _, _, _ = _init_phi_3d(d)
        np.testing.assert_allclose(phi, d)


class TestApplyResult3D:
    def test_writes_correct_region(self):
        phi = np.zeros((3, 8, 8, 8))
        sz, sy, sx = 3, 3, 3
        voxels = sz * sy * sx
        # Packing: [dx_flat, dy_flat, dz_flat]
        result_x = np.concatenate([
            np.full(voxels, 1.0),   # dx
            np.full(voxels, 2.0),   # dy
            np.full(voxels, 3.0),   # dz
        ])
        _apply_result_3d(phi, result_x, cz=4, cy=4, cx=4, sub_size=(3, 3, 3))
        np.testing.assert_array_equal(phi[2, 3:6, 3:6, 3:6], 1.0)  # dx
        np.testing.assert_array_equal(phi[1, 3:6, 3:6, 3:6], 2.0)  # dy
        np.testing.assert_array_equal(phi[0, 3:6, 3:6, 3:6], 3.0)  # dz

    def test_does_not_modify_outside(self):
        phi = np.zeros((3, 8, 8, 8))
        voxels = 27
        result_x = np.ones(3 * voxels)
        _apply_result_3d(phi, result_x, cz=4, cy=4, cx=4, sub_size=(3, 3, 3))
        phi[:, 3:6, 3:6, 3:6] = 0.0
        np.testing.assert_array_equal(phi, 0.0)


class TestPatchJacobian3D:
    def test_matches_full_recomputation(self):
        """Interior of patched sub-volume matches full recomputation.

        np.gradient uses different boundary stencils at the edges of a
        sub-array vs. the full array, so only the true interior (1 voxel
        inward from the patch boundary) is guaranteed to match.
        """
        rng = np.random.default_rng(42)
        phi = rng.standard_normal((3, 8, 8, 8)) * 0.2
        jac_full = jacobian_det3D(phi)

        jac_patched = np.ones((8, 8, 8))
        _patch_jacobian_3d(jac_patched, phi, center=(4, 4, 4), sub_size=(3, 3, 3))

        # Interior of the patched region (skip 1-voxel boundary of sub-array)
        np.testing.assert_allclose(
            jac_patched[3:6, 3:6, 3:6], jac_full[3:6, 3:6, 3:6], atol=1e-12)

    def test_corner_patch(self):
        """Corner patch: grid-edge boundary matches, far boundary does not."""
        rng = np.random.default_rng(99)
        phi = rng.standard_normal((3, 6, 6, 6)) * 0.2
        jac_full = jacobian_det3D(phi)

        jac_patched = np.ones((6, 6, 6))
        _patch_jacobian_3d(jac_patched, phi, center=(1, 1, 1), sub_size=(3, 3, 3))
        # Sub-array is [0:4]; interior safe region excludes far boundary (index 3)
        np.testing.assert_allclose(
            jac_patched[0:3, 0:3, 0:3], jac_full[0:3, 0:3, 0:3], atol=1e-12)

    def test_mutates_in_place(self):
        phi = np.zeros((3, 6, 6, 6))
        jac = np.zeros((6, 6, 6))
        result = _patch_jacobian_3d(jac, phi, center=(3, 3, 3), sub_size=(3, 3, 3))
        assert result is jac


class TestUpdateMetrics3D:
    def test_identity_field(self):
        phi = np.zeros((3, 6, 6, 6))
        phi_init = phi.copy()
        num_neg = []
        min_jdet = []
        error_list = []

        jac, neg, mn = _update_metrics_3d(
            phi, phi_init, num_neg, min_jdet, error_list)
        assert neg == 0
        np.testing.assert_allclose(mn, 1.0)
        np.testing.assert_allclose(error_list[0], 0.0)

    def test_with_patch(self):
        rng = np.random.default_rng(42)
        phi = rng.standard_normal((3, 8, 8, 8)) * 0.2
        phi_init = phi.copy()
        jac_full = jacobian_det3D(phi)

        num_neg = []
        min_jdet = []
        # Patch around center
        jac, _, _ = _update_metrics_3d(
            phi, phi_init, num_neg, min_jdet,
            jacobian_matrix=jac_full.copy(),
            patch_center=(4, 4, 4), patch_size=(3, 3, 3))

        # Interior of patched region matches (skip sub-array boundary)
        np.testing.assert_allclose(jac[3:6, 3:6, 3:6], jac_full[3:6, 3:6, 3:6], atol=1e-12)
