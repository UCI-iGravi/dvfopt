"""Tests for dvfopt.core.solver — internal helper functions."""

import numpy as np
import pytest

from dvfopt.core.solver import (
    _init_phi,
    _apply_result,
    _patch_jacobian_2d,
    _update_metrics,
    _setup_accumulators,
)
from dvfopt.core.constraints import _build_constraints
from dvfopt.jacobian.numpy_jdet import jacobian_det2D


class TestInitPhi:
    def test_shape(self):
        d = np.zeros((3, 1, 8, 12), dtype=np.float64)
        phi, phi_init, H, W = _init_phi(d)
        assert phi.shape == (2, 8, 12)
        assert phi_init.shape == (2, 8, 12)
        assert H == 8
        assert W == 12

    def test_channel_mapping(self):
        """phi[0] = dy (deformation[-2]), phi[1] = dx (deformation[-1])."""
        d = np.zeros((3, 1, 5, 5), dtype=np.float64)
        d[1, 0] = 1.0  # dy channel
        d[2, 0] = 2.0  # dx channel
        phi, phi_init, _, _ = _init_phi(d)
        np.testing.assert_array_equal(phi[0], 1.0)  # dy
        np.testing.assert_array_equal(phi[1], 2.0)  # dx

    def test_is_independent_copy(self):
        d = np.ones((3, 1, 5, 5), dtype=np.float64)
        phi, phi_init, _, _ = _init_phi(d)
        phi[0, 0, 0] = 999.0
        assert phi_init[0, 0, 0] != 999.0, "phi_init should be independent copy"

    def test_dz_channel_ignored(self):
        """The z-channel (d[0]) is not used in 2D phi."""
        d = np.zeros((3, 1, 5, 5), dtype=np.float64)
        d[0, 0] = 99.0  # dz — should be ignored
        phi, _, _, _ = _init_phi(d)
        np.testing.assert_array_equal(phi, 0.0)


class TestApplyResult:
    def test_writes_correct_region(self):
        phi = np.zeros((2, 10, 10))
        # result_x packing: [dx_flat, dy_flat]
        result_x = np.concatenate([np.ones(9), np.full(9, 2.0)])  # 3x3 window
        _apply_result(phi, result_x, cy=5, cx=5, sub_size=(3, 3))
        # dx (phi[1]) should be 1.0 in the 3x3 region
        np.testing.assert_array_equal(phi[1, 4:7, 4:7], 1.0)
        # dy (phi[0]) should be 2.0 in the 3x3 region
        np.testing.assert_array_equal(phi[0, 4:7, 4:7], 2.0)

    def test_does_not_modify_outside(self):
        phi = np.zeros((2, 10, 10))
        result_x = np.ones(2 * 3 * 3)
        _apply_result(phi, result_x, cy=5, cx=5, sub_size=(3, 3))
        # Outside the 3x3 region should still be zero
        phi[:, 4:7, 4:7] = 0.0
        np.testing.assert_array_equal(phi, 0.0)

    def test_rectangular_window(self):
        phi = np.zeros((2, 10, 10))
        sy, sx = 3, 5
        result_x = np.ones(2 * sy * sx) * 7.0
        _apply_result(phi, result_x, cy=5, cx=5, sub_size=(sy, sx))
        assert phi[1, 4:7, 3:8].sum() == 7.0 * sy * sx


class TestPatchJacobian2D:
    def test_matches_full_recomputation(self):
        """Patched Jacobian should match full recomputation."""
        rng = np.random.default_rng(42)
        phi = rng.standard_normal((2, 12, 12)) * 0.3
        jac_full = jacobian_det2D(phi)

        # Start with stale Jacobian (all ones), patch around center
        jac_patched = np.ones((1, 12, 12))
        _patch_jacobian_2d(jac_patched, phi, center=(6, 6), sub_size=(5, 5))

        # The patched region (and its border) should match
        np.testing.assert_allclose(
            jac_patched[0, 3:10, 3:10], jac_full[0, 3:10, 3:10], atol=1e-12)

    def test_patch_at_corner(self):
        """Patching near grid corner should not crash."""
        rng = np.random.default_rng(99)
        phi = rng.standard_normal((2, 8, 8)) * 0.2
        jac_full = jacobian_det2D(phi)

        jac_patched = np.ones((1, 8, 8))
        _patch_jacobian_2d(jac_patched, phi, center=(1, 1), sub_size=(3, 3))
        # Region around (1,1) should match full
        np.testing.assert_allclose(
            jac_patched[0, 0:4, 0:4], jac_full[0, 0:4, 0:4], atol=1e-12)

    def test_mutates_in_place(self):
        phi = np.zeros((2, 8, 8))
        jac = np.zeros((1, 8, 8))
        result = _patch_jacobian_2d(jac, phi, center=(4, 4), sub_size=(3, 3))
        assert result is jac


class TestUpdateMetrics:
    def test_appends_to_accumulators(self):
        phi = np.zeros((2, 6, 6))
        phi_init = np.zeros((2, 6, 6))
        num_neg = []
        min_jdet = []
        error_list = []

        _update_metrics(phi, phi_init, False, False,
                        num_neg, min_jdet, error_list)

        assert len(num_neg) == 1
        assert len(min_jdet) == 1
        assert len(error_list) == 1
        assert num_neg[0] == 0
        np.testing.assert_allclose(min_jdet[0], 1.0)
        np.testing.assert_allclose(error_list[0], 0.0)

    def test_counts_negatives(self):
        phi = np.zeros((2, 6, 6))
        phi[1, 3, 3] = 5.0  # spike creates negative Jdet
        num_neg = []
        min_jdet = []

        _update_metrics(phi, phi.copy(), False, False, num_neg, min_jdet)
        assert num_neg[0] > 0
        assert min_jdet[0] < 1.0

    def test_returns_jacobian_matrix(self):
        phi = np.zeros((2, 6, 6))
        phi_init = phi.copy()
        num_neg = []
        min_jdet = []

        jac, qm, neg, mn = _update_metrics(
            phi, phi_init, False, False, num_neg, min_jdet)
        assert jac.shape == (1, 6, 6)


class TestBuildConstraints:
    def test_identity_field_constraints_satisfied(self):
        """Identity field should satisfy all constraints."""
        sy, sx = 5, 5
        phi_flat = np.zeros(2 * sy * sx)
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=False,
            window_reached_max=False, threshold=0.01)
        # Should have Jdet constraint + boundary freeze constraint
        assert len(constraints) >= 2

    def test_boundary_freezing(self):
        """When not at edge, boundary pixels should be frozen via LinearConstraint."""
        sy, sx = 5, 5
        rng = np.random.default_rng(42)
        phi_flat = rng.standard_normal(2 * sy * sx) * 0.1
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=False,
            window_reached_max=False, threshold=0.01)

        # Find LinearConstraint (boundary freeze)
        from scipy.optimize import LinearConstraint
        linear_cs = [c for c in constraints if isinstance(c, LinearConstraint)]
        assert len(linear_cs) == 1

        lc = linear_cs[0]
        # Verify frozen values match original
        A_dense = lc.A.toarray()
        frozen_vals = A_dense @ phi_flat
        np.testing.assert_allclose(frozen_vals, lc.lb)
        np.testing.assert_allclose(frozen_vals, lc.ub)

    def test_at_edge_no_boundary_freeze(self):
        """When at grid edge, no boundary freeze constraint is added."""
        sy, sx = 5, 5
        phi_flat = np.zeros(2 * sy * sx)
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=True,
            window_reached_max=False, threshold=0.01)

        from scipy.optimize import LinearConstraint
        linear_cs = [c for c in constraints if isinstance(c, LinearConstraint)]
        assert len(linear_cs) == 0

    def test_shoelace_adds_constraint(self):
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        c_no = _build_constraints(phi_flat, (sy, sx), False, False, 0.01,
                                  enforce_shoelace=False)
        c_yes = _build_constraints(phi_flat, (sy, sx), False, False, 0.01,
                                   enforce_shoelace=True)
        assert len(c_yes) > len(c_no)

    def test_injectivity_adds_constraint(self):
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        c_no = _build_constraints(phi_flat, (sy, sx), False, False, 0.01,
                                  enforce_injectivity=False)
        c_yes = _build_constraints(phi_flat, (sy, sx), False, False, 0.01,
                                   enforce_injectivity=True)
        assert len(c_yes) > len(c_no)


class TestSetupAccumulators:
    def test_returns_five_structures(self):
        result = _setup_accumulators()
        assert len(result) == 5
        for i in range(4):
            assert isinstance(result[i], list)
