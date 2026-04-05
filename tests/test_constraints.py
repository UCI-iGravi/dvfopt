"""Tests for dvfopt.core.constraints — constraint functions and quality map."""

import numpy as np
import pytest

from dvfopt.core.constraints import jacobian_constraint, _quality_map
from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.jacobian.shoelace import shoelace_constraint
from dvfopt.jacobian.monotonicity import injectivity_constraint


class TestJacobianConstraint:
    def _make_identity_flat(self, sy, sx):
        """Create a flat phi (identity) for a sy x sx window: [dx_flat, dy_flat]."""
        return np.zeros(2 * sy * sx)

    def test_identity_exclude_boundaries(self):
        """Identity → Jdet = 1.0 for all interior pixels."""
        sy, sx = 5, 5
        phi_flat = self._make_identity_flat(sy, sx)
        vals = jacobian_constraint(phi_flat, (sy, sx), exclude_boundaries=True)
        assert vals.shape == ((sy - 2) * (sx - 2),)
        np.testing.assert_allclose(vals, 1.0)

    def test_identity_include_boundaries(self):
        """Identity → Jdet = 1.0 for all pixels."""
        sy, sx = 5, 5
        phi_flat = self._make_identity_flat(sy, sx)
        vals = jacobian_constraint(phi_flat, (sy, sx), exclude_boundaries=False)
        assert vals.shape == (sy * sx,)
        np.testing.assert_allclose(vals, 1.0)

    def test_output_size_rectangular(self):
        sy, sx = 4, 6
        phi_flat = self._make_identity_flat(sy, sx)
        vals = jacobian_constraint(phi_flat, (sy, sx), exclude_boundaries=True)
        assert vals.shape == ((sy - 2) * (sx - 2),)


class TestShoelaceConstraint:
    def test_identity_exclude_boundaries(self):
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        vals = shoelace_constraint(phi_flat, (sy, sx), exclude_boundaries=True)
        # Shoelace areas are (sy-1, sx-1), trimmed to (sy-3, sx-3) interior
        assert vals.shape == ((sy - 3) * (sx - 3),)
        np.testing.assert_allclose(vals, 1.0)

    def test_identity_include_boundaries(self):
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        vals = shoelace_constraint(phi_flat, (sy, sx), exclude_boundaries=False)
        assert vals.shape == ((sy - 1) * (sx - 1),)
        np.testing.assert_allclose(vals, 1.0)


class TestInjectivityConstraint:
    def test_identity_exclude_boundaries(self):
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        vals = injectivity_constraint(phi_flat, (sy, sx), exclude_boundaries=True)
        # h_mono interior: (sy-2)*(sx-3), v_mono interior: (sy-3)*(sx-2)
        # d1 and d2 diagonal: (sy-1)*(sx-1) cells each, minus 2 all-frozen corners
        n_diag = (sy - 1) * (sx - 1) - 2
        expected_len = (sy - 2) * (sx - 3) + (sy - 3) * (sx - 2) + 2 * n_diag
        assert vals.shape == (expected_len,)
        np.testing.assert_allclose(vals, 1.0)

    def test_identity_include_boundaries(self):
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        vals = injectivity_constraint(phi_flat, (sy, sx), exclude_boundaries=False)
        # h_mono: sy*(sx-1), v_mono: (sy-1)*sx, d1: (sy-1)*(sx-1), d2: (sy-1)*(sx-1)
        expected_len = sy * (sx - 1) + (sy - 1) * sx + 2 * (sy - 1) * (sx - 1)
        assert vals.shape == (expected_len,)
        np.testing.assert_allclose(vals, 1.0)


class TestQualityMap:
    def test_jdet_only(self, identity_phi_2d):
        """With no extra constraints, quality_map == jacobian_det2D."""
        qm = _quality_map(identity_phi_2d, enforce_shoelace=False, enforce_injectivity=False)
        jdet = jacobian_det2D(identity_phi_2d)
        np.testing.assert_array_equal(qm, jdet)

    def test_shoelace_lowers_quality_at_folds(self):
        """When shoelace is enforced, folded quads reduce the quality map below Jdet."""
        H, W = 8, 8
        phi = np.zeros((2, H, W))
        phi[1, :, 0] = 3.0   # dx: push left column right
        phi[1, :, -1] = -3.0  # dx: push right column left

        qm_no_shoe = _quality_map(phi, enforce_shoelace=False)
        qm_shoe = _quality_map(phi, enforce_shoelace=True)

        # With shoelace, worst quality should be <= without
        assert qm_shoe.min() <= qm_no_shoe.min()

    def test_injectivity_lowers_quality_at_crossings(self):
        H, W = 8, 8
        phi = np.zeros((2, H, W))
        phi[1, :, 1] = 3.0
        phi[1, :, 2] = -3.0

        qm_no_inj = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=False)
        qm_inj = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=True)

        assert qm_inj.min() <= qm_no_inj.min()

    def test_output_shape(self, identity_phi_2d):
        qm = _quality_map(identity_phi_2d, enforce_shoelace=True, enforce_injectivity=True)
        assert qm.shape == (1, 10, 10)

    def test_precomputed_jacobian(self, identity_phi_2d):
        """Passing precomputed jacobian_matrix avoids recomputation."""
        jdet = jacobian_det2D(identity_phi_2d)
        qm = _quality_map(identity_phi_2d, enforce_shoelace=False, jacobian_matrix=jdet)
        np.testing.assert_array_equal(qm, jdet)
