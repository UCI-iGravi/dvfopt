"""Tests for dvfopt.core.constraints3d — 3D constraint functions."""

import numpy as np
import pytest

from dvfopt.core.slsqp.constraints3d import jacobian_constraint_3d, _build_constraints_3d


class TestJacobianConstraint3D:
    def _make_identity_flat(self, sz, sy, sx):
        return np.zeros(3 * sz * sy * sx)

    def test_identity_all_ones(self):
        sz, sy, sx = 4, 4, 4
        phi_flat = self._make_identity_flat(sz, sy, sx)
        vals = jacobian_constraint_3d(phi_flat, (sz, sy, sx))
        assert vals.shape == (sz * sy * sx,)
        np.testing.assert_allclose(vals, 1.0)

    def test_with_freeze_mask(self):
        sz, sy, sx = 4, 4, 4
        phi_flat = self._make_identity_flat(sz, sy, sx)
        mask = np.zeros((sz, sy, sx), dtype=bool)
        mask[0, :, :] = True  # freeze first z-slice
        vals = jacobian_constraint_3d(phi_flat, (sz, sy, sx), freeze_mask=mask)
        n_frozen = int(mask.sum())
        assert vals.shape == (sz * sy * sx - n_frozen,)
        np.testing.assert_allclose(vals, 1.0)

    def test_no_mask_same_as_none(self):
        sz, sy, sx = 3, 3, 3
        phi_flat = self._make_identity_flat(sz, sy, sx)
        v1 = jacobian_constraint_3d(phi_flat, (sz, sy, sx), freeze_mask=None)
        v2 = jacobian_constraint_3d(phi_flat, (sz, sy, sx))
        np.testing.assert_array_equal(v1, v2)


class TestBuildConstraints3D:
    def test_no_freeze_only_jdet(self):
        sz, sy, sx = 4, 4, 4
        phi_flat = np.zeros(3 * sz * sy * sx)
        mask = np.zeros((sz, sy, sx), dtype=bool)
        constraints = _build_constraints_3d(phi_flat, (sz, sy, sx), mask, 0.01)
        # Only Jdet constraint, no LinearConstraint
        assert len(constraints) == 1

    def test_with_freeze_adds_linear(self):
        sz, sy, sx = 4, 4, 4
        phi_flat = np.zeros(3 * sz * sy * sx)
        mask = np.zeros((sz, sy, sx), dtype=bool)
        mask[0, :, :] = True
        constraints = _build_constraints_3d(phi_flat, (sz, sy, sx), mask, 0.01)
        assert len(constraints) == 2  # Jdet + LinearConstraint

    def test_frozen_values_match(self):
        """Frozen pixel values in LinearConstraint should match phi_flat."""
        sz, sy, sx = 4, 4, 4
        rng = np.random.default_rng(42)
        phi_flat = rng.standard_normal(3 * sz * sy * sx) * 0.1
        mask = np.zeros((sz, sy, sx), dtype=bool)
        mask[0, :, :] = True
        constraints = _build_constraints_3d(phi_flat, (sz, sy, sx), mask, 0.01)

        from scipy.optimize import LinearConstraint
        lc = [c for c in constraints if isinstance(c, LinearConstraint)][0]
        frozen_vals = lc.A.toarray() @ phi_flat
        np.testing.assert_allclose(frozen_vals, lc.lb)
