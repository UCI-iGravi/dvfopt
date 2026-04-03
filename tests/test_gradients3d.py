"""Tests for dvfopt.core.gradients3d — 3D analytical gradients vs finite differences."""

import numpy as np
import pytest

from dvfopt.core.constraints3d import jacobian_constraint_3d
from dvfopt.core.gradients3d import jdet_constraint_jacobian_3d, _gradient_stencil


def _numerical_jacobian(func, x0, eps=1e-6):
    f0 = func(x0)
    n_out = len(f0)
    n_in = len(x0)
    J = np.zeros((n_out, n_in))
    for j in range(n_in):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        J[:, j] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return J


class TestGradientStencil:
    def test_interior(self):
        idx, coeff = _gradient_stencil(2, 5)
        assert idx == [1, 3]
        assert coeff == [-0.5, 0.5]

    def test_left_boundary(self):
        idx, coeff = _gradient_stencil(0, 5)
        assert idx == [0, 1]
        assert coeff == [-1.0, 1.0]

    def test_right_boundary(self):
        idx, coeff = _gradient_stencil(4, 5)
        assert idx == [3, 4]
        assert coeff == [-1.0, 1.0]

    def test_single_element(self):
        idx, coeff = _gradient_stencil(0, 1)
        assert coeff == [0.0]


class TestJdetGradient3D:
    def test_matches_finite_diff_no_mask(self):
        sz, sy, sx = 4, 4, 4
        rng = np.random.default_rng(42)
        phi_flat = rng.standard_normal(3 * sz * sy * sx) * 0.2

        analytical = jdet_constraint_jacobian_3d(phi_flat, (sz, sy, sx))
        numerical = _numerical_jacobian(
            lambda p: jacobian_constraint_3d(p, (sz, sy, sx)),
            phi_flat,
        )
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-4)

    def test_matches_finite_diff_with_mask(self):
        sz, sy, sx = 4, 4, 4
        rng = np.random.default_rng(77)
        phi_flat = rng.standard_normal(3 * sz * sy * sx) * 0.2

        mask = np.zeros((sz, sy, sx), dtype=bool)
        mask[0, :, :] = True
        mask[-1, :, :] = True

        analytical = jdet_constraint_jacobian_3d(phi_flat, (sz, sy, sx), mask)
        numerical = _numerical_jacobian(
            lambda p: jacobian_constraint_3d(p, (sz, sy, sx), mask),
            phi_flat,
        )
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-4)

    def test_identity_field(self):
        sz, sy, sx = 3, 3, 3
        phi_flat = np.zeros(3 * sz * sy * sx)

        analytical = jdet_constraint_jacobian_3d(phi_flat, (sz, sy, sx))
        numerical = _numerical_jacobian(
            lambda p: jacobian_constraint_3d(p, (sz, sy, sx)),
            phi_flat,
        )
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-5)

    def test_output_shape(self):
        sz, sy, sx = 3, 4, 5
        phi_flat = np.zeros(3 * sz * sy * sx)
        J = jdet_constraint_jacobian_3d(phi_flat, (sz, sy, sx))
        assert J.shape == (sz * sy * sx, 3 * sz * sy * sx)
