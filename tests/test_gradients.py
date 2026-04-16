"""Tests for dvfopt.core.gradients — analytical gradients vs finite differences.

For each constraint type (Jdet, shoelace, injectivity), we verify that the
analytical Jacobian matrix matches a numerical finite-difference approximation.
"""

import numpy as np
import pytest

from dvfopt.core.slsqp.constraints import jacobian_constraint
from dvfopt.jacobian.shoelace import shoelace_constraint
from dvfopt.jacobian.monotonicity import injectivity_constraint
from dvfopt.core.slsqp.gradients import (
    jdet_constraint_jacobian_2d,
    shoelace_constraint_jacobian_2d,
    injectivity_constraint_jacobian_2d,
)


def _numerical_jacobian(func, x0, eps=1e-6):
    """Compute numerical Jacobian via central finite differences."""
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


class TestJdetGradient:
    @pytest.mark.parametrize("exclude_boundaries", [True, False])
    def test_matches_finite_diff(self, exclude_boundaries):
        sy, sx = 5, 5
        rng = np.random.default_rng(42)
        phi_flat = rng.standard_normal(2 * sy * sx) * 0.3

        analytical = jdet_constraint_jacobian_2d(phi_flat, (sy, sx), exclude_boundaries)
        numerical = _numerical_jacobian(
            lambda p: jacobian_constraint(p, (sy, sx), exclude_boundaries),
            phi_flat,
        )
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-5)

    def test_rectangular_window(self):
        sy, sx = 4, 6
        rng = np.random.default_rng(99)
        phi_flat = rng.standard_normal(2 * sy * sx) * 0.2

        analytical = jdet_constraint_jacobian_2d(phi_flat, (sy, sx), True)
        numerical = _numerical_jacobian(
            lambda p: jacobian_constraint(p, (sy, sx), True),
            phi_flat,
        )
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-5)

    def test_identity_field(self):
        """At identity (zero displacement), the analytical gradient should still match."""
        sy, sx = 5, 5
        phi_flat = np.zeros(2 * sy * sx)

        analytical = jdet_constraint_jacobian_2d(phi_flat, (sy, sx), True)
        numerical = _numerical_jacobian(
            lambda p: jacobian_constraint(p, (sy, sx), True),
            phi_flat,
        )
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-5)


class TestShoelaceGradient:
    @pytest.mark.parametrize("exclude_boundaries", [True, False])
    def test_matches_finite_diff(self, exclude_boundaries):
        sy, sx = 6, 6
        rng = np.random.default_rng(77)
        phi_flat = rng.standard_normal(2 * sy * sx) * 0.3

        analytical = shoelace_constraint_jacobian_2d(phi_flat, (sy, sx), exclude_boundaries)
        numerical = _numerical_jacobian(
            lambda p: shoelace_constraint(p, (sy, sx), exclude_boundaries),
            phi_flat,
        )
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-5)


class TestInjectivityGradient:
    @pytest.mark.parametrize("exclude_boundaries", [True, False])
    def test_matches_finite_diff(self, exclude_boundaries):
        sy, sx = 5, 5
        rng = np.random.default_rng(55)
        phi_flat = rng.standard_normal(2 * sy * sx) * 0.3

        analytical = injectivity_constraint_jacobian_2d(phi_flat, (sy, sx), exclude_boundaries)
        numerical = _numerical_jacobian(
            lambda p: injectivity_constraint(p, (sy, sx), exclude_boundaries),
            phi_flat,
        )
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-5)

    def test_constant_jacobian(self):
        """Injectivity constraint has constant Jacobian (does not depend on phi values)."""
        sy, sx = 5, 5
        rng = np.random.default_rng(33)
        phi1 = rng.standard_normal(2 * sy * sx)
        phi2 = rng.standard_normal(2 * sy * sx)

        j1 = injectivity_constraint_jacobian_2d(phi1, (sy, sx), True)
        j2 = injectivity_constraint_jacobian_2d(phi2, (sy, sx), True)
        np.testing.assert_array_equal(j1.toarray(), j2.toarray())
