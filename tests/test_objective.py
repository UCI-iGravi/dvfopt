"""Tests for dvfopt.core.objective — squared L2 objective function.

objective_euc computes 0.5 * ||phi - phi_init||^2 with gradient (phi - phi_init).
"""

import numpy as np
import pytest

from dvfopt.core.objective import objective_euc


class TestObjectiveEuc:
    def test_identical_inputs_zero(self):
        phi = np.array([1.0, 2.0, 3.0])
        val, grad = objective_euc(phi, phi.copy())
        assert val == 0.0
        np.testing.assert_array_equal(grad, 0.0)

    def test_value_is_half_squared_norm(self):
        """Value = 0.5 * (3^2 + 4^2) = 0.5 * 25 = 12.5, not the L2 norm 5.0."""
        phi = np.array([3.0, 4.0])
        phi_init = np.array([0.0, 0.0])
        val, grad = objective_euc(phi, phi_init)
        np.testing.assert_allclose(val, 12.5)

    def test_gradient_equals_diff(self):
        """Gradient = (phi - phi_init), not the unit vector diff/||diff||."""
        phi = np.array([3.0, 4.0])
        phi_init = np.array([0.0, 0.0])
        val, grad = objective_euc(phi, phi_init)
        np.testing.assert_allclose(grad, np.array([3.0, 4.0]))

    def test_gradient_direction(self):
        """Gradient should point from phi_init toward phi."""
        rng = np.random.default_rng(123)
        phi = rng.standard_normal(20)
        phi_init = rng.standard_normal(20)
        val, grad = objective_euc(phi, phi_init)
        diff = phi - phi_init
        # grad should be parallel to diff
        cos_sim = np.dot(grad, diff) / (np.linalg.norm(grad) * np.linalg.norm(diff))
        np.testing.assert_allclose(cos_sim, 1.0, atol=1e-12)

    def test_gradient_norm_equals_diff_norm(self):
        """Gradient norm = ||phi - phi_init|| (not 1.0 — this is not the L2 norm)."""
        rng = np.random.default_rng(456)
        phi = rng.standard_normal(50)
        phi_init = rng.standard_normal(50)
        val, grad = objective_euc(phi, phi_init)
        np.testing.assert_allclose(np.linalg.norm(grad),
                                   np.linalg.norm(phi - phi_init), atol=1e-12)

    def test_symmetry(self):
        """Objective is symmetric: 0.5||a-b||^2 == 0.5||b-a||^2."""
        rng = np.random.default_rng(789)
        a = rng.standard_normal(10)
        b = rng.standard_normal(10)
        val_ab, _ = objective_euc(a, b)
        val_ba, _ = objective_euc(b, a)
        np.testing.assert_allclose(val_ab, val_ba)
