"""Tests for dvfopt.core.objective — L2 norm objective function."""

import numpy as np
import pytest

from dvfopt.core.objective import objectiveEuc


class TestObjectiveEuc:
    def test_identical_inputs_zero(self):
        phi = np.array([1.0, 2.0, 3.0])
        val, grad = objectiveEuc(phi, phi.copy())
        assert val == 0.0
        np.testing.assert_array_equal(grad, 0.0)

    def test_value_is_l2_norm(self):
        phi = np.array([3.0, 4.0])
        phi_init = np.array([0.0, 0.0])
        val, grad = objectiveEuc(phi, phi_init)
        np.testing.assert_allclose(val, 5.0)

    def test_gradient_is_unit_diff(self):
        phi = np.array([3.0, 4.0])
        phi_init = np.array([0.0, 0.0])
        val, grad = objectiveEuc(phi, phi_init)
        expected_grad = np.array([3.0 / 5.0, 4.0 / 5.0])
        np.testing.assert_allclose(grad, expected_grad)

    def test_gradient_direction(self):
        """Gradient should point from phi_init toward phi."""
        rng = np.random.default_rng(123)
        phi = rng.standard_normal(20)
        phi_init = rng.standard_normal(20)
        val, grad = objectiveEuc(phi, phi_init)
        diff = phi - phi_init
        # grad should be parallel to diff
        cos_sim = np.dot(grad, diff) / (np.linalg.norm(grad) * np.linalg.norm(diff))
        np.testing.assert_allclose(cos_sim, 1.0, atol=1e-12)

    def test_gradient_unit_norm(self):
        """Gradient of ||x|| is a unit vector (when x != 0)."""
        rng = np.random.default_rng(456)
        phi = rng.standard_normal(50)
        phi_init = rng.standard_normal(50)
        val, grad = objectiveEuc(phi, phi_init)
        np.testing.assert_allclose(np.linalg.norm(grad), 1.0, atol=1e-12)

    def test_symmetry(self):
        """Objective is symmetric: ||a - b|| == ||b - a||."""
        rng = np.random.default_rng(789)
        a = rng.standard_normal(10)
        b = rng.standard_normal(10)
        val_ab, _ = objectiveEuc(a, b)
        val_ba, _ = objectiveEuc(b, a)
        np.testing.assert_allclose(val_ab, val_ba)
