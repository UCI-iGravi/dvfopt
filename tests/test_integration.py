"""Integration tests — run the iterative solver on small fields and verify correction.

These tests are slower (seconds each) but verify the full pipeline works end-to-end.
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.dvf.generation import generate_random_dvf


class TestIterativeSolver2D:
    """End-to-end tests for iterative_serial."""

    @staticmethod
    def _make_folded_field(H=10, W=10):
        """Create a small deformation with guaranteed negative Jacobians."""
        return generate_random_dvf((3, 1, H, W), max_magnitude=3.0, seed=42).astype(np.float64)

    def test_corrects_negative_jacobians(self):
        """After correction, all Jacobian determinants should be >= threshold."""
        from dvfopt.core.iterative import iterative_serial

        deformation = self._make_folded_field(10, 10)
        jdet_before = jacobian_det2D(deformation[[1, 2], 0])
        assert jdet_before.min() < 0.01, (
            f"Test field should have negative Jdet, got min={jdet_before.min():.4f}"
        )

        phi = iterative_serial(
            deformation, verbose=0, threshold=0.01, max_iterations=500,
        )
        jdet_after = jacobian_det2D(phi)
        assert jdet_after.min() >= 0.01 - 1e-5, (
            f"Expected all Jdet >= 0.01, got min={jdet_after.min():.6f}"
        )

    def test_output_shape(self):
        from dvfopt.core.iterative import iterative_serial

        deformation = self._make_folded_field(8, 12)
        phi = iterative_serial(deformation, verbose=0, max_iterations=50)
        assert phi.shape == (2, 8, 12)

    def test_identity_field_unchanged(self):
        """A field with no negative Jdet should pass through with minimal change."""
        from dvfopt.core.iterative import iterative_serial

        deformation = np.zeros((3, 1, 8, 8), dtype=np.float64)
        phi = iterative_serial(deformation, verbose=0, max_iterations=10)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_displacement_stays_close(self):
        """Corrected field should not deviate too far from the original."""
        from dvfopt.core.iterative import iterative_serial

        deformation = self._make_folded_field(10, 10)
        phi_init = np.stack([deformation[1, 0], deformation[2, 0]])
        phi = iterative_serial(deformation, verbose=0, max_iterations=500)
        max_change = np.abs(phi - phi_init).max()
        assert max_change < 10.0, f"Max displacement change too large: {max_change:.3f}"

    def test_with_shoelace(self):
        """Solver should also succeed with enforce_shoelace=True."""
        from dvfopt.core.iterative import iterative_serial

        deformation = self._make_folded_field(10, 10)
        phi = iterative_serial(
            deformation, verbose=0, threshold=0.01,
            max_iterations=500, enforce_shoelace=True,
        )
        jdet_after = jacobian_det2D(phi)
        assert jdet_after.min() >= 0.01 - 1e-5

    def test_with_injectivity(self):
        """Solver should also succeed with enforce_injectivity=True."""
        from dvfopt.core.iterative import iterative_serial

        deformation = self._make_folded_field(10, 10)
        phi = iterative_serial(
            deformation, verbose=0, threshold=0.01,
            max_iterations=1000, enforce_injectivity=True,
        )
        jdet_after = jacobian_det2D(phi)
        assert jdet_after.min() >= 0.01 - 1e-5


class TestParallelSolver2D:
    """End-to-end tests for iterative_parallel (same contract as serial)."""

    @staticmethod
    def _make_folded_field(H=10, W=10):
        return generate_random_dvf((3, 1, H, W), max_magnitude=3.0, seed=42).astype(np.float64)

    def test_corrects_negative_jacobians(self):
        from dvfopt.core.parallel import iterative_parallel

        deformation = self._make_folded_field(10, 10)
        phi = iterative_parallel(
            deformation, verbose=0, threshold=0.01,
            max_iterations=200, max_workers=1,
        )
        jdet_after = jacobian_det2D(phi)
        assert jdet_after.min() >= 0.01 - 1e-5

    def test_output_shape(self):
        from dvfopt.core.parallel import iterative_parallel

        deformation = self._make_folded_field(8, 12)
        phi = iterative_parallel(
            deformation, verbose=0, max_iterations=50, max_workers=1,
        )
        assert phi.shape == (2, 8, 12)


class TestRandomDvfCorrection:
    """Test correction on random DVFs — verifies the solver handles realistic noise."""

    def test_random_dvf_small_magnitude(self):
        """Small random displacements should be correctable."""
        from dvfopt.core.iterative import iterative_serial

        dvf = generate_random_dvf((3, 1, 12, 12), max_magnitude=1.5, seed=42)
        dvf = dvf.astype(np.float64)
        jdet_before = jacobian_det2D(dvf[[1, 2], 0])

        if jdet_before.min() >= 0.01:
            pytest.skip("Random DVF has no negative Jacobians to correct")

        phi = iterative_serial(
            dvf, verbose=0, threshold=0.01, max_iterations=500,
        )
        jdet_after = jacobian_det2D(phi)
        assert jdet_after.min() >= 0.01 - 1e-5
