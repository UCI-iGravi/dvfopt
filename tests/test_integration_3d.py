"""Integration tests for the 3D iterative solver.

These tests verify the full 3D pipeline works end-to-end:
after correction, zero negative Jacobian determinants remain.
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det3D
from dvfopt.dvf.generation import generate_random_dvf_3d


THRESHOLD = 0.01


def _assert_no_neg_jdet_3d(phi, threshold=THRESHOLD):
    jdet = jacobian_det3D(phi)
    n_neg = int((jdet <= 0).sum())
    min_j = float(jdet.min())
    assert n_neg == 0, f"Expected 0 negative Jdet voxels, got {n_neg} (min={min_j:.6f})"
    assert min_j >= threshold - 1e-5, f"Expected min Jdet >= {threshold}, got {min_j:.6f}"


class TestIterative3D:
    def test_identity_unchanged(self):
        from dvfopt.core.iterative3d import iterative_3d

        d = np.zeros((3, 4, 4, 4), dtype=np.float64)
        phi = iterative_3d(d, verbose=0, max_iterations=5)
        assert phi.shape == (3, 4, 4, 4)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_output_shape(self):
        from dvfopt.core.iterative3d import iterative_3d

        d = np.zeros((3, 4, 5, 6), dtype=np.float64)
        phi = iterative_3d(d, verbose=0, max_iterations=5)
        assert phi.shape == (3, 4, 5, 6)

    def test_corrects_single_spike(self):
        """A single large displacement spike should be correctable."""
        from dvfopt.core.iterative3d import iterative_3d

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0  # large dx spike
        jdet_before = jacobian_det3D(d)
        assert jdet_before.min() < THRESHOLD

        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=500)
        _assert_no_neg_jdet_3d(phi)

    def test_corrects_random_field(self):
        """Random 3D DVF with negative Jacobians should be fully corrected."""
        from dvfopt.core.iterative3d import iterative_3d

        d = generate_random_dvf_3d((3, 5, 5, 5), max_magnitude=2.0, seed=42)
        jdet_before = jacobian_det3D(d)
        if jdet_before.min() >= THRESHOLD:
            pytest.skip("Random 3D DVF has no negative Jacobians")

        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=1000)
        _assert_no_neg_jdet_3d(phi)

    def test_opposing_spikes(self):
        from dvfopt.core.iterative3d import iterative_3d

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 2] = 3.0
        d[2, 3, 3, 3] = -3.0
        jdet_before = jacobian_det3D(d)
        assert jdet_before.min() < THRESHOLD

        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=500)
        _assert_no_neg_jdet_3d(phi)

    def test_displacement_stays_close(self):
        from dvfopt.core.iterative3d import iterative_3d

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0
        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=500)
        max_change = np.abs(phi - d).max()
        assert max_change < 10.0

    def test_non_cubic_grid(self):
        """Non-cubic 3D grid triggers full-grid fallback path."""
        from dvfopt.core.iterative3d import iterative_3d

        d = np.zeros((3, 3, 5, 7), dtype=np.float64)
        d[2, 1, 2, 3] = 4.0
        d[2, 1, 2, 4] = -4.0
        jdet_before = jacobian_det3D(d)
        if jdet_before.min() >= THRESHOLD:
            pytest.skip("No negative Jdet in non-cubic field")

        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=500)
        _assert_no_neg_jdet_3d(phi)
