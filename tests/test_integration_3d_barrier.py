"""Integration tests for the 3D penalty -> log-barrier solvers.

Mirrors test_integration_3d.py but exercises the full-grid L-BFGS-B
(numpy/scipy) and torch L-BFGS backends.
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


class TestBarrier3DNumpy:
    def test_identity_unchanged(self):
        from dvfopt.core.iterative3d_barrier import iterative_3d_barrier

        d = np.zeros((3, 4, 4, 4), dtype=np.float64)
        phi = iterative_3d_barrier(d, verbose=0)
        assert phi.shape == (3, 4, 4, 4)
        np.testing.assert_allclose(phi, 0.0, atol=1e-3)

    def test_output_shape(self):
        from dvfopt.core.iterative3d_barrier import iterative_3d_barrier

        d = np.zeros((3, 4, 5, 6), dtype=np.float64)
        phi = iterative_3d_barrier(d, verbose=0)
        assert phi.shape == (3, 4, 5, 6)

    def test_corrects_single_spike(self):
        from dvfopt.core.iterative3d_barrier import iterative_3d_barrier

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0
        assert jacobian_det3D(d).min() < THRESHOLD

        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD,
                                    lam_schedule=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6))
        _assert_no_neg_jdet_3d(phi)

    def test_corrects_random_field(self):
        from dvfopt.core.iterative3d_barrier import iterative_3d_barrier

        d = generate_random_dvf_3d((3, 5, 5, 5), max_magnitude=2.0, seed=42)
        if jacobian_det3D(d).min() >= THRESHOLD:
            pytest.skip("Random DVF has no negative Jacobians")

        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD)
        _assert_no_neg_jdet_3d(phi)

    def test_opposing_spikes(self):
        from dvfopt.core.iterative3d_barrier import iterative_3d_barrier

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 2] = 3.0
        d[2, 3, 3, 3] = -3.0
        assert jacobian_det3D(d).min() < THRESHOLD

        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD)
        _assert_no_neg_jdet_3d(phi)

    def test_displacement_stays_close(self):
        from dvfopt.core.iterative3d_barrier import iterative_3d_barrier

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0
        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD)
        assert np.abs(phi - d).max() < 10.0

    def test_non_cubic_grid(self):
        from dvfopt.core.iterative3d_barrier import iterative_3d_barrier

        d = np.zeros((3, 3, 5, 7), dtype=np.float64)
        d[2, 1, 2, 3] = 4.0
        d[2, 1, 2, 4] = -4.0
        if jacobian_det3D(d).min() >= THRESHOLD:
            pytest.skip("No negative Jdet in non-cubic field")

        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD)
        _assert_no_neg_jdet_3d(phi)


class TestBarrier3DTorch:
    def test_identity_unchanged(self):
        from dvfopt.core.iterative3d_barrier_torch import iterative_3d_barrier_torch

        d = np.zeros((3, 4, 4, 4), dtype=np.float64)
        phi = iterative_3d_barrier_torch(d, verbose=0, device="cpu")
        assert phi.shape == (3, 4, 4, 4)
        np.testing.assert_allclose(phi, 0.0, atol=1e-3)

    def test_corrects_single_spike(self):
        from dvfopt.core.iterative3d_barrier_torch import iterative_3d_barrier_torch

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0
        assert jacobian_det3D(d).min() < THRESHOLD

        phi = iterative_3d_barrier_torch(d, verbose=0, threshold=THRESHOLD, device="cpu",
                                          lam_schedule=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6))
        _assert_no_neg_jdet_3d(phi)

    def test_opposing_spikes(self):
        from dvfopt.core.iterative3d_barrier_torch import iterative_3d_barrier_torch

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 2] = 3.0
        d[2, 3, 3, 3] = -3.0
        assert jacobian_det3D(d).min() < THRESHOLD

        phi = iterative_3d_barrier_torch(d, verbose=0, threshold=THRESHOLD, device="cpu")
        _assert_no_neg_jdet_3d(phi)
