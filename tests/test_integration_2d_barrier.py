"""Integration tests for the 2D penalty -> log-barrier solvers (numpy + torch)."""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.dvf.generation import generate_random_dvf


THRESHOLD = 0.01


def _make_folded_field(H=10, W=10):
    return generate_random_dvf((3, 1, H, W), max_magnitude=3.0, seed=42).astype(np.float64)


def _assert_no_neg_jdet_2d(phi, threshold=THRESHOLD):
    jdet = jacobian_det2D(phi)
    n_neg = int((jdet <= 0).sum())
    min_j = float(jdet.min())
    assert n_neg == 0, f"Expected 0 negative Jdet, got {n_neg} (min={min_j:.6f})"
    assert min_j >= threshold - 1e-5, f"Expected min Jdet >= {threshold}, got {min_j:.6f}"


class TestBarrier2DNumpy:
    def test_identity_unchanged(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = np.zeros((3, 1, 8, 8), dtype=np.float64)
        phi = iterative_2d_barrier(d, verbose=0)
        assert phi.shape == (2, 8, 8)
        np.testing.assert_allclose(phi, 0.0, atol=1e-3)

    def test_output_shape(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = _make_folded_field(8, 12)
        phi = iterative_2d_barrier(d, verbose=0)
        assert phi.shape == (2, 8, 12)

    def test_corrects_negative_jacobians(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = _make_folded_field(10, 10)
        if jacobian_det2D(d[[1, 2], 0]).min() >= THRESHOLD:
            pytest.skip("Test field already feasible")
        phi = iterative_2d_barrier(d, verbose=0, threshold=THRESHOLD)
        _assert_no_neg_jdet_2d(phi)

    def test_displacement_stays_close(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = _make_folded_field(10, 10)
        phi_init = np.stack([d[1, 0], d[2, 0]])
        phi = iterative_2d_barrier(d, verbose=0, threshold=THRESHOLD)
        assert np.linalg.norm(phi - phi_init) < 50.0


class TestBarrier2DTorch:
    def test_identity_unchanged(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier_torch

        d = np.zeros((3, 1, 8, 8), dtype=np.float64)
        phi = iterative_2d_barrier_torch(d, verbose=0, device="cpu")
        assert phi.shape == (2, 8, 8)
        np.testing.assert_allclose(phi, 0.0, atol=1e-3)

    def test_corrects_negative_jacobians(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier_torch

        d = _make_folded_field(10, 10)
        if jacobian_det2D(d[[1, 2], 0]).min() >= THRESHOLD:
            pytest.skip("Test field already feasible")
        phi = iterative_2d_barrier_torch(d, verbose=0, threshold=THRESHOLD, device="cpu")
        _assert_no_neg_jdet_2d(phi)
