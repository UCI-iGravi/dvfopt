"""Tests for dvfopt.dvf — generation and scaling utilities."""

import numpy as np
import pytest

from dvfopt.dvf.generation import generate_random_dvf, generate_random_dvf_3d
from dvfopt.dvf.scaling import scale_dvf, scale_dvf_3d


class TestGenerateRandomDvf:
    def test_shape(self):
        dvf = generate_random_dvf((3, 1, 10, 12), seed=42)
        assert dvf.shape == (3, 1, 10, 12)

    def test_dtype(self):
        dvf = generate_random_dvf((3, 1, 8, 8), seed=42)
        assert dvf.dtype == np.float64

    def test_magnitude_bounds(self):
        mag = 3.0
        dvf = generate_random_dvf((3, 1, 20, 20), max_magnitude=mag, seed=42)
        assert dvf.max() <= mag
        assert dvf.min() >= -mag

    def test_seed_reproducibility(self):
        a = generate_random_dvf((3, 1, 10, 10), seed=123)
        b = generate_random_dvf((3, 1, 10, 10), seed=123)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = generate_random_dvf((3, 1, 10, 10), seed=1)
        b = generate_random_dvf((3, 1, 10, 10), seed=2)
        assert not np.array_equal(a, b)

    def test_wrong_channels_raises(self):
        with pytest.raises(AssertionError):
            generate_random_dvf((2, 1, 10, 10))


class TestGenerateRandomDvf3D:
    def test_shape(self):
        dvf = generate_random_dvf_3d((3, 4, 6, 8), seed=42)
        assert dvf.shape == (3, 4, 6, 8)

    def test_dtype(self):
        dvf = generate_random_dvf_3d((3, 4, 6, 8), seed=42)
        assert dvf.dtype == np.float64

    def test_magnitude_bounds(self):
        mag = 2.5
        dvf = generate_random_dvf_3d((3, 4, 6, 8), max_magnitude=mag, seed=42)
        assert dvf.max() <= mag
        assert dvf.min() >= -mag


class TestScaleDvf:
    def test_output_shape(self):
        dvf = np.zeros((3, 1, 10, 10))
        scaled = scale_dvf(dvf, (20, 20))
        assert scaled.shape == (3, 1, 20, 20)

    def test_identity_stays_zero(self):
        dvf = np.zeros((3, 1, 10, 10))
        scaled = scale_dvf(dvf, (20, 20))
        np.testing.assert_allclose(scaled, 0.0, atol=1e-10)

    def test_magnitude_scales_proportionally(self):
        """Uniform displacement should scale with the spatial scaling factor."""
        H, W = 10, 10
        dvf = np.zeros((3, 1, H, W))
        dvf[1, 0] = 1.0  # dy = 1 everywhere
        dvf[2, 0] = 1.0  # dx = 1 everywhere

        new_H, new_W = 20, 30
        scaled = scale_dvf(dvf, (new_H, new_W))
        # dy should scale by new_H/H = 2.0
        np.testing.assert_allclose(scaled[1, 0], 2.0, atol=1e-6)
        # dx should scale by new_W/W = 3.0
        np.testing.assert_allclose(scaled[2, 0], 3.0, atol=1e-6)

    def test_downscale(self):
        dvf = np.zeros((3, 1, 20, 20))
        scaled = scale_dvf(dvf, (10, 10))
        assert scaled.shape == (3, 1, 10, 10)

    def test_preserves_dtype(self):
        dvf = np.zeros((3, 1, 10, 10), dtype=np.float32)
        scaled = scale_dvf(dvf, (20, 20))
        assert scaled.dtype == np.float32


class TestScaleDvf3D:
    def test_output_shape(self):
        dvf = np.zeros((3, 4, 6, 8))
        scaled = scale_dvf_3d(dvf, (8, 12, 16))
        assert scaled.shape == (3, 8, 12, 16)

    def test_identity_stays_zero(self):
        dvf = np.zeros((3, 4, 6, 8))
        scaled = scale_dvf_3d(dvf, (8, 12, 16))
        np.testing.assert_allclose(scaled, 0.0, atol=1e-10)

    def test_magnitude_scales_proportionally(self):
        D, H, W = 4, 6, 8
        dvf = np.zeros((3, D, H, W))
        dvf[0] = 1.0  # dz = 1
        dvf[1] = 1.0  # dy = 1
        dvf[2] = 1.0  # dx = 1

        new_D, new_H, new_W = 8, 12, 16
        scaled = scale_dvf_3d(dvf, (new_D, new_H, new_W))
        np.testing.assert_allclose(scaled[0], 2.0, atol=1e-6)   # dz * 2
        np.testing.assert_allclose(scaled[1], 2.0, atol=1e-6)   # dy * 2
        np.testing.assert_allclose(scaled[2], 2.0, atol=1e-6)   # dx * 2
