"""Tests for dvfopt.jacobian — Jacobian determinant, shoelace areas, monotonicity."""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import (
    _numpy_jdet_2d,
    jacobian_det2D,
    _numpy_jdet_3d,
    jacobian_det3D,
)
from dvfopt.jacobian.shoelace import _shoelace_areas_2d, shoelace_det2D
from dvfopt.jacobian.monotonicity import _monotonicity_diffs_2d


# ── 2D Jacobian determinant ──────────────────────────────────────────────

class TestNumpyJdet2D:
    def test_identity_field_all_ones(self):
        """Zero displacement → Jdet = 1 everywhere."""
        dy = np.zeros((10, 10))
        dx = np.zeros((10, 10))
        jdet = _numpy_jdet_2d(dy, dx)
        np.testing.assert_allclose(jdet, 1.0)

    def test_uniform_translation_all_ones(self):
        """Constant displacement → zero gradients → Jdet = 1."""
        dy = np.full((10, 10), 3.5)
        dx = np.full((10, 10), -2.0)
        jdet = _numpy_jdet_2d(dy, dx)
        np.testing.assert_allclose(jdet, 1.0)

    def test_pure_scaling(self):
        """dx = s*x, dy = s*y → Jdet = (1+s)^2 at interior points."""
        H, W = 11, 11
        yy, xx = np.mgrid[:H, :W].astype(float)
        s = 0.5
        dx = s * xx
        dy = s * yy
        jdet = _numpy_jdet_2d(dy, dx)
        # Interior pixels (central diff) should give exact (1+s)^2 = 2.25
        np.testing.assert_allclose(jdet[1:-1, 1:-1], (1 + s) ** 2, atol=1e-12)

    def test_known_negative_jacobian(self):
        """A crossing displacement should produce negative Jdet somewhere."""
        H, W = 5, 5
        dx = np.zeros((H, W))
        dy = np.zeros((H, W))
        # Create a fold: push left column right and right column left
        dx[:, 0] = 3.0
        dx[:, -1] = -3.0
        jdet = _numpy_jdet_2d(dy, dx)
        assert jdet.min() < 0, "Expected negative Jacobian from crossing displacement"

    def test_output_shape(self):
        dy = np.zeros((7, 13))
        dx = np.zeros((7, 13))
        jdet = _numpy_jdet_2d(dy, dx)
        assert jdet.shape == (7, 13)


class TestJacobianDet2D:
    def test_shape_2hw(self):
        phi = np.zeros((2, 8, 12))
        jdet = jacobian_det2D(phi)
        assert jdet.shape == (1, 8, 12)

    def test_shape_21hw(self):
        phi = np.zeros((2, 1, 8, 12))
        jdet = jacobian_det2D(phi)
        assert jdet.shape == (1, 8, 12)

    def test_identity_ones(self):
        phi = np.zeros((2, 10, 10))
        jdet = jacobian_det2D(phi)
        np.testing.assert_allclose(jdet, 1.0)


# ── 3D Jacobian determinant ──────────────────────────────────────────────

class TestNumpyJdet3D:
    def test_identity_field_all_ones(self):
        dz = np.zeros((6, 6, 6))
        dy = np.zeros((6, 6, 6))
        dx = np.zeros((6, 6, 6))
        jdet = _numpy_jdet_3d(dz, dy, dx)
        np.testing.assert_allclose(jdet, 1.0)

    def test_uniform_translation_all_ones(self):
        dz = np.full((6, 6, 6), 1.0)
        dy = np.full((6, 6, 6), -2.5)
        dx = np.full((6, 6, 6), 0.3)
        jdet = _numpy_jdet_3d(dz, dy, dx)
        np.testing.assert_allclose(jdet, 1.0)

    def test_pure_isotropic_scaling_interior(self):
        """dx = s*x, dy = s*y, dz = s*z → Jdet = (1+s)^3 at interior."""
        D, H, W = 9, 9, 9
        zz, yy, xx = np.mgrid[:D, :H, :W].astype(float)
        s = 0.3
        dx = s * xx
        dy = s * yy
        dz = s * zz
        jdet = _numpy_jdet_3d(dz, dy, dx)
        expected = (1 + s) ** 3
        np.testing.assert_allclose(jdet[1:-1, 1:-1, 1:-1], expected, atol=1e-12)

    def test_output_shape(self):
        jdet = _numpy_jdet_3d(np.zeros((4, 5, 6)), np.zeros((4, 5, 6)), np.zeros((4, 5, 6)))
        assert jdet.shape == (4, 5, 6)


class TestJacobianDet3D:
    def test_shape(self):
        phi = np.zeros((3, 4, 5, 6))
        jdet = jacobian_det3D(phi)
        assert jdet.shape == (4, 5, 6)

    def test_identity(self):
        phi = np.zeros((3, 6, 6, 6))
        jdet = jacobian_det3D(phi)
        np.testing.assert_allclose(jdet, 1.0)


# ── Shoelace (quad area) ─────────────────────────────────────────────────

class TestShoelace:
    def test_identity_unit_areas(self):
        """Zero displacement → each quad cell has area 1.0."""
        dy = np.zeros((6, 6))
        dx = np.zeros((6, 6))
        areas = _shoelace_areas_2d(dy, dx)
        assert areas.shape == (5, 5)
        np.testing.assert_allclose(areas, 1.0)

    def test_uniform_scaling(self):
        """dx = s*x, dy = s*y → each quad has area (1+s)^2."""
        H, W = 8, 8
        yy, xx = np.mgrid[:H, :W].astype(float)
        s = 0.5
        dx = s * xx
        dy = s * yy
        areas = _shoelace_areas_2d(dy, dx)
        np.testing.assert_allclose(areas, (1 + s) ** 2, atol=1e-12)

    def test_negative_area_from_fold(self):
        """A folding displacement should produce negative quad areas."""
        H, W = 5, 5
        dy = np.zeros((H, W))
        dx = np.zeros((H, W))
        dx[:, 0] = 3.0
        dx[:, -1] = -3.0
        areas = _shoelace_areas_2d(dy, dx)
        assert areas.min() < 0

    def test_shoelace_det2D_shape(self):
        phi = np.zeros((2, 6, 8))
        result = shoelace_det2D(phi)
        assert result.shape == (1, 5, 7)


# ── Monotonicity ─────────────────────────────────────────────────────────

class TestMonotonicity:
    def test_identity_all_positive(self):
        """Zero displacement → monotonicity diffs = 1.0 everywhere."""
        dy = np.zeros((6, 6))
        dx = np.zeros((6, 6))
        h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
        assert h_mono.shape == (6, 5)
        assert v_mono.shape == (5, 6)
        np.testing.assert_allclose(h_mono, 1.0)
        np.testing.assert_allclose(v_mono, 1.0)

    def test_uniform_translation_unchanged(self):
        """Constant displacement → diff = 0 → 1 + diff = 1."""
        dy = np.full((6, 6), 5.0)
        dx = np.full((6, 6), -3.0)
        h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
        np.testing.assert_allclose(h_mono, 1.0)
        np.testing.assert_allclose(v_mono, 1.0)

    def test_negative_monotonicity_from_crossing(self):
        """Displacements that reverse ordering → negative monotonicity."""
        H, W = 5, 5
        dx = np.zeros((H, W))
        dy = np.zeros((H, W))
        # Column 1 moves right by 3, column 2 moves left by 3
        dx[:, 1] = 3.0
        dx[:, 2] = -3.0
        h_mono, _ = _monotonicity_diffs_2d(dy, dx)
        assert h_mono.min() < 0
