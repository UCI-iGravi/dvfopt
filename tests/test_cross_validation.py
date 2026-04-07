"""Cross-validation: numpy Jacobian determinant vs SimpleITK Jacobian determinant.

The two implementations should agree on interior pixels (both use central
differences there). Boundary pixels may differ due to different boundary
stencils.
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.jacobian.sitk_jdet import sitk_jacobian_determinant


class TestNumpyVsSitk2D:
    def test_identity_field(self):
        d = np.zeros((3, 1, 10, 10), dtype=np.float64)
        jdet_np = jacobian_det2D(d[[1, 2], 0])[0]  # (H, W)
        jdet_sitk = sitk_jacobian_determinant(d)     # (1, H, W)
        # Interior should both be 1.0
        np.testing.assert_allclose(jdet_np[1:-1, 1:-1], 1.0, atol=1e-10)
        np.testing.assert_allclose(jdet_sitk[0, 1:-1, 1:-1], 1.0, atol=1e-10)

    def test_smooth_field_interior_agrees(self):
        """For a smooth field, interior Jdet should agree between numpy and SimpleITK."""
        H, W = 20, 20
        d = np.zeros((3, 1, H, W), dtype=np.float64)
        yy, xx = np.mgrid[:H, :W].astype(float)
        d[1, 0] = 0.3 * np.sin(2 * np.pi * yy / H)
        d[2, 0] = 0.3 * np.sin(2 * np.pi * xx / W)

        jdet_np = jacobian_det2D(d[[1, 2], 0])[0]
        jdet_sitk = sitk_jacobian_determinant(d)[0]

        # Interior pixels (central diff in both) should be very close
        np.testing.assert_allclose(
            jdet_np[2:-2, 2:-2], jdet_sitk[2:-2, 2:-2], atol=1e-6)

    def test_random_field_interior_agrees(self):
        rng = np.random.default_rng(42)
        d = np.zeros((3, 1, 15, 15), dtype=np.float64)
        d[1, 0] = rng.standard_normal((15, 15)) * 0.3
        d[2, 0] = rng.standard_normal((15, 15)) * 0.3

        jdet_np = jacobian_det2D(d[[1, 2], 0])[0]
        jdet_sitk = sitk_jacobian_determinant(d)[0]

        np.testing.assert_allclose(
            jdet_np[2:-2, 2:-2], jdet_sitk[2:-2, 2:-2], atol=1e-6)


class TestNumpyVsSitk3D:
    def test_identity_field_3d(self):
        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        jdet_np = jacobian_det3D(d)
        jdet_sitk = sitk_jacobian_determinant(d)
        np.testing.assert_allclose(jdet_np[1:-1, 1:-1, 1:-1], 1.0, atol=1e-10)
        np.testing.assert_allclose(jdet_sitk[1:-1, 1:-1, 1:-1], 1.0, atol=1e-10)

    def test_smooth_field_3d_interior_agrees(self):
        D, H, W = 10, 10, 10
        d = np.zeros((3, D, H, W), dtype=np.float64)
        zz, yy, xx = np.mgrid[:D, :H, :W].astype(float)
        d[0] = 0.2 * np.sin(2 * np.pi * zz / D)
        d[1] = 0.2 * np.sin(2 * np.pi * yy / H)
        d[2] = 0.2 * np.sin(2 * np.pi * xx / W)

        jdet_np = jacobian_det3D(d)
        jdet_sitk = sitk_jacobian_determinant(d)

        np.testing.assert_allclose(
            jdet_np[2:-2, 2:-2, 2:-2], jdet_sitk[2:-2, 2:-2, 2:-2], atol=1e-5)
