"""Cross-validation tests: compare numpy Jacobian determinant against SimpleITK
and a manual finite-difference reference implementation.

The goal is to verify that dvfopt's numpy-based Jacobian determinant computation
matches established library results on interior pixels (boundary handling differs
between methods, so only interior pixels are compared).
"""

import numpy as np
import pytest
import SimpleITK as sitk

from dvfopt.jacobian.numpy_jdet import (
    _numpy_jdet_2d,
    _numpy_jdet_3d,
    jacobian_det2D,
    jacobian_det3D,
)
from dvfopt.jacobian.sitk_jdet import sitk_jacobian_determinant
from dvfopt.dvf.generation import generate_random_dvf, generate_random_dvf_3d


# ── Reference implementation (manual central differences) ───────────────

def _reference_jdet_2d(dy, dx):
    """Manual central-difference Jacobian determinant (interior only).

    Computes the 2x2 deformation gradient tensor element-by-element using
    the standard central difference formula: f'(x) = (f(x+1) - f(x-1)) / 2.
    Returns an array of shape (H-2, W-2) covering only interior pixels.
    """
    H, W = dy.shape
    jdet = np.empty((H - 2, W - 2))
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            ddx_dx = (dx[i, j + 1] - dx[i, j - 1]) / 2.0
            ddx_dy = (dx[i + 1, j] - dx[i - 1, j]) / 2.0
            ddy_dx = (dy[i, j + 1] - dy[i, j - 1]) / 2.0
            ddy_dy = (dy[i + 1, j] - dy[i - 1, j]) / 2.0
            jdet[i - 1, j - 1] = (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx
    return jdet


def _reference_jdet_3d(dz, dy, dx):
    """Manual central-difference 3D Jacobian determinant (interior only).

    Returns shape (D-2, H-2, W-2).
    """
    D, H, W = dz.shape
    jdet = np.empty((D - 2, H - 2, W - 2))
    for k in range(1, D - 1):
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                ddx_dx = (dx[k, i, j + 1] - dx[k, i, j - 1]) / 2.0
                ddx_dy = (dx[k, i + 1, j] - dx[k, i - 1, j]) / 2.0
                ddx_dz = (dx[k + 1, i, j] - dx[k - 1, i, j]) / 2.0

                ddy_dx = (dy[k, i, j + 1] - dy[k, i, j - 1]) / 2.0
                ddy_dy = (dy[k, i + 1, j] - dy[k, i - 1, j]) / 2.0
                ddy_dz = (dy[k + 1, i, j] - dy[k - 1, i, j]) / 2.0

                ddz_dx = (dz[k, i, j + 1] - dz[k, i, j - 1]) / 2.0
                ddz_dy = (dz[k, i + 1, j] - dz[k, i - 1, j]) / 2.0
                ddz_dz = (dz[k + 1, i, j] - dz[k - 1, i, j]) / 2.0

                a11 = 1 + ddx_dx; a12 = ddx_dy;     a13 = ddx_dz
                a21 = ddy_dx;     a22 = 1 + ddy_dy;  a23 = ddy_dz
                a31 = ddz_dx;     a32 = ddz_dy;      a33 = 1 + ddz_dz

                jdet[k - 1, i - 1, j - 1] = (
                    a11 * (a22 * a33 - a23 * a32)
                    - a12 * (a21 * a33 - a23 * a31)
                    + a13 * (a21 * a32 - a22 * a31)
                )
    return jdet


def _sitk_jdet_2d(dy, dx):
    """Compute 2D Jacobian determinant via SimpleITK for a single 2D slice.

    Wraps the displacement as a (3,1,H,W) field with dz=0 so
    sitk_jacobian_determinant can process it.
    """
    H, W = dy.shape
    phi = np.zeros((3, 1, H, W), dtype=np.float64)
    phi[0, 0] = 0.0   # dz = 0
    phi[1, 0] = dy
    phi[2, 0] = dx
    jdet_3d = sitk_jacobian_determinant(phi)  # (1, H, W)
    return jdet_3d[0]  # (H, W)


# ── Helper to build test fields ────────────────────────────────────────

def _make_smooth_2d(H, W, seed):
    """Generate a smooth 2D displacement field via low-frequency random + zoom."""
    rng = np.random.default_rng(seed)
    small_dy = rng.standard_normal((4, 4))
    small_dx = rng.standard_normal((4, 4))
    from scipy.ndimage import zoom
    dy = zoom(small_dy, (H / 4, W / 4), order=3)
    dx = zoom(small_dx, (H / 4, W / 4), order=3)
    return dy, dx


def _make_smooth_3d(D, H, W, seed):
    """Generate a smooth 3D displacement field."""
    rng = np.random.default_rng(seed)
    small = rng.standard_normal((3, 3, 3, 3))
    from scipy.ndimage import zoom
    factors = (1, D / 3, H / 3, W / 3)
    big = zoom(small, factors, order=3)
    return big[0], big[1], big[2]


# ── 2D: numpy vs manual reference ──────────────────────────────────────

class TestNumpyVsReference2D:
    """Compare numpy Jdet against the explicit loop reference on interior pixels."""

    def test_identity(self):
        dy = np.zeros((10, 10))
        dx = np.zeros((10, 10))
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        ref_jdet = _reference_jdet_2d(dy, dx)
        np.testing.assert_allclose(np_jdet, ref_jdet, atol=1e-14)

    def test_pure_scaling(self):
        H, W = 15, 15
        yy, xx = np.mgrid[:H, :W].astype(float)
        s = 0.4
        dy, dx = s * yy, s * xx
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        ref_jdet = _reference_jdet_2d(dy, dx)
        np.testing.assert_allclose(np_jdet, ref_jdet, atol=1e-12)

    def test_rotation_field(self):
        """Small rotation: dx = -theta*y, dy = theta*x."""
        H, W = 20, 20
        yy, xx = np.mgrid[:H, :W].astype(float)
        theta = 0.05
        dx = -theta * yy
        dy = theta * xx
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        ref_jdet = _reference_jdet_2d(dy, dx)
        np.testing.assert_allclose(np_jdet, ref_jdet, atol=1e-12)

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_smooth_random(self, seed):
        dy, dx = _make_smooth_2d(20, 20, seed)
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        ref_jdet = _reference_jdet_2d(dy, dx)
        np.testing.assert_allclose(np_jdet, ref_jdet, atol=1e-10)


# ── 2D: numpy vs SimpleITK ────────────────────────────────────────────

class TestNumpyVsSimpleITK2D:
    """Compare numpy Jdet against SimpleITK's DisplacementFieldJacobianDeterminant.

    SimpleITK uses central differences for interior pixels, matching np.gradient.
    Boundary handling differs, so only interior pixels are compared.
    """

    def test_identity(self):
        dy = np.zeros((12, 12))
        dx = np.zeros((12, 12))
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        sitk_jdet = _sitk_jdet_2d(dy, dx)[1:-1, 1:-1]
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-14)

    def test_uniform_translation(self):
        dy = np.full((12, 12), 3.7)
        dx = np.full((12, 12), -1.2)
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        sitk_jdet = _sitk_jdet_2d(dy, dx)[1:-1, 1:-1]
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-14)

    def test_pure_scaling(self):
        H, W = 16, 16
        yy, xx = np.mgrid[:H, :W].astype(float)
        s = 0.5
        dy, dx = s * yy, s * xx
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        sitk_jdet = _sitk_jdet_2d(dy, dx)[1:-1, 1:-1]
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_smooth_random(self, seed):
        dy, dx = _make_smooth_2d(24, 24, seed)
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        sitk_jdet = _sitk_jdet_2d(dy, dx)[1:-1, 1:-1]
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-10)

    def test_field_with_negative_jdet(self):
        """Folding displacement — both methods should agree on negative values."""
        H, W = 12, 12
        dy = np.zeros((H, W))
        dx = np.zeros((H, W))
        dx[:, 0] = 8.0
        dx[:, -1] = -8.0
        np_jdet = _numpy_jdet_2d(dy, dx)[1:-1, 1:-1]
        sitk_jdet = _sitk_jdet_2d(dy, dx)[1:-1, 1:-1]
        # Both should detect negative Jdet
        assert np_jdet.min() < 0
        assert sitk_jdet.min() < 0
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-10)

    def test_random_dvf_from_generation(self):
        """Use the project's own DVF generator and compare via the public APIs."""
        dvf = generate_random_dvf((3, 1, 20, 20), max_magnitude=2.0, seed=123)
        np_jdet = jacobian_det2D(dvf[1:])  # (1, H, W) — pass (2,1,H,W)
        sitk_jdet = sitk_jacobian_determinant(dvf)  # (1, H, W)
        # Compare interior only
        np.testing.assert_allclose(
            np_jdet[0, 1:-1, 1:-1],
            sitk_jdet[0, 1:-1, 1:-1],
            atol=1e-10,
        )


# ── 3D: numpy vs manual reference ──────────────────────────────────────

class TestNumpyVsReference3D:
    """Compare numpy 3D Jdet against the explicit loop reference on interior voxels."""

    def test_identity(self):
        shape = (8, 8, 8)
        dz = np.zeros(shape)
        dy = np.zeros(shape)
        dx = np.zeros(shape)
        np_jdet = _numpy_jdet_3d(dz, dy, dx)[1:-1, 1:-1, 1:-1]
        ref_jdet = _reference_jdet_3d(dz, dy, dx)
        np.testing.assert_allclose(np_jdet, ref_jdet, atol=1e-14)

    def test_isotropic_scaling(self):
        D, H, W = 9, 9, 9
        zz, yy, xx = np.mgrid[:D, :H, :W].astype(float)
        s = 0.3
        dz, dy, dx = s * zz, s * yy, s * xx
        np_jdet = _numpy_jdet_3d(dz, dy, dx)[1:-1, 1:-1, 1:-1]
        ref_jdet = _reference_jdet_3d(dz, dy, dx)
        np.testing.assert_allclose(np_jdet, ref_jdet, atol=1e-12)

    def test_anisotropic_scaling(self):
        """Different scale per axis: Jdet = (1+sx)*(1+sy)*(1+sz)."""
        D, H, W = 9, 9, 9
        zz, yy, xx = np.mgrid[:D, :H, :W].astype(float)
        sx, sy, sz = 0.2, -0.1, 0.4
        dx = sx * xx
        dy = sy * yy
        dz = sz * zz
        np_jdet = _numpy_jdet_3d(dz, dy, dx)[1:-1, 1:-1, 1:-1]
        ref_jdet = _reference_jdet_3d(dz, dy, dx)
        np.testing.assert_allclose(np_jdet, ref_jdet, atol=1e-12)

    @pytest.mark.parametrize("seed", [0, 42])
    def test_smooth_random(self, seed):
        dz, dy, dx = _make_smooth_3d(10, 10, 10, seed)
        np_jdet = _numpy_jdet_3d(dz, dy, dx)[1:-1, 1:-1, 1:-1]
        ref_jdet = _reference_jdet_3d(dz, dy, dx)
        np.testing.assert_allclose(np_jdet, ref_jdet, atol=1e-10)


# ── 3D: numpy vs SimpleITK ────────────────────────────────────────────

class TestNumpyVsSimpleITK3D:
    """Compare numpy 3D Jdet against SimpleITK on interior voxels."""

    def test_identity(self):
        phi = np.zeros((3, 8, 8, 8))
        np_jdet = jacobian_det3D(phi)[1:-1, 1:-1, 1:-1]
        sitk_jdet = sitk_jacobian_determinant(phi)[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-14)

    def test_isotropic_scaling(self):
        D, H, W = 10, 10, 10
        zz, yy, xx = np.mgrid[:D, :H, :W].astype(float)
        s = 0.3
        phi = np.stack([s * zz, s * yy, s * xx])  # (3, D, H, W)
        np_jdet = jacobian_det3D(phi)[1:-1, 1:-1, 1:-1]
        sitk_jdet = sitk_jacobian_determinant(phi)[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 42])
    def test_smooth_random(self, seed):
        dz, dy, dx = _make_smooth_3d(10, 10, 10, seed)
        phi = np.stack([dz, dy, dx])
        np_jdet = jacobian_det3D(phi)[1:-1, 1:-1, 1:-1]
        sitk_jdet = sitk_jacobian_determinant(phi)[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-10)

    def test_random_dvf_3d(self):
        dvf = generate_random_dvf_3d((3, 8, 8, 8), max_magnitude=1.5, seed=77)
        np_jdet = jacobian_det3D(dvf)[1:-1, 1:-1, 1:-1]
        sitk_jdet = sitk_jacobian_determinant(dvf)[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-10)

    def test_field_with_negative_jdet_3d(self):
        """3D folding field — both methods should agree on negative values."""
        D, H, W = 8, 8, 8
        phi = np.zeros((3, D, H, W))
        phi[2, :, :, 0] = 5.0   # push x=0 plane right
        phi[2, :, :, -1] = -5.0  # push x=W-1 plane left
        np_jdet = jacobian_det3D(phi)[1:-1, 1:-1, 1:-1]
        sitk_jdet = sitk_jacobian_determinant(phi)[1:-1, 1:-1, 1:-1]
        assert np_jdet.min() < 0
        assert sitk_jdet.min() < 0
        np.testing.assert_allclose(np_jdet, sitk_jdet, atol=1e-10)


# ── Analytic sanity checks (all three methods agree) ───────────────────

class TestAnalyticCases:
    """Verify all methods produce the analytically known Jacobian determinant."""

    def test_2d_pure_shear(self):
        """dx = gamma * y, dy = 0 → Jdet = 1 everywhere."""
        H, W = 15, 15
        yy, xx = np.mgrid[:H, :W].astype(float)
        gamma = 0.3
        dx = gamma * yy
        dy = np.zeros_like(dx)
        interior = slice(1, -1)

        np_jdet = _numpy_jdet_2d(dy, dx)[interior, interior]
        ref_jdet = _reference_jdet_2d(dy, dx)
        sitk_jdet = _sitk_jdet_2d(dy, dx)[interior, interior]

        # Analytic: Jdet = (1+0)*(1+0) - gamma*0 = 1
        np.testing.assert_allclose(np_jdet, 1.0, atol=1e-12)
        np.testing.assert_allclose(ref_jdet, 1.0, atol=1e-12)
        np.testing.assert_allclose(sitk_jdet, 1.0, atol=1e-10)

    def test_2d_scaling_analytic(self):
        """dx = sx*x, dy = sy*y → Jdet = (1+sx)*(1+sy)."""
        H, W = 15, 15
        yy, xx = np.mgrid[:H, :W].astype(float)
        sx, sy = 0.5, -0.2
        dx = sx * xx
        dy = sy * yy
        expected = (1 + sx) * (1 + sy)
        interior = slice(1, -1)

        np_jdet = _numpy_jdet_2d(dy, dx)[interior, interior]
        ref_jdet = _reference_jdet_2d(dy, dx)
        sitk_jdet = _sitk_jdet_2d(dy, dx)[interior, interior]

        np.testing.assert_allclose(np_jdet, expected, atol=1e-12)
        np.testing.assert_allclose(ref_jdet, expected, atol=1e-12)
        np.testing.assert_allclose(sitk_jdet, expected, atol=1e-10)

    def test_3d_scaling_analytic(self):
        """dx=sx*x, dy=sy*y, dz=sz*z → Jdet = (1+sx)*(1+sy)*(1+sz)."""
        D, H, W = 9, 9, 9
        zz, yy, xx = np.mgrid[:D, :H, :W].astype(float)
        sx, sy, sz = 0.2, -0.1, 0.4
        dx, dy, dz = sx * xx, sy * yy, sz * zz
        expected = (1 + sx) * (1 + sy) * (1 + sz)
        interior = slice(1, -1)

        np_jdet = _numpy_jdet_3d(dz, dy, dx)[interior, interior, interior]
        ref_jdet = _reference_jdet_3d(dz, dy, dx)
        phi = np.stack([dz, dy, dx])
        sitk_jdet = sitk_jacobian_determinant(phi)[interior, interior, interior]

        np.testing.assert_allclose(np_jdet, expected, atol=1e-12)
        np.testing.assert_allclose(ref_jdet, expected, atol=1e-12)
        np.testing.assert_allclose(sitk_jdet, expected, atol=1e-10)
