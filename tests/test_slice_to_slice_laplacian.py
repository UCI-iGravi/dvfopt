"""Smoke tests for ``sliceToSlice3DLaplacian``.

Listed as a primary entry point in CLAUDE.md but previously not exercised
by any test. Guards two specific bugs the review found:
- output ``deformationField`` was hardcoded ``np.float32`` regardless of
  ``solver_dtype`` (silent precision loss),
- ``cg``/``lgmres`` were called with hardcoded ``rtol=`` (broken on
  SciPy < 1.12; we now use the inspect-based shim).
"""

import numpy as np
import pytest

# sliceToSlice3DLaplacian requires skimage and other heavy deps used only
# in the registration pipeline; skip cleanly if not installed.
skimage = pytest.importorskip("skimage")
from laplacian.correspondence import sliceToSlice3DLaplacian  # noqa: E402


def _two_circle_volumes():
    """Return (fixed, moving) tiny 3D test volumes with a single slice each
    containing offset circles, so the slice-to-slice path has something to
    register."""
    D, H, W = 1, 32, 32
    fixed = np.zeros((D, H, W), dtype=np.float32)
    moving = np.zeros((D, H, W), dtype=np.float32)

    yy, xx = np.mgrid[:H, :W]
    fixed[0] = ((yy - 16) ** 2 + (xx - 16) ** 2 <= 8**2).astype(np.float32)
    moving[0] = ((yy - 16) ** 2 + (xx - 18) ** 2 <= 8**2).astype(np.float32)
    return fixed, moving


class TestSliceToSlice3DLaplacian:
    def test_runs_with_default_settings(self):
        fixed, moving = _two_circle_volumes()
        out = sliceToSlice3DLaplacian(
            fixed,
            moving,
            rtol=1e-2,
            maxiter=50,
            log_fn=lambda *_a, **_k: None,
        )
        # Returned shape is (nd, D, H, W) with nd=3 channels [dz, dy, dx].
        assert out.ndim == 4
        assert out.shape[0] == 3
        assert out.shape[1:] == fixed.shape

    def test_solver_dtype_float64_is_honoured(self):
        """Regression: previously the output buffer was hardcoded float32."""
        fixed, moving = _two_circle_volumes()
        out = sliceToSlice3DLaplacian(
            fixed,
            moving,
            rtol=1e-2,
            maxiter=50,
            solver_dtype="float64",
            log_fn=lambda *_a, **_k: None,
        )
        assert out.dtype == np.float64

    def test_solver_dtype_float32_path(self):
        fixed, moving = _two_circle_volumes()
        out = sliceToSlice3DLaplacian(
            fixed,
            moving,
            rtol=1e-2,
            maxiter=50,
            solver_dtype="float32",
            log_fn=lambda *_a, **_k: None,
        )
        assert out.dtype == np.float32

    def test_dz_channel_is_zero(self):
        """sliceToSlice3DLaplacian operates in-plane; dz should be zero."""
        fixed, moving = _two_circle_volumes()
        out = sliceToSlice3DLaplacian(
            fixed,
            moving,
            rtol=1e-2,
            maxiter=50,
            solver_dtype="float64",
            log_fn=lambda *_a, **_k: None,
        )
        np.testing.assert_array_equal(out[0], 0.0)


def _asymmetric_circle_volumes():
    """(fixed, moving) volumes of asymmetric shape (2, 64, 80): two identical
    slices along axis 0, each containing offset circles (moving shifted +2
    along the slice-column direction).

    The circle radius must be large enough that its contour survives the
    <100-pixel component filter in ``getDataContours`` (perimeter of r=20
    is ~126 px)."""
    D, H, W = 2, 64, 80
    fixed = np.zeros((D, H, W), dtype=np.float32)
    moving = np.zeros((D, H, W), dtype=np.float32)
    yy, xx = np.mgrid[:H, :W]
    fcirc = ((yy - 32) ** 2 + (xx - 40) ** 2 <= 20**2).astype(np.float32)
    mcirc = ((yy - 32) ** 2 + (xx - 42) ** 2 <= 20**2).astype(np.float32)
    fixed[:] = fcirc
    moving[:] = mcirc
    return fixed, moving


class TestSliceToSlice3DLaplacianAxis:
    """Regression tests for the `axis` parameter.

    Previously the flat-index formula and the dy/dx channel assignment
    hard-coded axis=0 ordering, so axis=1/2 silently scattered boundary
    conditions to wrong voxels and wrote displacements to wrong channels.

    Strategy: run axis=0 on a reference volume, then present the *same*
    slices to the pipeline via a moveaxis'd volume with axis=1/2.  The
    correspondences are identical and the Laplacian systems are related by
    an exact grid permutation, so each displacement value must land at the
    permuted voxel and in the remapped channel (up to CG solve tolerance —
    hence the tight rtol below and the loose allclose atol).
    """

    _kw = dict(rtol=1e-10, maxiter=5000, solver_dtype="float64", log_fn=lambda *_a, **_k: None)

    def _reference(self):
        fixed, moving = _asymmetric_circle_volumes()
        out0 = sliceToSlice3DLaplacian(fixed, moving, axis=0, **self._kw)
        return fixed, moving, out0

    def test_axis0_regression(self):
        """axis=0 behaviour unchanged: dy/dx in channels 1/2, dz zero,
        moving circle shifted +x => dx positive near the contour."""
        _, _, out0 = self._reference()
        assert out0.shape == (3, 2, 64, 80)
        np.testing.assert_array_equal(out0[0], 0.0)
        assert np.abs(out0[2]).max() > 0.5, "expected in-plane dx displacement"
        assert out0[2].sum() > 0, "moving shifted +x => net dx positive"

    def test_axis1_matches_permuted_axis0(self):
        fixed, moving, out0 = self._reference()
        # New volume: (64, 2, 80); slices along axis=1 are the same images.
        fixed1 = np.moveaxis(fixed, 0, 1).copy()
        moving1 = np.moveaxis(moving, 0, 1).copy()
        out1 = sliceToSlice3DLaplacian(fixed1, moving1, axis=1, **self._kw)

        assert out1.shape == (3, 64, 2, 80)
        # Slice axis (volume axis 1) must carry no displacement.
        np.testing.assert_array_equal(out1[1], 0.0)
        # In-plane rows = volume axis 0, cols = volume axis 2.
        np.testing.assert_allclose(out1[0], np.moveaxis(out0[1], 0, 1), atol=1e-5)
        np.testing.assert_allclose(out1[2], np.moveaxis(out0[2], 0, 1), atol=1e-5)
        # Planted correspondence sanity: +x circle shift shows up in channel 2.
        assert np.abs(out1[2]).max() > 0.5

    def test_axis2_matches_permuted_axis0(self):
        fixed, moving, out0 = self._reference()
        # New volume: (64, 80, 2); slices along axis=2 are the same images.
        fixed2 = np.moveaxis(fixed, 0, 2).copy()
        moving2 = np.moveaxis(moving, 0, 2).copy()
        out2 = sliceToSlice3DLaplacian(fixed2, moving2, axis=2, **self._kw)

        assert out2.shape == (3, 64, 80, 2)
        # Slice axis (volume axis 2) must carry no displacement.
        np.testing.assert_array_equal(out2[2], 0.0)
        # In-plane rows = volume axis 0, cols = volume axis 1.
        np.testing.assert_allclose(out2[0], np.moveaxis(out0[1], 0, 2), atol=1e-5)
        np.testing.assert_allclose(out2[1], np.moveaxis(out0[2], 0, 2), atol=1e-5)
        # The circle shift is along the slice-column direction = volume axis 1.
        assert np.abs(out2[1]).max() > 0.5

    def test_invalid_axis_raises(self):
        fixed, moving = _asymmetric_circle_volumes()
        with pytest.raises(ValueError, match="axis"):
            sliceToSlice3DLaplacian(fixed, moving, axis=3, **self._kw)
