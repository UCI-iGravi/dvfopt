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
