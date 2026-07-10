"""Tests for dvfopt.io.nifti — NIfTI loading utilities."""

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
from dvfopt.io.nifti import load_nii_images  # noqa: E402


@pytest.fixture
def nii_path(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    path = tmp_path / "vol.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return str(path), data


class TestLoadNiiImages:
    def test_load_single_path_returns_array(self, nii_path):
        path, data = nii_path
        out = load_nii_images([path])
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, data)

    def test_loaded_array_is_writable(self, nii_path):
        """Loaded arrays are fresh allocations owned by the caller."""
        path, _ = nii_path
        out = load_nii_images([path])
        out[0, 0, 0] = -1.0  # must not raise

    def test_ndarray_input_returns_copy(self):
        """Caller-supplied ndarrays must be copied, never aliased."""
        arr = np.ones((2, 3, 4))
        out = load_nii_images([arr])
        np.testing.assert_array_equal(out, arr)
        out[0, 0, 0] = 99.0
        assert arr[0, 0, 0] == 1.0, "mutating the output must not touch the input"

    def test_multiple_images_returns_list(self, nii_path):
        path, data = nii_path
        arr = np.zeros((2, 3, 4))
        out = load_nii_images([path, arr])
        assert isinstance(out, list)
        assert len(out) == 2
        np.testing.assert_allclose(out[0], data)
        np.testing.assert_array_equal(out[1], arr)

    def test_scale_same_zooms_is_identity(self, nii_path):
        path, data = nii_path
        out = load_nii_images([path, path], scale=True)
        np.testing.assert_allclose(out[0], data)
        np.testing.assert_allclose(out[1], data)

    def test_scale_with_ndarray_first_disables_scaling(self):
        arr = np.ones((2, 3, 4))
        out = load_nii_images([arr], scale=True)
        np.testing.assert_array_equal(out, arr)
