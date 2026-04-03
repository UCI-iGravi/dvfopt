"""Tests for dvfopt.laplacian.solver — Laplacian interpolation solver."""

import numpy as np
import pytest

from dvfopt.laplacian.solver import (
    _prepare_correspondence_data,
    sliceToSlice3DLaplacian,
    createA,
)


class TestPrepareCorrespondenceData:
    def test_basic(self):
        shape = (1, 5, 5)
        mpoints = np.array([[0, 2, 3]])
        fpoints = np.array([[0, 1, 1]])
        Xd, Yd, Zd, bnd = _prepare_correspondence_data(shape, mpoints, fpoints)
        assert len(Xd) == 1 * 5 * 5
        assert len(bnd) == 1
        # Displacement = mpoint - fpoint
        idx = 0 * 5 * 5 + 1 * 5 + 1
        np.testing.assert_allclose(Xd[idx], 0 - 0)   # dz
        np.testing.assert_allclose(Yd[idx], 2 - 1)    # dy
        np.testing.assert_allclose(Zd[idx], 3 - 1)    # dx

    def test_multiple_correspondences(self):
        shape = (1, 10, 10)
        mpoints = np.array([[0, 3, 4], [0, 7, 8]])
        fpoints = np.array([[0, 1, 2], [0, 5, 6]])
        Xd, Yd, Zd, bnd = _prepare_correspondence_data(shape, mpoints, fpoints)
        assert len(bnd) == 2

    def test_no_correspondences_empty_boundary(self):
        shape = (1, 5, 5)
        mpoints = np.array([]).reshape(0, 3)
        fpoints = np.array([]).reshape(0, 3)
        Xd, Yd, Zd, bnd = _prepare_correspondence_data(shape, mpoints, fpoints)
        assert len(bnd) == 0
        np.testing.assert_array_equal(Xd, 0.0)


class TestSliceToSlice3DLaplacian:
    def test_output_shape(self):
        fixed = np.zeros((1, 8, 10))
        mpoints = np.array([[0, 2, 3], [0, 5, 7]])
        fpoints = np.array([[0, 1, 2], [0, 4, 6]])
        deformation, A, Xd, Yd, Zd = sliceToSlice3DLaplacian(fixed, mpoints, fpoints)
        assert deformation.shape == (3, 1, 8, 10)

    def test_zero_displacement_at_correspondences(self):
        """Where m==f, displacement should be ~0."""
        fixed = np.zeros((1, 10, 10))
        pts = np.array([[0, 3, 3], [0, 7, 7]])
        deformation, _, _, _, _ = sliceToSlice3DLaplacian(fixed, pts, pts)
        # At correspondence points, displacement should be near zero
        for p in pts:
            z, y, x = p
            assert abs(deformation[1, z, y, x]) < 0.1
            assert abs(deformation[2, z, y, x]) < 0.1

    def test_dz_channel_is_zero(self):
        """For 2D slices, the dz channel should be zero."""
        fixed = np.zeros((1, 8, 8))
        mpoints = np.array([[0, 2, 2]])
        fpoints = np.array([[0, 4, 4]])
        deformation, _, _, _, _ = sliceToSlice3DLaplacian(fixed, mpoints, fpoints)
        np.testing.assert_array_equal(deformation[0], 0.0)


class TestCreateA:
    def test_output_is_sparse(self):
        import scipy.sparse
        fixed = np.zeros((1, 5, 5))
        mpoints = np.array([[0, 2, 2]])
        fpoints = np.array([[0, 1, 1]])
        A = createA(fixed, mpoints, fpoints)
        assert scipy.sparse.issparse(A)

    def test_shape(self):
        fixed = np.zeros((1, 5, 5))
        mpoints = np.array([[0, 2, 2]])
        fpoints = np.array([[0, 1, 1]])
        A = createA(fixed, mpoints, fpoints)
        n = 1 * 5 * 5
        assert A.shape == (n, n)

    def test_no_correspondences_flag(self):
        fixed = np.zeros((1, 5, 5))
        mpoints = np.array([[0, 2, 2]])
        fpoints = np.array([[0, 1, 1]])
        A_with = createA(fixed, mpoints, fpoints, use_correspondences=True)
        A_without = createA(fixed, mpoints, fpoints, use_correspondences=False)
        assert A_without.dtype == np.int8
