"""Tests for laplacian.solver — Laplacian interpolation solver."""

import numpy as np
import pytest

from laplacian.solver import solveLaplacianFromCorrespondences


class TestSolveLaplacianFromCorrespondences:
    def test_output_shape(self):
        vol_shape = (1, 8, 10)
        source_pts = np.array([[0, 2, 3], [0, 5, 7]])
        target_pts = np.array([[0, 1, 2], [0, 4, 6]])
        deformation = solveLaplacianFromCorrespondences(vol_shape, source_pts, target_pts)
        assert deformation.shape == (3, 1, 8, 10)

    def test_zero_displacement_at_correspondences(self):
        """Where source==target, displacement should be ~0."""
        vol_shape = (1, 10, 10)
        pts = np.array([[0, 3, 3], [0, 7, 7]])
        deformation = solveLaplacianFromCorrespondences(vol_shape, pts, pts)
        for p in pts:
            z, y, x = p
            assert abs(deformation[1, z, y, x]) < 0.1
            assert abs(deformation[2, z, y, x]) < 0.1

    def test_dz_channel_is_zero(self):
        """For 2D slices with default axes=(1,2), the dz channel should be zero."""
        vol_shape = (1, 8, 8)
        source_pts = np.array([[0, 2, 2]])
        target_pts = np.array([[0, 4, 4]])
        deformation = solveLaplacianFromCorrespondences(vol_shape, source_pts, target_pts)
        np.testing.assert_array_equal(deformation[0], 0.0)

    def test_lgmres_solver(self):
        """The lgmres solver variant should also produce valid output."""
        vol_shape = (1, 8, 8)
        source_pts = np.array([[0, 2, 2]])
        target_pts = np.array([[0, 4, 4]])
        deformation = solveLaplacianFromCorrespondences(
            vol_shape, source_pts, target_pts, solver_method='lgmres'
        )
        assert deformation.shape == (3, 1, 8, 8)
        np.testing.assert_array_equal(deformation[0], 0.0)
