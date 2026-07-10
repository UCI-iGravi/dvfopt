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

    def test_nonconvergence_emits_warning(self):
        """Regression: CG hitting maxiter without converging was only sent
        to a (default no-op) logger — it must raise a RuntimeWarning."""
        vol_shape = (1, 16, 16)
        source_pts = np.array([[0, 2, 2], [0, 12, 13], [0, 5, 9]])
        target_pts = np.array([[0, 4, 4], [0, 10, 11], [0, 7, 7]])
        with pytest.warns(RuntimeWarning, match="did NOT converge"):
            solveLaplacianFromCorrespondences(
                vol_shape, source_pts, target_pts, rtol=1e-14, maxiter=1
            )

    def test_return_info_nonconverged(self):
        """return_info=True must expose the scipy info flag programmatically."""
        vol_shape = (1, 16, 16)
        source_pts = np.array([[0, 2, 2], [0, 12, 13], [0, 5, 9]])
        target_pts = np.array([[0, 4, 4], [0, 10, 11], [0, 7, 7]])
        with pytest.warns(RuntimeWarning):
            deformation, info = solveLaplacianFromCorrespondences(
                vol_shape, source_pts, target_pts, rtol=1e-14, maxiter=1, return_info=True
            )
        assert deformation.shape == (3, 1, 16, 16)
        assert set(info.keys()) == {"dy", "dx"}
        assert any(flag != 0 for flag in info.values())

    def test_return_info_converged(self):
        vol_shape = (1, 10, 10)
        pts = np.array([[0, 3, 3], [0, 7, 7]])
        deformation, info = solveLaplacianFromCorrespondences(vol_shape, pts, pts, return_info=True)
        assert deformation.shape == (3, 1, 10, 10)
        assert all(flag == 0 for flag in info.values())

    def test_default_return_is_single_value(self):
        """Backward compatibility: without return_info the return value is
        the bare field (existing callers unpack a single value)."""
        vol_shape = (1, 8, 8)
        pts = np.array([[0, 2, 2]])
        result = solveLaplacianFromCorrespondences(vol_shape, pts, pts)
        assert isinstance(result, np.ndarray)

    def test_return_info_empty_correspondences(self):
        deformation, info = solveLaplacianFromCorrespondences(
            (1, 8, 8), np.empty((0, 3)), np.empty((0, 3)), return_info=True
        )
        assert deformation.shape == (3, 1, 8, 8)
        assert info == {}
