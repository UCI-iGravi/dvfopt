"""Tests for laplacian.utils — sparse Laplacian matrix construction."""

import numpy as np
import pytest
import scipy.sparse

from laplacian.utils import (
    laplacianA1D,
    laplacianA2D,
    laplacianA3D,
    propagate_dirichlet_rhs,
)


class TestLaplacianA1D:
    def test_output_shape(self):
        n = 10
        A = laplacianA1D(n, np.array([0]))
        assert A.shape == (n, n)

    def test_sparse_type(self):
        A = laplacianA1D(5, np.array([0]))
        assert scipy.sparse.issparse(A)

    def test_diagonal_positive(self):
        A = laplacianA1D(8, np.array([0, 4])).toarray()
        assert np.all(np.diag(A) > 0)

    def test_boundary_row_isolated(self):
        A = laplacianA1D(8, np.array([3])).toarray()
        row = A[3].copy()
        row[3] = 0
        assert np.allclose(row, 0), "Boundary row should have no off-diagonal entries"

    def test_zero_rhs_trivial_solution(self):
        n = 10
        A = laplacianA1D(n, np.arange(3))
        b = np.zeros(n)
        x = scipy.sparse.linalg.spsolve(A.tocsc(), b)
        np.testing.assert_allclose(x, 0.0, atol=1e-10)


class TestLaplacianA2D:
    def test_output_shape(self):
        shape = (4, 5)
        N = 20
        A = laplacianA2D(shape, np.array([0]))
        assert A.shape == (N, N)

    def test_sparse_type(self):
        A = laplacianA2D((3, 3), np.array([0]))
        assert scipy.sparse.issparse(A)

    def test_diagonal_positive(self):
        A = laplacianA2D((5, 5), np.array([0, 12, 24])).toarray()
        assert np.all(np.diag(A) > 0)

    def test_boundary_row_isolated(self):
        A = laplacianA2D((4, 4), np.array([5])).toarray()
        row = A[5].copy()
        row[5] = 0
        assert np.allclose(row, 0), "Boundary row should have no off-diagonal entries"


class TestLaplacianA3D:
    def test_output_shape(self):
        shape = (1, 4, 4)
        n = shape[0] * shape[1] * shape[2]
        A = laplacianA3D(shape, np.array([0]))
        assert A.shape == (n, n)

    def test_sparse_type(self):
        shape = (1, 3, 3)
        A = laplacianA3D(shape, np.array([0]))
        assert scipy.sparse.issparse(A)

    def test_interior_rows_symmetric(self):
        """Non-boundary rows of the Laplacian should be symmetric with their columns."""
        shape = (1, 5, 5)
        n = 25
        boundary = np.array([0, 12, 24])
        A = laplacianA3D(shape, boundary).toarray()
        interior = [i for i in range(n) if i not in boundary]
        for i in interior:
            for j in interior:
                assert abs(A[i, j] - A[j, i]) < 1e-10

    def test_diagonal_positive(self):
        shape = (1, 5, 5)
        boundary = np.array([0, 12, 24])
        A = laplacianA3D(shape, boundary).toarray()
        diag = np.diag(A)
        assert np.all(diag > 0)

    def test_boundary_rows_are_isolated(self):
        """Boundary rows should have no off-diagonal entries."""
        shape = (1, 4, 4)
        bnd_idx = np.array([5])
        A = laplacianA3D(shape, bnd_idx).toarray()
        row = A[5].copy()
        row[5] = 0
        assert np.allclose(row, 0), "Boundary row should have no off-diagonal entries"

    def test_identity_solution_for_zero_rhs(self):
        """Ax = 0 should have the trivial solution for a well-conditioned system."""
        shape = (1, 5, 5)
        n = 25
        boundary = np.arange(5)
        A = laplacianA3D(shape, boundary)
        b = np.zeros(n)
        x = scipy.sparse.linalg.spsolve(A.tocsc(), b)
        np.testing.assert_allclose(x, 0.0, atol=1e-10)

    def test_spacing_changes_weights(self):
        """Anisotropic spacing should produce different off-diagonal weights."""
        shape = (2, 3, 4)
        bnd = np.array([0])
        A_iso = laplacianA3D(shape, bnd).toarray()
        A_aniso = laplacianA3D(shape, bnd, spacing=(1.0, 1.0, 2.0)).toarray()
        assert not np.allclose(A_iso, A_aniso)


class TestPropagateDirichletRhs:
    def test_modifies_rhs_in_place(self):
        shape = (1, 5, 5)
        N = 25
        bnd = np.array([12])  # center of 5x5
        rhs = np.zeros(N)
        rhs[12] = 1.0
        rhs_orig = rhs.copy()
        propagate_dirichlet_rhs(shape, bnd, rhs)
        # Neighbours of boundary should be modified
        assert not np.allclose(rhs, rhs_orig)

    def test_boundary_values_unchanged(self):
        shape = (1, 5, 5)
        N = 25
        bnd = np.array([12])
        rhs = np.zeros(N)
        rhs[12] = 1.0
        val_before = rhs[12]
        propagate_dirichlet_rhs(shape, bnd, rhs)
        assert rhs[12] == val_before
