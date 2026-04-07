"""Tests for laplacian_interp.matrix — sparse Laplacian matrix construction."""

import numpy as np
import pytest
import scipy.sparse

from laplacian_interp.matrix import (
    get_laplacian_index,
    get_adjacent_indices,
    laplacian_a_3d,
)


class TestGetLaplacianIndex:
    def test_origin(self):
        assert get_laplacian_index(0, 0, 0, (2, 3, 4)) == 0

    def test_x_varies_fastest(self):
        # (0,0,1) → 1; (0,1,0) → 4; (1,0,0) → 12 for shape (2,3,4)
        assert get_laplacian_index(0, 0, 1, (2, 3, 4)) == 1
        assert get_laplacian_index(0, 1, 0, (2, 3, 4)) == 4
        assert get_laplacian_index(1, 0, 0, (2, 3, 4)) == 12

    def test_last_element(self):
        shape = (2, 3, 4)
        assert get_laplacian_index(1, 2, 3, shape) == 2 * 3 * 4 - 1

    def test_consistent_formula(self):
        shape = (3, 5, 7)
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    idx = get_laplacian_index(z, y, x, shape)
                    assert idx == z * shape[1] * shape[2] + y * shape[2] + x


class TestGetAdjacentIndices:
    def test_interior_has_four_neighbors(self):
        shape = (1, 5, 5)
        adj = get_adjacent_indices(0, 2, 2, shape)
        assert all(a is not None for a in adj)

    def test_corner_has_two_neighbors(self):
        shape = (1, 5, 5)
        adj = get_adjacent_indices(0, 0, 0, shape)
        # Left and up are None
        assert adj[0] is None  # left
        assert adj[2] is None  # up
        assert adj[1] is not None  # right
        assert adj[3] is not None  # down

    def test_edge_has_three_neighbors(self):
        shape = (1, 5, 5)
        adj = get_adjacent_indices(0, 0, 2, shape)  # top row, middle col
        assert adj[2] is None  # up is None
        assert adj[0] is not None  # left
        assert adj[1] is not None  # right
        assert adj[3] is not None  # down

    def test_neighbor_indices_correct(self):
        shape = (1, 5, 5)
        adj = get_adjacent_indices(0, 2, 3, shape)
        # left: (0,2,2), right: (0,2,4), up: (0,1,3), down: (0,3,3)
        assert adj[0] == get_laplacian_index(0, 2, 2, shape)
        assert adj[1] == get_laplacian_index(0, 2, 4, shape)
        assert adj[2] == get_laplacian_index(0, 1, 3, shape)
        assert adj[3] == get_laplacian_index(0, 3, 3, shape)


class TestLaplacianA3D:
    def test_output_shape(self):
        shape = (1, 4, 4)
        n = shape[0] * shape[1] * shape[2]
        A = laplacian_a_3d(shape, np.array([0]))
        assert A.shape == (n, n)

    def test_sparse_type(self):
        shape = (1, 3, 3)
        A = laplacian_a_3d(shape, np.array([0]))
        assert scipy.sparse.issparse(A)

    def test_interior_rows_symmetric(self):
        """Non-boundary rows of the Laplacian should be symmetric with their columns."""
        shape = (1, 5, 5)
        n = 25
        boundary = np.array([0, 12, 24])
        A = laplacian_a_3d(shape, boundary).toarray()
        # Check symmetry for non-boundary interior rows
        interior = [i for i in range(n) if i not in boundary]
        for i in interior:
            for j in interior:
                assert abs(A[i, j] - A[j, i]) < 1e-10

    def test_diagonal_positive(self):
        """Diagonal elements should all be positive."""
        shape = (1, 5, 5)
        boundary = np.array([0, 12, 24])
        A = laplacian_a_3d(shape, boundary).toarray()
        diag = np.diag(A)
        assert np.all(diag > 0)

    def test_boundary_rows_are_isolated(self):
        """Boundary rows with correspondences should have no off-diagonal entries."""
        shape = (1, 4, 4)
        bnd_idx = np.array([5])
        A = laplacian_a_3d(shape, bnd_idx, use_correspondences=True).toarray()
        # Boundary row should only have diagonal entry (off-diag zeroed by bnd mask)
        row = A[5].copy()
        row[5] = 0
        assert np.allclose(row, 0), "Boundary row should have no off-diagonal entries"

    def test_no_correspondences_uses_int8(self):
        shape = (1, 3, 3)
        A = laplacian_a_3d(shape, np.array([0]), use_correspondences=False)
        assert A.dtype == np.int8

    def test_identity_solution_for_zero_rhs(self):
        """Ax = 0 should have the trivial solution for a well-conditioned system."""
        shape = (1, 5, 5)
        n = 25
        boundary = np.arange(5)  # first row as boundary
        A = laplacian_a_3d(shape, boundary)
        b = np.zeros(n)
        x = scipy.sparse.linalg.spsolve(A.tocsc(), b)
        np.testing.assert_allclose(x, 0.0, atol=1e-10)
