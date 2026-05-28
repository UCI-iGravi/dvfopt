"""Tests for ``triangle_constraint_jacobian_2d`` — the 4-triangle-per-cell
analytical Jacobian, newly vectorised.

Covers two things:

1. Standard correctness vs. numerical finite-difference of the constraint
   value function ``triangle_constraint``.
2. *Vertex coverage*: in a 4-triangle-per-cell scheme over an (H, W) DVF,
   every grid vertex — *including vertices on the boundary of the field* —
   must participate in at least two triangle constraints, so that no
   vertex can fold without one of the triangle areas going negative.
"""

import numpy as np
import pytest

from dvfopt.jacobian.shoelace import triangle_constraint
from dvfopt.core.slsqp.gradients import triangle_constraint_jacobian_2d


def _numerical_jacobian(func, x0, eps=1e-6):
    f0 = func(x0)
    n_out = len(f0)
    n_in = len(x0)
    J = np.zeros((n_out, n_in))
    for j in range(n_in):
        x_plus = x0.copy(); x_plus[j] += eps
        x_minus = x0.copy(); x_minus[j] -= eps
        J[:, j] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return J


# ---------------------------------------------------------------------------
# Math correctness vs. finite-difference reference
# ---------------------------------------------------------------------------

class TestTriangleJacobianValues:
    def test_matches_finite_diff_identity(self):
        sy, sx = 5, 5
        phi = np.zeros(2 * sy * sx)
        analytical = triangle_constraint_jacobian_2d(phi, (sy, sx),
                                                    exclude_boundaries=False)
        numerical = _numerical_jacobian(
            lambda p: triangle_constraint(p, (sy, sx), exclude_boundaries=False),
            phi)
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=1e-5)

    def test_matches_finite_diff_random(self):
        sy, sx = 6, 7
        rng = np.random.default_rng(42)
        phi = rng.standard_normal(2 * sy * sx) * 0.15
        analytical = triangle_constraint_jacobian_2d(phi, (sy, sx),
                                                    exclude_boundaries=False)
        numerical = _numerical_jacobian(
            lambda p: triangle_constraint(p, (sy, sx), exclude_boundaries=False),
            phi)
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=2e-4)

    def test_matches_finite_diff_exclude_boundaries(self):
        sy, sx = 6, 6
        rng = np.random.default_rng(7)
        phi = rng.standard_normal(2 * sy * sx) * 0.1
        analytical = triangle_constraint_jacobian_2d(phi, (sy, sx),
                                                    exclude_boundaries=True)
        numerical = _numerical_jacobian(
            lambda p: triangle_constraint(p, (sy, sx), exclude_boundaries=True),
            phi)
        np.testing.assert_allclose(analytical.toarray(), numerical, atol=2e-4)


# ---------------------------------------------------------------------------
# Row count and structure
# ---------------------------------------------------------------------------

class TestTriangleJacobianStructure:
    def test_row_count_full_cells(self):
        sy, sx = 5, 4
        phi = np.zeros(2 * sy * sx)
        J = triangle_constraint_jacobian_2d(phi, (sy, sx),
                                            exclude_boundaries=False)
        # 4 triangles per cell, (sy-1) * (sx-1) cells.
        assert J.shape == (4 * (sy - 1) * (sx - 1), 2 * sy * sx)

    def test_row_count_exclude_boundaries(self):
        sy, sx = 6, 7
        phi = np.zeros(2 * sy * sx)
        J = triangle_constraint_jacobian_2d(phi, (sy, sx),
                                            exclude_boundaries=True)
        # Interior cells in [1, sy-2) x [1, sx-2) -> (sy-3) * (sx-3) cells.
        assert J.shape == (4 * (sy - 3) * (sx - 3), 2 * sy * sx)

    def test_zero_rows_when_grid_too_small(self):
        # 2x2 grid has 1 cell; excluding boundaries removes it.
        phi = np.zeros(2 * 4)
        J = triangle_constraint_jacobian_2d(phi, (2, 2),
                                            exclude_boundaries=True)
        assert J.shape == (0, 8)


# ---------------------------------------------------------------------------
# Vertex coverage — the property the user specifically asked us to verify:
# in a DVF with full triangle coverage, every grid vertex (including
# boundary vertices) participates in at least 2 triangle constraints.
# ---------------------------------------------------------------------------

class TestTriangleVertexCoverage:
    """In the 4-tri-per-cell scheme each cell vertex participates in 3 of the
    4 cell triangles. So for a vertex of the DVF that touches K cells, the
    coverage is 3*K triangles. Vertices on the field boundary touch fewer
    cells than interior ones, but with K >= 1 (every vertex is in at least
    one cell) the coverage is >= 3. We assert >= 2 here for safety margin."""

    def _cell_count_for_vertex(self, r, c, H, W):
        """Number of (H-1)x(W-1) cells whose corners include vertex (r, c)."""
        # A vertex (r, c) is a corner of up to 4 cells:
        # cell (r-1, c-1) BR, cell (r-1, c) BL, cell (r, c-1) TR, cell (r, c) TL.
        count = 0
        for dr, dc in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
            cr, cc = r + dr, c + dc
            if 0 <= cr < H - 1 and 0 <= cc < W - 1:
                count += 1
        return count

    def _vertex_to_triangle_count(self, H, W, exclude_boundaries):
        """Build the (4*n_cells, 2*H*W) Jacobian on an identity field and
        count, for each vertex (r, c), how many triangle rows it has a
        non-zero entry in (combining dx and dy column blocks).
        """
        phi = np.zeros(2 * H * W)
        J = triangle_constraint_jacobian_2d(phi, (H, W),
                                            exclude_boundaries=exclude_boundaries)
        # Identity field has all triangle areas == 0.5, derivatives are non-zero.
        # Look up via lil for easy column access.
        J_lil = J.tolil()
        coverage = np.zeros((H, W), dtype=int)
        for r in range(H):
            for c in range(W):
                lin = r * W + c
                # dx column block: 0..H*W. dy column block: H*W..2*H*W.
                col_dx = lin
                col_dy = H * W + lin
                # Count rows in which EITHER dx or dy of this vertex has nz.
                col_csc = J.tocsc()
                rows_dx = set(col_csc.getcol(col_dx).nonzero()[0].tolist())
                rows_dy = set(col_csc.getcol(col_dy).nonzero()[0].tolist())
                coverage[r, c] = len(rows_dx | rows_dy)
        return coverage

    def test_every_vertex_in_at_least_two_triangles_full(self):
        """exclude_boundaries=False: every vertex of an H x W DVF (with H, W >= 2)
        belongs to >= 1 cell, contributing >= 3 triangles. So minimum coverage = 3."""
        H, W = 5, 6
        coverage = self._vertex_to_triangle_count(H, W, exclude_boundaries=False)
        # Specifically check corner vertices — they touch only 1 cell.
        for (r, c) in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
            assert coverage[r, c] >= 2, (
                f"corner vertex ({r},{c}) has coverage {coverage[r, c]}; "
                f"expected >= 2")
        # Edge vertices touch 2 cells, so >= 6.
        for c in range(1, W - 1):
            assert coverage[0, c] >= 6
            assert coverage[H - 1, c] >= 6
        for r in range(1, H - 1):
            assert coverage[r, 0] >= 6
            assert coverage[r, W - 1] >= 6
        # Interior: 4 cells, >= 12.
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                assert coverage[r, c] >= 12

    def test_cell_count_helper_matches_geometric_expectation(self):
        H, W = 4, 5
        # Corners: 1 cell each.
        assert self._cell_count_for_vertex(0, 0, H, W) == 1
        assert self._cell_count_for_vertex(H - 1, W - 1, H, W) == 1
        # Edge: 2 cells.
        assert self._cell_count_for_vertex(0, 2, H, W) == 2
        # Interior: 4 cells.
        assert self._cell_count_for_vertex(1, 1, H, W) == 4
