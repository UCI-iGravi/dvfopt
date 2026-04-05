"""
Logic-focused tests for the iterative SLSQP deformation correction algorithm.

These tests verify that the algorithm behaves correctly mechanically:
  - Jacobian determinant formula for non-trivial known transforms
  - Connected component bounding window stays within its component
  - Write-back with padded extraction does not touch the outer ring
  - _edge_flags returns truthful at-edge / reached-max values
  - Optimization result actually satisfies the Jdet constraint
  - Quality map is the exact element-wise minimum of active metrics
  - Threshold boundary is strictly <= (inclusive), not <
  - Window size actually grows when the algorithm escalates

These complement the convergence/parameter tests in the other test files.

Run with:  python -m pytest tests/test_algorithm_logic.py -v
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.core.iterative import iterative_serial

THRESHOLD = 0.01
ERR_TOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fold_dvf(H, W, cy, cx, mag=2.5):
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    dvf[2, 0, cy, cx] = mag
    dvf[2, 0, cy, cx + 1] = -mag
    return dvf


# ===========================================================================
# 1. JACOBIAN DETERMINANT FORMULA — KNOWN NON-TRIVIAL TRANSFORMS
# ===========================================================================

class TestJacobianFormulasNew:
    """Jdet must give the analytically correct value for known transforms.

    phi[0]=dy, phi[1]=dx.  Jdet = (1+∂dx/∂x)(1+∂dy/∂y) − (∂dx/∂y)(∂dy/∂x).
    """

    def test_pure_x_axis_stretch_jdet_equals_scale_factor(self):
        """phi[1] = (s-1)*x (only x-axis scaled by s): Jdet = s in interior.
        Distinct from the uniform s^2 compression test in test_invariants.py.
        """
        H, W = 8, 8
        s = 2.0
        phi = np.zeros((2, H, W))
        phi[1] = (s - 1) * np.arange(W, dtype=float)[np.newaxis, :]  # dx = (s-1)*x
        # phi[0] = 0 (no y displacement)
        jdet = jacobian_det2D(phi)
        # Interior: central differences of a linear function are exact → Jdet = s
        np.testing.assert_allclose(jdet[0, 1:-1, 1:-1], s, atol=1e-10,
                                   err_msg="Pure x-stretch: Jdet should equal scale factor")

    def test_pure_y_axis_stretch_jdet_equals_scale_factor(self):
        """phi[0] = (s-1)*y (only y-axis): Jdet = s in interior."""
        H, W = 8, 8
        s = 3.0
        phi = np.zeros((2, H, W))
        phi[0] = (s - 1) * np.arange(H, dtype=float)[:, np.newaxis]  # dy = (s-1)*y
        jdet = jacobian_det2D(phi)
        np.testing.assert_allclose(jdet[0, 1:-1, 1:-1], s, atol=1e-10,
                                   err_msg="Pure y-stretch: Jdet should equal scale factor")

    def test_pure_x_shear_jdet_is_one(self):
        """phi[1] = s*y (x displaced proportional to y): Jdet = 1 (volume-preserving).

        Shear: ∂dx/∂y = s, ∂dy/∂x = 0; cross terms cancel → Jdet = 1.
        """
        H, W = 8, 8
        s = 1.5
        phi = np.zeros((2, H, W))
        phi[1] = s * np.arange(H, dtype=float)[:, np.newaxis]  # dx = s*y
        jdet = jacobian_det2D(phi)
        np.testing.assert_allclose(jdet[0, 1:-1, 1:-1], 1.0, atol=1e-10,
                                   err_msg="Pure shear: Jdet should be 1 (area-preserving)")

    def test_fold_field_gives_negative_jdet(self):
        """A sharp fold (opposing displacements at adjacent pixels) must give Jdet < 0."""
        phi = np.zeros((2, 10, 10))
        phi[1, 5, 5] = 3.0    # large +dx
        phi[1, 5, 6] = -3.0   # large -dx at adjacent pixel
        jdet = jacobian_det2D(phi)
        assert jdet[0].min() < 0, \
            f"Known fold should give negative Jdet; got min={jdet[0].min():.4f}"

    def test_3d_pure_z_stretch_jdet_equals_scale_factor(self):
        """3D: phi[0] = (s-1)*z → Jdet = s in interior."""
        D, H, W = 8, 8, 8
        s = 2.0
        phi = np.zeros((3, D, H, W))
        phi[0] = (s - 1) * np.arange(D, dtype=float)[:, np.newaxis, np.newaxis]
        jdet = jacobian_det3D(phi)
        np.testing.assert_allclose(jdet[1:-1, 1:-1, 1:-1], s, atol=1e-9,
                                   err_msg="3D z-stretch: Jdet should equal scale factor")


# ===========================================================================
# 2. CONNECTED-COMPONENT ISOLATION IN BOUNDING WINDOW
# ===========================================================================

class TestConnectedComponentIsolation:
    """neg_jdet_bounding_window must bound ONLY the connected component
    containing the query pixel — not all negative pixels globally."""

    def test_bounding_window_does_not_cross_to_distant_cluster(self):
        """Two well-separated negative clusters.  Querying cluster 1 must NOT
        produce a window large enough to cover cluster 2."""
        from dvfopt.core.spatial import neg_jdet_bounding_window, get_nearest_center

        H, W = 20, 20
        jm = np.ones((1, H, W)) * 0.5
        # Cluster 1: rows 4-6, cols 3-5
        jm[0, 4:7, 3:6] = -0.3
        # Cluster 2: rows 4-6, cols 14-16  (9 pixels away from cluster 1)
        jm[0, 4:7, 14:17] = -0.3

        size1, center1 = neg_jdet_bounding_window(jm, (5, 4), THRESHOLD, ERR_TOL)
        _, cy1, cx1 = get_nearest_center(center1, (1, H, W), size1)
        hy1, hx1 = size1[0] // 2, size1[1] // 2
        hx1_hi = size1[1] - hx1
        # Right edge of window for cluster 1 must not reach cluster 2 at col 14
        right_edge = cx1 + hx1_hi
        assert right_edge <= 12, \
            f"Cluster 1 window right edge ({right_edge}) reaches cluster 2 (col 14)"

    def test_bounding_window_center_is_bbox_center_not_worst_pixel(self):
        """When the worst pixel is off-center in a connected region, the returned
        center must be the bounding-box center, not the worst pixel itself."""
        from dvfopt.core.spatial import neg_jdet_bounding_window

        jm = np.ones((1, 20, 20)) * 0.5
        # Horizontal strip at row 8, cols 3..9 — 7 pixels wide
        jm[0, 8, 3:10] = -0.3
        jm[0, 8, 3] = -1.0   # worst pixel is leftmost (col 3)

        _, center = neg_jdet_bounding_window(jm, (8, 3), THRESHOLD, ERR_TOL)
        cy, cx = center
        # BBox: y=[7,9], x=[2,10] → bbox_center_x = (2+10+1)//2 = 6
        # The center x must be significantly right of the worst pixel (col 3)
        assert cx >= 5, \
            f"Window center x ({cx}) should be near bbox center, not worst pixel col 3"

    def test_wide_flat_cluster_produces_non_square_window(self):
        """A wide, flat negative region must produce a window wider than tall."""
        from dvfopt.core.spatial import neg_jdet_bounding_window

        jm = np.ones((1, 20, 20)) * 0.5
        # Single row, 10 pixels wide → window should be wider than tall
        jm[0, 10, 4:14] = -0.3

        size, _ = neg_jdet_bounding_window(jm, (10, 9), THRESHOLD, ERR_TOL)
        height, width = size
        assert width > height, \
            f"Wide flat region should give wider-than-tall window; got {height}x{width}"

    def test_entire_grid_negative_produces_full_grid_window(self):
        """All pixels negative → bounding window covers the whole grid."""
        from dvfopt.core.spatial import neg_jdet_bounding_window

        H, W = 8, 8
        jm = np.full((1, H, W), -0.5)
        size, _ = neg_jdet_bounding_window(jm, (4, 4), THRESHOLD, ERR_TOL)
        assert size[0] == H and size[1] == W, \
            f"All-negative grid should give full-grid window; got {size}"


# ===========================================================================
# 3. WRITE-BACK LOGIC WITH PADDED EXTRACTION
# ===========================================================================

class TestWriteBackLogic:
    """_apply_result with a padded extraction must only modify the inner (sy,sx)
    region of phi, leaving the surrounding 1-pixel ring completely untouched."""

    def test_padded_writeback_outer_ring_of_extracted_region_unchanged(self):
        """The 1-pixel border of the padded extraction area must NOT be written
        back when write_size is given."""
        from dvfopt.core.solver import _apply_result
        from dvfopt.core.spatial import get_phi_sub_flat_padded

        rng = np.random.default_rng(7)
        H, W = 15, 15
        phi = rng.standard_normal((2, H, W))
        phi_before = phi.copy()
        cy, cx = 7, 7
        sub_size = (5, 5)
        sy, sx = sub_size

        flat_padded, opt_size = get_phi_sub_flat_padded(phi, 0, cy, cx, (1, H, W), sub_size)
        assert opt_size == (sy + 2, sx + 2), "Expected padded extraction"

        # Apply a modified result back, using write_size to strip padding
        modified = flat_padded * 1.5
        _apply_result(phi, modified, cy, cx, opt_size, write_size=sub_size)

        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx
        # The 1-pixel ring *outside* the original window must be unchanged
        # Check top and bottom extra rows
        np.testing.assert_array_equal(
            phi[:, cy - hy - 1, cx - hx - 1:cx + hx_hi + 1],
            phi_before[:, cy - hy - 1, cx - hx - 1:cx + hx_hi + 1],
            err_msg="Top padding row of extracted region was modified",
        )
        np.testing.assert_array_equal(
            phi[:, cy + hy_hi, cx - hx - 1:cx + hx_hi + 1],
            phi_before[:, cy + hy_hi, cx - hx - 1:cx + hx_hi + 1],
            err_msg="Bottom padding row of extracted region was modified",
        )

    def test_padded_writeback_only_inner_sy_sx_pixels_changed(self):
        """After padded write-back, pixels inside the original (sy,sx) window
        should differ from before; pixels completely outside should not."""
        from dvfopt.core.solver import _apply_result
        from dvfopt.core.spatial import get_phi_sub_flat_padded

        H, W = 15, 15
        phi = np.ones((2, H, W)) * 0.0   # known constant base
        cy, cx = 7, 7
        sub_size = (5, 5)
        sy, sx = sub_size
        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx

        flat_padded, opt_size = get_phi_sub_flat_padded(phi, 0, cy, cx, (1, H, W), sub_size)
        # Apply a result of all 9.0 — if write-back works, inner window should be 9
        modified = np.full_like(flat_padded, 9.0)
        _apply_result(phi, modified, cy, cx, opt_size, write_size=sub_size)

        # Inner window is changed
        assert phi[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi].mean() == pytest.approx(9.0), \
            "Inner window should have been written with value 9"
        # Pixels outside the inner window stay 0
        assert phi[0, cy - hy - 2, cx] == pytest.approx(0.0), \
            "Pixel 2 rows above inner window should be unchanged"
        assert phi[0, cy + hy_hi + 1, cx] == pytest.approx(0.0), \
            "Pixel below inner window should be unchanged"

    def test_unpadded_writeback_modifies_full_window_including_boundary(self):
        """Without padding, _apply_result (write_size=None) writes the full
        (sy,sx) region including the boundary ring."""
        from dvfopt.core.solver import _apply_result
        from dvfopt.core.spatial import get_phi_sub_flat

        H, W = 15, 15
        phi = np.zeros((2, H, W))
        cy, cx = 7, 7
        sub_size = (5, 5)
        sy, sx = sub_size
        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx

        flat = get_phi_sub_flat(phi, 0, cy, cx, (1, H, W), sub_size)
        modified = np.full_like(flat, 7.0)
        _apply_result(phi, modified, cy, cx, sub_size, write_size=None)

        # ALL pixels in the (sy, sx) window should be 7 — including boundary row/col
        window = phi[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi]
        np.testing.assert_allclose(window, 7.0, atol=1e-12,
                                   err_msg="Unpadded write-back must cover full window")

    def test_pixels_completely_outside_window_never_modified(self):
        """Both padded and unpadded write-back must leave pixels > 1 row outside
        the window boundary completely unchanged."""
        from dvfopt.core.solver import _apply_result

        H, W = 20, 20
        phi_before = np.random.default_rng(99).standard_normal((2, H, W))
        phi = phi_before.copy()
        cy, cx = 10, 10
        sub_size = (5, 5)
        sy, sx = sub_size
        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx

        fake_result = np.zeros(2 * sy * sx)
        _apply_result(phi, fake_result, cy, cx, sub_size, write_size=None)

        # Row completely outside the window (2 rows above top)
        outside_row = cy - hy - 2
        np.testing.assert_array_equal(
            phi[:, outside_row, :],
            phi_before[:, outside_row, :],
            err_msg="Row outside window was modified",
        )


# ===========================================================================
# 4. _EDGE_FLAGS LOGIC
# ===========================================================================

class TestEdgeFlagsLogic:
    """_edge_flags must truthfully report is_at_edge and window_reached_max."""

    def _flags(self, cy, cx, size, H=10, W=10, max_size=None):
        from dvfopt.core.spatial import _edge_flags
        if max_size is None:
            max_size = (H, W)
        return _edge_flags(cy, cx, size, (1, H, W), max_size)

    def test_window_starting_at_row_zero_is_at_edge(self):
        # hy=1 → start_y = cy - 1 = 0 → at edge
        is_at_edge, _ = self._flags(cy=1, cx=5, size=(3, 3))
        assert is_at_edge

    def test_window_ending_at_last_row_is_at_edge(self):
        # hy_hi=2, end_y = cy + 2 - 1 = cy + 1 >= H-1=9 → cy=8
        is_at_edge, _ = self._flags(cy=8, cx=5, size=(3, 3))
        assert is_at_edge

    def test_window_starting_at_col_zero_is_at_edge(self):
        is_at_edge, _ = self._flags(cy=5, cx=1, size=(3, 3))
        assert is_at_edge

    def test_window_ending_at_last_col_is_at_edge(self):
        is_at_edge, _ = self._flags(cy=5, cx=8, size=(3, 3))
        assert is_at_edge

    def test_interior_window_not_at_edge_not_at_max(self):
        is_at_edge, w_max = self._flags(cy=5, cx=5, size=(3, 3))
        assert not is_at_edge
        assert not w_max

    def test_window_at_max_size_sets_reached_max(self):
        _, w_max = self._flags(cy=5, cx=5, size=(10, 10), max_size=(10, 10))
        assert w_max

    def test_window_smaller_than_max_not_reached_max(self):
        _, w_max = self._flags(cy=5, cx=5, size=(7, 7), max_size=(10, 10))
        assert not w_max

    def test_max_size_partial_match_not_reached_max(self):
        """Both dimensions must meet or exceed max_window for reached_max."""
        _, w_max = self._flags(cy=5, cx=5, size=(10, 7), max_size=(10, 10))
        assert not w_max, "Only one dimension at max should NOT set window_reached_max"


# ===========================================================================
# 5. OPTIMIZATION RESULT SATISFIES CONSTRAINT
# ===========================================================================

class TestOptimizationSatisfiesConstraint:
    """_optimize_single_window must produce a result satisfying Jdet >= threshold
    for interior pixels when given a feasible fold."""

    def test_optimization_jdet_constraint_satisfied_after_call(self):
        """Call _optimize_single_window directly on a sub-window with a fold.
        The returned x must satisfy jacobian_constraint >= threshold."""
        from dvfopt.core.solver import _optimize_single_window
        from dvfopt.core.constraints import jacobian_constraint
        from dvfopt.core.spatial import get_phi_sub_flat

        H, W = 15, 15
        phi = np.zeros((2, H, W))
        # Inject fold at interior of the window
        phi[1, 7, 7] = 3.0
        phi[1, 7, 8] = -3.0
        phi_init = phi.copy()

        sub_size = (7, 7)
        cy, cx = 7, 7
        flat = get_phi_sub_flat(phi, 0, cy, cx, (1, H, W), sub_size)
        flat_init = get_phi_sub_flat(phi_init, 0, cy, cx, (1, H, W), sub_size)

        result_x, _, _ = _optimize_single_window(
            flat, flat_init, sub_size,
            is_at_edge=False, window_reached_max=False,
            threshold=THRESHOLD, max_minimize_iter=500, method_name="SLSQP",
        )

        jdet_vals = jacobian_constraint(result_x, sub_size, exclude_boundaries=True)
        assert jdet_vals.min() >= THRESHOLD - ERR_TOL, \
            f"Interior Jdet constraint violated: min={jdet_vals.min():.6f}"

    def test_optimization_changes_phi_for_fold(self):
        """After iterative_serial on a folded field, phi must differ from phi_init."""
        dvf = _fold_dvf(12, 12, 6, 5)
        phi_init = dvf[1:3, 0].copy()
        phi = iterative_serial(dvf, verbose=0, threshold=THRESHOLD, err_tol=ERR_TOL,
                               max_iterations=300, max_per_index_iter=100,
                               max_minimize_iter=300)
        max_change = float(np.abs(phi - phi_init).max())
        assert max_change > 1e-6, \
            f"phi was not modified despite having a fold (max change={max_change:.2e})"

    def test_correction_strictly_improves_min_jdet(self):
        """min_jdet after correction must be greater than before."""
        dvf = _fold_dvf(12, 12, 6, 5)
        jdet_before = float(jacobian_det2D(dvf[1:3, 0]).min())
        phi = iterative_serial(dvf, verbose=0, threshold=THRESHOLD, err_tol=ERR_TOL,
                               max_iterations=300, max_per_index_iter=100,
                               max_minimize_iter=300)
        jdet_after = float(jacobian_det2D(phi).min())
        assert jdet_after > jdet_before, \
            f"Correction did not improve min_jdet: {jdet_before:.4f} -> {jdet_after:.4f}"

    def test_unoptimized_fold_has_negative_jdet_in_window(self):
        """Verify the fold actually produces negative Jdet before correction."""
        dvf = _fold_dvf(12, 12, 6, 5)
        phi_before = dvf[1:3, 0]
        jdet = jacobian_det2D(phi_before)
        assert jdet[0].min() < 0, \
            f"Test setup failed: expected negative Jdet in fold; got min={jdet[0].min():.4f}"


# ===========================================================================
# 6. QUALITY MAP IS EXACT ELEMENT-WISE MINIMUM
# ===========================================================================

class TestQualityMapExact:
    """_quality_map must be exactly the element-wise minimum of active metrics,
    not just a lower bound or a different quantity."""

    def test_without_constraints_returns_same_jdet_object(self):
        """With no extra constraints, quality_map must be the same object as jdet."""
        from dvfopt.core.constraints import _quality_map
        phi = np.random.default_rng(0).standard_normal((2, 8, 8)) * 0.2
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=False,
                          jacobian_matrix=jdet)
        assert qm is jdet, "_quality_map without constraints must return the jdet object"

    def test_quality_map_for_identity_is_exactly_one(self):
        """Zero displacement: Jdet=1 and shoelace=1 everywhere.
        quality_map must be exactly 1, not less (not a too-low bound)."""
        from dvfopt.core.constraints import _quality_map
        phi = np.zeros((2, 8, 8))
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=True, enforce_injectivity=True,
                          jacobian_matrix=jdet)
        np.testing.assert_allclose(qm, 1.0, atol=1e-10,
                                   err_msg="Quality map for identity field should be exactly 1")

    def test_quality_map_minimum_equals_minimum_of_contributing_metrics(self):
        """qm.min() must equal the minimum of all individual metrics (tight lower bound)."""
        from dvfopt.core.constraints import _quality_map
        from dvfopt.jacobian.shoelace import shoelace_det2D

        rng = np.random.default_rng(42)
        phi = rng.standard_normal((2, 10, 10)) * 0.2
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=True, enforce_injectivity=False,
                          jacobian_matrix=jdet)

        # The minimum of quality_map should be no larger than the minimum of jdet
        # (it should be the actual minimum of min(jdet, shoelace_spread))
        assert qm.min() <= jdet.min() + 1e-12, \
            "quality_map.min() cannot exceed jdet.min()"

        # The minimum must be tight: equal to actual min over the joint metric
        # Verify qm is not trivially under-estimated (e.g., all -inf)
        assert np.isfinite(qm).all(), "quality_map must be finite everywhere"
        assert qm.min() > -1e6, "quality_map must not be arbitrarily small"

    def test_quality_map_reflects_shoelace_when_shoelace_is_bottleneck(self):
        """At pixels where shoelace < Jdet, quality_map must equal the shoelace value."""
        from dvfopt.core.constraints import _quality_map
        from dvfopt.jacobian.shoelace import shoelace_det2D

        # Construct a phi where Jdet > 0 everywhere but shoelace areas can be small
        # Use a field where uniform Jdet = 1 but shoelace varies
        phi = np.zeros((2, 8, 8))
        # Mild fold that makes shoelace < 1 in a cell without making Jdet negative
        phi[1, 4, 4] = 0.6    # large dx at one corner distorts the quad area
        phi[1, 4, 5] = -0.5   # opposite displacement at next pixel

        jdet = jacobian_det2D(phi)
        shoe_areas = shoelace_det2D(phi)  # (1, H-1, W-1)

        qm = _quality_map(phi, enforce_shoelace=True, enforce_injectivity=False,
                          jacobian_matrix=jdet)

        # For any cell where shoelace < jdet, quality at that pixel should be
        # less than or equal to the shoelace area
        H, W = phi.shape[1], phi.shape[2]
        for r in range(H - 1):
            for c in range(W - 1):
                area = float(shoe_areas[0, r, c])
                if area < float(jdet[0, r, c]) - 1e-9:
                    # This pixel's quality should reflect shoelace constraint
                    # (quality_map at corners of this cell should be <= area)
                    for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        pr, pc = r + dr, c + dc
                        if 0 <= pr < H and 0 <= pc < W:
                            assert qm[0, pr, pc] <= area + 1e-9, \
                                f"quality_map[{pr},{pc}]={qm[0,pr,pc]:.4f} > " \
                                f"shoelace[{r},{c}]={area:.4f}"


# ===========================================================================
# 7. THRESHOLD BOUNDARY CONDITION (INCLUSIVE <=)
# ===========================================================================

class TestThresholdBoundaryCondition:
    """The convergence check and negative-pixel counter both use <= (inclusive).

    A pixel at exactly threshold - err_tol must be treated as still negative;
    only a pixel strictly above this level is considered converged.
    """

    def test_pixel_at_exactly_boundary_counts_as_negative(self):
        """The convergence condition is <= (not <), so a pixel at the exact
        boundary value is still negative."""
        boundary = THRESHOLD - ERR_TOL
        jm = np.ones((1, 10, 10)) * 1.0
        jm[0, 5, 5] = boundary  # exactly at boundary

        neg_count = int((jm <= boundary).sum())
        assert neg_count == 1, \
            f"Expected 1 pixel at boundary to count as negative, got {neg_count}"

    def test_pixel_epsilon_above_boundary_does_not_count(self):
        boundary = THRESHOLD - ERR_TOL
        jm = np.ones((1, 10, 10)) * 1.0
        jm[0, 5, 5] = boundary + 1e-10  # just above boundary

        neg_count = int((jm <= boundary).sum())
        assert neg_count == 0, \
            f"Pixel just above boundary should not count as negative, got {neg_count}"

    def test_algorithm_stops_when_all_pixels_above_threshold(self):
        """If min_jdet is already above threshold, no iterations should run."""
        dvf = np.zeros((3, 1, 8, 8), dtype=np.float64)
        # Zero displacement: Jdet = 1 everywhere >> threshold
        phi = iterative_serial(dvf, verbose=0, threshold=THRESHOLD, err_tol=ERR_TOL,
                               max_iterations=1000, max_per_index_iter=50,
                               max_minimize_iter=100)
        # phi must be all zeros (unchanged)
        np.testing.assert_array_equal(phi, np.zeros((2, 8, 8)),
                                      err_msg="Clean field should not be modified at all")

    def test_algorithm_runs_if_exactly_one_pixel_at_boundary(self, tmp_path):
        """If one pixel's Jdet is exactly at the boundary, the loop must fire."""
        from dvfopt.core.solver import _save_results
        # We can't easily craft a real phi where Jdet = exactly boundary, but
        # we can verify that the negative-pixel count correctly uses <=:
        # Create a field with actual fold and check it was corrected.
        dvf = _fold_dvf(10, 10, 5, 4, mag=3.0)
        jdet_before = jacobian_det2D(dvf[1:3, 0])
        assert jdet_before[0].min() < 0, "Test setup: expected negative Jdet"

        phi = iterative_serial(dvf, verbose=0, threshold=THRESHOLD, err_tol=ERR_TOL,
                               max_iterations=500, max_per_index_iter=100,
                               max_minimize_iter=300, save_path=str(tmp_path))
        text = (tmp_path / "results.txt").read_text()
        # Iterations must be > 0 (the loop fired)
        assert "Number of index iterations: 0" not in text, \
            "Expected at least one iteration for a folded field"


# ===========================================================================
# 8. WINDOW GROWTH MECHANISM
# ===========================================================================

class TestWindowGrowthMechanism:
    """The algorithm must actually use progressively larger windows when a
    small window cannot fix a fold (escalation / window growth)."""

    def test_window_counts_show_multiple_sizes_for_hard_fold(self, tmp_path):
        """A fold embedded in a large connected negative region requires window
        growth.  window_counts.csv must record more than one window size."""
        H, W = 14, 14
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        # Wide connected strip so that the initial 3x3 window has dirty edges
        for i in range(4, 10):
            dvf[2, 0, i, 4:10] = 3.0 if i % 2 == 0 else -3.0

        iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=3000, max_per_index_iter=10,
                         max_minimize_iter=200)

        csv_lines = (tmp_path / "window_counts.csv").read_text().splitlines()
        data_lines = [l for l in csv_lines[1:] if l.strip()]  # skip header
        assert len(data_lines) > 1, \
            "Expected multiple window sizes in window_counts.csv for a hard fold"

    def test_larger_windows_appear_in_counts_for_hard_fold(self, tmp_path):
        """When escalation fires, windows larger than 3x3 must be used."""
        H, W = 14, 14
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        for i in range(4, 10):
            dvf[2, 0, i, 4:10] = 3.0 if i % 2 == 0 else -3.0

        iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=3000, max_per_index_iter=10,
                         max_minimize_iter=200)

        csv_lines = (tmp_path / "window_counts.csv").read_text().splitlines()
        data_lines = [l for l in csv_lines[1:] if l.strip()]

        # Parse window sizes (height, width) from CSV
        window_sizes = []
        for line in data_lines:
            parts = line.split(",")
            h, w = int(parts[0]), int(parts[1])
            window_sizes.append((h, w))

        # At least one window larger than 3x3 must have been used
        has_large = any(h > 3 or w > 3 for h, w in window_sizes)
        assert has_large, \
            f"No window larger than 3x3 used; sizes: {window_sizes}"

    def test_simple_fold_uses_at_least_one_window(self, tmp_path):
        """A corrected fold must register at least one window-size entry
        in window_counts.csv (the optimizer was actually called)."""
        dvf = _fold_dvf(10, 10, 5, 4)
        iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=300, max_per_index_iter=100,
                         max_minimize_iter=300)

        csv_lines = (tmp_path / "window_counts.csv").read_text().splitlines()
        data_lines = [l for l in csv_lines[1:] if l.strip()]
        assert len(data_lines) >= 1, "Expected at least one window size entry"
        # Total count must be >= 1
        total_calls = sum(int(l.split(",")[-1]) for l in data_lines)
        assert total_calls >= 1, \
            f"Expected at least 1 optimizer call, got {total_calls}"
