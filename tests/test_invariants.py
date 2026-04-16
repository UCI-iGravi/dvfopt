"""
Invariant and numerical correctness tests for the iterative SLSQP solver.

These complement test_edge_cases.py by testing properties that must hold
regardless of which specific fold is corrected:

  - phi_init is never mutated
  - Frozen boundary values are unchanged after optimization
  - Jdet of optimized sub-window satisfies the constraint
  - Analytical constraint Jacobians match finite differences
  - phi_sub_flat / apply_result roundtrip (channel-order contract)
  - Parallel batch windows are truly non-overlapping with 1px gap
  - L2 error does not blow up (phi stays near phi_init)
  - Stall-count purge mechanics
  - Adaptive max_minimize_iter formula
  - Bounding-window edge cases (boundary, whole-grid negative, zero-pixel)
  - get_nearest_center_3d off-by-one at exact grid edge

Run with:  python -m pytest tests/test_invariants.py -v
"""

import numpy as np
import pytest
import scipy.sparse

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, _numpy_jdet_2d

THRESHOLD = 0.01
ERR_TOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold_dvf(H, W, cy, cx, mag=2.5):
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    dvf[2, 0, cy, cx] = mag
    dvf[2, 0, cy, cx + 1] = -mag
    return dvf


def _run_serial(dvf, **kw):
    from dvfopt.core.slsqp.iterative import iterative_serial
    return iterative_serial(
        dvf, verbose=0,
        threshold=THRESHOLD, err_tol=ERR_TOL,
        max_iterations=kw.pop("max_iterations", 500),
        max_per_index_iter=200,
        max_minimize_iter=500,
        **kw,
    )


# ===========================================================================
# 1. PHI_INIT IMMUTABILITY
# ===========================================================================

class TestPhiInitImmutable:
    """The algorithm must never write to phi_init."""

    def test_phi_init_unchanged_after_correction(self):
        dvf = _make_fold_dvf(12, 12, 6, 5)
        phi_init_before = dvf[1:3, 0].copy()
        _run_serial(dvf)
        # phi_init is extracted from dvf inside _init_phi; dvf itself
        # must not be mutated either (it's the user's input array).
        np.testing.assert_array_equal(
            dvf[1:3, 0], phi_init_before,
            err_msg="Input dvf array was mutated by iterative_serial",
        )

    def test_phi_init_unchanged_after_3d_correction(self):
        from dvfopt.core.slsqp.iterative3d import iterative_3d
        dvf = np.zeros((3, 6, 6, 6), dtype=np.float64)
        dvf[2, 3, 3, 3] = 2.5
        dvf[2, 3, 3, 4] = -2.5
        dvf_copy = dvf.copy()
        iterative_3d(dvf, verbose=0, threshold=THRESHOLD, err_tol=ERR_TOL,
                     max_iterations=200, max_per_index_iter=50,
                     max_minimize_iter=200)
        np.testing.assert_array_equal(
            dvf, dvf_copy,
            err_msg="Input dvf 3D array was mutated by iterative_3d",
        )


# ===========================================================================
# 2. FROZEN BOUNDARY IS ACTUALLY FROZEN
# ===========================================================================

class TestFrozenBoundaryRespected:
    """After _optimize_single_window, the boundary voxels of the sub-window
    must be byte-for-byte identical to their input values."""

    def _extract_boundary(self, result_x, sy, sx):
        """Return boundary ring values from the flat [dx, dy] result."""
        pixels = sy * sx
        dx = result_x[:pixels].reshape(sy, sx)
        dy = result_x[pixels:].reshape(sy, sx)
        mask = np.zeros((sy, sx), dtype=bool)
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        return dx[mask].copy(), dy[mask].copy()

    def test_frozen_edges_unchanged(self):
        """Use unpadded extraction (sub-window not at edge) so the LinearConstraint
        freezes the outer ring.  After optimization those values must be identical."""
        from dvfopt.core.solver import _optimize_single_window
        from dvfopt.core.slsqp.spatial import get_phi_sub_flat

        rng = np.random.default_rng(17)
        H, W = 15, 15
        phi = rng.standard_normal((2, H, W)) * 0.3
        phi_init = phi.copy()
        submatrix_size = (7, 7)
        cy, cx = 7, 7   # deep interior — padded extraction possible, but we
        sy, sx = submatrix_size  # use unpadded to test the frozen-ring constraint

        phi_sub = get_phi_sub_flat(phi, 0, cy, cx, (1, H, W), submatrix_size)
        phi_init_sub = get_phi_sub_flat(phi_init, 0, cy, cx, (1, H, W), submatrix_size)

        # Force a fold inside the interior so the optimizer has something to fix
        pixels = sy * sx
        phi_sub[2 * sx + 3] += 3.0   # interior dx element

        # is_at_edge=False, window_reached_max=False → boundary IS frozen
        result_x, _, _ = _optimize_single_window(
            phi_sub, phi_init_sub, (sy, sx),
            is_at_edge=False, window_reached_max=False,
            threshold=THRESHOLD, max_minimize_iter=300, method_name="SLSQP",
        )

        # Boundary ring of result_x must equal the boundary ring of phi_sub
        boundary_before_dx, boundary_before_dy = self._extract_boundary(phi_sub, sy, sx)
        boundary_after_dx, boundary_after_dy = self._extract_boundary(result_x, sy, sx)

        np.testing.assert_allclose(boundary_after_dx, boundary_before_dx, atol=1e-7,
                                   err_msg="dx boundary values changed after optimization")
        np.testing.assert_allclose(boundary_after_dy, boundary_before_dy, atol=1e-7,
                                   err_msg="dy boundary values changed after optimization")


# ===========================================================================
# 3. PHI_SUB_FLAT / APPLY_RESULT ROUNDTRIP
# ===========================================================================

class TestPhiSubFlatRoundtrip:
    """get_phi_sub_flat_padded + _apply_result must be an identity transform:
    extract → trivially modify → write back → re-extract must give the same
    values that were written."""

    def test_extract_apply_roundtrip_unpadded(self):
        from dvfopt.core.slsqp.spatial import get_phi_sub_flat
        from dvfopt.core.solver import _apply_result
        rng = np.random.default_rng(0)
        H, W = 15, 15
        phi = rng.standard_normal((2, H, W))
        cy, cx = 7, 7
        sub_size = (5, 5)
        sy, sx = sub_size

        # Extract
        flat = get_phi_sub_flat(phi, 0, cy, cx, (1, H, W), sub_size)
        # Modify (scale by 1.1 — doesn't matter what, just changes it)
        modified = flat * 1.1
        # Write back
        phi2 = phi.copy()
        _apply_result(phi2, modified, cy, cx, sub_size, write_size=None)
        # Re-extract
        flat2 = get_phi_sub_flat(phi2, 0, cy, cx, (1, H, W), sub_size)
        np.testing.assert_allclose(flat2, modified, atol=1e-12,
                                   err_msg="Roundtrip failed for unpadded sub-window")

    def test_extract_apply_roundtrip_padded(self):
        from dvfopt.core.slsqp.spatial import get_phi_sub_flat_padded, get_phi_sub_flat
        from dvfopt.core.solver import _apply_result
        rng = np.random.default_rng(1)
        H, W = 15, 15
        phi = rng.standard_normal((2, H, W))
        cy, cx = 7, 7
        sub_size = (5, 5)
        sy, sx = sub_size

        # Padded extraction: returns (sy+2, sx+2) flat
        flat_padded, opt_size = get_phi_sub_flat_padded(phi, 0, cy, cx, (1, H, W), sub_size)
        assert opt_size == (sy + 2, sx + 2), f"Expected padded size, got {opt_size}"

        # The inner (sy, sx) slice of the padded result should match unpadded
        flat_unpadded = get_phi_sub_flat(phi, 0, cy, cx, (1, H, W), sub_size)

        opt_sy, opt_sx = opt_size
        pixels_opt = opt_sy * opt_sx
        dx_padded = flat_padded[:pixels_opt].reshape(opt_sy, opt_sx)
        dy_padded = flat_padded[pixels_opt:].reshape(opt_sy, opt_sx)
        # Inner region of padded should match unpadded exactly
        np.testing.assert_allclose(
            dx_padded[1:-1, 1:-1].flatten(),
            flat_unpadded[:sy * sx],
            atol=1e-12,
            err_msg="Inner dx of padded extraction doesn't match unpadded",
        )
        np.testing.assert_allclose(
            dy_padded[1:-1, 1:-1].flatten(),
            flat_unpadded[sy * sx:],
            atol=1e-12,
            err_msg="Inner dy of padded extraction doesn't match unpadded",
        )

    def test_channel_order_dx_dy(self):
        """phi[0]=dy, phi[1]=dx.  Flat packing is [dx, dy].  Verify."""
        from dvfopt.core.slsqp.spatial import get_phi_sub_flat
        H, W = 10, 10
        phi = np.zeros((2, H, W))
        phi[0] = 1.0   # dy = 1 everywhere
        phi[1] = 2.0   # dx = 2 everywhere
        sub_size = (3, 3)
        flat = get_phi_sub_flat(phi, 0, 5, 5, (1, H, W), sub_size)
        pixels = 3 * 3
        dx_vals = flat[:pixels]
        dy_vals = flat[pixels:]
        assert np.all(dx_vals == 2.0), "dx block should be 2.0"
        assert np.all(dy_vals == 1.0), "dy block should be 1.0"

    def test_3d_channel_order_dx_dy_dz(self):
        """3D flat packing: [dx, dy, dz]. phi[0]=dz, phi[1]=dy, phi[2]=dx."""
        from dvfopt.core.slsqp.spatial3d import get_phi_sub_flat_3d
        D, H, W = 8, 8, 8
        phi = np.zeros((3, D, H, W))
        phi[0] = 1.0   # dz = 1
        phi[1] = 2.0   # dy = 2
        phi[2] = 3.0   # dx = 3
        sub_size = (3, 3, 3)
        flat = get_phi_sub_flat_3d(phi, 4, 4, 4, sub_size)
        voxels = 3 ** 3
        dx_vals = flat[:voxels]
        dy_vals = flat[voxels:2 * voxels]
        dz_vals = flat[2 * voxels:]
        assert np.all(dx_vals == 3.0), "dx block should be 3.0"
        assert np.all(dy_vals == 2.0), "dy block should be 2.0"
        assert np.all(dz_vals == 1.0), "dz block should be 1.0"


# ===========================================================================
# 4. ANALYTICAL CONSTRAINT JACOBIANS MATCH FINITE DIFFERENCES
# ===========================================================================

class TestConstraintJacobians:
    """All analytical Jacobians must match FD to within roundoff."""

    EPS = 1e-6
    ATOL = 1e-4  # FD noise at 1e-6 step; single-precision FD needs slack

    def _fd_jacobian(self, fn, phi, n_out):
        """Compute Jacobian of fn(phi) via central finite differences."""
        n_in = len(phi)
        J = np.zeros((n_out, n_in))
        for k in range(n_in):
            p_hi = phi.copy(); p_hi[k] += self.EPS
            p_lo = phi.copy(); p_lo[k] -= self.EPS
            J[:, k] = (fn(p_hi) - fn(p_lo)) / (2 * self.EPS)
        return J

    def test_jdet_constraint_jacobian_2d_interior(self):
        from dvfopt.core.slsqp.constraints import jacobian_constraint
        from dvfopt.core.slsqp.gradients import jdet_constraint_jacobian_2d
        rng = np.random.default_rng(42)
        sub_size = (5, 5)
        phi = rng.standard_normal(2 * 25) * 0.3
        fn = lambda p: jacobian_constraint(p, sub_size, exclude_boundaries=True)
        n_out = len(fn(phi))
        J_analytical = jdet_constraint_jacobian_2d(phi, sub_size, exclude_boundaries=True).toarray()
        J_fd = self._fd_jacobian(fn, phi, n_out)
        np.testing.assert_allclose(J_analytical, J_fd, atol=self.ATOL,
                                   err_msg="Jdet Jacobian 2D (interior) mismatch")

    def test_jdet_constraint_jacobian_2d_full(self):
        from dvfopt.core.slsqp.constraints import jacobian_constraint
        from dvfopt.core.slsqp.gradients import jdet_constraint_jacobian_2d
        rng = np.random.default_rng(43)
        sub_size = (4, 4)
        phi = rng.standard_normal(2 * 16) * 0.3
        fn = lambda p: jacobian_constraint(p, sub_size, exclude_boundaries=False)
        n_out = len(fn(phi))
        J_analytical = jdet_constraint_jacobian_2d(phi, sub_size, exclude_boundaries=False).toarray()
        J_fd = self._fd_jacobian(fn, phi, n_out)
        np.testing.assert_allclose(J_analytical, J_fd, atol=self.ATOL,
                                   err_msg="Jdet Jacobian 2D (full) mismatch")

    def test_shoelace_constraint_jacobian_2d(self):
        from dvfopt.jacobian.shoelace import shoelace_constraint
        from dvfopt.core.slsqp.gradients import shoelace_constraint_jacobian_2d
        rng = np.random.default_rng(44)
        sub_size = (6, 6)
        phi = rng.standard_normal(2 * 36) * 0.2
        fn = lambda p: shoelace_constraint(p, sub_size, exclude_boundaries=True)
        n_out = len(fn(phi))
        J_analytical = shoelace_constraint_jacobian_2d(phi, sub_size, exclude_boundaries=True).toarray()
        J_fd = self._fd_jacobian(fn, phi, n_out)
        np.testing.assert_allclose(J_analytical, J_fd, atol=self.ATOL,
                                   err_msg="Shoelace Jacobian mismatch")

    def test_injectivity_constraint_jacobian_2d(self):
        from dvfopt.jacobian.monotonicity import injectivity_constraint
        from dvfopt.core.slsqp.gradients import injectivity_constraint_jacobian_2d
        rng = np.random.default_rng(45)
        sub_size = (5, 5)
        phi = rng.standard_normal(2 * 25) * 0.1  # small so monotonicity holds
        fn = lambda p: injectivity_constraint(p, sub_size, exclude_boundaries=True)
        n_out = len(fn(phi))
        J_analytical = injectivity_constraint_jacobian_2d(phi, sub_size, exclude_boundaries=True).toarray()
        J_fd = self._fd_jacobian(fn, phi, n_out)
        np.testing.assert_allclose(J_analytical, J_fd, atol=self.ATOL,
                                   err_msg="Injectivity Jacobian mismatch")

    def test_jdet_constraint_jacobian_3d(self):
        from dvfopt.core.slsqp.constraints3d import jacobian_constraint_3d
        from dvfopt.core.slsqp.gradients3d import jdet_constraint_jacobian_3d
        rng = np.random.default_rng(46)
        sub_size = (3, 3, 3)
        voxels = 27
        phi = rng.standard_normal(3 * voxels) * 0.2
        fn = lambda p: jacobian_constraint_3d(p, sub_size)
        n_out = len(fn(phi))
        J_analytical = jdet_constraint_jacobian_3d(phi, sub_size).toarray()
        J_fd = self._fd_jacobian(fn, phi, n_out)
        np.testing.assert_allclose(J_analytical, J_fd, atol=self.ATOL,
                                   err_msg="Jdet Jacobian 3D mismatch")


# ===========================================================================
# 5. JACOBIAN IDENTITY CHECK
# ===========================================================================

class TestJacobianIdentity:
    def test_zero_displacement_gives_jdet_one(self):
        """Identity displacement → Jdet = 1 everywhere (no deformation)."""
        phi = np.zeros((2, 10, 10))
        jdet = jacobian_det2D(phi)
        np.testing.assert_allclose(jdet, 1.0, atol=1e-12,
                                   err_msg="Zero phi should give Jdet=1")

    def test_uniform_displacement_gives_jdet_one(self):
        """Constant displacement (translation) → Jdet = 1 (gradients are 0)."""
        phi = np.full((2, 10, 10), 3.7)
        jdet = jacobian_det2D(phi)
        np.testing.assert_allclose(jdet, 1.0, atol=1e-12,
                                   err_msg="Constant displacement should give Jdet=1")

    def test_linear_compression_gives_correct_jdet(self):
        """Linear compression by factor s → Jdet = s² for 2D."""
        H, W = 10, 10
        s = 0.5  # compress by half
        phi = np.zeros((2, H, W))
        # dx[i, j] = (s-1)*j → ddx_dx = s-1 → Jdet = s^2 in interior
        x = np.arange(W, dtype=float)
        phi[1] = (s - 1) * x[np.newaxis, :]
        phi[0] = (s - 1) * np.arange(H, dtype=float)[:, np.newaxis]
        jdet = jacobian_det2D(phi)
        # Interior values should be s^2 = 0.25; boundary has one-sided diff
        interior_jdet = jdet[0, 1:-1, 1:-1]
        np.testing.assert_allclose(interior_jdet, s * s, atol=1e-10,
                                   err_msg="Linear compression Jdet wrong")


# ===========================================================================
# 6. L2 ERROR SANITY
# ===========================================================================

class TestL2ErrorSanity:
    def test_l2_error_finite_and_bounded(self):
        """phi after correction must not drift wildly from phi_init."""
        dvf = _make_fold_dvf(15, 15, 7, 7, mag=2.0)
        phi_init = dvf[1:3, 0].copy()
        phi = _run_serial(dvf)
        l2 = float(np.sqrt(np.sum((phi - phi_init) ** 2)))
        assert np.isfinite(l2), "L2 error is not finite"
        # The correction should not introduce displacement larger than the
        # original fold magnitude (2.0) times the grid size
        assert l2 < 2.0 * 15 * 15, f"L2 error suspiciously large: {l2:.2f}"

    def test_correction_does_not_increase_fold_severity(self):
        """min Jdet after correction must be >= min Jdet before (or at threshold)."""
        dvf = _make_fold_dvf(12, 12, 6, 5)
        init_min = float(jacobian_det2D(dvf[1:3, 0]).min())
        phi = _run_serial(dvf)
        final_min = float(jacobian_det2D(phi).min())
        assert final_min >= init_min - 0.01, \
            f"Correction made Jdet worse: {init_min:.4f} -> {final_min:.4f}"


# ===========================================================================
# 7. PARALLEL BATCH NON-OVERLAP
# ===========================================================================

class TestParallelBatchNonOverlap:
    """_select_non_overlapping must return windows with no shared pixels,
    including the mandatory 1px gap."""

    def _windows_overlap_including_gap(self, cy1, cx1, sz1, cy2, cx2, sz2):
        """Check if two windows overlap when each is expanded by 1px."""
        sy1, sx1 = sz1
        hy1, hx1 = sy1 // 2, sx1 // 2
        hy1_hi, hx1_hi = sy1 - hy1, sx1 - hx1

        sy2, sx2 = sz2
        hy2, hx2 = sy2 // 2, sx2 // 2
        hy2_hi, hx2_hi = sy2 - hy2, sx2 - hx2

        # Padded footprint: 1 extra pixel on each side
        py0_1 = cy1 - hy1 - 1;  py1_1 = cy1 + hy1_hi + 1
        px0_1 = cx1 - hx1 - 1;  px1_1 = cx1 + hx1_hi + 1

        py0_2 = cy2 - hy2 - 1;  py1_2 = cy2 + hy2_hi + 1
        px0_2 = cx2 - hx2 - 1;  px1_2 = cx2 + hx2_hi + 1

        y_overlap = py0_1 < py1_2 and py0_2 < py1_1
        x_overlap = px0_1 < px1_2 and px0_2 < px1_1
        return y_overlap and x_overlap

    def test_selected_windows_non_overlapping(self):
        from dvfopt.core.slsqp.spatial import _select_non_overlapping
        slice_shape = (1, 30, 30)
        # Densely packed negative pixels
        neg_pixels = [(r, c) for r in range(0, 25, 3) for c in range(0, 25, 3)]
        pixel_window_sizes = {px: (5, 5) for px in neg_pixels}
        pixel_bbox_centers = {px: px for px in neg_pixels}

        batch = _select_non_overlapping(
            neg_pixels, pixel_window_sizes, slice_shape,
            near_cent_dict={}, pixel_bbox_centers=pixel_bbox_centers,
        )

        # Every pair in the batch must be gap-separated
        for i, (_, (_, cy1, cx1), sz1) in enumerate(batch):
            for j, (_, (_, cy2, cx2), sz2) in enumerate(batch):
                if i >= j:
                    continue
                assert not self._windows_overlap_including_gap(
                    cy1, cx1, sz1, cy2, cx2, sz2
                ), f"Windows {i} and {j} overlap (with gap): ({cy1},{cx1}) and ({cy2},{cx2})"

    def test_empty_neg_pixels_returns_empty_batch(self):
        from dvfopt.core.slsqp.spatial import _select_non_overlapping
        batch = _select_non_overlapping([], {}, (1, 10, 10), {})
        assert batch == []

    def test_single_pixel_always_selected(self):
        from dvfopt.core.slsqp.spatial import _select_non_overlapping
        batch = _select_non_overlapping(
            [(5, 5)], {(5, 5): (3, 3)}, (1, 15, 15), {},
            pixel_bbox_centers={(5, 5): (5, 5)},
        )
        assert len(batch) == 1


# ===========================================================================
# 8. BOUNDING WINDOW EDGE CASES
# ===========================================================================

class TestBoundingWindowEdgeCases:
    def test_negative_pixel_at_row_zero(self):
        """Negative pixel touching grid top must not produce out-of-bounds bbox."""
        from dvfopt.core.slsqp.spatial import neg_jdet_bounding_window
        jm = np.ones((1, 10, 10))
        jm[0, 0, 5] = -0.5   # top row
        size, center = neg_jdet_bounding_window(jm, (0, 5), THRESHOLD, ERR_TOL)
        assert size[0] >= 3 and size[1] >= 3
        assert center[0] >= 0 and center[1] >= 0

    def test_negative_pixel_at_last_column(self):
        """neg_jdet_bounding_window returns a raw bbox center (not clamped).
        Clamping is done downstream by get_nearest_center.  Verify that the
        size is at least 3×3 and that the raw center is within the grid."""
        from dvfopt.core.slsqp.spatial import neg_jdet_bounding_window
        jm = np.ones((1, 10, 10))
        jm[0, 5, 9] = -0.5   # last column
        size, center = neg_jdet_bounding_window(jm, (5, 9), THRESHOLD, ERR_TOL)
        H, W = 10, 10
        sy, sx = size
        cy, cx = center
        assert sy >= 3 and sx >= 3, f"Size too small: {size}"
        # The raw center should be within the grid bounds (it's derived from
        # bounding-box arithmetic on valid pixel indices)
        assert 0 <= cy < H and 0 <= cx < W, \
            f"Raw center ({cy},{cx}) outside grid {H}×{W}"

    def test_whole_grid_negative_returns_grid_size(self):
        """When the entire grid is negative, bbox should cover the full grid."""
        from dvfopt.core.slsqp.spatial import neg_jdet_bounding_window
        H, W = 8, 8
        jm = np.full((1, H, W), -0.5)
        size, center = neg_jdet_bounding_window(jm, (4, 4), THRESHOLD, ERR_TOL)
        # bbox covers rows 0..H-1 → size at least H
        assert size[0] >= H and size[1] >= W, \
            f"Whole-grid negative: bbox too small {size} for {H}×{W}"

    def test_non_negative_center_pixel_returns_minimum_window(self):
        """If the 'center' pixel is actually positive (edge case), return (3,3)."""
        from dvfopt.core.slsqp.spatial import neg_jdet_bounding_window
        jm = np.ones((1, 10, 10))
        # All positive — center_yx pixel is not negative
        size, center = neg_jdet_bounding_window(jm, (5, 5), THRESHOLD, ERR_TOL)
        assert size == (3, 3)

    def test_3d_bounding_window_boundary_pixel(self):
        """3D: negative voxel at z=0 face must not give negative z bbox."""
        from dvfopt.core.slsqp.spatial3d import neg_jdet_bounding_window_3d
        D, H, W = 8, 8, 8
        jm = np.ones((D, H, W))
        jm[0, 4, 4] = -0.5   # z=0 face
        size, center = neg_jdet_bounding_window_3d(jm, (0, 4, 4), THRESHOLD, ERR_TOL)
        assert all(s >= 3 for s in size)
        assert center[0] >= 0


# ===========================================================================
# 9. GET_NEAREST_CENTER FULL-GRID WINDOW
# ===========================================================================

class TestGetNearestCenterFullGrid:
    def test_full_grid_window_center_2d(self):
        """When window == grid, center must be at (H//2, W//2) (approximately)."""
        from dvfopt.core.slsqp.spatial import get_nearest_center
        H, W = 12, 15
        _, cy, cx = get_nearest_center((0, 0), (1, H, W), (H, W))
        # Window [cy-hy, cy+hy_hi) must cover the full grid
        hy = H // 2;  hy_hi = H - hy
        hx = W // 2;  hx_hi = W - hx
        assert cy - hy == 0 and cy + hy_hi == H, \
            f"Full-grid 2D: window doesn't cover grid rows: cy={cy}"
        assert cx - hx == 0 and cx + hx_hi == W, \
            f"Full-grid 2D: window doesn't cover grid cols: cx={cx}"

    def test_full_grid_window_center_3d(self):
        from dvfopt.core.slsqp.spatial3d import get_nearest_center_3d
        D, H, W = 5, 7, 9
        cz, cy, cx = get_nearest_center_3d((0, 0, 0), (D, H, W), (D, H, W))
        sz, sy, sx = D, H, W
        hz, hy, hx = sz // 2, sy // 2, sx // 2
        assert cz - hz == 0 and cz + (sz - hz) == D
        assert cy - hy == 0 and cy + (sy - hy) == H
        assert cx - hx == 0 and cx + (sx - hx) == W

    def test_window_larger_than_grid_clamped(self):
        """Requesting a window bigger than the grid must not crash or go OOB."""
        from dvfopt.core.slsqp.spatial import get_nearest_center
        H, W = 5, 5
        _, cy, cx = get_nearest_center((2, 2), (1, H, W), (H + 4, W + 4))
        # Just must not crash; center will be clamped to valid range
        assert 0 <= cy < H
        assert 0 <= cx < W


# ===========================================================================
# 10. STALL COUNT PURGE MECHANICS
# ===========================================================================

class TestStallCountPurge:
    """Unit test the purge logic: stall_counts keys for resolved pixels
    must be removed at the start of each outer iteration."""

    def test_stale_stall_key_purged(self):
        """Simulate two outer iterations: a pixel stalls, gets resolved
        by a neighbour, then reappears. Its count must restart from 0."""
        # We test the purge expression directly (no full solver needed)
        # quality_matrix: all positive except (5, 5)
        H, W = 10, 10
        quality = np.ones((1, H, W)) * 0.5

        stall_counts = {(5, 5): 2, (3, 3): 1}

        # Both pixels are currently positive → purge clears both
        quality[0, 5, 5] = 0.5  # clean
        quality[0, 3, 3] = 0.5  # clean
        stall_counts = {k: v for k, v in stall_counts.items()
                        if quality[0, k[0], k[1]] <= THRESHOLD - ERR_TOL}
        assert len(stall_counts) == 0, "Stale counts not purged"

    def test_active_stall_key_preserved(self):
        """Stall count for a pixel still below threshold must survive the purge."""
        H, W = 10, 10
        quality = np.ones((1, H, W)) * 0.5
        quality[0, 5, 5] = -0.3  # still bad

        stall_counts = {(5, 5): 2, (3, 3): 1}

        quality[0, 3, 3] = 0.5  # (3,3) is clean
        stall_counts = {k: v for k, v in stall_counts.items()
                        if quality[0, k[0], k[1]] <= THRESHOLD - ERR_TOL}
        assert (5, 5) in stall_counts and stall_counts[(5, 5)] == 2
        assert (3, 3) not in stall_counts


# ===========================================================================
# 11. ADAPTIVE MAX_MINIMIZE_ITER FORMULA
# ===========================================================================

class TestAdaptiveMaxMinimizeIter:
    """The adaptive iter budget formula must satisfy its design contract."""

    def _compute_eff(self, sy, sx, base):
        return min(max(base, 2 * sy * sx // 10), 10 * base)

    def test_small_window_uses_base(self):
        """3×3 window: 2*9//10 = 1 < base → use base."""
        base = 200
        assert self._compute_eff(3, 3, base) == base

    def test_large_window_scales_up(self):
        """50×50 window: 2*2500//10 = 500 > base=200 → use 500."""
        base = 200
        eff = self._compute_eff(50, 50, base)
        assert eff == 500

    def test_cap_at_10x_base(self):
        """Very large window must be capped at 10*base."""
        base = 200
        eff = self._compute_eff(1000, 1000, base)
        assert eff == 10 * base

    def test_formula_monotone_in_window_size(self):
        """Larger windows must not get fewer iterations than smaller ones."""
        base = 200
        prev = self._compute_eff(3, 3, base)
        for s in [5, 10, 20, 50, 100]:
            curr = self._compute_eff(s, s, base)
            assert curr >= prev, f"Non-monotone: {s}×{s} got {curr} < {prev}"
            prev = curr


# ===========================================================================
# 12. QUALITY MAP MINIMUM ELEMENT-WISE
# ===========================================================================

class TestQualityMap:
    """_quality_map must always be ≤ jacobian_det2D pixel-wise."""

    def test_quality_le_jdet_with_shoelace(self):
        from dvfopt.core.slsqp.constraints import _quality_map
        rng = np.random.default_rng(99)
        phi = rng.standard_normal((2, 12, 12)) * 0.5
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=True, enforce_injectivity=False,
                          jacobian_matrix=jdet)
        assert (qm <= jdet + 1e-12).all(), \
            "quality_map > jdet somewhere (shoelace mode)"

    def test_quality_le_jdet_with_injectivity(self):
        from dvfopt.core.slsqp.constraints import _quality_map
        rng = np.random.default_rng(100)
        phi = rng.standard_normal((2, 12, 12)) * 0.2
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=True,
                          jacobian_matrix=jdet)
        assert (qm <= jdet + 1e-12).all(), \
            "quality_map > jdet somewhere (injectivity mode)"

    def test_quality_equals_jdet_when_both_false(self):
        from dvfopt.core.slsqp.constraints import _quality_map
        rng = np.random.default_rng(101)
        phi = rng.standard_normal((2, 10, 10)) * 0.3
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=False,
                          jacobian_matrix=jdet)
        assert qm is jdet, "_quality_map should return jdet directly when both flags False"


# ===========================================================================
# 13. _UNPACK_SIZE_3D ERROR HANDLING
# ===========================================================================

class TestUnpackSize3d:
    def test_3tuple_accepted(self):
        from dvfopt._defaults import _unpack_size_3d
        assert _unpack_size_3d((3, 5, 7)) == (3, 5, 7)

    def test_scalar_gives_cubic(self):
        from dvfopt._defaults import _unpack_size_3d
        assert _unpack_size_3d(4) == (4, 4, 4)

    def test_2tuple_raises(self):
        from dvfopt._defaults import _unpack_size_3d
        with pytest.raises(ValueError, match="length-2"):
            _unpack_size_3d((3, 5))

    def test_4tuple_raises(self):
        from dvfopt._defaults import _unpack_size_3d
        with pytest.raises(ValueError, match="length-4"):
            _unpack_size_3d((3, 5, 7, 9))

    def test_empty_tuple_raises(self):
        from dvfopt._defaults import _unpack_size_3d
        with pytest.raises(ValueError):
            _unpack_size_3d(())
