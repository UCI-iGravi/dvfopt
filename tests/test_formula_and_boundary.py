"""
Formula-correctness and boundary-condition tests for low-level helpers.

Verifies:
  1. get_phi_sub_flat_padded — exact can_pad boundary condition and fallback
  2. _frozen_edges_clean_3d — completely untested function
  3. _frozen_edges_clean (2D) at exact threshold value (strict >)
  4. _shoelace_areas_2d formula for non-trivial known geometry
  5. objectiveEuc exact value and gradient formula
  6. Injectivity diagonal components (d1, d2) reach quality_map
  7. _patch_jacobian_2d / _patch_jacobian_3d clamping at grid boundaries

Run with:  python -m pytest tests/test_formula_and_boundary.py -v
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D

THRESHOLD = 0.01
ERR_TOL = 1e-5


# ===========================================================================
# 1. GET_PHI_SUB_FLAT_PADDED — can_pad boundary condition
# ===========================================================================

class TestPaddedExtractionBoundary:
    """The can_pad condition is:
        cy - hy - 1 >= 0  AND  cy + hy_hi + 1 <= H
        cx - hx - 1 >= 0  AND  cx + hx_hi + 1 <= W
    This is tested for the exact boundary values (one step either side).
    """

    def test_interior_center_returns_padded_size(self):
        """Far from all edges: padded extraction returns (sy+2, sx+2)."""
        from dvfopt.core.spatial import get_phi_sub_flat_padded
        phi = np.random.default_rng(0).standard_normal((2, 15, 15))
        sy, sx = 3, 3
        flat, actual_size = get_phi_sub_flat_padded(phi, 0, 7, 7, (1, 15, 15), (sy, sx))
        assert actual_size == (sy + 2, sx + 2), f"Expected padded; got {actual_size}"
        assert len(flat) == 2 * (sy + 2) * (sx + 2)

    def test_exact_lower_boundary_still_pads(self):
        """cy - hy - 1 = 0 (exactly zero) → can still pad."""
        from dvfopt.core.spatial import get_phi_sub_flat_padded
        phi = np.random.default_rng(1).standard_normal((2, 15, 15))
        sy, sx = 3, 3
        hy = sy // 2   # = 1
        cy = hy + 1    # cy - hy - 1 = 0  (lower boundary, just barely pads)
        _, actual_size = get_phi_sub_flat_padded(phi, 0, cy, 7, (1, 15, 15), (sy, sx))
        assert actual_size == (sy + 2, sx + 2), \
            f"cy={cy}: cy-hy-1=0 should still enable padding; got {actual_size}"

    def test_one_step_inside_edge_falls_back(self):
        """cy - hy - 1 = -1 (one step inside lower edge) → falls back to unpadded."""
        from dvfopt.core.spatial import get_phi_sub_flat_padded
        phi = np.random.default_rng(2).standard_normal((2, 15, 15))
        sy, sx = 3, 3
        hy = sy // 2   # = 1
        cy = hy        # cy - hy - 1 = -1  → cannot pad
        flat, actual_size = get_phi_sub_flat_padded(phi, 0, cy, 7, (1, 15, 15), (sy, sx))
        assert actual_size == (sy, sx), \
            f"cy={cy}: cy-hy-1=-1 should fall back to unpadded; got {actual_size}"
        assert len(flat) == 2 * sy * sx

    def test_padded_fallback_inner_values_match_unpadded(self):
        """When can_pad=True, the inner (sy,sx) of the padded result equals
        the unpadded extraction �� verifying they describe the same region."""
        from dvfopt.core.spatial import get_phi_sub_flat_padded, get_phi_sub_flat
        phi = np.random.default_rng(3).standard_normal((2, 15, 15))
        sy, sx = 5, 5
        cy, cx = 7, 7
        flat_pad, opt_size = get_phi_sub_flat_padded(phi, 0, cy, cx, (1, 15, 15), (sy, sx))
        flat_unpad = get_phi_sub_flat(phi, 0, cy, cx, (1, 15, 15), (sy, sx))

        opt_sy, opt_sx = opt_size
        pixels_opt = opt_sy * opt_sx
        dx_pad = flat_pad[:pixels_opt].reshape(opt_sy, opt_sx)[1:-1, 1:-1]
        dy_pad = flat_pad[pixels_opt:].reshape(opt_sy, opt_sx)[1:-1, 1:-1]
        np.testing.assert_array_equal(dx_pad.flatten(), flat_unpad[:sy * sx],
                                      err_msg="Padded inner dx should match unpadded")
        np.testing.assert_array_equal(dy_pad.flatten(), flat_unpad[sy * sx:],
                                      err_msg="Padded inner dy should match unpadded")


# ===========================================================================
# 2. _FROZEN_EDGES_CLEAN_3D — completely untested function
# ===========================================================================

class TestFrozenEdgesClean3d:
    """_frozen_edges_clean_3d checks that all frozen boundary voxels
    have Jdet > threshold - err_tol (strict inequality)."""

    def _interior_mask(self, D=6, H=6, W=6):
        """Freeze all 6 faces of a 3x3x3 window at the centre."""
        from dvfopt.core.spatial3d import _frozen_boundary_mask_3d
        return _frozen_boundary_mask_3d(3, 3, 3, (3, 3, 3), (D, H, W))

    def test_all_frozen_voxels_positive_returns_true(self):
        from dvfopt.core.spatial3d import _frozen_edges_clean_3d
        jm = np.ones((6, 6, 6)) * 0.5   # all positive
        freeze_mask = self._interior_mask()
        result = _frozen_edges_clean_3d(jm, 3, 3, 3, (3, 3, 3), THRESHOLD, ERR_TOL,
                                        freeze_mask)
        assert bool(result)

    def test_one_frozen_voxel_negative_returns_false(self):
        from dvfopt.core.spatial3d import _frozen_edges_clean_3d, _frozen_boundary_mask_3d
        jm = np.ones((6, 6, 6)) * 0.5
        # Window at (3,3,3) size (3,3,3): hz=1, starts at z=2
        # Frozen z-min face is at z-index 0 of subvolume = grid index 2
        jm[2, 3, 3] = -0.5   # negative on the z-min face of the window
        freeze_mask = self._interior_mask()
        result = _frozen_edges_clean_3d(jm, 3, 3, 3, (3, 3, 3), THRESHOLD, ERR_TOL,
                                        freeze_mask)
        assert not bool(result)

    def test_empty_freeze_mask_returns_true(self):
        """If no voxels are frozen, the function must return True (safe path)."""
        from dvfopt.core.spatial3d import _frozen_edges_clean_3d
        D, H, W = 5, 5, 5
        jm = np.ones((D, H, W)) * (-0.5)   # everything negative — but no frozen voxels
        freeze_mask = np.zeros((3, 3, 3), dtype=bool)   # all False
        result = _frozen_edges_clean_3d(jm, 2, 2, 2, (3, 3, 3), THRESHOLD, ERR_TOL,
                                        freeze_mask)
        assert bool(result)

    def test_frozen_voxel_exactly_at_boundary_is_not_clean(self):
        """A frozen voxel with Jdet == threshold - err_tol must return False.
        The check is strict >: value at boundary is NOT clean."""
        from dvfopt.core.spatial3d import _frozen_edges_clean_3d
        boundary = THRESHOLD - ERR_TOL
        jm = np.ones((6, 6, 6)) * 0.5
        jm[2, 3, 3] = boundary   # exactly at boundary on frozen face
        freeze_mask = self._interior_mask()
        result = _frozen_edges_clean_3d(jm, 3, 3, 3, (3, 3, 3), THRESHOLD, ERR_TOL,
                                        freeze_mask)
        assert not bool(result), \
            "Value exactly at threshold-err_tol should NOT be considered clean (strict >)"

    def test_frozen_voxel_just_above_boundary_is_clean(self):
        """A frozen voxel with Jdet just above boundary must return True."""
        from dvfopt.core.spatial3d import _frozen_edges_clean_3d
        boundary = THRESHOLD - ERR_TOL
        jm = np.ones((6, 6, 6)) * 0.5
        jm[2, 3, 3] = boundary + 1e-10   # just above boundary
        freeze_mask = self._interior_mask()
        result = _frozen_edges_clean_3d(jm, 3, 3, 3, (3, 3, 3), THRESHOLD, ERR_TOL,
                                        freeze_mask)
        assert bool(result)


# ===========================================================================
# 3. _FROZEN_EDGES_CLEAN (2D) — exact threshold boundary (strict >)
# ===========================================================================

class TestFrozenEdgesClean2dBoundary:
    """The 2D _frozen_edges_clean checks edge_vals.min() > threshold - err_tol.
    This strict inequality means exactly-at-boundary is NOT clean."""

    def _window_has_boundary_at(self, H=10, W=10, cy=5, cx=5, size=(5, 5)):
        """Return the y,x coordinate of the top-left corner of the window boundary."""
        from dvfopt._defaults import _unpack_size
        sy, sx = _unpack_size(size)
        hy = sy // 2
        hx = sx // 2
        return cy - hy, cx - hx  # top-left boundary pixel

    def test_boundary_value_exactly_at_threshold_is_not_clean(self):
        """edge_vals.min() == threshold - err_tol → returns False (strict >)."""
        from dvfopt.core.spatial import _frozen_edges_clean
        boundary = THRESHOLD - ERR_TOL
        jm = np.ones((1, 10, 10)) * 0.5
        # Top-left corner of the (5,5) window at (5,5) is at y=3, x=3
        jm[0, 3, 3] = boundary
        result = _frozen_edges_clean(jm, 5, 5, (5, 5), THRESHOLD, ERR_TOL)
        assert not result, \
            "Boundary value exactly at threshold-err_tol must not be clean (strict >)"

    def test_boundary_value_just_above_threshold_is_clean(self):
        """edge_vals.min() just above threshold - err_tol → returns True."""
        from dvfopt.core.spatial import _frozen_edges_clean
        boundary = THRESHOLD - ERR_TOL
        jm = np.ones((1, 10, 10)) * 0.5
        jm[0, 3, 3] = boundary + 1e-10
        result = _frozen_edges_clean(jm, 5, 5, (5, 5), THRESHOLD, ERR_TOL)
        assert result, \
            "Value just above threshold-err_tol should be clean"

    def test_all_boundary_pixels_positive_returns_true(self):
        from dvfopt.core.spatial import _frozen_edges_clean
        jm = np.ones((1, 10, 10)) * 1.0
        assert _frozen_edges_clean(jm, 5, 5, (5, 5), THRESHOLD, ERR_TOL)

    def test_inner_pixel_negative_does_not_affect_edges(self):
        """A deeply interior negative pixel must not influence the edge check."""
        from dvfopt.core.spatial import _frozen_edges_clean
        jm = np.ones((1, 10, 10)) * 0.5
        jm[0, 5, 5] = -0.5   # interior of the window, not the boundary
        # Edge pixels are still all 0.5 > threshold - err_tol
        result = _frozen_edges_clean(jm, 5, 5, (5, 5), THRESHOLD, ERR_TOL)
        assert result, "Interior negative must not affect the frozen-edge check"


# ===========================================================================
# 4. _SHOELACE_AREAS_2D FORMULA — verified against analytic geometry
# ===========================================================================

class TestShoelaceFormula:
    """The shoelace formula computes the signed area of each deformed quad cell.

    For known deformations the area is analytically calculable:
      - Identity:            area = 1 everywhere
      - Uniform x-stretch s: area = s (rectangle of width s, height 1)
      - Folded corner:       area < 0 when quad is reversed
    """

    def test_identity_field_gives_area_one(self):
        from dvfopt.jacobian.shoelace import _shoelace_areas_2d
        H, W = 6, 6
        dy = np.zeros((H, W))
        dx = np.zeros((H, W))
        areas = _shoelace_areas_2d(dy, dx)
        np.testing.assert_allclose(areas, 1.0, atol=1e-12,
                                   err_msg="Identity: all quad areas must be 1")

    def test_uniform_x_stretch_gives_area_s(self):
        """phi[1] = (s-1)*x → each quad has width s, height 1 → area = s.

        Derivation: def_x = s*j, def_y = i.  Shoelace for TL→TR→BR→BL:
          = 0.5 * (−s*i + s*(j+1) + s*(i+1) − s*j) = s.
        """
        from dvfopt.jacobian.shoelace import _shoelace_areas_2d
        H, W = 6, 6
        s = 2.5
        dy = np.zeros((H, W))
        dx = (s - 1) * np.arange(W, dtype=float)[np.newaxis, :]
        areas = _shoelace_areas_2d(dy, dx)
        np.testing.assert_allclose(areas, s, atol=1e-10,
                                   err_msg=f"Uniform x-stretch s={s}: area should be {s}")

    def test_uniform_y_stretch_gives_area_s(self):
        """phi[0] = (s-1)*y → each quad has width 1, height s → area = s."""
        from dvfopt.jacobian.shoelace import _shoelace_areas_2d
        H, W = 6, 6
        s = 3.0
        dx = np.zeros((H, W))
        dy = (s - 1) * np.ones((H, W)) * np.arange(H, dtype=float)[:, np.newaxis]
        areas = _shoelace_areas_2d(dy, dx)
        np.testing.assert_allclose(areas, s, atol=1e-10,
                                   err_msg=f"Uniform y-stretch s={s}: area should be {s}")

    def test_reversed_quad_corner_gives_negative_area(self):
        """Pushing TR far to the left of TL reverses the quad orientation.

        Cell (0,0): TL=(0,0), TR=(-2,0), BR=(-2,1), BL=(0,1).
        Shoelace = 0.5*(0*0−(−2)*0) + (−2*1−(−2)*0) + (−2*1−0*1) + (0*0−0*1)
                 = 0.5*(0 + (−2) + (−2) + 0) = −2/2 = −1.
        """
        from dvfopt.jacobian.shoelace import _shoelace_areas_2d
        H, W = 4, 4
        dy = np.zeros((H, W))
        dx = np.zeros((H, W))
        dx[0, 1] = -3.0   # TR of cell (0,0) moved far left → def_x = 1 + (-3) = -2
        areas = _shoelace_areas_2d(dy, dx)
        assert areas[0, 0] < 0, \
            f"Reversed quad at (0,0) should have negative area; got {areas[0, 0]:.4f}"

    def test_zero_area_degenerate_quad(self):
        """Collapsing BR to overlap TL gives area = 0 (TL=BR makes a bowtie).

        Cell (0,0): TL=(0,0), TR=(1,0), BR=(0,0), BL=(0,1)
        Shoelace = 0 because TL and BR share the same position.
        """
        from dvfopt.jacobian.shoelace import _shoelace_areas_2d
        H, W = 4, 4
        dy = np.zeros((H, W))
        dx = np.zeros((H, W))
        dx[1, 1] = -1.0   # BR: def_x = 1 + (-1) = 0 = TL x-position
        dy[1, 1] = -1.0   # BR: def_y = 1 + (-1) = 0 = TL y-position
        areas = _shoelace_areas_2d(dy, dx)
        assert abs(areas[0, 0]) < 1e-10, \
            f"Degenerate quad (BR collapsed to TL) should have area ≈ 0; got {areas[0, 0]:.6f}"


# ===========================================================================
# 5. OBJECTIVEEUC — exact value and gradient
# ===========================================================================

class TestObjectiveEucFormula:
    """objectiveEuc minimises 0.5 * ||phi - phi_init||^2.
    The value and gradient must match this formula exactly."""

    def test_known_value_and_gradient(self):
        """phi = [1, 3, -2], phi_init = [0, 0, 0]:
        value = 0.5*(1+9+4) = 7.0, gradient = [1, 3, -2]."""
        from dvfopt.core.objective import objectiveEuc
        phi = np.array([1.0, 3.0, -2.0])
        phi_init = np.zeros(3)
        val, grad = objectiveEuc(phi, phi_init)
        assert val == pytest.approx(7.0, rel=1e-12), \
            f"Expected value=7.0, got {val}"
        np.testing.assert_allclose(grad, [1.0, 3.0, -2.0], atol=1e-12,
                                   err_msg="Gradient must equal phi - phi_init")

    def test_gradient_is_difference_not_twice_difference(self):
        """Gradient is (phi - phi_init), NOT 2*(phi - phi_init).
        The 0.5 factor in the objective cancels the factor of 2 from the derivative.
        """
        from dvfopt.core.objective import objectiveEuc
        phi = np.array([2.0, -1.0])
        phi_init = np.array([0.5, 0.5])
        _, grad = objectiveEuc(phi, phi_init)
        expected_grad = phi - phi_init   # [1.5, -1.5]
        np.testing.assert_allclose(grad, expected_grad, atol=1e-12,
                                   err_msg="Gradient = (phi - phi_init), not 2*(phi - phi_init)")

    def test_value_is_half_sum_of_squares(self):
        """Value must be exactly 0.5 * dot(diff, diff)."""
        from dvfopt.core.objective import objectiveEuc
        rng = np.random.default_rng(7)
        phi = rng.standard_normal(20)
        phi_init = rng.standard_normal(20)
        val, _ = objectiveEuc(phi, phi_init)
        expected = 0.5 * np.dot(phi - phi_init, phi - phi_init)
        assert val == pytest.approx(expected, rel=1e-12), \
            f"Value mismatch: got {val}, expected {expected}"


# ===========================================================================
# 6. INJECTIVITY DIAGONAL COMPONENTS (d1, d2) REACH QUALITY_MAP
# ===========================================================================

class TestInjectivityDiagonalComponents:
    """The injectivity constraint includes d1 and d2 (diagonal) terms
    beyond h_mono and v_mono.  A phi with valid h/v but invalid diagonal
    must still be flagged by quality_map.

    d1[r,c] = 1 + dx[r, c+1] - dx[r+1, c]
    d2[r,c] = 1 + dy[r+1, c] - dy[r, c+1]
    """

    def test_d1_violation_reflected_in_quality_map(self):
        """phi where h_mono > 0, v_mono > 0, but d1 < 0 at one cell.
        quality_map must be < threshold at the pixels adjoining that cell.
        """
        from dvfopt.core.constraints import _quality_map

        H, W = 6, 6
        phi = np.zeros((2, H, W))
        # d1[r=2, c=2] = 1 + dx[2,3] - dx[3,2] = 1 + 0 - 2.0 = -1.0  (violates)
        # h_mono[r, c] = 1 + dx[r,c+1] - dx[r,c]: all 0 + 0 = 1 (ok)
        # v_mono[r, c] = 1 + dy[r+1,c] - dy[r,c]: all 0 = 1 (ok)
        phi[1, 3, 2] = 2.0   # dx[3,2] = 2.0

        # Note: Jdet at pixel (3,3) becomes 0 due to the gradient at that point,
        # but the test only cares that d1[2,2] = -1 propagates into quality_map.
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=True,
                          jacobian_matrix=jdet)

        # d1 at (2,2) is negative → quality_map at pixels (2,3) and (3,2) should be ≤ d1
        d1_val = 1 + phi[1, 2, 3] - phi[1, 3, 2]   # = 1 + 0 - 2 = -1
        assert d1_val < 0, f"d1 at (2,2) should be negative; got {d1_val}"
        # quality_map must propagate this to the adjacent pixels
        assert qm[0, 2, 3] <= d1_val + 1e-9, \
            f"quality_map[2,3]={qm[0,2,3]:.4f} should reflect d1={d1_val}"
        assert qm[0, 3, 2] <= d1_val + 1e-9, \
            f"quality_map[3,2]={qm[0,3,2]:.4f} should reflect d1={d1_val}"

    def test_d2_violation_reflected_in_quality_map(self):
        """phi where h_mono > 0, v_mono > 0, but d2 < 0.
        d2[r,c] = 1 + dy[r+1,c] - dy[r,c+1].
        Set dy[r+1,c] = 2.0, all else zero → d2 = 1 + 2 - 0 = 3 > 0 ... not a violation.

        To get d2 < 0: dy[r,c+1] must be large positive.
        d2[2,2] = 1 + dy[3,2] - dy[2,3] = 1 + 0 - 2.5 = -1.5.
        """
        from dvfopt.core.constraints import _quality_map

        H, W = 6, 6
        phi = np.zeros((2, H, W))
        phi[0, 2, 3] = 2.5   # dy[2,3] = 2.5

        # Note: Jdet at pixel (3,3) becomes negative due to the y-gradient,
        # but the test only cares that d2[2,2] = -1.5 propagates into quality_map.
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=True,
                          jacobian_matrix=jdet)

        d2_val = 1 + phi[0, 3, 2] - phi[0, 2, 3]   # = 1 + 0 - 2.5 = -1.5
        assert d2_val < 0, f"d2 at (2,2) should be negative; got {d2_val}"
        # d2 is spread to pixels (3,2) and (2,3)
        assert qm[0, 3, 2] <= d2_val + 1e-9 or qm[0, 2, 3] <= d2_val + 1e-9, \
            f"quality_map should reflect d2={d2_val}"

    def test_identity_all_four_mono_are_one(self):
        """For zero displacement, h_mono=v_mono=d1=d2=1 everywhere.
        quality_map should be min(jdet=1, mono=1) = 1 everywhere.
        """
        from dvfopt.core.constraints import _quality_map

        phi = np.zeros((2, 8, 8))
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=True,
                          jacobian_matrix=jdet)
        np.testing.assert_allclose(qm, 1.0, atol=1e-10,
                                   err_msg="Identity: all four mono components = 1")

    def test_h_mono_violation_does_not_confuse_d1_check(self):
        """A pure h_mono violation (no d1/d2 issue) shows up in quality_map
        at the correct pixels, not at unrelated diagonal positions."""
        from dvfopt.core.constraints import _quality_map
        from dvfopt.jacobian.monotonicity import _monotonicity_diffs_2d

        H, W = 6, 6
        phi = np.zeros((2, H, W))
        # h_mono[3,3] = 1 + dx[3,4] - dx[3,3] = 1 + 0 - 1.5 = -0.5
        phi[1, 3, 3] = 1.5   # dx at (3,3)

        h_mono, v_mono = _monotonicity_diffs_2d(phi[0], phi[1])
        assert h_mono[3, 3] < 0, "Setup: h_mono should be negative at (3,3)"

        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=False, enforce_injectivity=True,
                          jacobian_matrix=jdet)
        # quality_map at (3,3) and (3,4) should reflect h_mono violation
        assert qm[0, 3, 3] < 0 or qm[0, 3, 4] < 0, \
            "h_mono violation should appear in quality_map at adjoining pixels"


# ===========================================================================
# 7. _PATCH_JACOBIAN AT GRID BOUNDARY ��� clamping correctness
# ===========================================================================

class TestPatchJacobianBoundary:
    """Patching a window near or at the grid boundary must produce the same
    Jdet as a full recompute — the clamping in wy0/wx0 must be correct."""

    def test_patch_2d_at_corner_matches_full_recompute(self):
        """Centre at (1,1): patch clamping must not produce wrong Jdet."""
        from dvfopt.core.solver import _patch_jacobian_2d

        rng = np.random.default_rng(10)
        H, W = 12, 12
        phi = rng.standard_normal((2, H, W)) * 0.3
        jac = jacobian_det2D(phi).copy()

        # Modify a corner sub-window then patch
        phi[1, 0:3, 0:3] += rng.standard_normal((3, 3)) * 0.1
        jac_full_new = jacobian_det2D(phi)
        _patch_jacobian_2d(jac, phi, (1, 1), (3, 3))

        # Affected region around the corner
        np.testing.assert_allclose(
            jac[0, 0:4, 0:4],
            jac_full_new[0, 0:4, 0:4],
            atol=1e-10,
            err_msg="_patch_jacobian_2d gave wrong values near grid corner",
        )

    def test_patch_2d_at_corner_leaves_distant_pixels_unchanged(self):
        """Patching near (1,1) must not corrupt pixels far from the patch."""
        from dvfopt.core.solver import _patch_jacobian_2d

        rng = np.random.default_rng(11)
        H, W = 12, 12
        phi = rng.standard_normal((2, H, W)) * 0.3
        jac = jacobian_det2D(phi).copy()
        jac_before_far = jac[0, 8:, 8:].copy()

        phi[1, 0:3, 0:3] += rng.standard_normal((3, 3)) * 0.1
        _patch_jacobian_2d(jac, phi, (1, 1), (3, 3))

        np.testing.assert_array_equal(
            jac[0, 8:, 8:], jac_before_far,
            err_msg="Patch near corner modified distant pixels",
        )

    def test_patch_3d_at_grid_face_matches_full_recompute(self):
        """3D patch with centre at (1,4,4): clamping on the z-axis must be correct."""
        from dvfopt.core.solver3d import _patch_jacobian_3d

        rng = np.random.default_rng(20)
        D, H, W = 10, 10, 10
        phi = rng.standard_normal((3, D, H, W)) * 0.2
        jac = jacobian_det3D(phi).copy()

        phi[0, 0:3, 3:6, 3:6] += rng.standard_normal((3, 3, 3)) * 0.1
        jac_full_new = jacobian_det3D(phi)
        _patch_jacobian_3d(jac, phi, (1, 4, 4), (3, 3, 3))

        # Affected z-range (clamped near z=0)
        np.testing.assert_allclose(
            jac[0:4, 3:7, 3:7],
            jac_full_new[0:4, 3:7, 3:7],
            atol=1e-10,
            err_msg="_patch_jacobian_3d gave wrong values near grid face",
        )

    def test_patch_3d_at_grid_face_leaves_distant_voxels_unchanged(self):
        """3D patch near z=0 must not alter voxels far from the affected region."""
        from dvfopt.core.solver3d import _patch_jacobian_3d

        rng = np.random.default_rng(21)
        D, H, W = 10, 10, 10
        phi = rng.standard_normal((3, D, H, W)) * 0.2
        jac = jacobian_det3D(phi).copy()
        jac_before_far = jac[7:, :, :].copy()

        phi[0, 0:3, 3:6, 3:6] += rng.standard_normal((3, 3, 3)) * 0.1
        _patch_jacobian_3d(jac, phi, (1, 4, 4), (3, 3, 3))

        np.testing.assert_array_equal(
            jac[7:, :, :], jac_before_far,
            err_msg="3D patch near z=0 face modified distant voxels",
        )
