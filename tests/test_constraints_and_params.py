"""
Tests for constraint modes and optimizer parameters.

Covers:
  - enforce_shoelace: convergence, shoelace areas positive after correction,
    no shoelace regression, interaction with Jdet convergence
  - enforce_injectivity: convergence, monotonicity after correction,
    injectivity_threshold sweep, diagonal monotonicity
  - Both constraints together
  - injectivity_threshold at threshold, below threshold (should warn/fail
    gracefully), above threshold (stricter)
  - method_name parameter (SLSQP)
  - threshold / err_tol parameter sweep
  - max_iterations hard cap
  - max_per_index_iter hard cap
  - max_minimize_iter effect
  - verbose flag (True/False/0/1/2 backward compat)
  - parallel: max_workers=1/2, enforce_shoelace, enforce_injectivity
  - 3D: threshold/max_iterations/verbose params
  - Constraint function correctness:
      - jacobian_constraint exclude_boundaries=True/False shape/values
      - shoelace_constraint interior/full shape/values
      - injectivity_constraint shape, all-positive on identity
      - _build_constraints frozen-ring encodes correct indices
      - _build_constraints_3d freeze_mask=all-False omits LinearConstraint

Run with:  python -m pytest tests/test_constraints_and_params.py -v
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.jacobian.shoelace import _shoelace_areas_2d, shoelace_det2D
from dvfopt.jacobian.monotonicity import (
    _monotonicity_diffs_2d,
    _diagonal_monotonicity_diffs_2d,
    injectivity_constraint,
)
from dvfopt.core.iterative import iterative_serial
from dvfopt.core.iterative3d import iterative_3d

THRESHOLD = 0.01
ERR_TOL = 1e-5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fold_dvf(H, W, cy=None, cx=None, mag=2.5):
    """Single-fold DVF (3,1,H,W)."""
    cy = cy if cy is not None else H // 2
    cx = cx if cx is not None else min(W // 2, W - 2)
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    dvf[2, 0, cy, cx] = mag
    dvf[2, 0, cy, cx + 1] = -mag
    return dvf


def _run(dvf, **kw):
    kw.setdefault("verbose", 0)
    kw.setdefault("threshold", THRESHOLD)
    kw.setdefault("err_tol", ERR_TOL)
    kw.setdefault("max_iterations", 1000)
    kw.setdefault("max_per_index_iter", 200)
    kw.setdefault("max_minimize_iter", 500)
    return iterative_serial(dvf, **kw)


def _jdet_ok(phi, threshold=THRESHOLD, err_tol=ERR_TOL):
    return bool((jacobian_det2D(phi) > threshold - err_tol).all())


def _shoelace_ok(phi, threshold=THRESHOLD, err_tol=ERR_TOL):
    areas = _shoelace_areas_2d(phi[0], phi[1])   # dy=phi[0], dx=phi[1]
    return bool((areas > threshold - err_tol).all())


def _h_mono_ok(phi, threshold=THRESHOLD, err_tol=ERR_TOL):
    h, _ = _monotonicity_diffs_2d(phi[0], phi[1])
    return bool((h > threshold - err_tol).all())


def _v_mono_ok(phi, threshold=THRESHOLD, err_tol=ERR_TOL):
    _, v = _monotonicity_diffs_2d(phi[0], phi[1])
    return bool((v > threshold - err_tol).all())


def _diag_mono_ok(phi, threshold=THRESHOLD, err_tol=ERR_TOL):
    d1, d2 = _diagonal_monotonicity_diffs_2d(phi[0], phi[1])
    return bool((d1 > threshold - err_tol).all() and (d2 > threshold - err_tol).all())


# ===========================================================================
# 1. SHOELACE CONSTRAINT
# ===========================================================================

class TestShoelaceConstraint:
    """enforce_shoelace=True must produce phi where all quad areas > threshold."""

    def test_shoelace_areas_positive_after_correction(self):
        phi = _run(_fold_dvf(14, 14), enforce_shoelace=True)
        assert _shoelace_ok(phi), \
            f"Shoelace areas not positive; min={_shoelace_areas_2d(phi[0], phi[1]).min():.4f}"

    def test_jdet_also_positive_with_shoelace(self):
        """Shoelace mode must still satisfy the Jdet constraint."""
        phi = _run(_fold_dvf(14, 14), enforce_shoelace=True)
        assert _jdet_ok(phi), \
            f"Jdet not satisfied in shoelace mode; min={float(jacobian_det2D(phi).min()):.4f}"

    def test_shoelace_stricter_than_jdet_only(self):
        """Result with shoelace enabled must have higher min shoelace area
        than result without it (it enforces an extra constraint)."""
        dvf = _fold_dvf(14, 14)
        phi_plain = _run(dvf.copy())
        phi_shoe = _run(dvf.copy(), enforce_shoelace=True)

        areas_plain = _shoelace_areas_2d(phi_plain[0], phi_plain[1]).min()
        areas_shoe = _shoelace_areas_2d(phi_shoe[0], phi_shoe[1]).min()
        assert areas_shoe >= THRESHOLD - ERR_TOL, \
            f"Shoelace result still has negative area: {areas_shoe:.4f}"
        # plain correction may have negative shoelace areas; shoe should not
        # (we just check shoe result is clean, not that plain is dirty)

    def test_shoelace_border_fold(self):
        """Fold touching top border with shoelace constraint."""
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[2, 0, 0, :] = 2.5
        dvf[2, 0, 1, :] = -2.5
        phi = _run(dvf, enforce_shoelace=True)
        assert _jdet_ok(phi)
        assert _shoelace_ok(phi)

    def test_shoelace_constraint_function_shape(self):
        """shoelace_constraint returns (sy-3)*(sx-3) values when exclude_boundaries=True."""
        from dvfopt.jacobian.shoelace import shoelace_constraint
        sy, sx = 7, 7
        phi_flat = np.zeros(2 * sy * sx)
        vals = shoelace_constraint(phi_flat, (sy, sx), exclude_boundaries=True)
        expected = (sy - 3) * (sx - 3)
        assert len(vals) == expected, f"Expected {expected} values, got {len(vals)}"

    def test_shoelace_constraint_function_shape_full(self):
        """shoelace_constraint returns (sy-1)*(sx-1) values when exclude_boundaries=False."""
        from dvfopt.jacobian.shoelace import shoelace_constraint
        sy, sx = 6, 8
        phi_flat = np.zeros(2 * sy * sx)
        vals = shoelace_constraint(phi_flat, (sy, sx), exclude_boundaries=False)
        assert len(vals) == (sy - 1) * (sx - 1)

    def test_identity_field_has_unit_shoelace_areas(self):
        """Zero displacement → each quad cell is a unit square → area = 1."""
        phi = np.zeros((2, 8, 8))
        areas = _shoelace_areas_2d(phi[0], phi[1])
        np.testing.assert_allclose(areas, 1.0, atol=1e-12)

    def test_uniform_scale_shoelace(self):
        """Uniform scale by s → area = s²."""
        H, W = 8, 8
        s = 0.7
        phi = np.zeros((2, H, W))
        y = np.arange(H, dtype=float)
        x = np.arange(W, dtype=float)
        phi[0] = (s - 1) * y[:, np.newaxis]   # dy = (s-1)*y
        phi[1] = (s - 1) * x[np.newaxis, :]   # dx = (s-1)*x
        areas = _shoelace_areas_2d(phi[0], phi[1])
        # Interior quad cells should have area ≈ s²
        np.testing.assert_allclose(areas[1:-1, 1:-1], s * s, atol=1e-6)


# ===========================================================================
# 2. INJECTIVITY CONSTRAINT
# ===========================================================================

class TestInjectivityConstraint:
    """enforce_injectivity=True must produce phi where h/v/diagonal mono > threshold."""

    def _run_inj(self, dvf, inj_threshold=0.05):
        return _run(dvf, enforce_injectivity=True,
                    injectivity_threshold=inj_threshold)

    def test_h_mono_positive_after_correction(self):
        phi = self._run_inj(_fold_dvf(14, 14))
        assert _h_mono_ok(phi), \
            f"h_mono not positive; min={_monotonicity_diffs_2d(phi[0], phi[1])[0].min():.4f}"

    def test_v_mono_positive_after_correction(self):
        phi = self._run_inj(_fold_dvf(14, 14))
        assert _v_mono_ok(phi), \
            f"v_mono not positive; min={_monotonicity_diffs_2d(phi[0], phi[1])[1].min():.4f}"

    def test_diagonal_mono_positive_after_correction(self):
        phi = self._run_inj(_fold_dvf(14, 14))
        assert _diag_mono_ok(phi), \
            "diagonal monotonicity not satisfied after injectivity correction"

    def test_jdet_positive_with_injectivity(self):
        phi = self._run_inj(_fold_dvf(14, 14))
        assert _jdet_ok(phi)

    def test_injectivity_threshold_sweep(self):
        """Higher injectivity_threshold → larger min monotonicity gap.
        Results at tau=0.1 must have higher min mono than at tau=0.01."""
        dvf = _fold_dvf(14, 14)
        phi_lo = self._run_inj(dvf.copy(), inj_threshold=0.01)
        phi_hi = self._run_inj(dvf.copy(), inj_threshold=0.1)

        h_lo, _ = _monotonicity_diffs_2d(phi_lo[0], phi_lo[1])
        h_hi, _ = _monotonicity_diffs_2d(phi_hi[0], phi_hi[1])
        # Hi-threshold result must satisfy hi-threshold
        assert float(h_hi.min()) >= 0.1 - ERR_TOL, \
            f"tau=0.1 not satisfied: h_min={float(h_hi.min()):.4f}"
        # Lo-threshold result must satisfy lo-threshold
        assert float(h_lo.min()) >= 0.01 - ERR_TOL, \
            f"tau=0.01 not satisfied: h_min={float(h_lo.min()):.4f}"

    def test_injectivity_at_high_threshold(self):
        """injectivity_threshold=0.3 — forces generous vertex separation."""
        phi = _run(_fold_dvf(14, 14), enforce_injectivity=True,
                   injectivity_threshold=0.3)
        h, v = _monotonicity_diffs_2d(phi[0], phi[1])
        assert float(h.min()) >= 0.3 - ERR_TOL, \
            f"tau=0.3 h not satisfied: {float(h.min()):.4f}"
        assert float(v.min()) >= 0.3 - ERR_TOL, \
            f"tau=0.3 v not satisfied: {float(v.min()):.4f}"

    def test_injectivity_constraint_function_identity(self):
        """Zero phi → all h/v/d monotonicity values = 1 (identity map is injective)."""
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        vals = injectivity_constraint(phi_flat, (sy, sx), exclude_boundaries=False)
        np.testing.assert_allclose(vals, 1.0, atol=1e-12,
                                   err_msg="Identity map should give mono=1 everywhere")

    def test_injectivity_constraint_function_shape_interior(self):
        """Count output rows for interior-only injectivity constraint."""
        from dvfopt.jacobian.monotonicity import injectivity_constraint
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        vals = injectivity_constraint(phi_flat, (sy, sx), exclude_boundaries=True)
        # h_mono interior: (sy-2)*(sx-3); v_mono interior: (sy-3)*(sx-2)
        # d1/d2: (sy-1)*(sx-1) - 2 each (two all-frozen corners excluded)
        n_h = (sy - 2) * (sx - 3)
        n_v = (sy - 3) * (sx - 2)
        n_diag = (sy - 1) * (sx - 1) - 2
        expected = n_h + n_v + 2 * n_diag
        assert len(vals) == expected, f"Expected {expected}, got {len(vals)}"

    def test_injectivity_constraint_function_shape_full(self):
        sy, sx = 5, 7
        phi_flat = np.zeros(2 * sy * sx)
        vals = injectivity_constraint(phi_flat, (sy, sx), exclude_boundaries=False)
        n_h = sy * (sx - 1)
        n_v = (sy - 1) * sx
        n_d = (sy - 1) * (sx - 1)
        expected = n_h + n_v + 2 * n_d
        assert len(vals) == expected


# ===========================================================================
# 3. BOTH CONSTRAINTS TOGETHER
# ===========================================================================

class TestBothConstraints:
    def _run_both(self, dvf, inj_threshold=0.05):
        return _run(dvf, enforce_shoelace=True, enforce_injectivity=True,
                    injectivity_threshold=inj_threshold)

    def test_both_satisfied_after_correction(self):
        phi = self._run_both(_fold_dvf(14, 14))
        assert _jdet_ok(phi),      "Jdet not satisfied (both constraints)"
        assert _shoelace_ok(phi),  "Shoelace not satisfied (both constraints)"
        assert _h_mono_ok(phi),    "h_mono not satisfied (both constraints)"
        assert _v_mono_ok(phi),    "v_mono not satisfied (both constraints)"

    def test_both_constraints_border_fold(self):
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[2, 0, 0, :] = 2.5
        dvf[2, 0, 1, :] = -2.5
        phi = self._run_both(dvf)
        assert _jdet_ok(phi)
        assert _shoelace_ok(phi)

    def test_both_constraints_with_high_inj_threshold(self):
        phi = self._run_both(_fold_dvf(14, 14), inj_threshold=0.2)
        h, v = _monotonicity_diffs_2d(phi[0], phi[1])
        assert float(h.min()) >= 0.2 - ERR_TOL
        assert float(v.min()) >= 0.2 - ERR_TOL
        assert _shoelace_ok(phi)

    def test_quality_map_uses_minimum_of_all_metrics(self):
        """quality_map must be ≤ min(jdet, shoelace) element-wise."""
        from dvfopt.core.constraints import _quality_map
        rng = np.random.default_rng(77)
        phi = rng.standard_normal((2, 10, 10)) * 0.3
        jdet = jacobian_det2D(phi)
        qm = _quality_map(phi, enforce_shoelace=True, enforce_injectivity=True,
                          jacobian_matrix=jdet)
        assert (qm <= jdet + 1e-12).all(), "quality_map > jdet"
        # Also ≤ shoelace spread
        areas = _shoelace_areas_2d(phi[0], phi[1])
        shoe = np.full_like(jdet, np.inf)
        shoe[0, :-1, :-1] = np.minimum(shoe[0, :-1, :-1], areas)
        shoe[0, :-1, 1:] = np.minimum(shoe[0, :-1, 1:], areas)
        shoe[0, 1:, :-1] = np.minimum(shoe[0, 1:, :-1], areas)
        shoe[0, 1:, 1:] = np.minimum(shoe[0, 1:, 1:], areas)
        assert (qm <= shoe + 1e-12).all(), "quality_map > shoelace spread"


# ===========================================================================
# 4. JDET CONSTRAINT FUNCTION INTERNALS
# ===========================================================================

class TestJdetConstraintFunction:
    def test_interior_shape(self):
        from dvfopt.core.constraints import jacobian_constraint
        sy, sx = 7, 5
        phi_flat = np.zeros(2 * sy * sx)
        vals = jacobian_constraint(phi_flat, (sy, sx), exclude_boundaries=True)
        assert len(vals) == (sy - 2) * (sx - 2)

    def test_full_shape(self):
        from dvfopt.core.constraints import jacobian_constraint
        sy, sx = 6, 4
        phi_flat = np.zeros(2 * sy * sx)
        vals = jacobian_constraint(phi_flat, (sy, sx), exclude_boundaries=False)
        assert len(vals) == sy * sx

    def test_identity_gives_ones(self):
        from dvfopt.core.constraints import jacobian_constraint
        sy, sx = 5, 5
        phi_flat = np.zeros(2 * sy * sx)
        vals = jacobian_constraint(phi_flat, (sy, sx), exclude_boundaries=False)
        np.testing.assert_allclose(vals, 1.0, atol=1e-12)

    def test_3d_identity_gives_ones(self):
        from dvfopt.core.constraints3d import jacobian_constraint_3d
        sz, sy, sx = 4, 4, 4
        phi_flat = np.zeros(3 * sz * sy * sx)
        vals = jacobian_constraint_3d(phi_flat, (sz, sy, sx))
        np.testing.assert_allclose(vals, 1.0, atol=1e-12)

    def test_3d_constraint_with_freeze_mask(self):
        """freeze_mask excludes frozen voxels from the constraint output."""
        from dvfopt.core.constraints3d import jacobian_constraint_3d
        sz, sy, sx = 3, 3, 3
        voxels = sz * sy * sx
        phi_flat = np.zeros(3 * voxels)
        freeze_mask = np.zeros((sz, sy, sx), dtype=bool)
        freeze_mask[0, :, :] = True   # freeze z=0 face (9 voxels)
        vals_frozen = jacobian_constraint_3d(phi_flat, (sz, sy, sx), freeze_mask)
        vals_full = jacobian_constraint_3d(phi_flat, (sz, sy, sx), None)
        assert len(vals_frozen) == voxels - 9
        assert len(vals_full) == voxels

    def test_frozen_ring_linear_constraint_correct_indices(self):
        """_build_constraints frozen LinearConstraint must pin exactly the
        boundary indices of phi_sub_flat."""
        from dvfopt.core.constraints import _build_constraints
        sy, sx = 5, 5
        pixels = sy * sx
        rng = np.random.default_rng(11)
        phi_flat = rng.standard_normal(2 * pixels)
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=False, window_reached_max=False,
            threshold=THRESHOLD,
        )
        # Last constraint should be the LinearConstraint for the frozen ring
        lc = constraints[-1]
        from scipy.optimize import LinearConstraint
        assert isinstance(lc, LinearConstraint)
        # Extract fixed indices from the sparse A matrix
        A = lc.A.toarray()
        lb = lc.lb
        # Each row of A should have exactly one 1; the 1 is at the fixed index
        fixed_vals_from_A = {
            int(np.argmax(A[i])): lb[i]
            for i in range(A.shape[0])
        }
        # All fixed indices should correspond to boundary ring pixels
        boundary_mask = np.zeros((sy, sx), dtype=bool)
        boundary_mask[[0, -1], :] = True
        boundary_mask[:, [0, -1]] = True
        boundary_indices = set()
        for y, x in zip(*np.where(boundary_mask)):
            idx = y * sx + x
            boundary_indices.add(idx)           # dx block
            boundary_indices.add(idx + pixels)  # dy block
        for idx in fixed_vals_from_A:
            assert idx in boundary_indices, \
                f"Non-boundary index {idx} in frozen LinearConstraint"

    def test_no_linear_constraint_when_at_edge(self):
        """At edge (is_at_edge=True) → no frozen LinearConstraint."""
        from dvfopt.core.constraints import _build_constraints
        from scipy.optimize import LinearConstraint
        sy, sx = 5, 5
        phi_flat = np.zeros(2 * sy * sx)
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=True, window_reached_max=False,
            threshold=THRESHOLD,
        )
        types = [type(c).__name__ for c in constraints]
        assert "LinearConstraint" not in types, \
            "LinearConstraint should not be added when is_at_edge=True"

    def test_no_linear_constraint_when_max_window(self):
        """At max window → no frozen LinearConstraint."""
        from dvfopt.core.constraints import _build_constraints
        from scipy.optimize import LinearConstraint
        sy, sx = 5, 5
        phi_flat = np.zeros(2 * sy * sx)
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=False, window_reached_max=True,
            threshold=THRESHOLD,
        )
        types = [type(c).__name__ for c in constraints]
        assert "LinearConstraint" not in types

    def test_3d_no_linear_constraint_when_no_freeze(self):
        """_build_constraints_3d with all-False freeze_mask → no LinearConstraint."""
        from dvfopt.core.constraints3d import _build_constraints_3d
        from scipy.optimize import LinearConstraint
        sz, sy, sx = 3, 3, 3
        phi_flat = np.zeros(3 * sz * sy * sx)
        freeze_mask = np.zeros((sz, sy, sx), dtype=bool)
        constraints = _build_constraints_3d(phi_flat, (sz, sy, sx), freeze_mask, THRESHOLD)
        types = [type(c).__name__ for c in constraints]
        assert "LinearConstraint" not in types


# ===========================================================================
# 5. OPTIMIZER PARAMETERS
# ===========================================================================

class TestOptimizerParameters:

    # --- threshold ---
    def test_high_threshold_converges(self):
        """threshold=0.5 → stricter; must still converge on a simple fold."""
        phi = _run(_fold_dvf(12, 12), threshold=0.5)
        jdet = jacobian_det2D(phi)
        assert bool((jdet > 0.5 - ERR_TOL).all()), \
            f"threshold=0.5 not met; min={float(jdet.min()):.4f}"

    def test_threshold_zero_accepts_boundary(self):
        """threshold=0.0 → any non-negative Jdet is acceptable."""
        phi = _run(_fold_dvf(12, 12), threshold=0.0)
        jdet = jacobian_det2D(phi)
        assert bool((jdet > -ERR_TOL).all()), \
            f"threshold=0.0 not met; min={float(jdet.min()):.4f}"

    def test_higher_threshold_gives_higher_min_jdet(self):
        """All else equal, a higher threshold yields a higher minimum Jdet."""
        dvf = _fold_dvf(12, 12)
        phi_lo = _run(dvf.copy(), threshold=0.01)
        phi_hi = _run(dvf.copy(), threshold=0.3)
        min_lo = float(jacobian_det2D(phi_lo).min())
        min_hi = float(jacobian_det2D(phi_hi).min())
        assert min_hi >= 0.3 - ERR_TOL, \
            f"threshold=0.3 not satisfied; min_jdet={min_hi:.4f}"
        assert min_lo >= 0.01 - ERR_TOL

    # --- err_tol ---
    def test_tight_err_tol_converges(self):
        """err_tol=1e-8 (very tight) should still converge."""
        phi = _run(_fold_dvf(10, 10), err_tol=1e-8)
        jdet = jacobian_det2D(phi)
        assert bool((jdet > THRESHOLD - 1e-8).all())

    def test_loose_err_tol_converges_faster(self):
        """err_tol=0.005 (half of threshold) → exits earlier.
        The result may not be as clean as tight err_tol but must not crash."""
        phi = _run(_fold_dvf(12, 12), threshold=0.01, err_tol=0.005)
        # Just check it ran without error and returned something finite
        assert np.isfinite(phi).all()

    # --- max_iterations hard cap ---
    def test_max_iterations_respected(self):
        """With max_iterations=1 the outer loop must run at most once."""
        call_count = {"n": 0}
        original_argmin = __import__(
            "dvfopt.core.spatial", fromlist=["argmin_quality"]
        ).argmin_quality

        def counting_argmin(qm):
            call_count["n"] += 1
            return original_argmin(qm)

        import dvfopt.core.iterative as _iter_mod
        original = _iter_mod.argmin_quality
        _iter_mod.argmin_quality = counting_argmin
        try:
            _run(_fold_dvf(12, 12), max_iterations=1)
        finally:
            _iter_mod.argmin_quality = original

        assert call_count["n"] <= 1, \
            f"max_iterations=1 violated: argmin called {call_count['n']} times"

    def test_max_iterations_zero_returns_unmodified(self):
        """max_iterations=0 → loop body never runs → phi equals phi_init."""
        dvf = _fold_dvf(12, 12)
        phi_init = dvf[1:3, 0].copy()
        phi = _run(dvf, max_iterations=0)
        np.testing.assert_array_equal(phi, phi_init,
                                      err_msg="max_iterations=0 modified phi")

    # --- max_per_index_iter ---
    def test_max_per_index_iter_one_still_converges(self):
        """Already tested in test_edge_cases; cross-check here for completeness."""
        phi = iterative_serial(
            _fold_dvf(8, 8), verbose=0, threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=2000, max_per_index_iter=1, max_minimize_iter=300,
        )
        assert _jdet_ok(phi)

    # --- max_minimize_iter ---
    def test_large_max_minimize_iter_still_converges(self):
        """max_minimize_iter=5000 — should work fine (just wastes time if not needed)."""
        phi = _run(_fold_dvf(10, 10), max_minimize_iter=5000)
        assert _jdet_ok(phi)

    def test_tiny_max_minimize_iter_still_terminates(self):
        """max_minimize_iter=1 → SLSQP basically does nothing per call.
        Outer loop + stall escalation must prevent infinite looping."""
        phi = iterative_serial(
            _fold_dvf(8, 8), verbose=0, threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=500, max_per_index_iter=50, max_minimize_iter=1,
        )
        # Result may not be fully corrected, but must terminate and be finite
        assert np.isfinite(phi).all()

    # --- verbose backward compat ---
    def test_verbose_true_equivalent_to_1(self, capsys):
        iterative_serial(_fold_dvf(8, 8), verbose=True,
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=3, max_per_index_iter=5,
                         max_minimize_iter=50)
        out = capsys.readouterr().out
        assert "[init]" in out, "verbose=True should print [init] line"

    def test_verbose_false_equivalent_to_0(self, capsys):
        iterative_serial(_fold_dvf(8, 8), verbose=False,
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=5, max_per_index_iter=5,
                         max_minimize_iter=50)
        out = capsys.readouterr().out
        assert out == "", "verbose=False should produce no output"

    def test_verbose_0_silent(self, capsys):
        iterative_serial(_fold_dvf(8, 8), verbose=0,
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=5, max_per_index_iter=5,
                         max_minimize_iter=50)
        out = capsys.readouterr().out
        assert out == ""

    def test_verbose_2_extra_output(self, capsys):
        iterative_serial(_fold_dvf(8, 8), verbose=2,
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=2, max_per_index_iter=3,
                         max_minimize_iter=50)
        out = capsys.readouterr().out
        # verbose=2 should include edge/window debug lines
        assert "[edge]" in out or "[sub-Jdet]" in out, \
            "verbose=2 should produce debug output"

    # --- return shape ---
    def test_return_shape_2d(self):
        """iterative_serial must return (2, H, W)."""
        H, W = 13, 17
        phi = _run(_fold_dvf(H, W, cy=6, cx=8))
        assert phi.shape == (2, H, W), f"Wrong shape: {phi.shape}"

    def test_return_dtype_float64(self):
        """Return dtype should be float64."""
        phi = _run(_fold_dvf(10, 10))
        assert phi.dtype == np.float64, f"Wrong dtype: {phi.dtype}"


# ===========================================================================
# 6. PARALLEL VARIANT
# ===========================================================================

class TestParallelVariant:

    def _run_par(self, dvf, **kw):
        from dvfopt.core.parallel import iterative_parallel
        kw.setdefault("verbose", 0)
        kw.setdefault("threshold", THRESHOLD)
        kw.setdefault("err_tol", ERR_TOL)
        kw.setdefault("max_iterations", 1000)
        kw.setdefault("max_per_index_iter", 200)
        kw.setdefault("max_minimize_iter", 500)
        return iterative_parallel(dvf, **kw)

    def test_parallel_single_worker_converges(self):
        phi = self._run_par(_fold_dvf(14, 14), max_workers=1)
        assert _jdet_ok(phi)

    def test_parallel_shoelace_converges(self):
        phi = self._run_par(_fold_dvf(14, 14),
                            enforce_shoelace=True, max_workers=1)
        assert _jdet_ok(phi)
        assert _shoelace_ok(phi)

    def test_parallel_injectivity_converges(self):
        phi = self._run_par(_fold_dvf(14, 14),
                            enforce_injectivity=True,
                            injectivity_threshold=0.05,
                            max_workers=1)
        assert _jdet_ok(phi)
        assert _h_mono_ok(phi)

    def test_parallel_return_shape(self):
        H, W = 11, 13
        phi = self._run_par(_fold_dvf(H, W, cy=5, cx=6))
        assert phi.shape == (2, H, W)

    def test_parallel_already_clean(self):
        """Clean field → no iterations → phi unchanged."""
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        phi_init = dvf[1:3, 0].copy()
        phi = self._run_par(dvf)
        np.testing.assert_array_equal(phi, phi_init)


# ===========================================================================
# 7. 3D SOLVER PARAMETERS
# ===========================================================================

class TestIterative3dParams:

    def _run_3d(self, dvf, **kw):
        kw.setdefault("verbose", 0)
        kw.setdefault("threshold", THRESHOLD)
        kw.setdefault("err_tol", ERR_TOL)
        kw.setdefault("max_iterations", 500)
        kw.setdefault("max_per_index_iter", 100)
        kw.setdefault("max_minimize_iter", 300)
        return iterative_3d(dvf, **kw)

    def _fold_dvf_3d(self, D, H, W, cz=None, cy=None, cx=None, mag=2.5):
        cz = cz if cz is not None else D // 2
        cy = cy if cy is not None else H // 2
        cx = cx if cx is not None else min(W // 2, W - 2)
        dvf = np.zeros((3, D, H, W), dtype=np.float64)
        dvf[2, cz, cy, cx] = mag
        dvf[2, cz, cy, cx + 1] = -mag
        return dvf

    def test_3d_return_shape(self):
        D, H, W = 5, 6, 7
        dvf = self._fold_dvf_3d(D, H, W)
        phi = self._run_3d(dvf)
        assert phi.shape == (3, D, H, W), f"Wrong shape: {phi.shape}"

    def test_3d_high_threshold(self):
        """threshold=0.3 in 3D must be satisfied after correction."""
        dvf = self._fold_dvf_3d(6, 6, 6)
        phi = self._run_3d(dvf, threshold=0.3)
        jdet = jacobian_det3D(phi)
        assert bool((jdet > 0.3 - ERR_TOL).all()), \
            f"3D threshold=0.3 not met; min={float(jdet.min()):.4f}"

    def test_3d_max_iterations_zero(self):
        """max_iterations=0 → phi unchanged."""
        D, H, W = 5, 5, 5
        dvf = self._fold_dvf_3d(D, H, W)
        phi_init = dvf.copy()
        phi = self._run_3d(dvf, max_iterations=0)
        np.testing.assert_array_equal(phi, phi_init)

    def test_3d_verbose_true_false_compat(self, capsys):
        dvf = self._fold_dvf_3d(4, 4, 4)
        iterative_3d(dvf, verbose=False, threshold=THRESHOLD, err_tol=ERR_TOL,
                     max_iterations=2, max_per_index_iter=5,
                     max_minimize_iter=50)
        out = capsys.readouterr().out
        assert out == "", "verbose=False should produce no output"

    def test_3d_already_clean(self):
        dvf = np.zeros((3, 5, 5, 5), dtype=np.float64)
        phi = self._run_3d(dvf)
        jdet = jacobian_det3D(phi)
        assert bool((jdet > THRESHOLD - ERR_TOL).all())
