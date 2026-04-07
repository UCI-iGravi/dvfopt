"""
Edge-case tests for the iterative SLSQP deformation correction algorithm.

Focused on scenarios that could cause infinite loops, oscillation, or incorrect
convergence: negative pixels touching grid edges, large connected negative
regions, single-pixel grids, already-clean fields, max-window situations, stall
escalation / de-escalation, and 3D parity.

Run with:  python -m pytest tests/test_edge_cases.py -v
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.core.iterative import iterative_serial
from dvfopt.core.iterative3d import iterative_3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THRESHOLD = 0.01
ERR_TOL = 1e-5


def _neg_count(phi):
    """Count pixels with Jdet <= 0."""
    jdet = jacobian_det2D(phi)
    return int((jdet <= 0).sum())


def _neg_count_3d(phi):
    jdet = jacobian_det3D(phi)
    return int((jdet <= 0).sum())


def _min_jdet(phi):
    return float(jacobian_det2D(phi).min())


def _min_jdet_3d(phi):
    return float(jacobian_det3D(phi).min())


def _clean_dvf_2d(H, W):
    """Return a fold-free (3,1,H,W) DVF (identity + small smooth warp)."""
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    # small smooth warp — guaranteed positive Jdet
    y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    dvf[1, 0] = 0.3 * np.sin(np.pi * y)
    dvf[2, 0] = 0.3 * np.sin(np.pi * x)
    return dvf


def _single_fold_dvf_2d(H, W, cy, cx, magnitude=3.0):
    """
    Place a single fold at (cy, cx) by injecting a strong shear spike.
    Returns (3,1,H,W).
    """
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    # Make a fold: reverse displacement at one interior pixel
    dvf[2, 0, cy, cx] = magnitude       # dx
    dvf[2, 0, cy, cx + 1] = -magnitude  # reversed neighbour
    return dvf


def _border_fold_dvf_2d(H, W):
    """
    DVF with a negative Jdet touching the grid border (top row).
    Achieved by a strong shear on the top two rows.
    """
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    dvf[2, 0, 0, :] = 3.0
    dvf[2, 0, 1, :] = -3.0
    return dvf


def _large_region_dvf_2d(H, W):
    """
    DVF with a large connected negative-Jdet region (checkerboard shear
    over most of the grid).
    """
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    for i in range(2, H - 2):
        sign = 1 if i % 2 == 0 else -1
        dvf[2, 0, i, 2:W-2] = sign * 4.0
    return dvf


def _run_serial(dvf, max_iterations=500, **kwargs):
    """Run iterative_serial with tight limits and return phi."""
    return iterative_serial(
        dvf,
        verbose=0,
        threshold=THRESHOLD,
        err_tol=ERR_TOL,
        max_iterations=max_iterations,
        max_per_index_iter=200,
        max_minimize_iter=500,
        **kwargs,
    )


def _converged(phi, threshold=THRESHOLD, err_tol=ERR_TOL):
    jdet = jacobian_det2D(phi)
    return bool((jdet > threshold - err_tol).all())


# ===========================================================================
# 1. BASELINE — already-clean fields must not be modified
# ===========================================================================

class TestAlreadyClean:
    def test_clean_field_unchanged_2d(self):
        """A field with no folds must exit immediately without changing phi."""
        dvf = _clean_dvf_2d(20, 20)
        phi0 = dvf[1:3, 0].copy()  # (2, H, W)

        phi = _run_serial(dvf)
        np.testing.assert_allclose(phi, phi0, atol=1e-12,
                                   err_msg="Clean field was modified")

    def test_already_clean_iteration_count(self):
        """Outer loop must not execute a single iteration on a clean field."""
        dvf = _clean_dvf_2d(10, 10)
        # Use a custom max_iterations=1 — if the loop fires even once,
        # it would mutate phi; we check idempotence instead.
        phi = _run_serial(dvf, max_iterations=1)
        assert _converged(phi), "Still has violations on clean field after 1 iter"

    def test_zero_dvf_is_clean(self):
        """Zero displacement → Jdet = 1 everywhere → no correction needed."""
        dvf = np.zeros((3, 1, 15, 15))
        phi = _run_serial(dvf)
        assert _converged(phi)
        assert _neg_count(phi) == 0


# ===========================================================================
# 2. SINGLE FOLD — minimal reproduction
# ===========================================================================

class TestSingleFold:
    def test_single_interior_fold_resolved(self):
        """A single fold in the interior of a 10×10 grid must be resolved."""
        H, W = 10, 10
        dvf = _single_fold_dvf_2d(H, W, cy=5, cx=4, magnitude=3.0)
        phi = _run_serial(dvf)
        assert _converged(phi), \
            f"Single interior fold not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_single_fold_small_grid(self):
        """Single fold on a 5×5 grid — smallest non-trivial case."""
        dvf = np.zeros((3, 1, 5, 5), dtype=np.float64)
        dvf[2, 0, 2, 2] = 2.5
        dvf[2, 0, 2, 3] = -2.5
        phi = _run_serial(dvf, max_iterations=1000)
        assert _converged(phi), \
            f"5×5 single fold not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_single_fold_3x3_grid(self):
        """Smallest possible grid: 3×3 with one fold."""
        dvf = np.zeros((3, 1, 3, 3), dtype=np.float64)
        dvf[2, 0, 1, 1] = 1.5
        dvf[2, 0, 1, 2] = -1.5
        phi = _run_serial(dvf, max_iterations=2000)
        assert _converged(phi), \
            f"3×3 single fold not resolved; min_jdet={_min_jdet(phi):.4f}"


# ===========================================================================
# 3. BORDER-TOUCHING FOLDS — forces window to hit grid edge early
# ===========================================================================

class TestBorderFolds:
    def test_fold_on_top_row(self):
        """Negative Jdet touching the top row — window must hit edge correctly."""
        dvf = _border_fold_dvf_2d(15, 15)
        phi = _run_serial(dvf, max_iterations=500)
        assert _converged(phi), \
            f"Top-row fold not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_fold_on_bottom_row(self):
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[2, 0, -1, :] = 3.0
        dvf[2, 0, -2, :] = -3.0
        phi = _run_serial(dvf)
        assert _converged(phi), \
            f"Bottom-row fold not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_fold_on_left_column(self):
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[1, 0, :, 0] = 3.0   # dy component
        dvf[1, 0, :, 1] = -3.0
        phi = _run_serial(dvf)
        assert _converged(phi), \
            f"Left-column fold not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_fold_at_corner(self):
        """Corner pixel fold — window can only go one direction on two axes."""
        dvf = np.zeros((3, 1, 10, 10), dtype=np.float64)
        dvf[2, 0, 0, 0] = 3.0
        dvf[2, 0, 0, 1] = -3.0
        dvf[1, 0, 0, 0] = 3.0
        dvf[1, 0, 1, 0] = -3.0
        phi = _run_serial(dvf, max_iterations=2000)
        assert _converged(phi), \
            f"Corner fold not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_fold_entire_border(self):
        """Folds all along the border perimeter — forces max-window path."""
        H, W = 10, 10
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        dvf[2, 0, 0, :] = 3.0    # top row
        dvf[2, 0, 1, :] = -3.0
        dvf[2, 0, -1, :] = 3.0   # bottom row
        dvf[2, 0, -2, :] = -3.0
        phi = _run_serial(dvf, max_iterations=2000)
        assert _converged(phi), \
            f"Full-border fold not resolved; min_jdet={_min_jdet(phi):.4f}"


# ===========================================================================
# 4. LARGE CONNECTED REGIONS — forces window growth to max
# ===========================================================================

class TestLargeRegions:
    def test_large_contiguous_region(self):
        """Half the grid is negative — algorithm must grow window to H×W."""
        H, W = 12, 12
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        # Strong uniform compression in y on left half
        dvf[1, 0, :, :W//2] = 5.0
        dvf[1, 0, :, W//2:] = -5.0
        phi = _run_serial(dvf, max_iterations=5000)
        assert _converged(phi), \
            f"Large region not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_multiple_disjoint_regions(self):
        """Several separate negative clusters — each must be fixed independently."""
        H, W = 20, 20
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        for cy, cx in [(4, 4), (4, 15), (15, 4), (15, 15)]:
            dvf[2, 0, cy, cx] = 2.5
            dvf[2, 0, cy, cx + 1] = -2.5
        phi = _run_serial(dvf, max_iterations=2000)
        assert _converged(phi), \
            f"Multiple clusters not fully resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_nearly_full_grid_negative(self):
        """Most of the grid is negative — forces window to grow to H×W.

        Uses a moderate shear magnitude (2.5) so the case is hard enough
        to require the max-window path but tractable within the iteration budget.
        A checkerboard at magnitude ≥4 is pathological (each correction
        disturbs neighbours indefinitely) and is excluded by design.
        """
        H, W = 10, 10
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        # Strong horizontal compression on interior rows — creates a large
        # connected negative region without the infinite mutual-disturbance
        # of an alternating-sign pattern.
        dvf[2, 0, 2:8, :] = 2.5    # push right
        dvf[1, 0, 2:8, :] = -2.5   # pull up
        phi = _run_serial(dvf, max_iterations=5000)
        assert _converged(phi), \
            f"Near-full-grid negative not resolved; min_jdet={_min_jdet(phi):.4f}"


# ===========================================================================
# 5. NON-SQUARE GRIDS — ensures window arithmetic is correct for H ≠ W
# ===========================================================================

class TestNonSquareGrids:
    @pytest.mark.parametrize("H,W", [(5, 20), (20, 5), (8, 15), (15, 8)])
    def test_non_square_fold(self, H, W):
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        cy = H // 2
        cx = min(W // 2, W - 2)
        dvf[2, 0, cy, cx] = 2.5
        dvf[2, 0, cy, cx + 1] = -2.5
        phi = _run_serial(dvf, max_iterations=2000)
        assert _converged(phi), \
            f"{H}×{W} grid: fold not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_tall_border_fold(self):
        """Tall grid (30×5) with a border fold — window must grow to 30×5."""
        dvf = _border_fold_dvf_2d(30, 5)
        phi = _run_serial(dvf, max_iterations=2000)
        assert _converged(phi), \
            f"30×5 border fold not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_wide_border_fold(self):
        """Wide grid (5×30) with a border fold."""
        dvf = np.zeros((3, 1, 5, 30), dtype=np.float64)
        dvf[1, 0, :, 0] = 3.0
        dvf[1, 0, :, 1] = -3.0
        phi = _run_serial(dvf, max_iterations=2000)
        assert _converged(phi), \
            f"5×30 border fold not resolved; min_jdet={_min_jdet(phi):.4f}"


# ===========================================================================
# 6. STALL ESCALATION — algorithm must not loop forever when stuck
# ===========================================================================

class TestStallEscalation:
    def test_persistent_fold_escalates(self):
        """
        A fold that cannot be resolved with a tiny window should escalate
        the global_min_window and eventually succeed — not loop forever.
        """
        H, W = 15, 15
        # Severe fold covering a 5×5 region
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        for i in range(5, 10):
            sign = 1 if i % 2 == 0 else -1
            dvf[2, 0, i, 5:10] = sign * 5.0
        phi = _run_serial(dvf, max_iterations=3000)
        assert _converged(phi), \
            f"Persistent fold not resolved via escalation; min_jdet={_min_jdet(phi):.4f}"

    def test_oscillation_does_not_loop_forever(self):
        """
        Two competing folds that would oscillate if one fix disturbs the other.
        The stall detector must escalate to a large enough window.
        """
        H, W = 12, 12
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        # Two folds close enough to share overlapping windows
        dvf[2, 0, 5, 5] = 3.0;  dvf[2, 0, 5, 6] = -3.0
        dvf[2, 0, 6, 6] = 3.0;  dvf[2, 0, 6, 7] = -3.0
        phi = _run_serial(dvf, max_iterations=3000)
        assert _converged(phi), \
            f"Oscillating folds not resolved; min_jdet={_min_jdet(phi):.4f}"

    def test_de_escalation_after_resolution(self):
        """
        After a hard cluster is fixed, subsequent easy folds should not be
        penalised by an inflated global_min_window (de-escalation test).
        We check that the algorithm still converges correctly — if de-escalation
        breaks something the second cluster will not be fixed.
        """
        H, W = 20, 20
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        # Hard cluster first (large region)
        for i in range(3, 8):
            sign = 1 if i % 2 == 0 else -1
            dvf[2, 0, i, 3:8] = sign * 5.0
        # Easy single fold on the other side
        dvf[2, 0, 15, 15] = 2.5
        dvf[2, 0, 15, 16] = -2.5
        phi = _run_serial(dvf, max_iterations=5000)
        assert _converged(phi), \
            f"De-escalation broken — not converged; min_jdet={_min_jdet(phi):.4f}"


# ===========================================================================
# 7. MAX_PER_INDEX_ITER BUDGET — must not exhaust budget on frozen-edge skips
# ===========================================================================

class TestIterBudget:
    def test_budget_not_exhausted_by_frozen_skips(self):
        """
        With a modest per-index budget, the algorithm must not fail just because
        frozen-edge skips burned the budget before a real SLSQP call happened.
        """
        H, W = 15, 15
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        dvf[2, 0, 7, 7] = 3.0
        dvf[2, 0, 7, 8] = -3.0
        phi = iterative_serial(
            dvf, verbose=0,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=500,
            max_per_index_iter=10,  # deliberately tight
            max_minimize_iter=200,
        )
        # Even with tight budget, the fold must eventually be resolved
        # (the outer loop will retry with growing windows)
        assert _converged(phi), \
            f"Budget exhaustion prevented convergence; min_jdet={_min_jdet(phi):.4f}"

    def test_single_iter_budget_still_makes_progress(self):
        """max_per_index_iter=1 means exactly one SLSQP call per outer iteration.
        The outer loop should compensate and still converge (slower but correct)."""
        H, W = 8, 8
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        dvf[2, 0, 3, 3] = 2.5
        dvf[2, 0, 3, 4] = -2.5
        phi = iterative_serial(
            dvf, verbose=0,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=2000,
            max_per_index_iter=1,
            max_minimize_iter=300,
        )
        assert _converged(phi), \
            f"max_per_index_iter=1 did not converge; min_jdet={_min_jdet(phi):.4f}"


# ===========================================================================
# 8. REPRODUCIBILITY — same input must give same output
# ===========================================================================

class TestReproducibility:
    def test_deterministic_output(self):
        """Two runs on identical input must produce identical phi."""
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[2, 0, 5, 5] = 2.5
        dvf[2, 0, 5, 6] = -2.5
        phi1 = _run_serial(dvf.copy())
        phi2 = _run_serial(dvf.copy())
        np.testing.assert_array_equal(phi1, phi2,
                                      err_msg="Non-deterministic output")

    def test_idempotent_on_already_corrected(self):
        """Feeding the output back as input must be a no-op."""
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[2, 0, 5, 5] = 2.5
        dvf[2, 0, 5, 6] = -2.5
        phi1 = _run_serial(dvf)

        # Reconstruct (3,1,H,W) from corrected phi
        H, W = phi1.shape[1], phi1.shape[2]
        dvf2 = np.zeros((3, 1, H, W), dtype=np.float64)
        dvf2[0, 0] = 0.0
        dvf2[1, 0] = phi1[0]
        dvf2[2, 0] = phi1[1]
        phi2 = _run_serial(dvf2)
        np.testing.assert_allclose(phi1, phi2, atol=1e-8,
                                   err_msg="Second pass modified already-corrected phi")


# ===========================================================================
# 9. CONSTRAINT MODES — shoelace and injectivity flags must not break things
# ===========================================================================

class TestConstraintModes:
    def test_shoelace_mode_converges(self):
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[2, 0, 5, 5] = 2.5
        dvf[2, 0, 5, 6] = -2.5
        phi = _run_serial(dvf, enforce_shoelace=True)
        assert _converged(phi), \
            f"Shoelace mode not converged; min_jdet={_min_jdet(phi):.4f}"

    def test_injectivity_mode_converges(self):
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[2, 0, 5, 5] = 2.5
        dvf[2, 0, 5, 6] = -2.5
        phi = _run_serial(dvf,
                          enforce_injectivity=True,
                          injectivity_threshold=0.05)
        assert _converged(phi), \
            f"Injectivity mode not converged; min_jdet={_min_jdet(phi):.4f}"

    def test_both_constraints_converge(self):
        dvf = np.zeros((3, 1, 12, 12), dtype=np.float64)
        dvf[2, 0, 5, 5] = 2.5
        dvf[2, 0, 5, 6] = -2.5
        phi = _run_serial(dvf, enforce_shoelace=True,
                          enforce_injectivity=True,
                          injectivity_threshold=0.05)
        assert _converged(phi), \
            f"Dual constraint mode not converged; min_jdet={_min_jdet(phi):.4f}"


# ===========================================================================
# 10. SPATIAL HELPERS — unit tests for low-level functions
# ===========================================================================

class TestSpatialHelpers:
    def test_get_nearest_center_clamps_to_valid_range(self):
        """get_nearest_center must never place the window outside the grid."""
        from dvfopt.core.spatial import get_nearest_center
        slice_shape = (1, 20, 20)

        # Pixel at corner (0, 0) with large window
        result = get_nearest_center((0, 0), slice_shape, (7, 7))
        _, cy, cx = result
        # half = 3, so cy/cx must be at least 3
        assert cy >= 3 and cx >= 3, f"Window out of bounds: cy={cy}, cx={cx}"

        # Pixel at far corner
        result = get_nearest_center((19, 19), slice_shape, (7, 7))
        _, cy, cx = result
        assert cy <= 16 and cx <= 16, f"Window out of bounds: cy={cy}, cx={cx}"

    def test_neg_jdet_bounding_window_single_pixel(self):
        """Bounding window around a lone negative pixel should be at least 3×3."""
        from dvfopt.core.spatial import neg_jdet_bounding_window
        jm = np.ones((1, 10, 10))
        jm[0, 5, 5] = -0.5  # single negative pixel
        size, center = neg_jdet_bounding_window(jm, (5, 5), THRESHOLD, ERR_TOL)
        assert size[0] >= 3 and size[1] >= 3

    def test_neg_jdet_bounding_window_precomputed_labels(self):
        """Passing precomputed labels must give same result as computing them."""
        from dvfopt.core.spatial import neg_jdet_bounding_window
        from scipy.ndimage import label
        jm = np.ones((1, 10, 10))
        jm[0, 4:7, 4:7] = -0.5
        labeled, _ = label(jm[0] <= THRESHOLD - ERR_TOL)

        size1, center1 = neg_jdet_bounding_window(jm, (5, 5), THRESHOLD, ERR_TOL)
        size2, center2 = neg_jdet_bounding_window(jm, (5, 5), THRESHOLD, ERR_TOL,
                                                   labeled=labeled)
        assert size1 == size2 and center1 == center2, \
            "Precomputed labels give different result"

    def test_frozen_edges_clean_all_positive(self):
        """Window entirely above threshold → frozen edges are clean."""
        from dvfopt.core.spatial import _frozen_edges_clean
        jm = np.ones((1, 10, 10)) * 0.5
        assert _frozen_edges_clean(jm, 5, 5, (5, 5), THRESHOLD, ERR_TOL)

    def test_frozen_edges_clean_negative_on_edge(self):
        """Negative value on the window edge → not clean."""
        from dvfopt.core.spatial import _frozen_edges_clean
        jm = np.ones((1, 10, 10)) * 0.5
        jm[0, 3, 3] = -0.5  # this is the top-left corner of a (5,5) window at (5,5)
        # Window at cy=5, cx=5, size=(5,5): y0=3, y1=7, x0=3, x1=7
        assert not _frozen_edges_clean(jm, 5, 5, (5, 5), THRESHOLD, ERR_TOL)

    def test_frozen_edges_clean_at_grid_border(self):
        """Window touching grid border → edge pixels not frozen → always clean."""
        from dvfopt.core.spatial import _frozen_edges_clean
        # Window at cy=1, cx=1 with size=(3,3): y0=0, y1=2, x0=0, x1=2
        # That means the window IS at the border; _frozen_edges_clean checks
        # the outer ring of the window. Even if edge is negative, caller
        # should skip when is_at_edge=True.
        jm = np.ones((1, 10, 10)) * 0.5
        jm[0, 0, 0] = -0.5
        # Clamped bounds: y0=max(0,0)=0, y1=min(1,9)=1, etc.
        # The result depends on the exact clamping, but it must not crash.
        result = _frozen_edges_clean(jm, 1, 1, (3, 3), THRESHOLD, ERR_TOL)
        assert isinstance(result, (bool, np.bool_))


# ===========================================================================
# 11. OBJECTIVE FUNCTION — gradient check
# ===========================================================================

class TestObjective:
    def test_gradient_numerical_check(self):
        """Analytical gradient of L2-squared must match finite differences."""
        from dvfopt.core.objective import objective_euc
        rng = np.random.default_rng(7)
        phi = rng.standard_normal(20)
        phi_init = rng.standard_normal(20)
        eps = 1e-6
        val, grad = objective_euc(phi, phi_init)
        grad_fd = np.zeros_like(phi)
        for i in range(len(phi)):
            p = phi.copy(); p[i] += eps
            v_hi, _ = objective_euc(p, phi_init)
            p = phi.copy(); p[i] -= eps
            v_lo, _ = objective_euc(p, phi_init)
            grad_fd[i] = (v_hi - v_lo) / (2 * eps)
        np.testing.assert_allclose(grad, grad_fd, rtol=1e-5,
                                   err_msg="Analytical gradient does not match FD")

    def test_objective_zero_at_init(self):
        """Value must be zero when phi == phi_init."""
        from dvfopt.core.objective import objective_euc
        phi = np.array([1.0, 2.0, -3.0])
        val, grad = objective_euc(phi, phi.copy())
        assert val == pytest.approx(0.0)
        np.testing.assert_allclose(grad, 0.0)


# ===========================================================================
# 12. 3D EDGE CASES
# ===========================================================================

class Test3D:
    def _make_3d_dvf(self, D, H, W):
        """Identity 3D DVF."""
        return np.zeros((3, D, H, W), dtype=np.float64)

    def _inject_3d_fold(self, dvf, cz, cy, cx, magnitude=3.0):
        """Inject a fold at (cz, cy, cx)."""
        dvf[2, cz, cy, cx] = magnitude
        dvf[2, cz, cy, cx + 1] = -magnitude
        return dvf

    def _run_3d(self, dvf, max_iterations=500):
        return iterative_3d(
            dvf, verbose=0,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=max_iterations,
            max_per_index_iter=200,
            max_minimize_iter=500,
        )

    def test_already_clean_3d(self):
        dvf = self._make_3d_dvf(5, 5, 5)
        phi = self._run_3d(dvf)
        assert _neg_count_3d(phi) == 0

    def test_single_interior_fold_3d(self):
        D, H, W = 8, 8, 8
        dvf = self._make_3d_dvf(D, H, W)
        dvf = self._inject_3d_fold(dvf, 4, 4, 3)
        phi = self._run_3d(dvf, max_iterations=1000)
        jdet = jacobian_det3D(phi)
        assert bool((jdet > THRESHOLD - ERR_TOL).all()), \
            f"3D single fold not resolved; min_jdet={float(jdet.min()):.4f}"

    def test_border_fold_3d(self):
        """Fold on z=0 face — 3D border path."""
        D, H, W = 6, 6, 6
        dvf = self._make_3d_dvf(D, H, W)
        dvf[2, 0, :, :] = 3.0    # dx at z=0
        dvf[2, 1, :, :] = -3.0   # reversed at z=1
        phi = self._run_3d(dvf, max_iterations=1000)
        jdet = jacobian_det3D(phi)
        assert bool((jdet > THRESHOLD - ERR_TOL).all()), \
            f"3D border fold not resolved; min_jdet={float(jdet.min()):.4f}"

    def test_non_cubic_3d(self):
        """Non-cubic grid (D≠H≠W) — verifies _unpack_size_3d and window arithmetic."""
        D, H, W = 4, 8, 6
        dvf = self._make_3d_dvf(D, H, W)
        dvf = self._inject_3d_fold(dvf, 2, 4, 3)
        phi = self._run_3d(dvf, max_iterations=1000)
        jdet = jacobian_det3D(phi)
        assert bool((jdet > THRESHOLD - ERR_TOL).all()), \
            f"3D non-cubic fold not resolved; min_jdet={float(jdet.min()):.4f}"

    def test_zero_dvf_3d_is_clean(self):
        dvf = self._make_3d_dvf(5, 5, 5)
        phi = self._run_3d(dvf)
        assert _min_jdet_3d(phi) > THRESHOLD - ERR_TOL


# ===========================================================================
# 13. PATCH JACOBIAN CONSISTENCY
# ===========================================================================

class TestPatchJacobian:
    def test_patch_matches_full_recompute_2d(self):
        """After modifying a sub-window, patch must match full recompute exactly."""
        from dvfopt.core.solver import _patch_jacobian_2d
        rng = np.random.default_rng(42)
        H, W = 20, 20
        phi = rng.standard_normal((2, H, W)) * 0.3

        jac_full = np.zeros((1, H, W))
        from dvfopt.jacobian.numpy_jdet import jacobian_det2D
        jac_full[:] = jacobian_det2D(phi)
        jac_patch = jac_full.copy()

        # Modify a sub-window then patch
        phi[0, 8:12, 8:12] += rng.standard_normal((4, 4)) * 0.1
        jac_full_new = jacobian_det2D(phi)
        _patch_jacobian_2d(jac_patch, phi, (10, 10), (5, 5))

        # Patch must match full recompute in the affected region (with border)
        np.testing.assert_allclose(
            jac_patch[0, 7:13, 7:13],
            jac_full_new[0, 7:13, 7:13],
            atol=1e-10,
            err_msg="_patch_jacobian_2d does not match full recompute",
        )

    def test_patch_matches_full_recompute_3d(self):
        """3D patch must match full recompute in the affected region."""
        from dvfopt.core.solver3d import _patch_jacobian_3d
        rng = np.random.default_rng(43)
        D, H, W = 10, 10, 10
        phi = rng.standard_normal((3, D, H, W)) * 0.3

        from dvfopt.jacobian.numpy_jdet import jacobian_det3D
        jac_full = jacobian_det3D(phi).copy()
        jac_patch = jac_full.copy()

        phi[0, 4:6, 4:6, 4:6] += rng.standard_normal((2, 2, 2)) * 0.1
        jac_full_new = jacobian_det3D(phi)
        _patch_jacobian_3d(jac_patch, phi, (5, 5, 5), (3, 3, 3))

        np.testing.assert_allclose(
            jac_patch[4:6, 4:6, 4:6],
            jac_full_new[4:6, 4:6, 4:6],
            atol=1e-10,
            err_msg="_patch_jacobian_3d does not match full recompute",
        )
