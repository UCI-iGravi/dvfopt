"""
Coverage-gap tests addressing areas identified as untested after the main
test suite (test_edge_cases, test_invariants, test_constraints_and_params).

Covers:
  1.  save_path file output (results.txt, .npy arrays, window_counts.csv)
  2.  argmin_quality / argmin_worst_voxel correctness
  3.  iterative_parallel with actual >1 workers (true parallel code-path)
  4.  get_nearest_center_3d clamping behaviour
  5.  neg_jdet_bounding_window_3d unit tests
  6.  _frozen_boundary_mask_3d logic
  7.  get_phi_sub_flat_3d channel-order roundtrip
  8.  Multiple disjoint 3D folds
  9.  Accumulator (error_list / num_neg_jac / min_jdet_list) length invariants
 10.  float32 input handling
 11.  injectivity_threshold=None -> _adaptive_injectivity_loop path
 12.  Frozen-edge skip regression (skip must NOT consume per_index_iter budget)

Run with:  python -m pytest tests/test_coverage_gaps.py -v
"""

import os

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.core.slsqp.iterative import iterative_serial
from dvfopt.core.slsqp.iterative3d import iterative_3d
from dvfopt.core.slsqp.parallel import iterative_parallel

THRESHOLD = 0.01
ERR_TOL = 1e-5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fold_dvf(H, W, cy, cx, mag=2.5):
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    dvf[2, 0, cy, cx] = mag
    dvf[2, 0, cy, cx + 1] = -mag
    return dvf


def _run(dvf, max_iterations=500, **kw):
    return iterative_serial(
        dvf, verbose=0,
        threshold=THRESHOLD, err_tol=ERR_TOL,
        max_iterations=max_iterations,
        max_per_index_iter=200,
        max_minimize_iter=500,
        **kw,
    )


def _converged(phi):
    jdet = jacobian_det2D(phi)
    return bool((jdet > THRESHOLD - ERR_TOL).all())


def _converged_3d(phi):
    jdet = jacobian_det3D(phi)
    return bool((jdet > THRESHOLD - ERR_TOL).all())


# ===========================================================================
# 1. SAVE_PATH — file output
# ===========================================================================

class TestSavePath:
    """_save_results must create the expected files with correct content."""

    def _simple_dvf(self):
        return _fold_dvf(10, 10, 5, 4)

    def test_creates_expected_files(self, tmp_path):
        dvf = self._simple_dvf()
        iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=300, max_per_index_iter=100,
                         max_minimize_iter=300)
        expected = [
            "results.txt", "phi.npy", "error_list.npy",
            "num_neg_jac.npy", "iter_times.npy",
            "min_jdet_list.npy", "window_counts.csv",
        ]
        for fname in expected:
            assert os.path.exists(tmp_path / fname), f"Missing file: {fname}"

    def test_saved_phi_matches_return(self, tmp_path):
        dvf = self._simple_dvf()
        phi_ret = iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                                   threshold=THRESHOLD, err_tol=ERR_TOL,
                                   max_iterations=300, max_per_index_iter=100,
                                   max_minimize_iter=300)
        phi_saved = np.load(str(tmp_path / "phi.npy"))
        np.testing.assert_array_equal(phi_ret, phi_saved,
                                      err_msg="Saved phi.npy doesn't match returned phi")

    def test_results_txt_contains_key_fields(self, tmp_path):
        dvf = self._simple_dvf()
        iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=300, max_per_index_iter=100,
                         max_minimize_iter=300)
        text = (tmp_path / "results.txt").read_text()
        for keyword in ["Method", "Threshold", "Max iterations",
                        "L2 error", "Jacobian"]:
            assert keyword in text, f"results.txt missing field: {keyword!r}"

    def test_window_counts_csv_header(self, tmp_path):
        dvf = self._simple_dvf()
        iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=300, max_per_index_iter=100,
                         max_minimize_iter=300)
        lines = (tmp_path / "window_counts.csv").read_text().splitlines()
        assert len(lines) >= 1, "window_counts.csv is empty"
        header = lines[0]
        assert "window_height" in header and "window_width" in header and "count" in header

    def test_save_3d(self, tmp_path):
        dvf = np.zeros((3, 5, 6, 6), dtype=np.float64)
        dvf[2, 2, 3, 3] = 2.5
        dvf[2, 2, 3, 4] = -2.5
        iterative_3d(dvf, verbose=0, save_path=str(tmp_path),
                     threshold=THRESHOLD, err_tol=ERR_TOL,
                     max_iterations=300, max_per_index_iter=100,
                     max_minimize_iter=300)
        text = (tmp_path / "results.txt").read_text()
        assert "5 x 6 x 6" in text, "3D results.txt missing grid shape"

    def test_save_already_clean_still_writes_files(self, tmp_path):
        """An already-clean field (0 iterations) still writes all files."""
        dvf = np.zeros((3, 1, 8, 8), dtype=np.float64)
        iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=100, max_per_index_iter=10,
                         max_minimize_iter=50)
        assert os.path.exists(tmp_path / "phi.npy")
        assert os.path.exists(tmp_path / "results.txt")
        text = (tmp_path / "results.txt").read_text()
        assert "Number of index iterations: 0" in text


# ===========================================================================
# 2. ARGMIN FUNCTIONS
# ===========================================================================

class TestArgminFunctions:
    """argmin_quality and argmin_worst_voxel must return the correct index."""

    def test_argmin_quality_known_minimum(self):
        from dvfopt.core.slsqp.spatial import argmin_quality
        jm = np.ones((1, 8, 8)) * 0.5
        jm[0, 3, 6] = -0.3  # unique minimum
        y, x = argmin_quality(jm)
        assert (y, x) == (3, 6), f"Expected (3,6), got ({y},{x})"

    def test_argmin_quality_returns_int_tuple(self):
        from dvfopt.core.slsqp.spatial import argmin_quality
        jm = np.ones((1, 5, 5))
        jm[0, 2, 2] = -1.0
        result = argmin_quality(jm)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_argmin_worst_voxel_known_minimum(self):
        from dvfopt.core.slsqp.spatial3d import argmin_worst_voxel
        jm = np.ones((4, 5, 6)) * 0.5
        jm[1, 2, 4] = -0.7  # unique minimum
        z, y, x = argmin_worst_voxel(jm)
        assert (z, y, x) == (1, 2, 4), f"Expected (1,2,4), got ({z},{y},{x})"

    def test_argmin_worst_voxel_returns_int_tuple(self):
        from dvfopt.core.slsqp.spatial3d import argmin_worst_voxel
        jm = np.ones((3, 4, 5))
        jm[0, 0, 0] = -1.0
        result = argmin_worst_voxel(jm)
        assert len(result) == 3
        assert all(isinstance(v, (int, np.integer)) for v in result)


# ===========================================================================
# 3. PARALLEL WITH ACTUAL >1 WORKERS
# ===========================================================================

class TestParallelMultiWorker:
    """iterative_parallel with max_workers=2 exercises the true parallel batch path."""

    def _two_fold_dvf(self):
        """30x30 grid, two folds far enough apart to be batched in parallel."""
        dvf = np.zeros((3, 1, 30, 30), dtype=np.float64)
        # Fold 1 at (8, 5)
        dvf[2, 0, 8, 5] = 2.5
        dvf[2, 0, 8, 6] = -2.5
        # Fold 2 at (8, 23) — 17 columns away, windows cannot overlap
        dvf[2, 0, 8, 23] = 2.5
        dvf[2, 0, 8, 24] = -2.5
        return dvf

    def test_parallel_two_workers_converges(self):
        """Two well-separated folds; with max_workers=2 the batch has >1 window."""
        dvf = self._two_fold_dvf()
        phi = iterative_parallel(
            dvf, verbose=0, max_workers=2,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=500, max_per_index_iter=100,
            max_minimize_iter=300,
        )
        assert phi.shape == (2, 30, 30)
        assert _converged(phi), \
            f"Parallel (workers=2) did not converge; min_jdet={float(jacobian_det2D(phi).min()):.4f}"

    def test_parallel_two_workers_result_shape_and_dtype(self):
        """Return shape is (2, H, W) and dtype is float64."""
        dvf = self._two_fold_dvf()
        phi = iterative_parallel(
            dvf, verbose=0, max_workers=2,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=200, max_per_index_iter=50,
            max_minimize_iter=200,
        )
        assert phi.shape == (2, 30, 30)
        assert phi.dtype == np.float64


# ===========================================================================
# 4. GET_NEAREST_CENTER_3D
# ===========================================================================

class TestGetNearestCenter3d:
    """get_nearest_center_3d must clamp to valid range in all 3 axes."""

    def test_interior_point_unchanged(self):
        from dvfopt.core.slsqp.spatial3d import get_nearest_center_3d
        cz, cy, cx = get_nearest_center_3d((5, 5, 5), (10, 10, 10), (3, 3, 3))
        assert (cz, cy, cx) == (5, 5, 5)

    def test_clamp_at_origin_corner(self):
        from dvfopt.core.slsqp.spatial3d import get_nearest_center_3d
        cz, cy, cx = get_nearest_center_3d((0, 0, 0), (10, 10, 10), (5, 5, 5))
        hz = 5 // 2  # = 2
        assert cz >= hz and cy >= hz and cx >= hz, \
            f"Origin corner not clamped: ({cz},{cy},{cx})"

    def test_clamp_at_far_corner(self):
        from dvfopt.core.slsqp.spatial3d import get_nearest_center_3d
        D, H, W = 8, 10, 12
        sz, sy, sx = 5, 5, 5
        cz, cy, cx = get_nearest_center_3d((D - 1, H - 1, W - 1), (D, H, W), (sz, sy, sx))
        # Window must not overflow the grid
        assert cz + (sz - sz // 2) <= D
        assert cy + (sy - sy // 2) <= H
        assert cx + (sx - sx // 2) <= W

    def test_clamp_asymmetric_volume(self):
        from dvfopt.core.slsqp.spatial3d import get_nearest_center_3d
        D, H, W = 4, 8, 6
        # Full-grid window at corner; center should land at (D//2, H//2, W//2)
        cz, cy, cx = get_nearest_center_3d((0, 0, 0), (D, H, W), (D, H, W))
        assert cz == D // 2
        assert cy == H // 2
        assert cx == W // 2


# ===========================================================================
# 5. NEG_JDET_BOUNDING_WINDOW_3D
# ===========================================================================

class TestBoundingWindow3d:
    """neg_jdet_bounding_window_3d correctness."""

    def test_single_negative_voxel_at_least_3x3x3(self):
        from dvfopt.core.slsqp.spatial3d import neg_jdet_bounding_window_3d
        jm = np.ones((6, 6, 6)) * 0.5
        jm[3, 3, 3] = -0.5
        size, _ = neg_jdet_bounding_window_3d(jm, (3, 3, 3), THRESHOLD, ERR_TOL)
        assert size[0] >= 3 and size[1] >= 3 and size[2] >= 3

    def test_region_label_zero_safe_path(self):
        """Calling with a voxel NOT in the negative region returns (3,3,3)."""
        from dvfopt.core.slsqp.spatial3d import neg_jdet_bounding_window_3d
        jm = np.ones((6, 6, 6)) * 0.5
        jm[1, 1, 1] = -0.5  # negative at (1,1,1)
        # Ask for bounding window centred on a POSITIVE voxel
        size, _ = neg_jdet_bounding_window_3d(jm, (4, 4, 4), THRESHOLD, ERR_TOL)
        assert size == (3, 3, 3)

    def test_precomputed_labels_same_result(self):
        """Passing pre-computed labels gives identical result."""
        from dvfopt.core.slsqp.spatial3d import neg_jdet_bounding_window_3d
        from scipy.ndimage import label
        jm = np.ones((8, 8, 8)) * 0.5
        jm[3:5, 3:5, 3:5] = -0.3
        structure = np.ones((3, 3, 3))
        labeled, _ = label(jm <= THRESHOLD - ERR_TOL, structure=structure)

        size1, c1 = neg_jdet_bounding_window_3d(jm, (4, 4, 4), THRESHOLD, ERR_TOL)
        size2, c2 = neg_jdet_bounding_window_3d(jm, (4, 4, 4), THRESHOLD, ERR_TOL,
                                                labeled_array=labeled)
        assert size1 == size2 and c1 == c2

    def test_connected_region_sizing(self):
        """A 3x3x3 negative cube should yield a window >= 5x5x5 (cube + 1px border)."""
        from dvfopt.core.slsqp.spatial3d import neg_jdet_bounding_window_3d
        jm = np.ones((10, 10, 10)) * 0.5
        jm[3:6, 3:6, 3:6] = -0.3
        size, _ = neg_jdet_bounding_window_3d(jm, (4, 4, 4), THRESHOLD, ERR_TOL)
        assert size[0] >= 5 and size[1] >= 5 and size[2] >= 5, \
            f"Window too small for 3x3x3 negative cube: {size}"


# ===========================================================================
# 6. _FROZEN_BOUNDARY_MASK_3D
# ===========================================================================

class TestFrozenBoundaryMask3d:
    """Interior-facing faces must be frozen; grid-edge faces must not."""

    def test_interior_window_all_six_faces_frozen(self):
        from dvfopt.core.slsqp.spatial3d import _frozen_boundary_mask_3d
        # (3,3,3) window centred at (5,5,5) in a 10x10x10 volume
        mask = _frozen_boundary_mask_3d(5, 5, 5, (3, 3, 3), (10, 10, 10))
        assert mask.shape == (3, 3, 3)
        assert mask[0, :, :].all(),  "z-min face not frozen"
        assert mask[-1, :, :].all(), "z-max face not frozen"
        assert mask[:, 0, :].all(),  "y-min face not frozen"
        assert mask[:, -1, :].all(), "y-max face not frozen"
        assert mask[:, :, 0].all(),  "x-min face not frozen"
        assert mask[:, :, -1].all(), "x-max face not frozen"

    def test_corner_window_grid_edge_faces_not_frozen(self):
        from dvfopt.core.slsqp.spatial3d import _frozen_boundary_mask_3d
        # hz=1 => cz=1 => start_z = 1 - 1 = 0 (at grid edge) => z-min NOT frozen
        mask = _frozen_boundary_mask_3d(1, 1, 1, (3, 3, 3), (6, 6, 6))
        assert not mask[0, :, :].all(),  "z-min face (at grid edge) should not be frozen"
        assert not mask[:, 0, :].all(),  "y-min face (at grid edge) should not be frozen"
        assert not mask[:, :, 0].all(),  "x-min face (at grid edge) should not be frozen"
        # Interior faces on the high side are away from the edge — should be frozen
        assert mask[-1, :, :].all(), "z-max interior face should be frozen"
        assert mask[:, -1, :].all(), "y-max interior face should be frozen"
        assert mask[:, :, -1].all(), "x-max interior face should be frozen"

    def test_full_grid_window_no_faces_frozen(self):
        from dvfopt.core.slsqp.spatial3d import _frozen_boundary_mask_3d
        D, H, W = 5, 5, 5
        mask = _frozen_boundary_mask_3d(D // 2, H // 2, W // 2, (D, H, W), (D, H, W))
        assert not mask.any(), "Full-grid window should have zero frozen faces"

    def test_mask_shape_matches_subvolume(self):
        from dvfopt.core.slsqp.spatial3d import _frozen_boundary_mask_3d
        mask = _frozen_boundary_mask_3d(5, 5, 5, (3, 5, 7), (10, 10, 10))
        assert mask.shape == (3, 5, 7)


# ===========================================================================
# 7. GET_PHI_SUB_FLAT_3D — channel-order roundtrip
# ===========================================================================

class TestPhiSubFlat3d:
    """get_phi_sub_flat_3d packs [dx, dy, dz]; values must round-trip exactly."""

    def test_roundtrip_values_preserved(self):
        from dvfopt.core.slsqp.spatial3d import get_phi_sub_flat_3d
        rng = np.random.default_rng(0)
        phi = rng.standard_normal((3, 8, 8, 8))
        sz, sy, sx = 3, 3, 3
        cz, cy, cx = 4, 4, 4
        flat = get_phi_sub_flat_3d(phi, cz, cy, cx, (sz, sy, sx))

        voxels = sz * sy * sx
        hz, hy, hx = sz // 2, sy // 2, sx // 2
        hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx
        slc = (slice(cz - hz, cz + hz_hi),
               slice(cy - hy, cy + hy_hi),
               slice(cx - hx, cx + hx_hi))

        np.testing.assert_array_equal(flat[:voxels], phi[2][slc].flatten(),
                                      err_msg="dx values (first block) wrong")
        np.testing.assert_array_equal(flat[voxels:2*voxels], phi[1][slc].flatten(),
                                      err_msg="dy values (middle block) wrong")
        np.testing.assert_array_equal(flat[2*voxels:], phi[0][slc].flatten(),
                                      err_msg="dz values (last block) wrong")

    def test_flat_length_is_3_times_voxels(self):
        from dvfopt.core.slsqp.spatial3d import get_phi_sub_flat_3d
        phi = np.zeros((3, 10, 10, 10))
        flat = get_phi_sub_flat_3d(phi, 5, 5, 5, (3, 5, 7))
        assert len(flat) == 3 * 3 * 5 * 7

    def test_channel_order_dx_dy_dz(self):
        """First voxels block = dx(=1), middle = dy(=2), last = dz(=3)."""
        from dvfopt.core.slsqp.spatial3d import get_phi_sub_flat_3d
        phi = np.zeros((3, 6, 6, 6))
        phi[2] = 1.0   # dx channel
        phi[1] = 2.0   # dy channel
        phi[0] = 3.0   # dz channel
        flat = get_phi_sub_flat_3d(phi, 3, 3, 3, (3, 3, 3))
        voxels = 27
        assert flat[:voxels].mean() == pytest.approx(1.0), "dx block should be 1"
        assert flat[voxels:2*voxels].mean() == pytest.approx(2.0), "dy block should be 2"
        assert flat[2*voxels:].mean() == pytest.approx(3.0), "dz block should be 3"


# ===========================================================================
# 8. MULTIPLE DISJOINT 3D FOLDS
# ===========================================================================

class Test3dMultipleFolds:
    """3D solver must fix multiple spatially independent folds."""

    def _run_3d(self, dvf, max_iterations=1000):
        return iterative_3d(dvf, verbose=0,
                            threshold=THRESHOLD, err_tol=ERR_TOL,
                            max_iterations=max_iterations,
                            max_per_index_iter=200,
                            max_minimize_iter=500)

    def test_two_disjoint_folds_both_fixed(self):
        D, H, W = 10, 10, 10
        dvf = np.zeros((3, D, H, W), dtype=np.float64)
        dvf[2, 2, 4, 4] = 2.5;  dvf[2, 2, 4, 5] = -2.5   # fold 1
        dvf[2, 7, 4, 4] = 2.5;  dvf[2, 7, 4, 5] = -2.5   # fold 2, 5 slices away
        phi = self._run_3d(dvf)
        assert _converged_3d(phi), \
            f"Two 3D folds not both fixed; min_jdet={float(jacobian_det3D(phi).min()):.4f}"

    def test_four_disjoint_folds_all_fixed(self):
        D, H, W = 12, 12, 12
        dvf = np.zeros((3, D, H, W), dtype=np.float64)
        for cz, cy, cx in [(2, 3, 3), (2, 3, 8), (8, 3, 3), (8, 3, 8)]:
            dvf[2, cz, cy, cx] = 2.5
            dvf[2, cz, cy, cx + 1] = -2.5
        phi = self._run_3d(dvf, max_iterations=2000)
        assert _converged_3d(phi), \
            f"Four 3D folds not all fixed; min_jdet={float(jacobian_det3D(phi).min()):.4f}"


# ===========================================================================
# 9. ACCUMULATOR LENGTH INVARIANTS
# ===========================================================================

class TestAccumulatorContent:
    """num_neg_jac, min_jdet_list, error_list must satisfy fixed length relationships.

    For N total SLSQP calls:
      len(num_neg_jac)  = N + 2   (init + N sub-iter appends + 1 final append)
      len(min_jdet_list)= N + 1   (init + N sub-iter appends)
      len(error_list)   = N       (N sub-iter appends only)
    """

    def _load_accumulators(self, tmp_path):
        num_neg   = np.load(str(tmp_path / "num_neg_jac.npy"))
        min_jdet  = np.load(str(tmp_path / "min_jdet_list.npy"))
        err_list  = np.load(str(tmp_path / "error_list.npy"))
        return num_neg, min_jdet, err_list

    def _run_to_path(self, dvf, tmp_path, max_iterations=300):
        iterative_serial(dvf, verbose=0, save_path=str(tmp_path),
                         threshold=THRESHOLD, err_tol=ERR_TOL,
                         max_iterations=max_iterations,
                         max_per_index_iter=100,
                         max_minimize_iter=200)

    def test_length_invariant_with_fold(self, tmp_path):
        dvf = _fold_dvf(10, 10, 5, 4)
        self._run_to_path(dvf, tmp_path)
        num_neg, min_jdet, err_list = self._load_accumulators(tmp_path)
        assert len(num_neg) >= 2
        assert len(num_neg) == len(min_jdet) + 1, \
            f"len(num_neg_jac)={len(num_neg)} != len(min_jdet_list)+1={len(min_jdet)+1}"
        assert len(num_neg) == len(err_list) + 2, \
            f"len(num_neg_jac)={len(num_neg)} != len(error_list)+2={len(err_list)+2}"

    def test_length_invariant_already_clean(self, tmp_path):
        """Zero SLSQP calls: num_neg_jac=2, min_jdet_list=1, error_list=0."""
        dvf = np.zeros((3, 1, 8, 8), dtype=np.float64)
        self._run_to_path(dvf, tmp_path)
        num_neg, min_jdet, err_list = self._load_accumulators(tmp_path)
        assert len(num_neg) == 2, f"Expected 2, got {len(num_neg)}"
        assert len(min_jdet) == 1, f"Expected 1, got {len(min_jdet)}"
        assert len(err_list) == 0, f"Expected 0, got {len(err_list)}"

    def test_min_jdet_list_non_decreasing(self, tmp_path):
        """min_jdet_list must be non-decreasing for a tractable problem."""
        dvf = _fold_dvf(10, 10, 5, 4)
        self._run_to_path(dvf, tmp_path)
        _, min_jdet, _ = self._load_accumulators(tmp_path)
        diffs = np.diff(min_jdet.astype(float))
        assert diffs.min() >= -1e-6, \
            f"min_jdet_list decreased: min diff = {diffs.min():.6f}"

    def test_final_num_neg_jac_zero_after_convergence(self, tmp_path):
        """Last entry in num_neg_jac should be 0 (strictly negative count)."""
        dvf = _fold_dvf(10, 10, 5, 4)
        self._run_to_path(dvf, tmp_path)
        num_neg, _, _ = self._load_accumulators(tmp_path)
        assert int(num_neg[-1]) == 0, \
            f"Final num_neg_jac should be 0 after convergence, got {num_neg[-1]}"


# ===========================================================================
# 10. FLOAT32 INPUT
# ===========================================================================

class TestFloat32Input:
    """float32 DVF input must work correctly; output phi is always float64."""

    def test_float32_dvf_converges(self):
        dvf = _fold_dvf(10, 10, 5, 4).astype(np.float32)
        phi = _run(dvf)
        assert _converged(phi), \
            f"float32 input did not converge; min_jdet={float(jacobian_det2D(phi).min()):.4f}"

    def test_float32_output_is_float64(self):
        dvf = _fold_dvf(10, 10, 5, 4).astype(np.float32)
        phi = _run(dvf)
        assert phi.dtype == np.float64, f"Expected float64 output, got {phi.dtype}"

    def test_float32_3d_converges(self):
        dvf = np.zeros((3, 6, 6, 6), dtype=np.float32)
        dvf[2, 3, 3, 3] = 2.5
        dvf[2, 3, 3, 4] = -2.5
        phi = iterative_3d(dvf, verbose=0,
                           threshold=THRESHOLD, err_tol=ERR_TOL,
                           max_iterations=300, max_per_index_iter=100,
                           max_minimize_iter=300)
        assert _converged_3d(phi), \
            f"float32 3D input did not converge; min_jdet={float(jacobian_det3D(phi).min()):.4f}"


# ===========================================================================
# 11. INJECTIVITY_THRESHOLD=NONE -> _ADAPTIVE_INJECTIVITY_LOOP
# ===========================================================================

class TestInjectivityThresholdNone:
    """enforce_injectivity=True with no explicit threshold triggers adaptive loop."""

    def test_serial_adaptive_loop_converges(self):
        dvf = _fold_dvf(12, 12, 6, 5)
        phi = iterative_serial(
            dvf, verbose=0,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=500, max_per_index_iter=100,
            max_minimize_iter=300,
            enforce_injectivity=True,
            # injectivity_threshold intentionally omitted -> adaptive loop
        )
        assert phi.shape == (2, 12, 12)
        assert _converged(phi), \
            f"Adaptive injectivity loop (serial) did not converge; " \
            f"min={float(jacobian_det2D(phi).min()):.4f}"

    def test_parallel_adaptive_loop_converges(self):
        dvf = _fold_dvf(12, 12, 6, 5)
        phi = iterative_parallel(
            dvf, verbose=0, max_workers=1,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=500, max_per_index_iter=100,
            max_minimize_iter=300,
            enforce_injectivity=True,
        )
        assert phi.shape == (2, 12, 12)
        assert _converged(phi), \
            f"Adaptive injectivity loop (parallel) did not converge; " \
            f"min={float(jacobian_det2D(phi).min()):.4f}"

    def test_adaptive_result_has_positive_monotonicity(self):
        """h_mono and v_mono must be positive after the adaptive loop."""
        from dvfopt.jacobian.monotonicity import _monotonicity_diffs_2d
        dvf = _fold_dvf(12, 12, 6, 5)
        phi = iterative_serial(
            dvf, verbose=0,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=500, max_per_index_iter=100,
            max_minimize_iter=300,
            enforce_injectivity=True,
        )
        h_mono, v_mono = _monotonicity_diffs_2d(phi[0], phi[1])
        assert float(h_mono.min()) > 0, "h_mono not positive after adaptive loop"
        assert float(v_mono.min()) > 0, "v_mono not positive after adaptive loop"


# ===========================================================================
# 12. FROZEN-EDGE SKIP REGRESSION
# ===========================================================================

class TestFrozenEdgeSkipRegression:
    """The frozen-edge `continue` must NOT consume the per_index_iter budget.

    Before the fix, `per_index_iter` was incremented on every skip, so
    `max_per_index_iter=1` could result in zero actual SLSQP calls when a
    skip fired.  These tests would fail if that regression is re-introduced.
    """

    def test_tight_budget_converges_despite_dirty_frozen_edges(self):
        """A fold embedded in a large connected negative region forces frozen-edge
        skips at the initial small window.  max_per_index_iter=1 must still mean
        exactly one SLSQP call per outer iteration so the algorithm converges via
        repeated outer iterations with growing windows.
        """
        H, W = 14, 14
        dvf = np.zeros((3, 1, H, W), dtype=np.float64)
        # Wide connected strip
        for i in range(4, 10):
            dvf[2, 0, i, 4:10] = 3.0 if i % 2 == 0 else -3.0
        phi = iterative_serial(
            dvf, verbose=0,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=5000,
            max_per_index_iter=1,
            max_minimize_iter=300,
        )
        assert _converged(phi), \
            f"Frozen-edge skip regression: did not converge; " \
            f"min_jdet={float(jacobian_det2D(phi).min()):.4f}"

    def test_single_fold_max_per_index_iter_one_converges(self):
        """Single clean fold with max_per_index_iter=1.  No frozen-edge skips
        should fire, and convergence must still happen — if skips secretly
        consumed budget this test would also fail.
        """
        phi = iterative_serial(
            _fold_dvf(10, 10, 5, 4), verbose=0,
            threshold=THRESHOLD, err_tol=ERR_TOL,
            max_iterations=2000,
            max_per_index_iter=1,
            max_minimize_iter=300,
        )
        assert _converged(phi), \
            f"Single fold with max_per_index_iter=1 did not converge; " \
            f"min_jdet={float(jacobian_det2D(phi).min()):.4f}"
