"""Solver primitives: init, metrics, save, sub-window optimisation, serial fix loop."""

import os
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from dvfopt._defaults import _log, _unpack_size
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d, jacobian_det2D
from dvfopt.jacobian.shoelace import _shoelace_areas_2d
from dvfopt.jacobian.monotonicity import _monotonicity_diffs_2d
from dvfopt.core.objective import objectiveEuc
from dvfopt.core.constraints import (
    _build_constraints,
    _quality_map,
)
from dvfopt.core.spatial import (
    get_nearest_center,
    neg_jdet_bounding_window,
    _frozen_edges_clean,
    get_phi_sub_flat,
    _edge_flags,
)


# ---------------------------------------------------------------------------
# Shared iteration helpers
# ---------------------------------------------------------------------------
def _setup_accumulators():
    """Return the five tracking structures used by every iterative loop."""
    return [], [], [], [], defaultdict(int)
    # error_list, num_neg_jac, iter_times, min_jdet_list, window_counts


def _print_summary(verbose, method_label, grid_shape, iteration,
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed, extra_lines=""):
    """Print the end-of-run summary block shared by all iterative solvers."""
    grid_str = " x ".join(str(d) for d in grid_shape)
    _log(verbose, 1, "")
    _log(verbose, 1, "=" * 60)
    _log(verbose, 1, f"  SUMMARY  ({method_label})")
    _log(verbose, 1, "-" * 60)
    _log(verbose, 1, f"  Grid size        : {grid_str}")
    iter_line = f"  Iterations       : {iteration}"
    if extra_lines:
        iter_line += f"  {extra_lines}"
    _log(verbose, 1, iter_line)
    _log(verbose, 1, f"  Neg-Jdet  {init_neg:>5d} -> {final_neg:>5d}")
    _log(verbose, 1, f"  Min Jdet  {init_min:+.6f} -> {final_min:+.6f}")
    _log(verbose, 1, f"  L2 error         : {final_err:.6f}")
    _log(verbose, 1, f"  Time             : {elapsed:.2f}s")
    _log(verbose, 1, "=" * 60)


# ---------------------------------------------------------------------------
# Init / metrics / save helpers
# ---------------------------------------------------------------------------
def _init_phi(deformation_i):
    """Create the initial ``phi`` working array from a ``(3,1,H,W)`` deformation.

    Returns ``(phi, phi_init, H, W)``.
    """
    H, W = deformation_i.shape[-2:]
    phi = np.zeros((2, H, W))
    phi[1] = deformation_i[-1]
    phi[0] = deformation_i[-2]
    phi_init = phi.copy()
    return phi, phi_init, H, W


def _update_metrics(phi, phi_init, enforce_shoelace, enforce_injectivity,
                    num_neg_jac, min_jdet_list, error_list=None,
                    jacobian_matrix=None, patch_center=None, patch_size=None):
    """Recompute Jacobian/quality matrices and append to accumulator lists.

    Parameters
    ----------
    error_list : list or None
        When not ``None``, the L2 error is appended.
    jacobian_matrix : ndarray or None
        When provided along with *patch_center* and *patch_size*, only the
        affected sub-region (+ 1px gradient border) is recomputed, avoiding
        a full-grid Jacobian computation.
    patch_center : tuple or None
        ``(cy, cx)`` center of the optimised sub-window.
    patch_size : tuple or None
        ``(sy, sx)`` size of the optimised sub-window.

    Returns
    -------
    jacobian_matrix, quality_matrix, cur_neg, cur_min
    """
    if jacobian_matrix is not None and patch_center is not None and patch_size is not None:
        jac = _patch_jacobian_2d(jacobian_matrix, phi, patch_center, patch_size)
    elif jacobian_matrix is not None and patch_center is None:
        # Jacobian already patched externally (e.g., parallel batch)
        jac = jacobian_matrix
    else:
        jac = jacobian_det2D(phi)
    use_q = enforce_shoelace or enforce_injectivity
    qm = _quality_map(phi, enforce_shoelace, enforce_injectivity, jacobian_matrix=jac) if use_q else jac
    cur_neg = int((jac <= 0).sum())
    cur_min = float(jac.min())
    num_neg_jac.append(cur_neg)
    min_jdet_list.append(cur_min)
    if error_list is not None:
        error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))
    return jac, qm, cur_neg, cur_min


def _patch_jacobian_2d(jacobian_matrix, phi, center, sub_size):
    """Recompute Jacobian only in the modified sub-region + 1px border.

    The computation region is expanded by an extra pixel beyond the
    write-back region so that ``np.gradient`` uses central differences
    at the write-back boundary (matching full-grid computation).

    Mutates *jacobian_matrix* in place and returns it.
    """
    cy, cx = center
    sy, sx = _unpack_size(sub_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    H, W = phi.shape[1], phi.shape[2]

    # Write-back region: sub-window + 1px border, clamped to grid
    wy0 = max(cy - hy - 1, 0)
    wy1 = min(cy + hy_hi + 1, H)
    wx0 = max(cx - hx - 1, 0)
    wx1 = min(cx + hx_hi + 1, W)

    # Computation region: 1 extra pixel for central-difference context
    cy0 = max(wy0 - 1, 0)
    cy1 = min(wy1 + 1, H)
    cx0 = max(wx0 - 1, 0)
    cx1 = min(wx1 + 1, W)

    jdet_comp = _numpy_jdet_2d(phi[0, cy0:cy1, cx0:cx1],
                                phi[1, cy0:cy1, cx0:cx1])

    # Trim to write-back region
    ty0 = wy0 - cy0
    tx0 = wx0 - cx0
    jacobian_matrix[0, wy0:wy1, wx0:wx1] = \
        jdet_comp[ty0:ty0 + wy1 - wy0, tx0:tx0 + wx1 - wx0]
    return jacobian_matrix


def _save_results(save_path, *, method, threshold, err_tol, max_iterations,
                  max_per_index_iter, max_minimize_iter,
                  grid_shape, elapsed, final_err, init_neg, final_neg,
                  init_min, final_min, iteration, phi, error_list,
                  num_neg_jac, iter_times, min_jdet_list, window_counts,
                  extra_settings="", extra_results=""):
    """Write correction results to *save_path*.

    Parameters
    ----------
    grid_shape : tuple
        ``(H, W)`` for 2D or ``(D, H, W)`` for 3D.
    """
    os.makedirs(save_path, exist_ok=True)

    ndim = len(grid_shape)
    if ndim == 2:
        res_label = "height x width"
        dim_names = ["window_height", "window_width"]
    else:
        res_label = "D x H x W"
        dim_names = ["window_depth", "window_height", "window_width"]
    res_str = " x ".join(str(d) for d in grid_shape)

    output_text = "Settings:\n"
    output_text += f"\tMethod: {method}\n"
    output_text += f"\tThreshold: {threshold}\n"
    output_text += f"\tError tolerance: {err_tol}\n"
    output_text += f"\tMax iterations: {max_iterations}\n"
    output_text += f"\tMax per index iterations: {max_per_index_iter}\n"
    output_text += f"\tMax minimize iterations: {max_minimize_iter}\n"
    if extra_settings:
        output_text += extra_settings
    output_text += "\nResults:\n"
    output_text += f"\tInput deformation field resolution ({res_label}): {res_str}\n"
    output_text += f"\tTotal run-time: {elapsed} seconds\n"
    output_text += f"\tFinal L2 error: {final_err}\n"
    output_text += f"\tStarting number of non-positive Jacobian determinants: {init_neg}\n"
    output_text += f"\tFinal number of non-positive Jacobian determinants: {final_neg}\n"
    output_text += f"\tStarting Jacobian determinant minimum value: {init_min}\n"
    output_text += f"\tFinal Jacobian determinant minimum value: {final_min}\n"
    output_text += f"\tNumber of index iterations: {iteration}"
    if extra_results:
        output_text += "\n" + extra_results

    with open(os.path.join(save_path, "results.txt"), "w") as f:
        f.write(output_text)

    np.save(os.path.join(save_path, "phi.npy"), phi)
    np.save(os.path.join(save_path, "error_list.npy"), error_list)
    np.save(os.path.join(save_path, "num_neg_jac.npy"), num_neg_jac)
    np.save(os.path.join(save_path, "iter_times.npy"), iter_times)
    np.save(os.path.join(save_path, "min_jdet_list.npy"), min_jdet_list)

    csv_header = ",".join(dim_names) + ",count\n"
    with open(os.path.join(save_path, "window_counts.csv"), "w") as f:
        f.write(csv_header)
        for ws in sorted(window_counts):
            dims = ws if isinstance(ws, tuple) else (ws,)
            f.write(",".join(str(d) for d in dims) + f",{window_counts[ws]}\n")


# ---------------------------------------------------------------------------
# Full-grid optimisation fallback (non-square grids)
# ---------------------------------------------------------------------------
def _full_grid_step(phi, phi_init, H, W, threshold, max_minimize_iter,
                    methodName, verbose, enforce_shoelace, enforce_injectivity):
    """Optimize the entire H×W grid at once.

    Used as a fallback when the square sub-window (capped at
    ``min(H, W)``) cannot cover the full grid.  Constraints are applied
    to **all** pixels (including boundary), matching the behaviour of
    windowed optimisations whose windows touch the grid edge.
    """
    pixels = H * W
    phi_flat = np.concatenate([phi[1].flatten(), phi[0].flatten()])
    phi_init_flat = np.concatenate([phi_init[1].flatten(), phi_init[0].flatten()])

    def jac_con(phi_xy):
        dx = phi_xy[:pixels].reshape(H, W)
        dy = phi_xy[pixels:].reshape(H, W)
        return _numpy_jdet_2d(dy, dx).flatten()

    constraints = [NonlinearConstraint(jac_con, threshold, np.inf)]

    if enforce_shoelace:
        def shoe_con(phi_xy):
            dx = phi_xy[:pixels].reshape(H, W)
            dy = phi_xy[pixels:].reshape(H, W)
            return _shoelace_areas_2d(dy, dx).flatten()
        constraints.append(NonlinearConstraint(shoe_con, threshold, np.inf))

    if enforce_injectivity:
        def inject_con(phi_xy):
            dx = phi_xy[:pixels].reshape(H, W)
            dy = phi_xy[pixels:].reshape(H, W)
            h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
            return np.concatenate([h_mono.flatten(), v_mono.flatten()])
        constraints.append(NonlinearConstraint(inject_con, threshold, np.inf))

    _log(verbose, 1,
         f"  [full-grid] Optimizing entire {H}x{W} grid "
         f"({2 * pixels} variables)")

    result = minimize(
        lambda phi1: objectiveEuc(phi1, phi_init_flat),
        phi_flat,
        jac=True,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": verbose >= 2},
        method=methodName,
    )

    phi[1] = result.x[:pixels].reshape(H, W)
    phi[0] = result.x[pixels:].reshape(H, W)


# ---------------------------------------------------------------------------
# SLSQP worker for a single window
# ---------------------------------------------------------------------------
def _optimize_single_window(
    phi_sub_flat,
    phi_init_sub_flat,
    submatrix_size,
    is_at_edge,
    window_reached_max,
    threshold,
    max_minimize_iter,
    method_name,
    enforce_shoelace=False,
    enforce_injectivity=False,
):
    """Run SLSQP on one sub-window.  Returns ``(result_x, elapsed)``."""

    constraints = _build_constraints(
        phi_sub_flat, submatrix_size, is_at_edge, window_reached_max, threshold,
        enforce_shoelace=enforce_shoelace,
        enforce_injectivity=enforce_injectivity,
    )

    t0 = time.time()
    result = minimize(
        lambda phi1: objectiveEuc(phi1, phi_init_sub_flat),
        phi_sub_flat,
        jac=True,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": False},
        method=method_name,
    )
    elapsed = time.time() - t0
    return result.x, elapsed


# ---------------------------------------------------------------------------
# Apply result back into phi
# ---------------------------------------------------------------------------
def _apply_result(phi, result_x, cy, cx, sub_size):
    """Write optimised sub-window back into *phi*."""
    sy, sx = _unpack_size(sub_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    pixels = sy * sx
    phi[1, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi] = \
        result_x[:pixels].reshape(sy, sx)
    phi[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi] = \
        result_x[pixels:].reshape(sy, sx)


# ---------------------------------------------------------------------------
# Serial inner loop — fix a single pixel
# ---------------------------------------------------------------------------
def _serial_fix_pixel(
    neg_index_tuple, phi, phi_init, jacobian_matrix,
    slice_shape, near_cent_dict, window_counts,
    max_per_index_iter, max_minimize_iter,
    max_window, threshold, err_tol, methodName, verbose,
    error_list, num_neg_jac, min_jdet_list, iter_times,
    enforce_shoelace=False,
    enforce_injectivity=False,
    plot_callback=None,
    deformation_i=None,
    min_window=(3, 3),
):
    """Fix a single pixel using the serial adaptive-window inner loop.

    Start from the bounding-box-derived window, then grow by 2 each
    sub-iteration until the local region is clean or the window hits the
    grid boundary.

    Mutates *phi* and the accumulator lists in-place.

    Returns
    -------
    jacobian_matrix, quality_matrix, submatrix_size, per_index_iter, (cy, cx)
    """
    _use_quality = enforce_shoelace or enforce_injectivity
    quality_matrix = _quality_map(phi, enforce_shoelace, enforce_injectivity, jacobian_matrix=jacobian_matrix) if _use_quality else jacobian_matrix

    # Adaptive starting size from negative-Jdet bounding box
    submatrix_size, bbox_center = neg_jdet_bounding_window(
        quality_matrix, neg_index_tuple, threshold, err_tol)
    max_sy, max_sx = _unpack_size(max_window)
    min_sy, min_sx = _unpack_size(min_window)
    submatrix_size = (max(min(submatrix_size[0], max_sy), min_sy),
                      max(min(submatrix_size[1], max_sx), min_sx))

    per_index_iter = 0
    window_reached_max = False

    while (
        per_index_iter == 0
        or (
            (not window_reached_max)
            and per_index_iter < max_per_index_iter
            and (quality_matrix[0,
                    cy - hy:cy + hy_hi,
                    cx - hx:cx + hx_hi]
                 < threshold - err_tol).any()
        )
    ):
        per_index_iter += 1

        window_counts[_unpack_size(submatrix_size)] += 1

        cz, cy, cx = get_nearest_center(
            bbox_center, slice_shape, submatrix_size, near_cent_dict)
        sy, sx = _unpack_size(submatrix_size)
        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx

        phi_init_sub_flat = get_phi_sub_flat(phi_init, cz, cy, cx, slice_shape, submatrix_size)
        phi_sub_flat = get_phi_sub_flat(phi, cz, cy, cx, slice_shape, submatrix_size)

        if per_index_iter > 1:
            _log(verbose, 2, f"  [window] Index {neg_index_tuple}: window grew to {sy}x{sx} (sub-iter {per_index_iter})")

        is_at_edge, w_max = _edge_flags(cy, cx, submatrix_size, slice_shape, max_window)
        window_reached_max = window_reached_max or w_max

        _log(verbose, 2, f"  [edge] at_edge={is_at_edge}  window_reached_max={window_reached_max}")

        # Skip optimizer if frozen edges have negative Jdet (likely infeasible)
        if (not is_at_edge and not window_reached_max
                and not _frozen_edges_clean(quality_matrix, cy, cx,
                                           submatrix_size, threshold, err_tol)):
            _log(verbose, 2, f"  [skip] Frozen edges have neg Jdet at win {sy}x{sx} — growing")
            sy, sx = _unpack_size(submatrix_size)
            if sy < max_sy or sx < max_sx:
                submatrix_size = (min(sy + 2, max_sy), min(sx + 2, max_sx))
            continue

        # Run optimisation directly — no process pool
        result_x, elapsed = _optimize_single_window(
            phi_sub_flat, phi_init_sub_flat, submatrix_size,
            is_at_edge, window_reached_max,
            threshold, max_minimize_iter, methodName,
            enforce_shoelace=enforce_shoelace,
            enforce_injectivity=enforce_injectivity,
        )
        iter_times.append(elapsed)

        _apply_result(phi, result_x, cy, cx, submatrix_size)

        jacobian_matrix, quality_matrix, cur_neg, cur_min = _update_metrics(
            phi, phi_init, enforce_shoelace, enforce_injectivity,
            num_neg_jac, min_jdet_list, error_list,
            jacobian_matrix=jacobian_matrix, patch_center=(cy, cx),
            patch_size=submatrix_size)

        _log(verbose, 2, f"  [sub-Jdet] centre ({cy},{cx}) window {sy}x{sx}:\n"
             + np.array2string(
                 jacobian_matrix[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi],
                 precision=4, suppress_small=True))

        if plot_callback is not None:
            plot_callback(deformation_i, phi)

        if float(quality_matrix[0].min()) > threshold - err_tol:
            break

        # Grow window for next sub-iteration
        sy, sx = _unpack_size(submatrix_size)
        if sy < max_sy or sx < max_sx:
            submatrix_size = (min(sy + 2, max_sy), min(sx + 2, max_sx))
        else:
            window_reached_max = True

    return jacobian_matrix, quality_matrix, submatrix_size, per_index_iter, (cy, cx)
