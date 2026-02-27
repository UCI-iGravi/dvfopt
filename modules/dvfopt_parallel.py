"""
Parallelized iterative SLSQP optimisation for correcting negative Jacobian
determinants in 2D deformation (displacement) fields.

This module mirrors ``dvfopt.iterative_with_jacobians2`` but processes
multiple non-overlapping negative-Jdet windows in parallel using
``concurrent.futures.ProcessPoolExecutor``.

Usage::

    from modules.dvfopt_parallel import iterative_parallel

"""

import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


# ---------------------------------------------------------------------------
# Re-export shared helpers from dvfopt
# ---------------------------------------------------------------------------
from modules.dvfopt import (
    DEFAULT_PARAMS,
    _log,
    objectiveEuc,
    jacobian_det2D,
    jacobian_constraint,
    nearest_center,
    get_nearest_center,
    argmin_excluding_edges,
    get_phi_sub_flat,
)


# ---------------------------------------------------------------------------
# Standalone worker for ProcessPoolExecutor (must be picklable)
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
):
    """Run SLSQP on a single sub-window.  Designed to be called in a worker
    process — all data is passed explicitly (no closures / lambdas that
    capture mutable state).

    Returns
    -------
    result_x : ndarray
        Optimised flattened phi for this window.
    elapsed : float
        Wall-clock time for this sub-optimisation.
    """
    # Build constraints ------------------------------------------------
    if window_reached_max:
        nlc = NonlinearConstraint(
            lambda phi1: jacobian_constraint(phi1, submatrix_size, False),
            threshold,
            np.inf,
        )
        constraints = [nlc]
    else:
        nlc = NonlinearConstraint(
            lambda phi1: jacobian_constraint(phi1, submatrix_size, not is_at_edge),
            threshold,
            np.inf,
        )

        if not is_at_edge:
            edge_mask = np.zeros((submatrix_size, submatrix_size), dtype=bool)
            edge_mask[[0, -1], :] = True
            edge_mask[:, [0, -1]] = True

            edge_indices = np.argwhere(edge_mask)
            fixed_indices = []
            y_offset_sub = submatrix_size * submatrix_size
            for y, x in edge_indices:
                idx = y * submatrix_size + x
                fixed_indices.extend([idx, idx + y_offset_sub])

            fixed_values = phi_sub_flat[fixed_indices]
            A_eq = np.zeros((len(fixed_indices), phi_sub_flat.size))
            for row, idx in enumerate(fixed_indices):
                A_eq[row, idx] = 1
            linear_constraint = LinearConstraint(A_eq, fixed_values, fixed_values)
            constraints = [nlc, linear_constraint]
        else:
            constraints = [nlc]

    # Optimise ---------------------------------------------------------
    t0 = time.time()
    result = minimize(
        lambda phi1: objectiveEuc(phi1, phi_init_sub_flat),
        phi_sub_flat,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": False},
        method=method_name,
    )
    elapsed = time.time() - t0
    return result.x, elapsed


# ---------------------------------------------------------------------------
# Helpers for batching non-overlapping windows
# ---------------------------------------------------------------------------
def _find_negative_pixels(jacobian_matrix, threshold, err_tol):
    """Return list of (y, x) for all inner pixels below threshold."""
    inner = jacobian_matrix[0, 1:-1, 1:-1]
    ys, xs = np.where(inner <= threshold - err_tol)
    # Sort by Jdet value (worst first)
    vals = inner[ys, xs]
    order = np.argsort(vals)
    return [(int(ys[i]) + 1, int(xs[i]) + 1) for i in order]


def _window_bounds(cy, cx, d):
    """Return (y_start, y_end, x_start, x_end) for a window centred at (cy,cx)."""
    return (cy - d, cy + d, cx - d, cx + d)


def _windows_overlap(b1, b2):
    """Check whether two bounding boxes (y0, y1, x0, x1) overlap."""
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])


def _select_non_overlapping(neg_pixels, slice_shape, submatrix_size, near_cent_dict):
    """Greedily select non-overlapping windows for negative pixels.

    Returns list of (neg_index, (cz, cy, cx), submatrix_size) tuples.
    """
    d = submatrix_size // 2
    selected = []
    used_bounds = []

    for neg_idx in neg_pixels:
        cz, cy, cx = get_nearest_center(neg_idx, slice_shape, submatrix_size, near_cent_dict)
        bounds = _window_bounds(cy, cx, d)

        overlaps = False
        for ub in used_bounds:
            if _windows_overlap(bounds, ub):
                overlaps = True
                break

        if not overlaps:
            selected.append((neg_idx, (cz, cy, cx), submatrix_size))
            used_bounds.append(bounds)

    return selected


# ---------------------------------------------------------------------------
# Main parallelized algorithm
# ---------------------------------------------------------------------------
def iterative_parallel(
    deformation_i,
    methodName="SLSQP",
    verbose=1,
    save_path=None,
    plot_every=0,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
    starting_window_size=None,
    max_workers=None,
):
    """Parallelized iterative SLSQP correction.

    Same interface as ``iterative_with_jacobians2`` but dispatches
    non-overlapping window optimisations to a process pool.

    Parameters
    ----------
    max_workers : int or None
        Number of worker processes.  ``None`` → ``os.cpu_count()``.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    """
    # Resolve parameters
    p = dict(DEFAULT_PARAMS)
    for name, val in [
        ("threshold", threshold),
        ("err_tol", err_tol),
        ("max_iterations", max_iterations),
        ("max_per_index_iter", max_per_index_iter),
        ("max_minimize_iter", max_minimize_iter),
        ("starting_window_size", starting_window_size),
    ]:
        if val is not None:
            p[name] = val
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]
    starting_window_size = p["starting_window_size"]

    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

    if max_workers is None:
        max_workers = os.cpu_count()

    # Accumulators
    error_list = []
    num_neg_jac = []
    iter_times = []
    min_jdet_list = []
    window_counts = defaultdict(int)

    start_time = time.time()
    H, W = deformation_i.shape[-2:]
    slice_shape = (1, H, W)
    near_cent_dict = {}

    # Working phi
    phi = np.zeros((2, H, W))
    phi[1] = deformation_i[-1]
    phi[0] = deformation_i[-2]
    phi_init = phi.copy()

    _log(verbose, 1, f"[init] Grid {H}x{W}  |  threshold={threshold}  |  method={methodName}  |  workers={max_workers}")

    # Initial Jacobian
    jacobian_matrix = jacobian_det2D(phi)
    init_neg = int((jacobian_matrix <= 0).sum())
    init_min = float(jacobian_matrix.min())
    min_jdet_list.append(init_min)
    num_neg_jac.append(init_neg)

    _log(verbose, 1, f"[init] Neg-Jdet pixels: {init_neg}  |  min Jdet: {init_min:.6f}")

    iteration = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while iteration < max_iterations and (jacobian_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any():
            iteration += 1

            # Find all negative pixels and select non-overlapping batch
            neg_pixels = _find_negative_pixels(jacobian_matrix, threshold, err_tol)
            if not neg_pixels:
                break

            # Use starting_window_size + 2 (matching the serial version's
            # initial bump: it enters the inner while loop with
            # submatrix_size = starting_window_size, then immediately does
            # submatrix_size += 2)
            batch_window = starting_window_size + 2
            batch = _select_non_overlapping(neg_pixels, slice_shape, batch_window, near_cent_dict)

            if not batch:
                # All negative pixels too close together – fall back to
                # single worst pixel
                neg_idx = neg_pixels[0]
                cz, cy, cx = get_nearest_center(neg_idx, slice_shape, batch_window, near_cent_dict)
                batch = [(neg_idx, (cz, cy, cx), batch_window)]

            _log(verbose, 1,
                 f"[iter {iteration:4d}]  batch_size={len(batch)}  "
                 f"neg_pixels={len(neg_pixels)}  window={batch_window}")

            # Submit all windows to the pool
            futures = {}
            for neg_idx, (cz, cy, cx), sub_size in batch:
                d = sub_size // 2
                window_counts[sub_size] += 1

                phi_init_sub_flat = get_phi_sub_flat(phi_init, cz, cy, cx, slice_shape, d)
                phi_sub_flat = get_phi_sub_flat(phi, cz, cy, cx, slice_shape, d)

                start_y = cy - d
                end_y = cy + d
                start_x = cx - d
                end_x = cx + d
                max_y, max_x = slice_shape[1:]

                is_at_edge = start_y == 0 or end_y >= max_y - 1 or start_x == 0 or end_x >= max_x - 1
                window_reached_max = sub_size >= min(slice_shape[1:]) - 1

                fut = executor.submit(
                    _optimize_single_window,
                    phi_sub_flat,
                    phi_init_sub_flat,
                    sub_size,
                    is_at_edge,
                    window_reached_max,
                    threshold,
                    max_minimize_iter,
                    methodName,
                )
                futures[fut] = (neg_idx, cz, cy, cx, sub_size)

            # Collect results
            batch_time = 0.0
            for fut in as_completed(futures):
                neg_idx, cz, cy, cx, sub_size = futures[fut]
                result_x, elapsed = fut.result()
                batch_time = max(batch_time, elapsed)  # wall-clock = longest

                d = sub_size // 2
                pixels = sub_size * sub_size
                phi_x_res = result_x[:pixels].reshape(sub_size, sub_size)
                phi_y_res = result_x[pixels:].reshape(sub_size, sub_size)

                phi[1, cy - d:cy + d + 1, cx - d:cx + d + 1] = phi_x_res
                phi[0, cy - d:cy + d + 1, cx - d:cx + d + 1] = phi_y_res

            iter_times.append(batch_time)

            # Recompute Jacobian after batch
            jacobian_matrix = jacobian_det2D(phi)
            cur_neg = int((jacobian_matrix <= 0).sum())
            cur_min = float(jacobian_matrix.min())
            num_neg_jac.append(cur_neg)
            min_jdet_list.append(cur_min)
            error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))

            cur_err = error_list[-1]
            _log(verbose, 1,
                 f"         -> neg_jdet {cur_neg:5d}  "
                 f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}")

            if plot_every and iteration % plot_every == 0:
                from modules.dvfviz import plot_step_snapshot
                plot_step_snapshot(jacobian_matrix, iteration, cur_neg, cur_min)

            if cur_min > threshold - err_tol:
                _log(verbose, 1, f"[done] All Jdet > threshold after iter {iteration}")
                break

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _log(verbose, 1, "")
    _log(verbose, 1, "=" * 60)
    _log(verbose, 1, f"  SUMMARY  ({methodName} — parallel, {max_workers} workers)")
    _log(verbose, 1, "-" * 60)
    _log(verbose, 1, f"  Grid size        : {H} x {W}")
    _log(verbose, 1, f"  Iterations       : {iteration}")
    _log(verbose, 1, f"  Neg-Jdet  {init_neg:>5d} -> {final_neg:>5d}")
    _log(verbose, 1, f"  Min Jdet  {init_min:+.6f} -> {final_min:+.6f}")
    _log(verbose, 1, f"  L2 error         : {final_err:.6f}")
    _log(verbose, 1, f"  Time             : {elapsed:.2f}s")
    _log(verbose, 1, "=" * 60)

    num_neg_jac.append(final_neg)

    # Save results
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        output_text = "Settings:\n"
        output_text += f"\tMethod: {methodName} (parallel, {max_workers} workers)\n"
        output_text += f"\tThreshold: {threshold}\n"
        output_text += f"\tError tolerance: {err_tol}\n"
        output_text += f"\tMax iterations: {max_iterations}\n"
        output_text += f"\tMax per index iterations: {max_per_index_iter}\n"
        output_text += f"\tMax minimize iterations: {max_minimize_iter}\n"
        output_text += f"\tStarting window size: {starting_window_size + 2}\n\n"

        output_text += "Results:\n"
        output_text += f"\tInput deformation field resolution (height x width): {H} x {W}\n"
        output_text += f"\tTotal run-time: {elapsed} seconds\n"
        output_text += f"\tFinal L2 error: {final_err}\n"
        output_text += f"\tStarting number of non-positive Jacobian determinants: {init_neg}\n"
        output_text += f"\tFinal number of non-positive Jacobian determinants: {final_neg}\n"
        output_text += f"\tStarting Jacobian determinant minimum value: {init_min}\n"
        output_text += f"\tFinal Jacobian determinant minimum value: {final_min}\n"
        output_text += f"\tNumber of index iterations: {iteration}"

        with open(save_path + "/results.txt", "w") as f:
            f.write(output_text)

        np.save(save_path + "/phi.npy", phi)
        np.save(save_path + "/error_list.npy", error_list)
        np.save(save_path + "/num_neg_jac.npy", num_neg_jac)
        np.save(save_path + "/iter_times.npy", iter_times)
        np.save(save_path + "/min_jdet_list.npy", min_jdet_list)

        with open(save_path + "/window_counts.csv", "w") as f:
            f.write("window_size,count\n")
            for ws in sorted(window_counts):
                f.write(f"{ws},{window_counts[ws]}\n")

    return phi
