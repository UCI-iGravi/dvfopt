"""
Iterative SLSQP optimisation for correcting negative Jacobian determinants
in 2D deformation (displacement) fields.

This module contains only the core algorithm and helpers — no matplotlib or
pandas dependency.  Visualisation lives in ``modules.dvfviz``.

Usage::

    from modules.dvfopt import iterative_with_jacobians2, jacobian_det2D

Verbosity levels (``verbose`` parameter):

* ``0`` — silent, no output
* ``1`` — one-line progress per outer iteration + final summary
* ``2`` — full debug output (edge masks, constraints, sub-matrices)
"""

import os
import time
from collections import defaultdict

import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

import modules.jacobian as jacobian


# ---------------------------------------------------------------------------
# DVF generation utilities
# ---------------------------------------------------------------------------
def generate_random_dvf(shape, max_magnitude=5.0, seed=None):
    """Generate a random 2D deformation vector field (DVF).

    Parameters
    ----------
    shape : tuple
        ``(3, 1, H, W)`` — standard deformation field shape.
    max_magnitude : float
        Max displacement in pixels (uniform in ``[-mag, +mag]``).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(3, 1, H, W)``
    """
    if seed is not None:
        np.random.seed(seed)

    C, _, H, W = shape
    assert C == 3, "DVF must have 3 channels (dz, dy, dx)"
    return np.random.uniform(-max_magnitude, max_magnitude, size=shape).astype(np.float32)


def scale_dvf(dvf, new_size):
    """Rescale a ``(3, 1, H, W)`` deformation field to *new_size* ``(new_H, new_W)``.

    Spatial interpolation is bilinear (``order=1``) and displacement
    magnitudes are scaled proportionally.
    """
    C, _, H, W = dvf.shape
    new_H, new_W = new_size
    scale_y = new_H / H
    scale_x = new_W / W

    dvf_resized = np.zeros((C, 1, new_H, new_W), dtype=dvf.dtype)
    for c in range(C):
        dvf_resized[c, 0] = zoom(dvf[c, 0], (scale_y, scale_x), order=1)

    dvf_resized[2, 0] *= scale_x  # dx
    dvf_resized[1, 0] *= scale_y  # dy
    return dvf_resized


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "threshold": 0.01,
    "err_tol": 1e-5,
    "max_iterations": 10000,
    "max_per_index_iter": 50,
    "max_minimize_iter": 1000,
    "starting_window_size": 7,
}


# ---------------------------------------------------------------------------
# Internal logging helpers
# ---------------------------------------------------------------------------
def _log(verbose, level, msg):
    """Print *msg* if *verbose* >= *level*."""
    if verbose >= level:
        print(msg)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------
def objectiveEuc(phi, phi_init):
    """L2 norm objective function."""
    return np.linalg.norm(phi - phi_init)


def objectiveManh(phi, phi_init):
    """L1 norm objective function."""
    return np.linalg.norm(phi - phi_init, ord=1)


# ---------------------------------------------------------------------------
# Jacobian helpers
# ---------------------------------------------------------------------------
def compute_jacobian_determinant(deformation):
    """
    Compute the Jacobian determinant of a 2D deformation field (z,y,x) using:
    - Central differences for internal pixels
    - Forward/backward differences at edges
    Returns: (1, H, W) Jacobian determinant

    NOTE: This is a reference implementation. The main code uses
    ``jacobian.sitk_jacobian_determinant`` instead.
    """
    _, _, H, W = deformation.shape
    dy = deformation[1, 0]
    dx = deformation[2, 0]

    def gradient_central_with_fallback(f):
        df_dx = np.zeros_like(f)
        df_dy = np.zeros_like(f)
        # Central
        df_dx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0
        df_dy[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0
        # Forward (left/top edge)
        df_dx[:, 0] = f[:, 1] - f[:, 0]
        df_dy[0, :] = f[1, :] - f[0, :]
        # Backward (right/bottom edge)
        df_dx[:, -1] = f[:, -1] - f[:, -2]
        df_dy[-1, :] = f[-1, :] - f[-2, :]
        return df_dx, df_dy

    dy_dx, dy_dy = gradient_central_with_fallback(dy)
    dx_dx, dx_dy = gradient_central_with_fallback(dx)

    det_J = (1 + dx_dx) * (1 + dy_dy) - dx_dy * dy_dx
    det_J = det_J[np.newaxis, :, :]
    return det_J


def jacobian_det2D(phi_xy):
    """Compute the Jacobian determinant from a ``(2, H, W)`` phi array."""
    deformation = np.zeros((3, 1, *phi_xy.shape[-2:]))
    deformation[2] = phi_xy[1]
    deformation[1] = phi_xy[0]
    return jacobian.sitk_jacobian_determinant(deformation)


def jacobian_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened Jacobian determinant values for optimiser constraints."""
    deformation = np.zeros((3, 1, submatrix_size, submatrix_size))
    pixels = submatrix_size * submatrix_size
    deformation[2] = phi_xy[:pixels].reshape((submatrix_size, submatrix_size))
    deformation[1] = phi_xy[pixels:].reshape((submatrix_size, submatrix_size))
    jacobian_mat = jacobian.sitk_jacobian_determinant(deformation)
    if exclude_boundaries:
        return jacobian_mat[0, 1:-1, 1:-1].flatten()
    else:
        return jacobian_mat.flatten()


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
def nearest_center(shape, n):
    """Build a dict mapping every (z,y,x) to the nearest valid sub-window centre."""
    near_cent = {}
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                y_cent = y
                x_cent = x
                if x - n < 0:
                    x_cent = n
                elif shape[2] - x < n + 1:
                    x_cent = shape[2] - n - 1
                if y - n < 0:
                    y_cent = n
                elif shape[1] - y < n + 1:
                    y_cent = shape[1] - n - 1
                near_cent[(z, y, x)] = [z, y_cent, x_cent]
    return near_cent


def get_nearest_center(neg_index, slice_shape, submatrix_size, near_cent_dict):
    """Look up (or compute) the nearest valid centre for *neg_index*."""
    if submatrix_size in near_cent_dict:
        return near_cent_dict[submatrix_size][(0, *neg_index)]
    else:
        near_cent = nearest_center(slice_shape, submatrix_size // 2)
        near_cent_dict[submatrix_size] = near_cent
        return near_cent[(0, *neg_index)]


def argmin_excluding_edges(jacobian_matrix):
    """Index of the pixel with the lowest Jacobian determinant, excluding edges."""
    inner = jacobian_matrix[0, 1:-1, 1:-1]
    flat_index = np.argmin(inner)
    inner_idx = np.unravel_index(flat_index, inner.shape)
    return (inner_idx[0] + 1, inner_idx[1] + 1)


def get_phi_sub_flat(phi, cz, cy, cx, shape, d):
    """Extract and flatten a square sub-window of *phi* around (cy, cx)."""
    phix = phi[1, cy - d:cy + d + 1, cx - d:cx + d + 1]
    phiy = phi[0, cy - d:cy + d + 1, cx - d:cx + d + 1]
    return np.concatenate([phix.flatten(), phiy.flatten()])


# ---------------------------------------------------------------------------
# Main iterative SLSQP algorithm
# ---------------------------------------------------------------------------
def iterative_with_jacobians2(
    deformation_i,
    methodName="SLSQP",
    verbose=1,
    save_path=None,
    plot_every=0,
    plot_callback=None,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
    starting_window_size=None,
):
    """Iterative SLSQP correction of negative Jacobian determinants.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
        Input deformation field with channels ``[dz, dy, dx]``.
    methodName : str
        Optimiser method passed to ``scipy.optimize.minimize``.
    verbose : int
        Verbosity level. ``0`` = silent, ``1`` = per-iteration progress
        line + final summary, ``2`` = full debug output (edge masks,
        constraints, sub-Jacobian matrices).  Accepts ``True``/``False``
        for backward compatibility (mapped to 1/0).
    save_path : str or None
        Directory to save results. ``None`` disables saving.
    plot_every : int
        Show a Jacobian heatmap snapshot every *plot_every* outer
        iterations.  ``0`` disables (default).
    plot_callback : callable or None
        Optional callback receiving ``(deformation_i, phi)``
        after each sub-optimisation.
    threshold, err_tol, max_iterations, max_per_index_iter,
    max_minimize_iter, starting_window_size :
        Override the corresponding default parameters.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
        Corrected displacement field ``[dy, dx]``.
    """
    # Resolve parameters – use defaults where not overridden
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

    # Normalise verbose: bool → int for backward compatibility
    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

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

    # Working phi – updated iteratively
    phi = np.zeros((2, H, W))
    phi[1] = deformation_i[-1]
    phi[0] = deformation_i[-2]
    phi_init = phi.copy()

    _log(verbose, 1, f"[init] Grid {H}x{W}  |  threshold={threshold}  |  method={methodName}")
    _log(verbose, 2, f"[init] deformation_i shape: {deformation_i.shape}, phi shape: {phi.shape}")

    # Initial Jacobian
    jacobian_matrix = jacobian_det2D(phi)
    init_neg = int((jacobian_matrix <= 0).sum())
    init_min = float(jacobian_matrix.min())
    min_jdet_list.append(init_min)
    num_neg_jac.append(init_neg)

    _log(verbose, 1, f"[init] Neg-Jdet pixels: {init_neg}  |  min Jdet: {init_min:.6f}")

    iteration = 0
    while iteration < max_iterations and (jacobian_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any():
        iteration += 1
        window_reached_max = False

        neg_index_tuple = argmin_excluding_edges(jacobian_matrix)

        submatrix_size = starting_window_size
        per_index_iter = 0

        while (
            submatrix_size == starting_window_size
            or (
                (not window_reached_max)
                and per_index_iter < max_per_index_iter
                and (jacobian_matrix[0, cy - center_distance:cy + center_distance + 1,
                                     cx - center_distance:cx + center_distance + 1] < threshold - err_tol).any()
            )
        ):
            per_index_iter += 1

            if submatrix_size < min(slice_shape[1:]) - 1:
                submatrix_size += 2
                window_counts[submatrix_size] += 1

                sub_shape = (1, submatrix_size, submatrix_size)
                cz, cy, cx = get_nearest_center(neg_index_tuple, slice_shape, submatrix_size, near_cent_dict)

                phi_init_sub_flat = get_phi_sub_flat(phi_init, cz, cy, cx, slice_shape, submatrix_size // 2)
                phi_sub_flat = get_phi_sub_flat(phi, cz, cy, cx, slice_shape, submatrix_size // 2)

                if submatrix_size > starting_window_size:
                    _log(verbose, 2, f"  [window] Index {neg_index_tuple}: window grew to {submatrix_size}x{submatrix_size} (sub-iter {per_index_iter})")

            # Build constraints
            if submatrix_size >= min(slice_shape[1:]) - 1:
                window_reached_max = True
                nonlinear_constraints = NonlinearConstraint(
                    lambda phi1: jacobian_constraint(phi1, submatrix_size, False), threshold, np.inf
                )
                constraints = [nonlinear_constraints]
            else:
                start_y = cy - submatrix_size // 2
                end_y = cy + submatrix_size // 2
                start_x = cx - submatrix_size // 2
                end_x = cx + submatrix_size // 2
                max_y, max_x = slice_shape[1:]

                is_at_edge = start_y == 0 or end_y >= max_y - 1 or start_x == 0 or end_x >= max_x - 1
                _log(verbose, 2, f"  [edge] at_edge={is_at_edge}  y=[{start_y},{end_y}]  x=[{start_x},{end_x}]  grid=[{max_y-1},{max_x-1}]")

                nonlinear_constraints = NonlinearConstraint(
                    lambda phi1: jacobian_constraint(phi1, submatrix_size, not is_at_edge), threshold, np.inf
                )

                edge_mask = np.zeros((submatrix_size, submatrix_size), dtype=bool)
                if not is_at_edge:
                    edge_mask[[0, -1], :] = True
                    edge_mask[:, [0, -1]] = True

                _log(verbose, 2, f"  [edge] Edge mask ({submatrix_size}x{submatrix_size}):\n{edge_mask}")

                if not is_at_edge:
                    edge_indices = np.argwhere(edge_mask)
                    fixed_indices = []
                    y_offset_sub = submatrix_size * submatrix_size
                    for y, x in edge_indices:
                        idx = y * submatrix_size + x
                        fixed_indices.extend([idx, idx + y_offset_sub])

                    _log(verbose, 2, f"  [edge] Freezing {len(fixed_indices)} boundary DOFs")

                    fixed_values = phi_sub_flat[fixed_indices]

                    A_eq = np.zeros((len(fixed_indices), phi_sub_flat.size))
                    for row, idx in enumerate(fixed_indices):
                        A_eq[row, idx] = 1

                    linear_constraint = LinearConstraint(A_eq, fixed_values, fixed_values)
                    constraints = [nonlinear_constraints, linear_constraint]
                else:
                    constraints = [nonlinear_constraints]

            # Run optimisation
            iter_start = time.time()
            result = minimize(
                lambda phi1: objectiveEuc(phi1, phi_init_sub_flat),
                phi_sub_flat,
                constraints=constraints,
                options={"maxiter": max_minimize_iter, "disp": verbose >= 2},
                method=methodName,
            )
            iter_end = time.time()
            iter_times.append(iter_end - iter_start)

            phi_x_res = result.x[: len(result.x) // 2].reshape(sub_shape[-2:])
            phi_y_res = result.x[len(result.x) // 2 :].reshape(sub_shape[-2:])

            center_distance = submatrix_size // 2

            # Update phi
            phi[1, cy - center_distance:cy + center_distance + 1, cx - center_distance:cx + center_distance + 1] = phi_x_res
            phi[0, cy - center_distance:cy + center_distance + 1, cx - center_distance:cx + center_distance + 1] = phi_y_res

            jacobian_matrix = jacobian_det2D(phi)
            cur_neg = int((jacobian_matrix <= 0).sum())
            cur_min = float(jacobian_matrix.min())
            num_neg_jac.append(cur_neg)
            min_jdet_list.append(cur_min)

            _log(verbose, 2, f"  [sub-Jdet] centre ({cy},{cx}) window {submatrix_size}x{submatrix_size}:\n"
                 + np.array2string(
                     jacobian_matrix[0, cy - center_distance:cy + center_distance + 1,
                                     cx - center_distance:cx + center_distance + 1],
                     precision=4, suppress_small=True))

            if plot_callback is not None:
                plot_callback(deformation_i, phi)

            error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))

            if cur_min > threshold - err_tol:
                _log(verbose, 1, f"[done] All Jdet > threshold after iter {iteration}")
                break

        # One-line progress per outer iteration
        cur_neg = int((jacobian_matrix <= 0).sum())
        cur_min = float(jacobian_matrix.min())
        cur_err = error_list[-1] if error_list else 0.0
        _log(verbose, 1,
             f"[iter {iteration:4d}]  fix ({neg_index_tuple[0]:3d},{neg_index_tuple[1]:3d})  "
             f"win {submatrix_size:3d}  neg_jdet {cur_neg:5d}  "
             f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
             f"sub-iters {per_index_iter}")

        # Per-step snapshot
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
    _log(verbose, 1, f"  SUMMARY  ({methodName})")
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
        output_text += f"\tMethod: {methodName}\n"
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

        # Write window_counts CSV without pandas
        with open(save_path + "/window_counts.csv", "w") as f:
            f.write("window_size,count\n")
            for ws in sorted(window_counts):
                f.write(f"{ws},{window_counts[ws]}\n")

    return phi
