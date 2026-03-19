"""
Iterative SLSQP optimisation for correcting negative Jacobian determinants
in 2D deformation (displacement) fields.

This module contains both the serial and hybrid-parallel algorithms, plus
all shared helpers.  No matplotlib or pandas dependency — visualisation
lives in ``modules.dvfviz``.

Usage::

    from modules.dvfopt import iterative_with_jacobians2, jacobian_det2D
    from modules.dvfopt import iterative_parallel  # hybrid parallel variant

Verbosity levels (``verbose`` parameter):

* ``0`` — silent, no output
* ``1`` — one-line progress per outer iteration + final summary
* ``2`` — full debug output (edge masks, constraints, sub-matrices)
"""

import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


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
    "min_window_size": 9,
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


# ---------------------------------------------------------------------------
# Jacobian helpers
# ---------------------------------------------------------------------------
def _numpy_jdet_2d(dy, dx):
    """Compute 2D Jacobian determinant from displacement components (numpy).

    Uses central differences for interior pixels, matching SimpleITK's
    ``DisplacementFieldJacobianDeterminant`` for interior values.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    ndarray, shape ``(H, W)``
    """
    ddx_dx = np.gradient(dx, axis=1)  # ∂dx/∂x
    ddy_dy = np.gradient(dy, axis=0)  # ∂dy/∂y
    ddx_dy = np.gradient(dx, axis=0)  # ∂dx/∂y
    ddy_dx = np.gradient(dy, axis=1)  # ∂dy/∂x
    return (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx


def jacobian_det2D(phi_xy):
    """Compute the Jacobian determinant from a ``(2, H, W)`` phi array.

    Also accepts ``(2, 1, H, W)`` (the extra unit dimension is squeezed).
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    jdet = _numpy_jdet_2d(dy, dx)
    return jdet[np.newaxis, :, :]  # (1, H, W) to match existing API


def jacobian_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened Jacobian determinant values for optimiser constraints."""
    pixels = submatrix_size * submatrix_size
    dx = phi_xy[:pixels].reshape((submatrix_size, submatrix_size))
    dy = phi_xy[pixels:].reshape((submatrix_size, submatrix_size))
    jdet = _numpy_jdet_2d(dy, dx)
    if exclude_boundaries:
        return jdet[1:-1, 1:-1].flatten()
    else:
        return jdet.flatten()


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
def nearest_center(shape, submatrix_size):
    """Build a dict mapping every (z,y,x) to the nearest valid sub-window centre."""
    d = submatrix_size // 2
    max_y = shape[1] - submatrix_size + d
    max_x = shape[2] - submatrix_size + d
    near_cent = {}
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                near_cent[(z, y, x)] = [z, max(d, min(y, max_y)),
                                           max(d, min(x, max_x))]
    return near_cent


def get_nearest_center(neg_index, slice_shape, submatrix_size, near_cent_dict):
    """Look up (or compute) the nearest valid centre for *neg_index*."""
    if submatrix_size in near_cent_dict:
        return near_cent_dict[submatrix_size][(0, *neg_index)]
    else:
        near_cent = nearest_center(slice_shape, submatrix_size)
        near_cent_dict[submatrix_size] = near_cent
        return near_cent[(0, *neg_index)]


def argmin_excluding_edges(jacobian_matrix):
    """Index of the pixel with the lowest Jacobian determinant, excluding edges."""
    inner = jacobian_matrix[0, 1:-1, 1:-1]
    flat_index = np.argmin(inner)
    inner_idx = np.unravel_index(flat_index, inner.shape)
    return (inner_idx[0] + 1, inner_idx[1] + 1)


def neg_jdet_bounding_window(jacobian_matrix, center_yx, threshold, err_tol,
                             min_window_size=5):
    """Compute the smallest window enclosing the negative-Jdet region around *center_yx*.

    The window is the bounding box of all pixels with Jdet <= *threshold* - *err_tol*
    that are **connected** (8-connectivity) to *center_yx*, expanded by 1 pixel on
    each side so the frozen edges sit on positive-Jdet pixels.

    The size is computed so that a window centred on *center_yx* (via
    ``nearest_center``) is guaranteed to cover the entire bounding box,
    regardless of how off-centre the worst pixel is within the negative
    region.  The result is always odd and at least *min_window_size*.

    Parameters
    ----------
    jacobian_matrix : ndarray, shape ``(1, H, W)``
    center_yx : tuple of int
        ``(y, x)`` of the worst pixel.
    threshold, err_tol : float
    min_window_size : int
        Absolute minimum window size (>= 5).

    Returns
    -------
    int
        Window side length (always odd).
    """
    from scipy.ndimage import label

    neg_mask = jacobian_matrix[0] <= threshold - err_tol
    labeled, _ = label(neg_mask)  # 8-connectivity is the scipy default via structure
    region_label = labeled[center_yx[0], center_yx[1]]

    if region_label == 0:
        # Pixel is not negative (shouldn't happen, but be safe)
        return min_window_size

    region_ys, region_xs = np.where(labeled == region_label)
    # Bounding box of the connected negative region + 1 pixel border,
    # clamped to the grid so edge-touching regions don't go out of bounds.
    H, W = jacobian_matrix.shape[1:]
    y_min = max(int(region_ys.min()) - 1, 0)
    y_max = min(int(region_ys.max()) + 1, H - 1)
    x_min = max(int(region_xs.min()) - 1, 0)
    x_max = min(int(region_xs.max()) + 1, W - 1)

    # Compute the maximum distance from the worst pixel to any bbox edge.
    # A window of size 2*d_needed+1 centred on center_yx is guaranteed to
    # cover [y_min..y_max] × [x_min..x_max] because d >= d_needed on each
    # side.
    d_needed = max(
        center_yx[0] - y_min,
        y_max - center_yx[0],
        center_yx[1] - x_min,
        x_max - center_yx[1],
    )
    side = 2 * d_needed + 1
    side = max(side, min_window_size)
    if side % 2 == 0:
        side += 1

    return side


def _frozen_edges_clean(jacobian_matrix, cy, cx, submatrix_size, threshold, err_tol):
    """Return True if the frozen edges of the window have positive Jdet.

    Checks the outer ring of the ``submatrix_size`` window centred at
    ``(cy, cx)``.  If any edge pixel has Jdet <= threshold - err_tol,
    the optimiser's frozen-edge constraint would be infeasible, so the
    caller should grow the window instead of running the optimiser.
    """
    d = submatrix_size // 2
    d_hi = submatrix_size - d
    y0, y1 = cy - d, cy + d_hi - 1
    x0, x1 = cx - d, cx + d_hi - 1
    edge_vals = np.concatenate([
        jacobian_matrix[0, y0, x0:x1 + 1].ravel(),
        jacobian_matrix[0, y1, x0:x1 + 1].ravel(),
        jacobian_matrix[0, y0:y1 + 1, x0].ravel(),
        jacobian_matrix[0, y0:y1 + 1, x1].ravel(),
    ])
    return edge_vals.min() > threshold - err_tol


def get_phi_sub_flat(phi, cz, cy, cx, shape, submatrix_size):
    """Extract and flatten a square sub-window of *phi* around (cy, cx)."""
    d = submatrix_size // 2
    d_hi = submatrix_size - d
    phix = phi[1, cy - d:cy + d_hi, cx - d:cx + d_hi]
    phiy = phi[0, cy - d:cy + d_hi, cx - d:cx + d_hi]
    return np.concatenate([phix.flatten(), phiy.flatten()])


# ---------------------------------------------------------------------------
# Shared constraint builder
# ---------------------------------------------------------------------------
def _build_constraints(phi_sub_flat, submatrix_size, is_at_edge,
                       window_reached_max, threshold):
    """Build SLSQP constraints for a sub-window optimisation.

    Returns a list of constraint objects suitable for
    ``scipy.optimize.minimize``.
    """
    if window_reached_max:
        nlc = NonlinearConstraint(
            lambda phi1: jacobian_constraint(phi1, submatrix_size, False),
            threshold, np.inf,
        )
        return [nlc]

    nlc = NonlinearConstraint(
        lambda phi1: jacobian_constraint(phi1, submatrix_size, not is_at_edge),
        threshold, np.inf,
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
        return [nlc, linear_constraint]

    return [nlc]


# ---------------------------------------------------------------------------
# Shared init / save helpers
# ---------------------------------------------------------------------------
def _resolve_params(**overrides):
    """Merge *overrides* into ``DEFAULT_PARAMS``, returning resolved dict."""
    p = dict(DEFAULT_PARAMS)
    for name, val in overrides.items():
        if val is not None:
            p[name] = val
    return p


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


def _save_results(save_path, *, method, threshold, err_tol, max_iterations,
                  max_per_index_iter, max_minimize_iter, min_window_size,
                  H, W, elapsed, final_err, init_neg, final_neg, init_min,
                  final_min, iteration, phi, error_list, num_neg_jac,
                  iter_times, min_jdet_list, window_counts,
                  extra_settings="", extra_results=""):
    """Write correction results to *save_path*."""
    os.makedirs(save_path, exist_ok=True)

    output_text = "Settings:\n"
    output_text += f"\tMethod: {method}\n"
    output_text += f"\tThreshold: {threshold}\n"
    output_text += f"\tError tolerance: {err_tol}\n"
    output_text += f"\tMax iterations: {max_iterations}\n"
    output_text += f"\tMax per index iterations: {max_per_index_iter}\n"
    output_text += f"\tMax minimize iterations: {max_minimize_iter}\n"
    output_text += f"\tMin window size: {min_window_size}\n"
    if extra_settings:
        output_text += extra_settings
    output_text += "\nResults:\n"
    output_text += f"\tInput deformation field resolution (height x width): {H} x {W}\n"
    output_text += f"\tTotal run-time: {elapsed} seconds\n"
    output_text += f"\tFinal L2 error: {final_err}\n"
    output_text += f"\tStarting number of non-positive Jacobian determinants: {init_neg}\n"
    output_text += f"\tFinal number of non-positive Jacobian determinants: {final_neg}\n"
    output_text += f"\tStarting Jacobian determinant minimum value: {init_min}\n"
    output_text += f"\tFinal Jacobian determinant minimum value: {final_min}\n"
    output_text += f"\tNumber of index iterations: {iteration}"
    if extra_results:
        output_text += "\n" + extra_results

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
    min_window_size=None,
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
    max_minimize_iter, min_window_size :
        Override the corresponding default parameters.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
        Corrected displacement field ``[dy, dx]``.
    """
    p = _resolve_params(threshold=threshold, err_tol=err_tol,
                        max_iterations=max_iterations,
                        max_per_index_iter=max_per_index_iter,
                        max_minimize_iter=max_minimize_iter,
                        min_window_size=min_window_size)
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]
    min_window_size = p["min_window_size"]

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
    phi, phi_init, H, W = _init_phi(deformation_i)
    slice_shape = (1, H, W)
    max_window = min(H, W) - 1
    near_cent_dict = {}

    _log(verbose, 1, f"[init] Grid {H}x{W}  |  threshold={threshold}  |  method={methodName}  |  min_window={min_window_size}")
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

        submatrix_size = min_window_size

        per_index_iter = 0

        while (
            per_index_iter == 0
            or (
                (not window_reached_max)
                and per_index_iter < max_per_index_iter
                and (jacobian_matrix[0, cy - d:cy + d_hi,
                                     cx - d:cx + d_hi] < threshold - err_tol).any()
            )
        ):
            per_index_iter += 1

            window_counts[submatrix_size] += 1

            cz, cy, cx = get_nearest_center(neg_index_tuple, slice_shape, submatrix_size, near_cent_dict)
            d = submatrix_size // 2
            d_hi = submatrix_size - d

            phi_init_sub_flat = get_phi_sub_flat(phi_init, cz, cy, cx, slice_shape, submatrix_size)
            phi_sub_flat = get_phi_sub_flat(phi, cz, cy, cx, slice_shape, submatrix_size)

            if per_index_iter > 1:
                _log(verbose, 2, f"  [window] Index {neg_index_tuple}: window grew to {submatrix_size}x{submatrix_size} (sub-iter {per_index_iter})")

            # Build constraints and edge flags
            is_at_edge, w_max = _edge_flags(cy, cx, submatrix_size, slice_shape, max_window)
            window_reached_max = window_reached_max or w_max

            _log(verbose, 2, f"  [edge] at_edge={is_at_edge}  window_reached_max={window_reached_max}")

            # If frozen edges contain negative Jdet the constraint is likely
            # infeasible — skip the expensive optimizer and grow immediately.
            if (not is_at_edge and not window_reached_max
                    and not _frozen_edges_clean(jacobian_matrix, cy, cx,
                                               submatrix_size, threshold, err_tol)):
                _log(verbose, 2, f"  [skip] Frozen edges have neg Jdet at win {submatrix_size} — growing")
                if submatrix_size < max_window:
                    submatrix_size += 2
                    submatrix_size = min(submatrix_size, max_window)
                continue

            constraints = _build_constraints(
                phi_sub_flat, submatrix_size, is_at_edge, window_reached_max, threshold
            )

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

            _apply_result(phi, result.x, cy, cx, submatrix_size)

            jacobian_matrix = jacobian_det2D(phi)
            cur_neg = int((jacobian_matrix <= 0).sum())
            cur_min = float(jacobian_matrix.min())
            num_neg_jac.append(cur_neg)
            min_jdet_list.append(cur_min)

            _log(verbose, 2, f"  [sub-Jdet] centre ({cy},{cx}) window {submatrix_size}x{submatrix_size}:\n"
                 + np.array2string(
                     jacobian_matrix[0, cy - d:cy + d_hi, cx - d:cx + d_hi],
                     precision=4, suppress_small=True))

            if plot_callback is not None:
                plot_callback(deformation_i, phi)

            error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))

            if cur_min > threshold - err_tol:
                _log(verbose, 1, f"[done] All Jdet > threshold after iter {iteration}")
                break

            # Grow window for next sub-iteration
            if submatrix_size < max_window:
                submatrix_size += 2
                submatrix_size = min(submatrix_size, max_window)

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

    if save_path is not None:
        _save_results(
            save_path, method=methodName, threshold=threshold, err_tol=err_tol,
            max_iterations=max_iterations, max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter, min_window_size=min_window_size,
            H=H, W=W, elapsed=elapsed, final_err=final_err,
            init_neg=init_neg, final_neg=final_neg, init_min=init_min,
            final_min=final_min, iteration=iteration, phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi


# ===================================================================
# Hybrid parallel iterative SLSQP
# ===================================================================

# ---------------------------------------------------------------------------
# Standalone worker (must be picklable — top-level function)
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
    """Run SLSQP on one sub-window.  Returns ``(result_x, elapsed)``."""

    constraints = _build_constraints(
        phi_sub_flat, submatrix_size, is_at_edge, window_reached_max, threshold
    )

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
# Parallel helpers
# ---------------------------------------------------------------------------
def _find_negative_pixels(jacobian_matrix, threshold, err_tol):
    """Return list of (y, x) for inner pixels below threshold, worst first."""
    inner = jacobian_matrix[0, 1:-1, 1:-1]
    ys, xs = np.where(inner <= threshold - err_tol)
    vals = inner[ys, xs]
    order = np.argsort(vals)
    return [(int(ys[i]) + 1, int(xs[i]) + 1) for i in order]


def _window_bounds(cy, cx, submatrix_size):
    d = submatrix_size // 2
    d_hi = submatrix_size - d
    return (cy - d, cy + d_hi - 1, cx - d, cx + d_hi - 1)


def _windows_overlap(b1, b2):
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])


def _select_non_overlapping(neg_pixels, pixel_window_sizes, slice_shape,
                             near_cent_dict):
    """Greedily select non-overlapping windows (each pixel has its own size)."""
    selected = []
    used_bounds = []

    for neg_idx in neg_pixels:
        ws = pixel_window_sizes[neg_idx]
        cz, cy, cx = get_nearest_center(neg_idx, slice_shape, ws, near_cent_dict)
        bounds = _window_bounds(cy, cx, ws)

        overlaps = False
        for ub in used_bounds:
            if _windows_overlap(bounds, ub):
                overlaps = True
                break

        if not overlaps:
            selected.append((neg_idx, (cz, cy, cx), ws))
            used_bounds.append(bounds)

    return selected


def _apply_result(phi, result_x, cy, cx, sub_size):
    """Write optimised sub-window back into *phi*."""
    d = sub_size // 2
    d_hi = sub_size - d
    pixels = sub_size * sub_size
    phi[1, cy - d:cy + d_hi, cx - d:cx + d_hi] = \
        result_x[:pixels].reshape(sub_size, sub_size)
    phi[0, cy - d:cy + d_hi, cx - d:cx + d_hi] = \
        result_x[pixels:].reshape(sub_size, sub_size)


def _edge_flags(cy, cx, submatrix_size, slice_shape, max_window):
    """Return (is_at_edge, window_reached_max) for a window."""
    d = submatrix_size // 2
    d_hi = submatrix_size - d
    start_y = cy - d
    end_y = cy + d_hi - 1          # inclusive last pixel
    start_x = cx - d
    end_x = cx + d_hi - 1
    max_y, max_x = slice_shape[1:]
    is_at_edge = (start_y == 0 or end_y >= max_y - 1
                  or start_x == 0 or end_x >= max_x - 1)
    window_reached_max = submatrix_size >= max_window
    return is_at_edge, window_reached_max


# ---------------------------------------------------------------------------
# Serial inner loop — runs when batch_size == 1
# ---------------------------------------------------------------------------
def _serial_fix_pixel(
    neg_index_tuple, phi, phi_init, jacobian_matrix,
    slice_shape, near_cent_dict, window_counts,
    min_window_size, max_per_index_iter, max_minimize_iter,
    max_window, threshold, err_tol, methodName, verbose,
    error_list, num_neg_jac, min_jdet_list, iter_times,
):
    """Fix a single pixel using the serial adaptive-window inner loop.

    Mirrors the inner ``while`` loop of ``iterative_with_jacobians2``
    exactly: start from the bounding-box-derived window, then grow by 2
    each sub-iteration until the local region is clean or the window
    hits the grid boundary.

    Mutates *phi*, *jacobian_matrix*, and the accumulator lists in-place,
    returns the updated jacobian_matrix.
    """
    # Adaptive starting size from negative-Jdet bounding box
    submatrix_size = min_window_size

    per_index_iter = 0
    window_reached_max = False

    while (
        per_index_iter == 0
        or (
            (not window_reached_max)
            and per_index_iter < max_per_index_iter
            and (jacobian_matrix[0,
                    cy - d:cy + d_hi,
                    cx - d:cx + d_hi]
                 < threshold - err_tol).any()
        )
    ):
        per_index_iter += 1

        window_counts[submatrix_size] += 1

        cz, cy, cx = get_nearest_center(
            neg_index_tuple, slice_shape, submatrix_size, near_cent_dict)
        d = submatrix_size // 2
        d_hi = submatrix_size - d

        phi_init_sub_flat = get_phi_sub_flat(phi_init, cz, cy, cx, slice_shape, submatrix_size)
        phi_sub_flat = get_phi_sub_flat(phi, cz, cy, cx, slice_shape, submatrix_size)

        is_at_edge, w_max = _edge_flags(cy, cx, submatrix_size, slice_shape, max_window)
        window_reached_max = window_reached_max or w_max

        # Skip optimizer if frozen edges have negative Jdet (likely infeasible)
        if (not is_at_edge and not window_reached_max
                and not _frozen_edges_clean(jacobian_matrix, cy, cx,
                                           submatrix_size, threshold, err_tol)):
            if submatrix_size < max_window:
                submatrix_size += 2
                submatrix_size = min(submatrix_size, max_window)
            continue

        # Run optimisation directly — no process pool
        result_x, elapsed = _optimize_single_window(
            phi_sub_flat, phi_init_sub_flat, submatrix_size,
            is_at_edge, window_reached_max,
            threshold, max_minimize_iter, methodName,
        )
        iter_times.append(elapsed)

        _apply_result(phi, result_x, cy, cx, submatrix_size)

        jacobian_matrix = jacobian_det2D(phi)
        cur_neg = int((jacobian_matrix <= 0).sum())
        cur_min = float(jacobian_matrix.min())
        num_neg_jac.append(cur_neg)
        min_jdet_list.append(cur_min)
        error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))

        if cur_min > threshold - err_tol:
            break

        # Grow window for next sub-iteration
        if submatrix_size < max_window:
            submatrix_size += 2
            submatrix_size = min(submatrix_size, max_window)
        else:
            window_reached_max = True

    return jacobian_matrix, submatrix_size, per_index_iter


# ---------------------------------------------------------------------------
# Main hybrid parallel algorithm
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
    min_window_size=None,
    max_workers=None,
):
    """Hybrid serial/parallel iterative SLSQP correction.

    * batch_size > 1 → ``ProcessPoolExecutor`` (true parallelism)
    * batch_size == 1 → direct in-process with full adaptive window
      growth (avoids Windows ``spawn`` overhead)

    Parameters
    ----------
    max_workers : int or None
        Number of worker processes.  ``None`` → ``os.cpu_count()``.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    """
    # Resolve parameters
    p = _resolve_params(threshold=threshold, err_tol=err_tol,
                        max_iterations=max_iterations,
                        max_per_index_iter=max_per_index_iter,
                        max_minimize_iter=max_minimize_iter,
                        min_window_size=min_window_size)
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]
    min_window_size = p["min_window_size"]

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
    phi, phi_init, H, W = _init_phi(deformation_i)
    slice_shape = (1, H, W)
    max_window = min(H, W) - 1
    near_cent_dict = {}

    _log(verbose, 1,
         f"[init] Grid {H}x{W}  |  threshold={threshold}  "
         f"|  method={methodName}  |  workers={max_workers}  |  min_window={min_window_size}")

    jacobian_matrix = jacobian_det2D(phi)
    init_neg = int((jacobian_matrix <= 0).sum())
    init_min = float(jacobian_matrix.min())
    min_jdet_list.append(init_min)
    num_neg_jac.append(init_neg)

    _log(verbose, 1,
         f"[init] Neg-Jdet pixels: {init_neg}  |  min Jdet: {init_min:.6f}")

    # Per-pixel window size tracker for parallel batches
    pixel_window_sizes = {}

    iteration = 0
    serial_iters = 0
    parallel_iters = 0
    executor = None  # lazy — only created if we actually need parallelism

    try:
        while (iteration < max_iterations
               and (jacobian_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any()):
            iteration += 1

            neg_pixels = _find_negative_pixels(jacobian_matrix, threshold, err_tol)
            if not neg_pixels:
                break

            # Assign / grow window sizes for batching decision
            current_neg_set = set(neg_pixels)
            pixel_window_sizes = {
                k: v for k, v in pixel_window_sizes.items()
                if k in current_neg_set
            }
            for px in neg_pixels:
                if px in pixel_window_sizes:
                    if pixel_window_sizes[px] < max_window:
                        pixel_window_sizes[px] += 2
                else:
                    # Adaptive starting size from bounding box
                    bbox_size = neg_jdet_bounding_window(
                        jacobian_matrix, px, threshold, err_tol, min_window_size
                    )
                    pixel_window_sizes[px] = bbox_size
                pixel_window_sizes[px] = min(pixel_window_sizes[px], max_window)

            # Select non-overlapping batch
            batch = _select_non_overlapping(
                neg_pixels, pixel_window_sizes, slice_shape, near_cent_dict
            )

            if len(batch) <= 1:
                # ──────────────────────────────────────────────────────
                # SERIAL PATH — run directly, no process pool overhead
                # ──────────────────────────────────────────────────────
                serial_iters += 1
                neg_idx = neg_pixels[0]

                _log(verbose, 1,
                     f"[iter {iteration:4d}]  serial  "
                     f"fix ({neg_idx[0]:3d},{neg_idx[1]:3d})  "
                     f"neg_pixels={len(neg_pixels)}")

                jacobian_matrix, sub_size, sub_iters = _serial_fix_pixel(
                    neg_idx, phi, phi_init, jacobian_matrix,
                    slice_shape, near_cent_dict, window_counts,
                    min_window_size, max_per_index_iter,
                    max_minimize_iter, max_window,
                    threshold, err_tol, methodName, verbose,
                    error_list, num_neg_jac, min_jdet_list, iter_times,
                )

                cur_neg = int((jacobian_matrix <= 0).sum())
                cur_min = float(jacobian_matrix.min())
                cur_err = error_list[-1] if error_list else 0.0
                _log(verbose, 1,
                     f"         -> neg_jdet {cur_neg:5d}  "
                     f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
                     f"win {sub_size}  sub-iters {sub_iters}")

            else:
                # ──────────────────────────────────────────────────────
                # PARALLEL PATH — submit batch to process pool
                # ──────────────────────────────────────────────────────
                parallel_iters += 1

                # Lazy-create executor on first parallel batch
                if executor is None:
                    executor = ProcessPoolExecutor(max_workers=max_workers)

                batch_sizes = [ws for _, _, ws in batch]
                _log(verbose, 1,
                     f"[iter {iteration:4d}]  parallel  batch={len(batch)}  "
                     f"neg_pixels={len(neg_pixels)}  "
                     f"windows={min(batch_sizes)}-{max(batch_sizes)}")

                futures = {}
                for neg_idx, (cz, cy, cx), sub_size in batch:
                    window_counts[sub_size] += 1

                    phi_init_sub_flat = get_phi_sub_flat(
                        phi_init, cz, cy, cx, slice_shape, sub_size)
                    phi_sub_flat = get_phi_sub_flat(
                        phi, cz, cy, cx, slice_shape, sub_size)

                    is_at_edge, window_reached_max = _edge_flags(
                        cy, cx, sub_size, slice_shape, max_window)

                    fut = executor.submit(
                        _optimize_single_window,
                        phi_sub_flat, phi_init_sub_flat, sub_size,
                        is_at_edge, window_reached_max,
                        threshold, max_minimize_iter, methodName,
                    )
                    futures[fut] = (neg_idx, cz, cy, cx, sub_size)

                batch_time = 0.0
                for fut in as_completed(futures):
                    neg_idx, cz, cy, cx, sub_size = futures[fut]
                    result_x, elapsed = fut.result()
                    batch_time = max(batch_time, elapsed)
                    _apply_result(phi, result_x, cy, cx, sub_size)

                iter_times.append(batch_time)

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

            # Per-step snapshot
            if plot_every and iteration % plot_every == 0:
                from modules.dvfviz import plot_step_snapshot
                cur_neg = int((jacobian_matrix <= 0).sum())
                cur_min = float(jacobian_matrix.min())
                plot_step_snapshot(jacobian_matrix, iteration, cur_neg, cur_min)

            if float(jacobian_matrix.min()) > threshold - err_tol:
                _log(verbose, 1,
                     f"[done] All Jdet > threshold after iter {iteration}")
                break

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _log(verbose, 1, "")
    _log(verbose, 1, "=" * 60)
    _log(verbose, 1, f"  SUMMARY  ({methodName} — hybrid parallel)")
    _log(verbose, 1, "-" * 60)
    _log(verbose, 1, f"  Grid size        : {H} x {W}")
    _log(verbose, 1, f"  Iterations       : {iteration}  "
         f"(serial={serial_iters}, parallel={parallel_iters})")
    _log(verbose, 1, f"  Neg-Jdet  {init_neg:>5d} -> {final_neg:>5d}")
    _log(verbose, 1, f"  Min Jdet  {init_min:+.6f} -> {final_min:+.6f}")
    _log(verbose, 1, f"  L2 error         : {final_err:.6f}")
    _log(verbose, 1, f"  Time             : {elapsed:.2f}s")
    _log(verbose, 1, "=" * 60)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results(
            save_path, method=f"{methodName} (hybrid parallel)",
            threshold=threshold, err_tol=err_tol,
            max_iterations=max_iterations, max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter, min_window_size=min_window_size,
            H=H, W=W, elapsed=elapsed, final_err=final_err,
            init_neg=init_neg, final_neg=final_neg, init_min=init_min,
            final_min=final_min, iteration=iteration, phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
            extra_results=(f"\tSerial iterations: {serial_iters}\n"
                           f"\tParallel iterations: {parallel_iters}"),
        )

    return phi
