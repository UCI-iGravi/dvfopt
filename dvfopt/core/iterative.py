"""Serial iterative SLSQP algorithm for 2D deformation field correction."""

import time

import numpy as np

from dvfopt._defaults import _log, _resolve_params, _unpack_size
from dvfopt.core.spatial import argmin_quality, neg_jdet_bounding_window, get_nearest_center, _edge_flags
from dvfopt.core.solver import (
    _setup_accumulators,
    _print_summary,
    _init_phi,
    _update_metrics,
    _save_results,
    _full_grid_step,
    _serial_fix_pixel,
)
from dvfopt.core.constraints import _quality_map


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
    enforce_shoelace=False,
    enforce_injectivity=False,
    injectivity_threshold=None,
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
    max_minimize_iter :
        Override the corresponding default parameters.
    enforce_shoelace : bool
        When ``True``, the optimiser also enforces positive shoelace
        quad-cell areas (geometric fold detection) in addition to the
        gradient-based Jacobian determinant.  Convergence and pixel
        selection use both metrics.
    enforce_injectivity : bool
        When ``True``, the optimiser enforces monotonicity of deformed
        coordinates (horizontal, vertical, and anti-diagonal) — together
        a sufficient condition for each quad cell to be convex and thus
        non-self-intersecting.
    injectivity_threshold : float or None
        Lower bound for the injectivity constraint.  Defaults to the same
        value as *threshold* when ``None``.  Increasing this (e.g. ``0.3``)
        forces greater vertex separation in deformed space, preventing
        distant cells from overlapping under large shear.  Recommended
        range: ``0.05`` – ``0.3`` when ``enforce_injectivity=True``.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
        Corrected displacement field ``[dy, dx]``.
    """
    p = _resolve_params(threshold=threshold, err_tol=err_tol,
                        max_iterations=max_iterations,
                        max_per_index_iter=max_per_index_iter,
                        max_minimize_iter=max_minimize_iter)
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]

    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

    error_list, num_neg_jac, iter_times, min_jdet_list, window_counts = _setup_accumulators()

    start_time = time.time()
    phi, phi_init, H, W = _init_phi(deformation_i)
    slice_shape = (1, H, W)
    max_window = (H, W)
    near_cent_dict = {}

    _log(verbose, 1, f"[init] Grid {H}x{W}  |  threshold={threshold}  |  method={methodName}")
    _log(verbose, 2, f"[init] deformation_i shape: {deformation_i.shape}, phi shape: {phi.shape}")

    # Initial Jacobian
    jacobian_matrix, quality_matrix, init_neg, init_min = _update_metrics(
        phi, phi_init, enforce_shoelace, enforce_injectivity,
        num_neg_jac, min_jdet_list)

    _log(verbose, 1, f"[init] Neg-Jdet pixels: {init_neg}  |  min Jdet: {init_min:.6f}")

    iteration = 0
    prev_neg = init_neg
    global_min_window = (3, 3)

    while iteration < max_iterations and (quality_matrix[0] <= threshold - err_tol).any():
        iteration += 1

        neg_index_tuple = argmin_quality(quality_matrix)

        # Snapshot bookkeeping — capture pre-step state and initial window.
        _show_snap = plot_every and iteration % plot_every == 0
        if _show_snap:
            _snap_before = jacobian_matrix.copy()
            _init_size, _init_center = neg_jdet_bounding_window(
                quality_matrix, neg_index_tuple, threshold, err_tol)
            _init_size = (min(_init_size[0], H), min(_init_size[1], W))
            cz0, cy0, cx0 = get_nearest_center(
                _init_center, slice_shape, _init_size, near_cent_dict)
            edge0, _ = _edge_flags(cy0, cx0, _init_size,
                                   slice_shape, max_window)
            _snap_windows = [(cy0, cx0, _init_size, edge0)]

        # Delegate to shared serial inner loop
        jacobian_matrix, quality_matrix, submatrix_size, per_index_iter, (cy, cx) = \
            _serial_fix_pixel(
                neg_index_tuple, phi, phi_init, jacobian_matrix,
                slice_shape, near_cent_dict, window_counts,
                max_per_index_iter, max_minimize_iter,
                max_window, threshold, err_tol, methodName, verbose,
                error_list, num_neg_jac, min_jdet_list, iter_times,
                enforce_shoelace=enforce_shoelace,
                enforce_injectivity=enforce_injectivity,
                injectivity_threshold=injectivity_threshold,
                plot_callback=plot_callback,
                deformation_i=deformation_i,
                min_window=global_min_window,
            )

        # Escalate minimum window when no progress (avoids oscillation
        # between neighbouring pixels with overlapping windows).
        cur_neg = int((jacobian_matrix <= 0).sum())
        gsy, gsx = global_min_window
        if cur_neg >= prev_neg and (gsy < H or gsx < W):
            global_min_window = (min(gsy + 2, H), min(gsx + 2, W))
            _log(verbose, 1,
                 f"  [escalate] no improvement ({prev_neg}->{cur_neg}), "
                 f"min window -> {global_min_window[0]}x{global_min_window[1]}")
        prev_neg = cur_neg

        # Side-by-side before/after snapshot for this iteration.
        if _show_snap:
            from dvfopt.viz.snapshots import plot_step_snapshot
            plot_step_snapshot(jacobian_matrix, iteration,
                               int((jacobian_matrix <= 0).sum()),
                               float(jacobian_matrix.min()),
                               windows=_snap_windows,
                               jacobian_before=_snap_before)

        # Full-grid fallback for non-square grids where the window
        # can't cover the entire grid.
        _sub_sy, _sub_sx = _unpack_size(submatrix_size)
        if (_sub_sy >= H and _sub_sx >= W
                and H != W
                and (quality_matrix[0] <= threshold - err_tol).any()):
            iter_start = time.time()
            _full_grid_step(phi, phi_init, H, W, threshold,
                            max_minimize_iter, methodName, verbose,
                            enforce_shoelace, enforce_injectivity,
                            injectivity_threshold=injectivity_threshold)
            iter_times.append(time.time() - iter_start)

            jacobian_matrix, quality_matrix, cur_neg, cur_min = _update_metrics(
                phi, phi_init, enforce_shoelace, enforce_injectivity,
                num_neg_jac, min_jdet_list, error_list)

        # One-line progress per outer iteration
        cur_neg = int((jacobian_matrix <= 0).sum())
        cur_min = float(jacobian_matrix.min())
        cur_err = error_list[-1] if error_list else 0.0
        _sy, _sx = _unpack_size(submatrix_size)
        _log(verbose, 1,
             f"[iter {iteration:4d}]  fix ({neg_index_tuple[0]:3d},{neg_index_tuple[1]:3d})  "
             f"win {_sy}x{_sx}  neg_jdet {cur_neg:5d}  "
             f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
             f"sub-iters {per_index_iter}")

        if float(quality_matrix[0].min()) > threshold - err_tol:
            _log(verbose, 1, f"[done] All Jdet > threshold after iter {iteration}")
            break

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _print_summary(verbose, methodName, (H, W), iteration,
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results(
            save_path, method=methodName, threshold=threshold, err_tol=err_tol,
            max_iterations=max_iterations, max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            grid_shape=(H, W), elapsed=elapsed, final_err=final_err,
            init_neg=init_neg, final_neg=final_neg, init_min=init_min,
            final_min=final_min, iteration=iteration, phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi
