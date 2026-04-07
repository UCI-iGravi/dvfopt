"""Serial iterative SLSQP algorithm for 2D deformation field correction."""

import time

import numpy as np

from dvfopt._defaults import _log, _resolve_params, _unpack_size
from scipy.ndimage import label as _scipy_label
from dvfopt.core.spatial import argmin_quality, neg_jdet_bounding_window, get_nearest_center, _edge_flags
from dvfopt.core.solver import (
    _setup_accumulators,
    _print_summary,
    _init_phi,
    _update_metrics,
    _save_results,

    _serial_fix_pixel,
    _adaptive_injectivity_loop,
)
from dvfopt.core.constraints import _quality_map


def iterative_serial(
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
    max_doublings=5,
    debug=None,
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
    max_doublings : int
        Maximum number of tau doublings in the adaptive injectivity loop
        (only used when ``enforce_injectivity=True`` and
        ``injectivity_threshold=None``).  Default 5.

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

    if debug is True:
        from dvfopt.viz.debug import DebugTracer
        debug = DebugTracer()

    # Adaptive outer loop: when enforce_injectivity=True and no explicit
    # threshold is given, double tau until globally injective.
    if enforce_injectivity and injectivity_threshold is None:
        return _adaptive_injectivity_loop(
            deformation_i, iterative_serial, verbose,
            max_doublings=max_doublings,
            methodName=methodName,
            save_path=save_path,
            plot_every=plot_every,
            plot_callback=plot_callback,
            threshold=threshold,
            err_tol=err_tol,
            max_iterations=max_iterations,
            max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            enforce_shoelace=enforce_shoelace,
            enforce_injectivity=enforce_injectivity,
            debug=debug,
        )

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

    if debug is not None:
        debug._on_init(jacobian_matrix, phi, H, W)

    iteration = 0
    prev_neg = init_neg
    global_min_window = (3, 3)
    stall_counts = {}        # neg_index -> consecutive no-improvement count
    consecutive_improving = 0
    _STALL_THRESHOLD = 3     # escalate after this many consecutive stalls on same pixel
    _DE_ESCALATE_AFTER = 5   # de-escalate after this many consecutive improving iterations

    while iteration < max_iterations and (quality_matrix[0] <= threshold - err_tol).any():
        iteration += 1

        neg_index_tuple = argmin_quality(quality_matrix)

        # Purge stall_counts for pixels no longer below threshold so that
        # stale counts don't cause premature escalation if they reappear.
        stall_counts = {k: v for k, v in stall_counts.items()
                        if quality_matrix[0, k[0], k[1]] <= threshold - err_tol}

        # Pre-compute connected-component labels once per outer iteration
        # so neg_jdet_bounding_window doesn't re-run scipy.ndimage.label
        # for every pixel visited inside the inner loop.
        _neg_mask = quality_matrix[0] <= threshold - err_tol
        _labeled_neg, _ = _scipy_label(_neg_mask)

        # Snapshot bookkeeping — capture pre-step state and initial window.
        _show_snap = plot_every and iteration % plot_every == 0
        if _show_snap:
            _snap_before = jacobian_matrix.copy()
            _init_size, _init_center = neg_jdet_bounding_window(
                quality_matrix, neg_index_tuple, threshold, err_tol,
                labeled=_labeled_neg)
            _init_size = (min(_init_size[0], H), min(_init_size[1], W))
            cz0, cy0, cx0 = get_nearest_center(
                _init_center, slice_shape, _init_size, near_cent_dict)
            edge0, _ = _edge_flags(cy0, cx0, _init_size,
                                   slice_shape, max_window)
            _snap_windows = [(cy0, cx0, _init_size, edge0)]

        # Delegate to shared serial inner loop
        if debug is not None:
            debug._on_iter_start(iteration, neg_index_tuple, jacobian_matrix, phi)
        _cb = debug.plot_callback if debug is not None else plot_callback
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
                plot_callback=_cb,
                deformation_i=deformation_i,
                min_window=global_min_window,
                labeled=_labeled_neg,
            )
        if debug is not None:
            debug._on_iter_end(iteration, neg_index_tuple, (cy, cx),
                               jacobian_matrix, phi, submatrix_size, per_index_iter)

        # Escalate minimum window only when the same pixel stalls repeatedly.
        # De-escalate back to 3x3 after sustained global improvement.
        cur_neg = int((quality_matrix[0] <= threshold - err_tol).sum())
        gsy, gsx = global_min_window
        if cur_neg >= prev_neg:
            consecutive_improving = 0
            stall_counts[neg_index_tuple] = stall_counts.get(neg_index_tuple, 0) + 1
            if stall_counts[neg_index_tuple] >= _STALL_THRESHOLD and (gsy < H or gsx < W):
                global_min_window = (min(gsy + 2, H), min(gsx + 2, W))
                stall_counts[neg_index_tuple] = 0
                _log(verbose, 1,
                     f"  [escalate] pixel ({neg_index_tuple[0]},{neg_index_tuple[1]}) "
                     f"stalled {_STALL_THRESHOLD}x, "
                     f"min window -> {global_min_window[0]}x{global_min_window[1]}")
        else:
            stall_counts.pop(neg_index_tuple, None)
            consecutive_improving += 1
            if consecutive_improving >= _DE_ESCALATE_AFTER and (gsy > 3 or gsx > 3):
                global_min_window = (3, 3)
                consecutive_improving = 0
                _log(verbose, 1, "  [de-escalate] consistent improvement, min window -> 3x3")
        prev_neg = cur_neg

        # Side-by-side before/after snapshot for this iteration.
        if _show_snap:
            from dvfopt.viz.snapshots import plot_step_snapshot
            plot_step_snapshot(jacobian_matrix, iteration,
                               int((jacobian_matrix <= 0).sum()),
                               float(jacobian_matrix.min()),
                               windows=_snap_windows,
                               jacobian_before=_snap_before)


        # One-line progress per outer iteration (threshold-consistent with stall detection)
        cur_neg = int((quality_matrix[0] <= threshold - err_tol).sum())
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
