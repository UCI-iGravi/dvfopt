"""Iterative SLSQP algorithm for 3D deformation field correction."""

import time

import numpy as np
from scipy.ndimage import label

from dvfopt._defaults import _log, _resolve_params, _unpack_size_3d
from dvfopt.core.slsqp.spatial3d import argmin_worst_voxel
from dvfopt.core.solver import _setup_accumulators, _print_summary, _save_results
from dvfopt.core.solver3d import (
    _init_phi_3d,
    _update_metrics_3d,
    _serial_fix_voxel,
)


def iterative_3d(
    deformation,
    method_name="SLSQP",
    verbose=1,
    save_path=None,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
    max_window=None,
    max_window_voxels=None,
    max_window_voxels_ceiling=None,
    voxel_cap_growth=2.0,
    voxel_cap_stall_threshold=5,
):
    """Iterative SLSQP correction of negative Jacobian determinants in 3D.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``
        Input deformation field with channels ``[dz, dy, dx]``.
    method_name : str
        Optimiser method passed to ``scipy.optimize.minimize``.
    verbose : int
        ``0`` = silent, ``1`` = per-iteration progress, ``2`` = debug.
    save_path : str or None
        Directory to save results.  ``None`` disables saving.
    threshold, err_tol, max_iterations, max_per_index_iter,
    max_minimize_iter :
        Override the corresponding default parameters.
    max_window : tuple of int or None
        Upper bound ``(sz, sy, sx)`` on sub-volume size used by the
        serial inner loop. ``None`` (default) uses the full grid
        ``(D, H, W)``. Set to a small tuple like ``(3, 3, 3)`` to force
        the solver to always work on tiny SLSQP sub-problems — useful
        on pathological fields where large initial windows cause SLSQP
        to hang inside its Fortran active-set QP.
    max_window_voxels : int or None
        Upper bound on the *total* voxel count ``sz * sy * sx`` of every
        sub-volume. When the connected-component bounding box would
        exceed this budget, it is shrunk aspect-preserving until the
        product fits. Captures SLSQP's actual cost cliff (QP work scales
        with voxel count, not per-axis dim) and admits elongated shapes
        a per-axis ``max_window`` would reject. Combine with ``max_window``
        to cap both axes and volume simultaneously. ``None`` (default) =
        unlimited.
    max_window_voxels_ceiling : int or None
        Hard upper bound on how far the voxel cap may be raised by the
        livelock-escalation heuristic. ``None`` (default) disables
        escalation entirely — useful when you want a strict, reproducible
        budget. When set (e.g. ``2000``), the solver starts at
        ``max_window_voxels`` and bumps the effective cap by
        ``voxel_cap_growth`` whenever ``neg_count`` has failed to decrease
        for ``voxel_cap_stall_threshold`` consecutive outer iterations,
        up to this ceiling.
    voxel_cap_growth : float
        Multiplier applied to the current voxel cap each time escalation
        fires (default ``2.0``).
    voxel_cap_stall_threshold : int
        How many consecutive non-improving outer iterations trigger a
        voxel-cap bump (default ``5``).

    Returns
    -------
    phi : ndarray, shape ``(3, D, H, W)``
        Corrected displacement field ``[dz, dy, dx]``.
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
    phi, phi_init, D, H, W = _init_phi_3d(deformation)
    volume_shape = (D, H, W)
    if max_window is None:
        max_window = (D, H, W)
    else:
        max_window = (min(int(max_window[0]), D),
                      min(int(max_window[1]), H),
                      min(int(max_window[2]), W))

    _log(verbose, 1,
         f"[init] Grid {D}x{H}x{W}  |  threshold={threshold}  "
         f"|  method={method_name}")

    jacobian_matrix, init_neg, init_min = _update_metrics_3d(
        phi, phi_init, num_neg_jac, min_jdet_list)

    _log(verbose, 1,
         f"[init] Neg-Jdet voxels: {init_neg}  |  min Jdet: {init_min:.6f}")

    iteration = 0
    prev_neg = init_neg
    global_min_window = (3, 3, 3)
    stall_counts = {}
    consecutive_improving = 0
    _STALL_THRESHOLD = 3
    _DE_ESCALATE_AFTER = 5

    # Voxel-cap escalation state. Tracks iters since neg-count hit a new
    # best; counting consecutive non-improvements would miss the typical
    # SLSQP livelock where neg oscillates (e.g. 12 -> 13 -> 12) without
    # either direction being sustained "progress".
    cur_voxel_cap = max_window_voxels
    best_neg_seen = init_neg
    iters_since_best = 0
    _escalate_cap = (max_window_voxels is not None
                     and max_window_voxels_ceiling is not None
                     and max_window_voxels_ceiling > max_window_voxels)

    while (iteration < max_iterations
           and (jacobian_matrix <= threshold - err_tol).any()):
        iteration += 1

        neg_index = argmin_worst_voxel(jacobian_matrix)

        # Compute connected-component labels for the current negative mask.
        # Used by the bounding-window and component-aware freeze mask.
        neg_mask = jacobian_matrix <= threshold - err_tol
        structure = np.ones((3, 3, 3))  # 26-connectivity
        labeled_array, _ = label(neg_mask, structure=structure)

        # Purge stale stall_counts for voxels no longer below threshold.
        stall_counts = {k: v for k, v in stall_counts.items()
                        if jacobian_matrix[k[0], k[1], k[2]] <= threshold - err_tol}

        jacobian_matrix, subvolume_size, per_index_iter, (cz, cy, cx) = \
            _serial_fix_voxel(
                neg_index, phi, phi_init, jacobian_matrix,
                volume_shape, window_counts,
                max_per_index_iter, max_minimize_iter,
                max_window, threshold, err_tol, method_name, verbose,
                error_list, num_neg_jac, min_jdet_list, iter_times,
                min_window=global_min_window,
                labeled_array=labeled_array,
                max_window_voxels=cur_voxel_cap,
            )

        sz, sy, sx = _unpack_size_3d(subvolume_size)
        cur_neg = int((jacobian_matrix <= threshold - err_tol).sum())
        cur_min = float(jacobian_matrix.min())
        cur_err = error_list[-1] if error_list else 0.0
        _log(verbose, 1,
             f"[iter {iteration:4d}]  fix ({neg_index[0]:3d},"
             f"{neg_index[1]:3d},{neg_index[2]:3d})  "
             f"win {sz}x{sy}x{sx}  neg_jdet {cur_neg:5d}  "
             f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
             f"sub-iters {per_index_iter}")

        # Per-voxel stall detection and de-escalation (same threshold as loop condition)
        neg_count = cur_neg
        gsz, gsy, gsx = global_min_window
        if neg_count >= prev_neg:
            consecutive_improving = 0
            stall_counts[neg_index] = stall_counts.get(neg_index, 0) + 1
            mwz, mwy, mwx = max_window
            if stall_counts[neg_index] >= _STALL_THRESHOLD and (
                    gsz < mwz or gsy < mwy or gsx < mwx):
                global_min_window = (min(gsz + 2, mwz),
                                     min(gsy + 2, mwy),
                                     min(gsx + 2, mwx))
                stall_counts[neg_index] = 0
                _log(verbose, 1,
                     f"  [escalate] voxel ({neg_index[0]},{neg_index[1]},"
                     f"{neg_index[2]}) stalled {_STALL_THRESHOLD}x, "
                     f"min window -> {global_min_window[0]}x"
                     f"{global_min_window[1]}x{global_min_window[2]}")
        else:
            stall_counts.pop(neg_index, None)
            consecutive_improving += 1
            if consecutive_improving >= _DE_ESCALATE_AFTER and (
                    gsz > 3 or gsy > 3 or gsx > 3):
                global_min_window = (3, 3, 3)
                consecutive_improving = 0
                _log(verbose, 1,
                     "  [de-escalate] consistent improvement, min window -> 3x3x3")
        prev_neg = neg_count

        # Voxel-cap escalation: bump the cap when we fail to reach a new
        # best neg-count for `voxel_cap_stall_threshold` iters in a row.
        # "New best" catches oscillation around a plateau that simple
        # "strictly decreased" would misread as alternating progress.
        if neg_count < best_neg_seen:
            best_neg_seen = neg_count
            iters_since_best = 0
        else:
            iters_since_best += 1
            if (_escalate_cap
                    and iters_since_best >= voxel_cap_stall_threshold
                    and cur_voxel_cap < max_window_voxels_ceiling):
                new_cap = min(int(cur_voxel_cap * voxel_cap_growth),
                              max_window_voxels_ceiling)
                if new_cap > cur_voxel_cap:
                    cur_voxel_cap = new_cap
                    iters_since_best = 0
                    _log(verbose, 1,
                         f"  [escalate-vox] no new best for "
                         f"{voxel_cap_stall_threshold} iters, "
                         f"voxel cap -> {cur_voxel_cap}")

        if float(jacobian_matrix.min()) > threshold - err_tol:
            _log(verbose, 1,
                 f"[done] All Jdet > threshold after iter {iteration}")
            break

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _print_summary(verbose, f"{method_name} — 3D", (D, H, W), iteration,
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results(
            save_path, method=method_name, threshold=threshold,
            err_tol=err_tol, max_iterations=max_iterations,
            max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            grid_shape=(D, H, W), elapsed=elapsed, final_err=final_err,
            init_neg=init_neg, final_neg=final_neg, init_min=init_min,
            final_min=final_min, iteration=iteration, phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi
