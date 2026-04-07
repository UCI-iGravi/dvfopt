"""Iterative SLSQP algorithm for 3D deformation field correction."""

import time

import numpy as np

from dvfopt._defaults import _log, _resolve_params, _unpack_size_3d
from dvfopt.core.spatial3d import argmin_worst_voxel
from dvfopt.core.solver import _setup_accumulators, _print_summary, _save_results
from dvfopt.core.solver3d import (
    _init_phi_3d,
    _update_metrics_3d,
    _serial_fix_voxel,
)


def iterative_3d(
    deformation,
    methodName="SLSQP",
    verbose=1,
    save_path=None,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
):
    """Iterative SLSQP correction of negative Jacobian determinants in 3D.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``
        Input deformation field with channels ``[dz, dy, dx]``.
    methodName : str
        Optimiser method passed to ``scipy.optimize.minimize``.
    verbose : int
        ``0`` = silent, ``1`` = per-iteration progress, ``2`` = debug.
    save_path : str or None
        Directory to save results.  ``None`` disables saving.
    threshold, err_tol, max_iterations, max_per_index_iter,
    max_minimize_iter :
        Override the corresponding default parameters.

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
    max_window = (D, H, W)

    _log(verbose, 1,
         f"[init] Grid {D}x{H}x{W}  |  threshold={threshold}  "
         f"|  method={methodName}")

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

    while (iteration < max_iterations
           and (jacobian_matrix <= threshold - err_tol).any()):
        iteration += 1

        neg_index = argmin_worst_voxel(jacobian_matrix)

        # Purge stale stall_counts for voxels no longer below threshold.
        stall_counts = {k: v for k, v in stall_counts.items()
                        if jacobian_matrix[k[0], k[1], k[2]] <= threshold - err_tol}

        jacobian_matrix, subvolume_size, per_index_iter, (cz, cy, cx) = \
            _serial_fix_voxel(
                neg_index, phi, phi_init, jacobian_matrix,
                volume_shape, window_counts,
                max_per_index_iter, max_minimize_iter,
                max_window, threshold, err_tol, methodName, verbose,
                error_list, num_neg_jac, min_jdet_list, iter_times,
                min_window=global_min_window,
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
            if stall_counts[neg_index] >= _STALL_THRESHOLD and (
                    gsz < D or gsy < H or gsx < W):
                global_min_window = (min(gsz + 2, D),
                                     min(gsy + 2, H),
                                     min(gsx + 2, W))
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

        if float(jacobian_matrix.min()) > threshold - err_tol:
            _log(verbose, 1,
                 f"[done] All Jdet > threshold after iter {iteration}")
            break

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _print_summary(verbose, f"{methodName} — 3D", (D, H, W), iteration,
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results(
            save_path, method=methodName, threshold=threshold,
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
