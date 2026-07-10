"""Hybrid serial/parallel iterative SLSQP algorithm."""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import label as _scipy_label

from dvfopt._defaults import _adaptive_maxiter, _log, _resolve_params, _unpack_size
from dvfopt.core._internal.metrics import _patch_jacobian_2d, _patch_quality_2d
from dvfopt.core.slsqp.spatial import (
    _edge_flags,
    _select_non_overlapping,
    get_phi_sub_flat_padded,
    neg_jdet_bounding_window,
)
from dvfopt.core.solver import (
    _adaptive_injectivity_loop,
    _apply_result,
    _init_phi,
    _optimize_single_window,
    _print_summary,
    _save_results,
    _serial_fix_pixel,
    _setup_accumulators,
    _update_metrics,
)


def _find_negative_pixels(jacobian_matrix, threshold, err_tol):
    """Return list of (y, x) for all pixels below threshold, worst first."""
    full = jacobian_matrix[0]
    ys, xs = np.where(full <= threshold - err_tol)
    vals = full[ys, xs]
    order = np.argsort(vals)
    return [(int(ys[i]), int(xs[i])) for i in order]


def iterative_parallel(
    deformation_i,
    method_name="SLSQP",
    verbose=1,
    save_path=None,
    plot_every=0,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
    max_workers=None,
    enforce_shoelace=False,
    enforce_injectivity=False,
    injectivity_threshold=None,
    enforce_triangles=False,
    max_doublings=5,
):
    """Hybrid serial/parallel iterative SLSQP correction.

    * batch_size > 1 → ``ProcessPoolExecutor`` (true parallelism)
    * batch_size == 1 → direct in-process with full adaptive window
      growth (avoids Windows ``spawn`` overhead)

    Parameters
    ----------
    max_workers : int or None
        Number of worker processes.  ``None`` → ``os.cpu_count()``.
    enforce_shoelace : bool
        When ``True``, also enforce positive shoelace quad-cell areas
        (geometric fold detection) alongside gradient-based Jacobian.
    enforce_injectivity : bool
        When ``True``, enforce monotonicity of deformed coordinates
        (sufficient condition for global injectivity on structured grids).

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    """
    # NaN-aware entry guard: the outer-loop gate `(quality <= thr).any()`
    # is NaN-blind while `np.argmin` / `np.argsort` are NaN-attracted, so
    # a single NaN pixel would livelock the solver on failed SLSQP calls
    # targeting the NaN window while real folds go untouched.
    if not np.isfinite(deformation_i).all():
        raise ValueError(
            'phi contains non-finite values (NaN/Inf); '
            'iterative_parallel requires a finite field.'
        )

    # Resolve parameters
    p = _resolve_params(
        threshold=threshold,
        err_tol=err_tol,
        max_iterations=max_iterations,
        max_per_index_iter=max_per_index_iter,
        max_minimize_iter=max_minimize_iter,
    )
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]

    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

    # Adaptive outer loop: when enforce_injectivity=True and no explicit
    # threshold is given, double tau until globally injective.
    if enforce_injectivity and injectivity_threshold is None:
        return _adaptive_injectivity_loop(
            deformation_i,
            iterative_parallel,
            verbose,
            max_doublings=max_doublings,
            method_name=method_name,
            save_path=save_path,
            plot_every=plot_every,
            threshold=threshold,
            err_tol=err_tol,
            max_iterations=max_iterations,
            max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            max_workers=max_workers,
            enforce_shoelace=enforce_shoelace,
            enforce_injectivity=enforce_injectivity,
            enforce_triangles=enforce_triangles,
        )

    if max_workers is None:
        max_workers = os.cpu_count()

    error_list, num_neg_jac, iter_times, min_jdet_list, window_counts = _setup_accumulators()

    start_time = time.time()
    phi, phi_init, H, W = _init_phi(deformation_i)
    slice_shape = (1, H, W)
    max_window = (H, W)
    near_cent_dict = {}

    _log(
        verbose,
        1,
        f"[init] Grid {H}x{W}  |  threshold={threshold}  "
        f"|  method={method_name}  |  workers={max_workers}",
    )

    jacobian_matrix, quality_matrix, init_neg, init_min = _update_metrics(
        phi,
        phi_init,
        enforce_shoelace,
        enforce_injectivity,
        num_neg_jac,
        min_jdet_list,
        enforce_triangles=enforce_triangles,
    )

    _log(verbose, 1, f"[init] Neg-Jdet pixels: {init_neg}  |  min Jdet: {init_min:.6f}")

    # Per-pixel window size tracker for parallel batches
    pixel_window_sizes = {}
    pixel_bbox_centers = {}

    iteration = 0
    serial_iters = 0
    parallel_iters = 0
    global_min_window = (3, 3)
    prev_neg = init_neg
    attempted_pixels = set()  # pixels attempted in the last batch
    executor = None  # lazy — only created if we actually need parallelism
    stall_counts = {}  # neg_pixel -> consecutive no-improvement count
    consecutive_improving = 0
    _STALL_THRESHOLD = 3
    _DE_ESCALATE_AFTER = 5

    try:
        while iteration < max_iterations and (quality_matrix[0] <= threshold - err_tol).any():
            iteration += 1

            neg_pixels = _find_negative_pixels(quality_matrix, threshold, err_tol)
            if not neg_pixels:
                break

            # Pre-compute connected-component labels once for all neg pixels
            # this iteration (avoids O(n_neg * H*W) label calls).
            _neg_mask = quality_matrix[0] <= threshold - err_tol
            _labeled_neg, _ = _scipy_label(_neg_mask)

            worst_pixel = neg_pixels[0]  # captured before the step for stall tracking

            # Purge stale stall_counts for pixels no longer below threshold.
            current_neg_set = set(neg_pixels)
            stall_counts = {k: v for k, v in stall_counts.items() if k in current_neg_set}

            # Assign / grow window sizes for batching decision
            new_window_sizes = {}
            new_bbox_centers = {}
            for px in neg_pixels:
                # Always recompute bounding box for current Jacobian state
                bbox_size, bbox_center = neg_jdet_bounding_window(
                    quality_matrix, px, threshold, err_tol, labeled=_labeled_neg
                )
                bsy, bsx = _unpack_size(bbox_size)
                max_sy, max_sx = _unpack_size(max_window)
                bsy = min(max(bsy, 3), max_sy)
                bsx = min(max(bsx, 3), max_sx)
                bbox_size = (bsy, bsx)
                new_bbox_centers[px] = bbox_center

                if px in attempted_pixels and px in pixel_window_sizes:
                    # Was attempted last iteration but still negative → grow
                    old_sy, old_sx = _unpack_size(pixel_window_sizes[px])
                    new_sy = min(max(bsy, old_sy + 2), max_sy)
                    new_sx = min(max(bsx, old_sx + 2), max_sx)
                    new_window_sizes[px] = (new_sy, new_sx)
                else:
                    new_window_sizes[px] = bbox_size

                gsy, gsx = _unpack_size(global_min_window)
                ws_sy, ws_sx = _unpack_size(new_window_sizes[px])
                new_window_sizes[px] = (max(ws_sy, gsy), max(ws_sx, gsx))
            # Merge into persistent dict so skipped pixels retain their
            # grown window sizes across iterations.
            pixel_window_sizes.update(new_window_sizes)
            pixel_bbox_centers.update(new_bbox_centers)
            # Purge entries for pixels no longer negative.
            pixel_window_sizes = {
                k: v for k, v in pixel_window_sizes.items() if k in current_neg_set
            }
            pixel_bbox_centers = {
                k: v for k, v in pixel_bbox_centers.items() if k in current_neg_set
            }

            # Select non-overlapping batch
            batch = _select_non_overlapping(
                neg_pixels,
                pixel_window_sizes,
                slice_shape,
                near_cent_dict,
                pixel_bbox_centers=pixel_bbox_centers,
            )

            # Snapshot bookkeeping — save pre-step state for side-by-side.
            _show_snap = plot_every and iteration % plot_every == 0
            if _show_snap:
                _snap_before = jacobian_matrix.copy()
                _snap_windows = []
                for _, (_, b_cy, b_cx), b_sz in batch:
                    b_edge, _ = _edge_flags(b_cy, b_cx, b_sz, slice_shape, max_window)
                    _snap_windows.append((b_cy, b_cx, b_sz, b_edge))

            if len(batch) <= 1:
                # ──────────────────────────────────────────────────────
                # SERIAL PATH — run directly, no process pool overhead
                # ──────────────────────────────────────────────────────
                serial_iters += 1
                neg_idx = neg_pixels[0]

                _log(
                    verbose,
                    1,
                    f"[iter {iteration:4d}]  serial  "
                    f"fix ({neg_idx[0]:3d},{neg_idx[1]:3d})  "
                    f"neg_pixels={len(neg_pixels)}",
                )

                jacobian_matrix, quality_matrix, sub_size, sub_iters, _final_center = (
                    _serial_fix_pixel(
                        neg_idx,
                        phi,
                        phi_init,
                        jacobian_matrix,
                        slice_shape,
                        near_cent_dict,
                        window_counts,
                        max_per_index_iter,
                        max_minimize_iter,
                        max_window,
                        threshold,
                        err_tol,
                        method_name,
                        verbose,
                        error_list,
                        num_neg_jac,
                        min_jdet_list,
                        iter_times,
                        enforce_shoelace=enforce_shoelace,
                        enforce_injectivity=enforce_injectivity,
                        injectivity_threshold=injectivity_threshold,
                        enforce_triangles=enforce_triangles,
                        min_window=global_min_window,
                        labeled=_labeled_neg,
                        quality_matrix=quality_matrix,
                    )
                )

                cur_neg = int((quality_matrix[0] <= threshold - err_tol).sum())
                cur_min = float(jacobian_matrix.min())
                cur_err = error_list[-1] if error_list else 0.0
                _ssy, _ssx = _unpack_size(sub_size)
                _log(
                    verbose,
                    1,
                    f"         -> neg_jdet {cur_neg:5d}  "
                    f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
                    f"win {_ssy}x{_ssx}  sub-iters {sub_iters}",
                )

                gsy, gsx = _unpack_size(global_min_window)
                if cur_neg >= prev_neg:
                    consecutive_improving = 0
                    stall_counts[neg_idx] = stall_counts.get(neg_idx, 0) + 1
                    if stall_counts[neg_idx] >= _STALL_THRESHOLD and (gsy < H or gsx < W):
                        global_min_window = (min(gsy + 2, H), min(gsx + 2, W))
                        stall_counts[neg_idx] = 0
                        _log(
                            verbose,
                            1,
                            f"  [escalate] serial pixel ({neg_idx[0]},{neg_idx[1]}) "
                            f"stalled {_STALL_THRESHOLD}x, "
                            f"global min window -> "
                            f"{global_min_window[0]}x{global_min_window[1]}",
                        )
                else:
                    stall_counts.pop(neg_idx, None)
                    consecutive_improving += 1
                    if consecutive_improving >= _DE_ESCALATE_AFTER and (gsy > 3 or gsx > 3):
                        global_min_window = (3, 3)
                        consecutive_improving = 0
                        _log(
                            verbose, 1, "  [de-escalate] consistent improvement, min window -> 3x3"
                        )
                prev_neg = cur_neg
                attempted_pixels = {neg_idx}

            else:
                # ──────────────────────────────────────────────────────
                # PARALLEL PATH — submit batch to process pool
                # ──────────────────────────────────────────────────────
                parallel_iters += 1

                # Lazy-create executor on first parallel batch
                if executor is None:
                    executor = ProcessPoolExecutor(max_workers=max_workers)

                batch_sizes = [_unpack_size(ws) for _, _, ws in batch]
                batch_strs = [f"{sy}x{sx}" for sy, sx in batch_sizes]
                _log(
                    verbose,
                    1,
                    f"[iter {iteration:4d}]  parallel  batch={len(batch)}  "
                    f"neg_pixels={len(neg_pixels)}  "
                    f"windows={','.join(batch_strs)}",
                )

                futures = {}
                for neg_idx, (cz, cy, cx), sub_size in batch:
                    window_counts[_unpack_size(sub_size)] += 1

                    # Use padded extraction so the full original window
                    # (including boundary ring) is optimised with correct
                    # central-difference Jacobian context.
                    phi_sub_flat, opt_size = get_phi_sub_flat_padded(
                        phi, cz, cy, cx, slice_shape, sub_size
                    )
                    phi_init_sub_flat, _ = get_phi_sub_flat_padded(
                        phi_init, cz, cy, cx, slice_shape, sub_size
                    )
                    is_padded = opt_size != _unpack_size(sub_size)

                    if is_padded:
                        opt_is_at_edge = False
                        opt_window_reached_max = False
                    else:
                        opt_is_at_edge, opt_window_reached_max = _edge_flags(
                            cy, cx, sub_size, slice_shape, max_window
                        )

                    _opt_sy, _opt_sx = _unpack_size(opt_size)
                    _eff_max_iter = _adaptive_maxiter(2 * _opt_sy * _opt_sx, max_minimize_iter)

                    fut = executor.submit(
                        _optimize_single_window,
                        phi_sub_flat,
                        phi_init_sub_flat,
                        opt_size,
                        opt_is_at_edge,
                        opt_window_reached_max,
                        threshold,
                        _eff_max_iter,
                        method_name,
                        enforce_shoelace,
                        enforce_injectivity,
                        injectivity_threshold,
                        enforce_triangles,
                    )
                    futures[fut] = (neg_idx, cz, cy, cx, sub_size, is_padded, opt_size)

                batch_time = 0.0
                completed = []
                for fut in as_completed(futures):
                    neg_idx, cz, cy, cx, sub_size, is_padded, opt_size = futures[fut]
                    result_x, elapsed, opt_success = fut.result()
                    if not opt_success:
                        _log(
                            verbose,
                            2,
                            f"  [warn] SLSQP did not converge for window "
                            f"at ({cy},{cx}) size {_unpack_size(opt_size)}",
                        )
                    batch_time = max(batch_time, elapsed)
                    completed.append((cy, cx, sub_size, is_padded, opt_size, result_x))

                iter_times.append(batch_time)

                # Apply each window with a compare-and-rollback guard:
                # patch the Jacobian (and quality map) window-locally and
                # reject any result that leaves its window locally
                # *strictly* worse ((n_neg_local, -min_local)
                # lexicographic) instead of applying failed SLSQP
                # results unconditionally.
                use_q = enforce_shoelace or enforce_injectivity or enforce_triangles
                for cy, cx, sub_size, is_padded, opt_size, result_x in completed:
                    w_sy, w_sx = _unpack_size(sub_size)
                    w_hy, w_hx = w_sy // 2, w_sx // 2
                    w_hy_hi, w_hx_hi = w_sy - w_hy, w_sx - w_hx
                    _wy0, _wy1 = max(cy - w_hy - 1, 0), min(cy + w_hy_hi + 1, H)
                    _wx0, _wx1 = max(cx - w_hx - 1, 0), min(cx + w_hx_hi + 1, W)
                    _phi_snap = phi[:, cy - w_hy : cy + w_hy_hi, cx - w_hx : cx + w_hx_hi].copy()
                    _jac_snap = jacobian_matrix[:, _wy0:_wy1, _wx0:_wx1].copy()
                    _qual_snap = (
                        quality_matrix[:, _wy0:_wy1, _wx0:_wx1].copy() if use_q else None
                    )
                    _old_loc = quality_matrix[0, _wy0:_wy1, _wx0:_wx1]
                    _old_n = int((_old_loc <= threshold - err_tol).sum())
                    _old_min = float(_old_loc.min())

                    _apply_result(
                        phi,
                        result_x,
                        cy,
                        cx,
                        opt_size,
                        write_size=sub_size if is_padded else None,
                    )
                    _patch_jacobian_2d(jacobian_matrix, phi, (cy, cx), sub_size)
                    if use_q:
                        _patch_quality_2d(
                            quality_matrix,
                            phi,
                            jacobian_matrix,
                            (cy, cx),
                            sub_size,
                            enforce_shoelace,
                            enforce_injectivity,
                            enforce_triangles=enforce_triangles,
                        )

                    _new_loc = quality_matrix[0, _wy0:_wy1, _wx0:_wx1]
                    _new_n = int((_new_loc <= threshold - err_tol).sum())
                    _new_min = float(_new_loc.min())
                    if _new_n > _old_n or (_new_n == _old_n and _new_min < _old_min):
                        phi[:, cy - w_hy : cy + w_hy_hi, cx - w_hx : cx + w_hx_hi] = _phi_snap
                        jacobian_matrix[:, _wy0:_wy1, _wx0:_wx1] = _jac_snap
                        if use_q:
                            quality_matrix[:, _wy0:_wy1, _wx0:_wx1] = _qual_snap
                        _log(
                            verbose,
                            1,
                            f"  [rollback] window at ({cy},{cx}) locally worse "
                            f"(neg {_old_n}->{_new_n}, "
                            f"min {_old_min:+.4f}->{_new_min:+.4f}) — reverted",
                        )

                jacobian_matrix, quality_matrix, cur_neg, cur_min = _update_metrics(
                    phi,
                    phi_init,
                    enforce_shoelace,
                    enforce_injectivity,
                    num_neg_jac,
                    min_jdet_list,
                    error_list,
                    jacobian_matrix=jacobian_matrix,
                    enforce_triangles=enforce_triangles,
                    quality_matrix=quality_matrix if use_q else None,
                )

                cur_err = error_list[-1]
                _log(
                    verbose,
                    1,
                    f"         -> neg_jdet {cur_neg:5d}  min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}",
                )

                # Escalate global min window only when worst pixel stalls
                # repeatedly; de-escalate after sustained improvement.
                # Only track stall for worst_pixel if it was actually in the batch.
                batch_pixels = {item[0] for item in batch}
                gsy, gsx = _unpack_size(global_min_window)
                if cur_neg >= prev_neg and worst_pixel in batch_pixels:
                    consecutive_improving = 0
                    stall_counts[worst_pixel] = stall_counts.get(worst_pixel, 0) + 1
                    if stall_counts[worst_pixel] >= _STALL_THRESHOLD and (gsy < H or gsx < W):
                        global_min_window = (min(gsy + 2, H), min(gsx + 2, W))
                        stall_counts[worst_pixel] = 0
                        _log(
                            verbose,
                            1,
                            f"  [escalate] parallel pixel "
                            f"({worst_pixel[0]},{worst_pixel[1]}) "
                            f"stalled {_STALL_THRESHOLD}x, "
                            f"global min window -> "
                            f"{global_min_window[0]}x{global_min_window[1]}",
                        )
                elif cur_neg >= prev_neg:
                    consecutive_improving = 0
                else:
                    stall_counts.pop(worst_pixel, None)
                    consecutive_improving += 1
                    if consecutive_improving >= _DE_ESCALATE_AFTER and (gsy > 3 or gsx > 3):
                        global_min_window = (3, 3)
                        consecutive_improving = 0
                        _log(
                            verbose, 1, "  [de-escalate] consistent improvement, min window -> 3x3"
                        )
                prev_neg = cur_neg
                attempted_pixels = batch_pixels

            # Side-by-side before/after snapshot for this iteration.
            if _show_snap:
                from dvfopt.viz.snapshots import plot_step_snapshot

                plot_step_snapshot(
                    jacobian_matrix,
                    iteration,
                    int((jacobian_matrix <= 0).sum()),
                    float(jacobian_matrix.min()),
                    windows=_snap_windows,
                    jacobian_before=_snap_before,
                )

            if float(quality_matrix[0].min()) > threshold - err_tol:
                _log(verbose, 1, f"[done] All Jdet > threshold after iter {iteration}")
                break

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _print_summary(
        verbose,
        f"{method_name} — hybrid parallel",
        (H, W),
        iteration,
        init_neg,
        final_neg,
        init_min,
        final_min,
        final_err,
        elapsed,
        extra_lines=f"(serial={serial_iters}, parallel={parallel_iters})",
    )

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results(
            save_path,
            method=f"{method_name} (hybrid parallel)",
            threshold=threshold,
            err_tol=err_tol,
            max_iterations=max_iterations,
            max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            grid_shape=(H, W),
            elapsed=elapsed,
            final_err=final_err,
            init_neg=init_neg,
            final_neg=final_neg,
            init_min=init_min,
            final_min=final_min,
            iteration=iteration,
            phi=phi,
            error_list=error_list,
            num_neg_jac=num_neg_jac,
            iter_times=iter_times,
            min_jdet_list=min_jdet_list,
            window_counts=window_counts,
            extra_results=(
                f"\tSerial iterations: {serial_iters}\n\tParallel iterations: {parallel_iters}"
            ),
        )

    return phi
