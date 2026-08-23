"""2D-solver coordinator: the serial per-pixel fix loop + adaptive outer.

The lower-level building blocks have moved into focused submodules:

* :mod:`dvfopt.core.slsqp_windowed._io`       — setup helpers, summary print, save_results, init_phi
* :mod:`dvfopt.core.slsqp_windowed._metrics`  — `_update_metrics`, `_patch_jacobian_2d`
* :mod:`dvfopt.core.slsqp_windowed._window`   — `_full_grid_step`, `_optimize_single_window`, `_apply_result`

They are re-exported from this module so existing imports
``from dvfopt.core.slsqp_windowed.coordinator import ...`` keep working.
"""

import numpy as np

from dvfopt._defaults import _adaptive_maxiter, _log, _unpack_size

# Re-exported building blocks. The actual implementations live in
# :mod:`dvfopt.core.slsqp_windowed._io`, ``._metrics``, and ``._window``;
# this module continues to surface them for back-compat with existing
# import sites (callers should migrate to those submodules directly).
from dvfopt.core.slsqp_windowed._io import (
    _init_phi,
    _print_summary,
    _save_results,
    _setup_accumulators,
)
from dvfopt.core.slsqp_windowed._metrics import _patch_jacobian_2d, _update_metrics
from dvfopt.core.slsqp_windowed._window import (
    _apply_result,
    _full_grid_step,
    _optimize_single_window,
)
from dvfopt.core.slsqp_windowed.constraints import _quality_map
from dvfopt.core.slsqp_windowed.spatial import (
    _edge_flags,
    _frozen_edges_clean,
    get_nearest_center,
    get_phi_sub_flat_padded,
    neg_jdet_bounding_window,
)

__all__ = [
    "_adaptive_injectivity_loop",
    "_apply_result",
    # per-window
    "_full_grid_step",
    "_init_phi",
    "_optimize_single_window",
    "_patch_jacobian_2d",
    "_print_summary",
    "_save_results",
    # coordinators (defined below)
    "_serial_fix_pixel",
    # io
    "_setup_accumulators",
    # metrics
    "_update_metrics",
]


def _serial_fix_pixel(
    neg_index_tuple,
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
    enforce_shoelace=False,
    enforce_injectivity=False,
    injectivity_threshold=None,
    enforce_triangles=False,
    plot_callback=None,
    step_callback=None,
    outer_iter=None,
    deformation_i=None,
    min_window=(3, 3),
    labeled=None,
    quality_matrix=None,
    objective=None,
):
    """Fix a single pixel using the serial adaptive-window inner loop.

    Start from the bounding-box-derived window, then grow by 2 each
    sub-iteration until the local region is clean or the window hits the
    grid boundary.

    Mutates *phi* and the accumulator lists in-place.

    Parameters
    ----------
    quality_matrix : ndarray or None
        The caller's current quality map (must be consistent with *phi*
        and *jacobian_matrix*).  When ``None`` it is (re)computed here —
        a full-grid ``_quality_map`` pass when any ``enforce_*`` flag is
        set.  Callers that already track the quality map should pass it
        to avoid that recomputation.

    Returns
    -------
    jacobian_matrix, quality_matrix, submatrix_size, per_index_iter, (cy, cx)
    """
    _use_quality = enforce_shoelace or enforce_injectivity or enforce_triangles
    if quality_matrix is None:
        quality_matrix = (
            _quality_map(
                phi,
                enforce_shoelace,
                enforce_injectivity,
                enforce_triangles=enforce_triangles,
                jacobian_matrix=jacobian_matrix,
            )
            if _use_quality
            else jacobian_matrix
        )

    # Adaptive starting size from negative-Jdet bounding box
    submatrix_size, bbox_center = neg_jdet_bounding_window(
        quality_matrix, neg_index_tuple, threshold, err_tol, labeled=labeled
    )
    max_sy, max_sx = _unpack_size(max_window)
    min_sy, min_sx = _unpack_size(min_window)
    submatrix_size = (
        max(min(submatrix_size[0], max_sy), min_sy),
        max(min(submatrix_size[1], max_sx), min_sx),
    )

    per_index_iter = 0
    window_reached_max = False
    # Check bounds used in while condition from iteration 2+.
    # Expanded by 1px when padded because phi[cy-hy] is freely optimised,
    # making J[cy-hy-1] subject to change (patched by _patch_jacobian_2d).
    _check_y0 = _check_y1 = _check_x0 = _check_x1 = 0  # placeholders, short-circuited on first eval

    while per_index_iter == 0 or (
        per_index_iter < max_per_index_iter
        and (
            quality_matrix[0, _check_y0:_check_y1, _check_x0:_check_x1] < threshold - err_tol
        ).any()
    ):
        cz, cy, cx = get_nearest_center(bbox_center, slice_shape, submatrix_size, near_cent_dict)
        sy, sx = _unpack_size(submatrix_size)
        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx

        # Try padded extraction: (sy+2)x(sx+2) so the full original window
        # (including its boundary ring) is optimised and its Jacobian is
        # constrained with proper central-difference context.
        phi_sub_flat, opt_size = get_phi_sub_flat_padded(
            phi, cz, cy, cx, slice_shape, submatrix_size
        )
        phi_init_sub_flat, _ = get_phi_sub_flat_padded(
            phi_init, cz, cy, cx, slice_shape, submatrix_size
        )
        is_padded = opt_size != (sy, sx)

        # Update check region for the NEXT while-condition evaluation.
        # Must be done before any `continue` so the bounds are always current.
        # Clamp to actual grid bounds (H, W), not max_window, so the check
        # remains correct if max_window is ever set smaller than the grid.
        _H, _W = slice_shape[1], slice_shape[2]
        _pad = 1 if is_padded else 0
        _check_y0 = max(cy - hy - _pad, 0)
        _check_y1 = min(cy + hy_hi + _pad, _H)
        _check_x0 = max(cx - hx - _pad, 0)
        _check_x1 = min(cx + hx_hi + _pad, _W)

        is_at_edge, w_max = _edge_flags(cy, cx, submatrix_size, slice_shape, max_window)
        window_reached_max = window_reached_max or w_max

        # When padded, the frozen boundary is the outer ring of the padded
        # window (1px outside the original); override edge flags accordingly.
        opt_is_at_edge = False if is_padded else is_at_edge
        opt_window_reached_max = False if is_padded else window_reached_max

        _log(
            verbose,
            2,
            f"  [edge] at_edge={is_at_edge}  window_reached_max={window_reached_max}  padded={is_padded}",
        )

        # Skip optimizer if frozen edges have negative Jdet (likely infeasible).
        # Does NOT consume per_index_iter budget — only actual optimizer calls do.
        # For padded windows check the padded outer ring (opt_size); for
        # unpacked windows check the original boundary ring (submatrix_size).
        check_size = opt_size if is_padded else submatrix_size
        if (
            not opt_is_at_edge
            and not opt_window_reached_max
            and not _frozen_edges_clean(quality_matrix, cy, cx, check_size, threshold, err_tol)
        ):
            _log(verbose, 2, f"  [skip] Frozen edges have neg Jdet at win {sy}x{sx} — growing")
            sy, sx = _unpack_size(submatrix_size)
            if sy < max_sy or sx < max_sx:
                submatrix_size = (min(sy + 2, max_sy), min(sx + 2, max_sx))
            continue

        # Frozen edges are clean (or not applicable): run the optimiser.
        per_index_iter += 1
        window_counts[_unpack_size(submatrix_size)] += 1

        if per_index_iter > 1:
            _log(
                verbose,
                2,
                f"  [window] Index {neg_index_tuple}: window grew to {sy}x{sx} (opt-iter {per_index_iter})",
            )

        _opt_sy, _opt_sx = _unpack_size(opt_size)
        _eff_max_iter = _adaptive_maxiter(2 * _opt_sy * _opt_sx, max_minimize_iter)

        # Run optimisation directly — no process pool
        result_x, elapsed, opt_success = _optimize_single_window(
            phi_sub_flat,
            phi_init_sub_flat,
            opt_size,
            opt_is_at_edge,
            opt_window_reached_max,
            threshold,
            _eff_max_iter,
            method_name,
            enforce_shoelace=enforce_shoelace,
            enforce_injectivity=enforce_injectivity,
            injectivity_threshold=injectivity_threshold,
            enforce_triangles=enforce_triangles,
            objective=objective,
        )
        iter_times.append(elapsed)
        if not opt_success:
            _log(
                verbose,
                1,
                f"  [warn] SLSQP did not converge at win {sy}x{sx} (sub-iter {per_index_iter})",
            )

        # Compare-and-rollback guard: snapshot the write region and the
        # local metric patch region (window + 1px border — exactly what
        # _patch_jacobian_2d / _patch_quality_2d rewrite) so a
        # failed/worse SLSQP result is rejected instead of applied
        # unconditionally.
        _wy0, _wy1 = max(cy - hy - 1, 0), min(cy + hy_hi + 1, _H)
        _wx0, _wx1 = max(cx - hx - 1, 0), min(cx + hx_hi + 1, _W)
        _phi_snap = phi[:, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi].copy()
        _jac_snap = jacobian_matrix[:, _wy0:_wy1, _wx0:_wx1].copy()
        _qual_snap = quality_matrix[:, _wy0:_wy1, _wx0:_wx1].copy() if _use_quality else None
        _old_loc = quality_matrix[0, _wy0:_wy1, _wx0:_wx1]
        _old_n = int((_old_loc <= threshold - err_tol).sum())
        _old_min = float(_old_loc.min())

        _apply_result(
            phi, result_x, cy, cx, opt_size, write_size=submatrix_size if is_padded else None
        )

        jacobian_matrix, quality_matrix, _cur_neg, _cur_min = _update_metrics(
            phi,
            phi_init,
            enforce_shoelace,
            enforce_injectivity,
            num_neg_jac,
            min_jdet_list,
            error_list,
            jacobian_matrix=jacobian_matrix,
            patch_center=(cy, cx),
            patch_size=submatrix_size,
            enforce_triangles=enforce_triangles,
            quality_matrix=quality_matrix if _use_quality else None,
        )

        # Roll back if the sub-solve made the window locally *strictly*
        # worse ((n_neg_local, -min_local) lexicographic).
        _new_loc = quality_matrix[0, _wy0:_wy1, _wx0:_wx1]
        _new_n = int((_new_loc <= threshold - err_tol).sum())
        _new_min = float(_new_loc.min())
        if _new_n > _old_n or (_new_n == _old_n and _new_min < _old_min):
            phi[:, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi] = _phi_snap
            jacobian_matrix[:, _wy0:_wy1, _wx0:_wx1] = _jac_snap
            if _use_quality:
                quality_matrix[:, _wy0:_wy1, _wx0:_wx1] = _qual_snap
            _cur_neg = int((jacobian_matrix <= 0).sum())
            _cur_min = float(jacobian_matrix.min())
            num_neg_jac[-1] = _cur_neg
            min_jdet_list[-1] = _cur_min
            if error_list:
                error_list[-1] = float(np.sqrt(np.sum((phi - phi_init) ** 2)))
            _log(
                verbose,
                1,
                f"  [rollback] sub-solve left window locally worse "
                f"(neg {_old_n}->{_new_n}, min {_old_min:+.4f}->{_new_min:+.4f}) — reverted",
            )

        _log(
            verbose,
            2,
            f"  [sub-Jdet] centre ({cy},{cx}) window {sy}x{sx}:\n"
            + np.array2string(
                jacobian_matrix[0, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi],
                precision=4,
                suppress_small=True,
            ),
        )

        if plot_callback is not None:
            plot_callback(deformation_i, phi)
        if step_callback is not None:
            # Rich state hook for live-visualization tools. The callback
            # receives a snapshot of the per-pixel inner-loop state; the
            # consumer is responsible for any copying it needs (most
            # arrays here are mutated in subsequent iterations).
            step_callback(
                {
                    'phi': phi,
                    'phi_init': phi_init,
                    'jacobian': jacobian_matrix,
                    'quality': quality_matrix,
                    'neg_index': neg_index_tuple,
                    'window_center': (cy, cx),
                    'window_size': submatrix_size,
                    'opt_size': opt_size,
                    'is_padded': is_padded,
                    'per_index_iter': per_index_iter,
                    'outer_iter': outer_iter,
                    'window_reached_max': window_reached_max,
                    'n_neg': int(_cur_neg) if _cur_neg is not None else -1,
                    'min_T': float(_cur_min) if _cur_min is not None else float('nan'),
                }
            )

        if float(quality_matrix[0].min()) > threshold - err_tol:
            break

        # Grow window for next sub-iteration
        sy, sx = _unpack_size(submatrix_size)
        if sy < max_sy or sx < max_sx:
            submatrix_size = (min(sy + 2, max_sy), min(sx + 2, max_sx))
        else:
            window_reached_max = True

    return jacobian_matrix, quality_matrix, submatrix_size, per_index_iter, (cy, cx)


def _adaptive_injectivity_loop(deformation_i, correct_fn, verbose, max_doublings=5, **kwargs):
    """Run *correct_fn* with doubling ``injectivity_threshold`` until globally injective.

    Called automatically when ``enforce_injectivity=True`` and
    ``injectivity_threshold=None``.  Each pass reruns the full correction
    from the **original** ``deformation_i`` (so the L2 objective always
    measures displacement from the original field).

    Parameters
    ----------
    correct_fn : callable
        One of ``iterative_serial`` or ``iterative_parallel``.
        Must accept ``injectivity_threshold=<float>`` as a keyword.
    verbose : int
        Outer-loop verbosity.  The inner correction runs silently (``verbose=0``).
    max_doublings : int
        Maximum number of times to double ``tau`` before giving up.
        Default 5 covers 0.05 → 0.10 → 0.20 → 0.40 → 0.80 → 1.60.
    **kwargs
        All other arguments forwarded to *correct_fn* (threshold, err_tol,
        max_iterations, enforce_shoelace, enforce_injectivity, …).
        ``injectivity_threshold`` must NOT be present — it is managed here.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    """
    from dvfopt.jacobian.intersection import has_quad_self_intersections

    tau = 0.05

    for attempt in range(max_doublings + 1):
        _log(
            verbose,
            1,
            f"[adaptive-injectivity] attempt {attempt + 1}/{max_doublings + 1}  "
            f"injectivity_threshold={tau:.4f}  max_doublings={max_doublings}",
        )

        phi = correct_fn(
            deformation_i.copy(),
            verbose=0,
            injectivity_threshold=tau,
            **kwargs,
        )

        if not has_quad_self_intersections(phi):
            _log(verbose, 1, f"[adaptive-injectivity] globally injective at tau={tau:.4f}")
            return phi

        _log(verbose, 1, "[adaptive-injectivity] intersections remain — doubling tau")
        tau *= 2.0

    _log(verbose, 1, "[adaptive-injectivity] max doublings reached; returning best result")
    return phi
