"""Generic Schwarz domain decomposition for the wallbreaker family.

Provides two functions:

* :func:`cluster_schwarz_2d_tri` — 2D, 2-triangle constraint
* :func:`cluster_schwarz_3d_tet` — 3D, 6-tet constraint

Both share the same algorithm shape: detect connected fold components,
crop each with padding, run an *arbitrary* user-supplied
``inner_solve(phi_crop) -> phi_corrected`` callable on the crop, splice
the result back, and (optionally) finish with a global polish.

These functions are what the legacy
``iterative_2d_tri_refine_repair_schwarz`` /
``iterative_3d_tet_refine_repair_schwarz`` wrappers delegate to today —
they are also what the public :class:`SchwarzWrapperStrategy` uses to
schwarz-wrap any compatible inner strategy.

The "inner_solve" callback contract
-----------------------------------

``inner_solve(phi_crop, time_budget_s=None)`` receives a
``(C, *crop_shape)`` array (plus an *optional* per-cluster wall-clock
suggestion in seconds) and returns the corrected field of the same
shape. The callback is responsible for any objective / constraint
setup it needs — Schwarz just hands it crops and splices the results
back. Callbacks that don't care about per-cluster budget apportionment
can ignore the ``time_budget_s`` kwarg; Schwarz only enforces the
*total* wall-clock budget at the outer level.

``final_polish_fn`` (when supplied) has the same shape contract but
takes no ``time_budget_s`` — it's invoked at most once post-sweep on
the full field.

Both callbacks are expected to be robust to "already-feasible" input
since the Schwarz pipeline may re-invoke them on near-feasible regions.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    generate_binary_structure,
)
from scipy.ndimage import (
    label as cc_label,
)

from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

# Minimum wall-clock (seconds) that must remain in the budget for the
# global inner_solve fallback to be worth launching at all. Below this,
# the fallback is skipped and the best-so-far field is returned — a
# too-small budget yields an infeasible best-effort result rather than
# overrunning the requested budget by minutes.
_FALLBACK_MIN_REMAINING_S = 5.0

# ---------------------------------------------------------------------------
# 2D — triangle-area cluster detection
# ---------------------------------------------------------------------------


def _stats_2d(phi: np.ndarray) -> tuple[int, float]:
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    n_neg = int((np.minimum(T1, T2) <= 0).sum())
    min_T = float(min(T1.min(), T2.min()))
    return n_neg, min_T


def _fold_clusters_2d(phi: np.ndarray, merge_dilation: int = 2):
    """Connected components of folded 2D cells, dilated for grouping."""
    if merge_dilation < 0:
        raise ValueError(f'merge_dilation must be >= 0, got {merge_dilation}')
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    fold_mask = np.minimum(T1, T2) <= 0
    if not fold_mask.any():
        return [], fold_mask
    # scipy treats iterations < 1 as "repeat until convergence" (fills
    # the grid), so only dilate for merge_dilation >= 1.
    grouped = (
        binary_dilation(fold_mask, iterations=merge_dilation) if merge_dilation >= 1 else fold_mask
    )
    labels, n_comp = cc_label(grouped, structure=generate_binary_structure(2, 2))
    bboxes = []
    for comp_id in range(1, n_comp + 1):
        comp = (labels == comp_id) & fold_mask
        if not comp.any():
            continue
        cy_idx, cx_idx = np.where(comp)
        bboxes.append(
            dict(
                comp_id=int(comp_id),
                cy0=int(cy_idx.min()),
                cy1=int(cy_idx.max()),
                cx0=int(cx_idx.min()),
                cx1=int(cx_idx.max()),
                n_folds=int(comp.sum()),
            )
        )
    return bboxes, fold_mask


def cluster_schwarz_2d_tri(
    phi_in: np.ndarray,
    inner_solve: Callable[..., np.ndarray],
    *,
    threshold: float,
    pad: int = 4,
    merge_dilation: int = 2,
    max_outer_iters: int = 3,
    fallback_size_ratio: float = 0.7,
    time_budget_s: float = 600.0,
    final_polish_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    verbose: int = 1,
    record_history: bool = False,
    step_callback=None,
):
    """Generic Schwarz domain decomposition for 2-triangle 2D fields.

    Detects connected fold components, runs ``inner_solve`` on each
    padded crop, splices results back, optionally polishes globally
    once the sweep is done. Falls back to running ``inner_solve`` on
    the whole field when a single cluster dominates or when outer
    rounds stall.

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)`` or ``(3, 1, H, W)``
        Input deformation field. Coerced to ``(2, H, W)`` internally.
    inner_solve : callable
        ``(phi_crop) -> phi_corrected`` of matching shape. Receives the
        cropped sub-window plus padding; should return a corrected
        sub-window. Robust to already-feasible input.
    threshold : float
        Lower bound for both triangle areas — used for the
        already-feasible early-out check and the final-polish trigger.
    pad : int
        Cells of context around each cluster's bounding box.
    merge_dilation : int
        Dilation applied to the fold mask before CCL (merges
        near-touching clusters).
    max_outer_iters : int
        Outer-loop budget if splicing introduces new folds at crop
        boundaries.
    fallback_size_ratio : float
        Single cluster covering > this fraction of either axis →
        fall back to ``inner_solve`` on the whole field.
    time_budget_s : float
        Total wall-clock budget for the sweep + polish. Checked at the
        top of every outer round, before each cluster, and before the
        global fallback — the fallback only receives the REMAINING
        budget (never more), and is skipped entirely when fewer than
        ~5 s remain. Consequence: a budget too small for the work
        returns the best-so-far field (possibly still infeasible,
        i.e. ``feasible=False`` in spirit) instead of overrunning the
        requested budget several-fold.
    final_polish_fn : callable, optional
        ``(phi) -> phi`` run once post-sweep if ``min_T < threshold + 1e-5``
        or ``n_neg > 0``. ``None`` skips polishing.
    verbose : int
    record_history : bool
        If True, returns ``(phi, info)`` instead of just ``phi``.

    Returns
    -------
    phi_out : ndarray, shape ``(2, H, W)``.
    info : dict, only when ``record_history=True``.
    """
    # Coerce input shape.
    if phi_in.ndim == 4:
        if phi_in.shape[0] == 3:
            phi_in = np.stack([phi_in[1, 0], phi_in[2, 0]])
        else:
            phi_in = phi_in[:, 0]
    if phi_in.dtype != np.float64:
        phi_in = phi_in.astype(np.float64)

    H, W = phi_in.shape[1], phi_in.shape[2]
    phi_out = phi_in.copy()
    t0 = time.time()

    def _fire(stage: str, phi):
        """Forward an intermediate phi snapshot to ``step_callback`` so
        the live-viz GUI can scrub through each cluster splice. Buggy
        callbacks are silenced; KeyboardInterrupt propagates as the
        documented stop signal."""
        if step_callback is None:
            return
        try:
            step_callback({'phi': np.asarray(phi).copy(), 'stage': stage})
        except KeyboardInterrupt:
            raise
        except Exception:
            pass

    init_n_neg, init_min_T = _stats_2d(phi_in)
    history: dict[str, Any] = {
        'cluster_runs': [],
        'outer_rounds': [],
        'fallback_to_global': False,
        'final_polish_fired': False,
        'init': dict(n_neg=init_n_neg, min_T=init_min_T),
    }

    if init_n_neg == 0:
        if final_polish_fn is not None and init_min_T < threshold + 1e-5:
            phi_out = final_polish_fn(phi_out)
            history['final_polish_fired'] = True
        final_n_neg, final_min_T = _stats_2d(phi_out)
        history['final'] = dict(n_neg=final_n_neg, min_T=final_min_T, wall=time.time() - t0)
        history['reason'] = 'already-feasible'
        return (phi_out, history) if record_history else phi_out

    prev_n_neg = init_n_neg
    no_progress_rounds = 0

    for outer in range(max_outer_iters):
        # Top-of-loop budget check (mirrors the 3D variant): never start
        # a new outer round past the requested wall-clock budget.
        if time.time() - t0 > time_budget_s:
            if verbose:
                print('  time budget exhausted; stopping outer loop', flush=True)
            break
        bboxes, fold_mask = _fold_clusters_2d(phi_out, merge_dilation=merge_dilation)
        if not bboxes:
            break
        n_neg = int(fold_mask.sum())
        if verbose:
            print(
                f'[outer {outer}] {n_neg} folds in {len(bboxes)} '
                f'clusters (elapsed {time.time() - t0:.1f}s)',
                flush=True,
            )

        if len(bboxes) == 1:
            b = bboxes[0]
            span_y_cells = b['cy1'] - b['cy0'] + 1
            span_x_cells = b['cx1'] - b['cx0'] + 1
            n_cells_y = H - 1
            n_cells_x = W - 1
            if (
                span_y_cells >= fallback_size_ratio * n_cells_y
                or span_x_cells >= fallback_size_ratio * n_cells_x
            ):
                # Grant the fallback only what REMAINS of the budget —
                # never a fresh floor on top of an exhausted one — and
                # skip it entirely when too little remains to be useful.
                fallback_budget = time_budget_s - (time.time() - t0)
                if fallback_budget <= _FALLBACK_MIN_REMAINING_S:
                    if verbose:
                        print(
                            '  budget exhausted; skipping global fallback (returning best-so-far)',
                            flush=True,
                        )
                    break
                if verbose:
                    print(
                        f'  single cluster spans {span_y_cells}x'
                        f'{span_x_cells} cells of {n_cells_y}x{n_cells_x}'
                        f' (HxW={H}x{W}): falling back to inner_solve(global)',
                        flush=True,
                    )
                history['fallback_to_global'] = True
                phi_out = inner_solve(phi_out, time_budget_s=fallback_budget)
                break

        round_runs = []
        for b in bboxes:
            if time.time() - t0 > time_budget_s:
                if verbose:
                    print('  budget exhausted; stopping cluster sweep', flush=True)
                break
            y0 = max(0, b['cy0'] - pad)
            y1 = min(H, b['cy1'] + pad + 2)
            x0 = max(0, b['cx0'] - pad)
            x1 = min(W, b['cx1'] + pad + 2)
            crop_h = y1 - y0
            crop_w = x1 - x0
            if crop_h < 4 or crop_w < 4:
                continue

            phi_win = phi_out[:, y0:y1, x0:x1].copy()
            n_before, min_before = _stats_2d(phi_win)
            t_cluster = time.time()
            cluster_budget = max(20.0, (time_budget_s - (time.time() - t0)) / max(1, len(bboxes)))
            try:
                phi_win_out = inner_solve(phi_win, time_budget_s=cluster_budget)
            except Exception as exc:
                if verbose:
                    print(
                        f'  cluster {b["comp_id"]} FAILED: {type(exc).__name__}: {exc}',
                        flush=True,
                    )
                continue
            wall = time.time() - t_cluster
            n_after, min_after = _stats_2d(phi_win_out)
            phi_out[:, y0:y1, x0:x1] = phi_win_out
            _fire(f'cluster_{b["comp_id"]}_outer{outer}', phi_out)

            round_runs.append(
                dict(
                    outer=outer,
                    comp_id=b['comp_id'],
                    crop=(crop_h, crop_w),
                    bbox=(y0, y1, x0, x1),
                    n_before=n_before,
                    min_before=min_before,
                    n_after=n_after,
                    min_after=min_after,
                    wall=wall,
                )
            )
            if verbose:
                print(
                    f'  cluster {b["comp_id"]} crop=({crop_h}x{crop_w}) '
                    f'@({y0},{x0})  n_neg: {n_before} -> {n_after}  '
                    f'min_T: {min_before:+.3f} -> {min_after:+.4f}  '
                    f'({wall:.1f}s)',
                    flush=True,
                )
        history['cluster_runs'].extend(round_runs)

        post_n_neg, post_min = _stats_2d(phi_out)
        history['outer_rounds'].append(
            dict(
                outer=outer,
                before_n_neg=n_neg,
                after_n_neg=post_n_neg,
                min_T=post_min,
                wall=time.time() - t0,
            )
        )
        if verbose:
            print(
                f'  round {outer} done: global n_neg {n_neg} -> '
                f'{post_n_neg}, min_T {post_min:+.4f}',
                flush=True,
            )

        if post_n_neg == 0:
            break
        if post_n_neg >= prev_n_neg:
            no_progress_rounds += 1
            if no_progress_rounds >= 2:
                # Remaining budget only; skip when effectively exhausted
                # (returns the best-so-far field rather than blowing the
                # requested budget on a from-scratch global solve).
                fallback_budget = time_budget_s - (time.time() - t0)
                if fallback_budget <= _FALLBACK_MIN_REMAINING_S:
                    if verbose:
                        print(
                            '  budget exhausted; skipping global fallback (returning best-so-far)',
                            flush=True,
                        )
                    break
                if verbose:
                    print(
                        '  no progress for 2 rounds -> falling back to inner_solve(global)',
                        flush=True,
                    )
                history['fallback_to_global'] = True
                phi_out = inner_solve(phi_out, time_budget_s=fallback_budget)
                break
        else:
            no_progress_rounds = 0
        prev_n_neg = post_n_neg

    post_n_neg, post_min = _stats_2d(phi_out)
    if (
        final_polish_fn is not None
        and (post_n_neg > 0 or post_min < threshold + 1e-5)
        and time.time() - t0 < time_budget_s
    ):
        if verbose:
            print(
                f'[final polish] min_T={post_min:+.4f} < threshold; running polish',
                flush=True,
            )
        t_polish = time.time()
        phi_out = final_polish_fn(phi_out)
        history['final_polish_fired'] = True
        history['final_polish_wall'] = time.time() - t_polish
        _fire('final_polish', phi_out)
        if verbose:
            final_n, final_m = _stats_2d(phi_out)
            print(
                f'  polish done: min_T {post_min:+.4f} -> {final_m:+.4f}  '
                f'n_neg {post_n_neg} -> {final_n}  '
                f'({time.time() - t_polish:.1f}s)',
                flush=True,
            )

    final_n, final_min = _stats_2d(phi_out)
    history['final'] = dict(n_neg=final_n, min_T=final_min, wall=time.time() - t0)
    return (phi_out, history) if record_history else phi_out


# ---------------------------------------------------------------------------
# 3D — 6-tet volume cluster detection
# ---------------------------------------------------------------------------


def _stats_3d(phi: np.ndarray) -> tuple[int, float]:
    # Fused per-cube min kernel — avoids materialising the full
    # (6, Dc, Hc, Wc) volume array. NOTE: ``n_neg`` therefore counts
    # folded CUBES (cells whose worst tet is <= 0), not folded tets —
    # matching the per-cell semantics of :func:`_stats_2d`. All internal
    # consumers only compare n_neg relatively (== 0, >= previous round),
    # so the change is behaviour-preserving for control flow.
    min_V = six_tet_min_volume_3d(phi)
    return int((min_V <= 0).sum()), float(min_V.min())


def _fold_clusters_3d(phi: np.ndarray, threshold: float, merge_dilation: int = 2):
    """Connected components of folded 3D voxel cells, dilated for grouping."""
    if merge_dilation < 0:
        raise ValueError(f'merge_dilation must be >= 0, got {merge_dilation}')
    fold_cells = six_tet_min_volume_3d(phi) < threshold
    if not fold_cells.any():
        return [], fold_cells
    # scipy treats iterations < 1 as "repeat until convergence" (fills
    # the grid), so only dilate for merge_dilation >= 1.
    grouped = (
        binary_dilation(fold_cells, iterations=merge_dilation)
        if merge_dilation >= 1
        else fold_cells
    )
    labels, n_comp = cc_label(grouped, structure=generate_binary_structure(3, 3))
    bboxes = []
    for comp_id in range(1, n_comp + 1):
        comp = (labels == comp_id) & fold_cells
        if not comp.any():
            continue
        cz, cy, cx = np.where(comp)
        bboxes.append(
            dict(
                comp_id=int(comp_id),
                cz0=int(cz.min()),
                cz1=int(cz.max()),
                cy0=int(cy.min()),
                cy1=int(cy.max()),
                cx0=int(cx.min()),
                cx1=int(cx.max()),
                n_folds=int(comp.sum()),
            )
        )
    return bboxes, fold_cells


def cluster_schwarz_3d_tet(
    phi_in: np.ndarray,
    inner_solve: Callable[..., np.ndarray],
    *,
    threshold: float,
    pad: int = 4,
    merge_dilation: int = 2,
    max_outer_iters: int = 3,
    fallback_size_ratio: float = 0.7,
    time_budget_s: float = 600.0,
    final_polish_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    verbose: int = 1,
    record_history: bool = False,
):
    """Generic Schwarz domain decomposition for 6-tet 3D fields.

    See :func:`cluster_schwarz_2d_tri` — same algorithm, 3D analog.
    Uses 6-tet volumes for fold detection and 26-connectivity CCL.

    Parameters
    ----------
    phi_in : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
    inner_solve, threshold, pad, merge_dilation, max_outer_iters,
    fallback_size_ratio, time_budget_s, final_polish_fn, verbose,
    record_history : see :func:`cluster_schwarz_2d_tri`.

    Returns
    -------
    phi_out : ndarray, shape ``(3, D, H, W)``.
    info : dict, only when ``record_history=True``.
    """
    phi_in = np.asarray(phi_in, dtype=np.float64)
    if phi_in.ndim != 4 or phi_in.shape[0] != 3:
        raise ValueError(f'expected (3, D, H, W) input; got shape {phi_in.shape}')
    _, D, H, W = phi_in.shape
    phi_out = phi_in.copy()
    t0 = time.time()

    init_n_neg, init_min = _stats_3d(phi_in)
    history: dict[str, Any] = {
        'init': dict(n_neg=init_n_neg, min_T=init_min),
        'cluster_runs': [],
        'outer_rounds': [],
        'fallback_to_global': False,
        'final_polish_fired': False,
    }

    if init_n_neg == 0:
        if final_polish_fn is not None and init_min < threshold + 1e-5:
            phi_out = final_polish_fn(phi_out)
            history['final_polish_fired'] = True
        final_n_neg, final_min = _stats_3d(phi_out)
        history['final'] = dict(n_neg=final_n_neg, min_T=final_min, mode='no-folds')
        return (phi_out, history) if record_history else phi_out

    last_n_neg = init_n_neg
    last_round_no_progress = 0

    for outer_round in range(max_outer_iters):
        if time.time() - t0 > time_budget_s:
            break
        bboxes, _fold_cells = _fold_clusters_3d(phi_out, threshold, merge_dilation)
        if not bboxes:
            break

        Dc, Hc, Wc = D - 1, H - 1, W - 1
        triggered_fallback = False
        for b in bboxes:
            span_z = (b['cz1'] - b['cz0'] + 1) / max(1, Dc)
            span_y = (b['cy1'] - b['cy0'] + 1) / max(1, Hc)
            span_x = (b['cx1'] - b['cx0'] + 1) / max(1, Wc)
            if max(span_z, span_y, span_x) > fallback_size_ratio:
                # Remaining budget only (see the 2D variant): never grant
                # a fresh floor past exhaustion; skip when ~nothing left.
                fallback_budget = time_budget_s - (time.time() - t0)
                if fallback_budget <= _FALLBACK_MIN_REMAINING_S:
                    if verbose:
                        print(
                            '  [schwarz] budget exhausted; skipping global '
                            'fallback (returning best-so-far)',
                            flush=True,
                        )
                    triggered_fallback = True
                    break
                history['fallback_to_global'] = True
                if verbose:
                    print(
                        f'  [schwarz outer {outer_round}] cluster {b["comp_id"]} '
                        f'spans {max(span_z, span_y, span_x):.2f} > '
                        f'{fallback_size_ratio} — falling back to inner_solve(global)',
                        flush=True,
                    )
                phi_out = inner_solve(phi_out, time_budget_s=fallback_budget)
                triggered_fallback = True
                break
        if triggered_fallback:
            break

        if verbose:
            print(
                f'  [schwarz outer {outer_round}] {len(bboxes)} cluster(s); n_neg={last_n_neg}',
                flush=True,
            )
        for b in bboxes:
            z0 = max(0, b['cz0'] - pad)
            z1 = min(D - 1, b['cz1'] + 1 + pad)
            y0 = max(0, b['cy0'] - pad)
            y1 = min(H - 1, b['cy1'] + 1 + pad)
            x0 = max(0, b['cx0'] - pad)
            x1 = min(W - 1, b['cx1'] + 1 + pad)
            if z1 - z0 < 2 or y1 - y0 < 2 or x1 - x0 < 2:
                continue
            crop = phi_out[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1].copy()
            cluster_budget = max(
                30.0,
                (time_budget_s - (time.time() - t0)) / max(1, len(bboxes)),
            )
            try:
                crop_out = inner_solve(crop, time_budget_s=cluster_budget)
            except Exception as exc:
                if verbose:
                    print(
                        f'  cluster {b["comp_id"]} FAILED: {type(exc).__name__}: {exc}',
                        flush=True,
                    )
                continue
            phi_out[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = crop_out
            history['cluster_runs'].append(
                dict(
                    outer=outer_round,
                    comp_id=b['comp_id'],
                    bbox=(z0, z1, y0, y1, x0, x1),
                    n_folds=b['n_folds'],
                )
            )

        cur_n_neg, cur_min = _stats_3d(phi_out)
        history['outer_rounds'].append(
            dict(outer=outer_round, n_neg=cur_n_neg, min_T=cur_min, wall=time.time() - t0)
        )
        if verbose:
            print(
                f'  [schwarz outer {outer_round}] done: n_neg={cur_n_neg}  '
                f'min_V={cur_min:+.5f}  ({time.time() - t0:.1f}s)',
                flush=True,
            )
        if cur_n_neg == 0:
            break
        if cur_n_neg >= last_n_neg:
            last_round_no_progress += 1
            if last_round_no_progress >= 2:
                fallback_budget = time_budget_s - (time.time() - t0)
                if fallback_budget <= _FALLBACK_MIN_REMAINING_S:
                    if verbose:
                        print(
                            '  [schwarz] budget exhausted; skipping global '
                            'fallback (returning best-so-far)',
                            flush=True,
                        )
                    break
                history['fallback_to_global'] = True
                if verbose:
                    print(
                        '  [schwarz] no progress for 2 rounds — falling back to inner_solve(global)',
                        flush=True,
                    )
                phi_out = inner_solve(phi_out, time_budget_s=fallback_budget)
                break
        else:
            last_round_no_progress = 0
        last_n_neg = cur_n_neg

    post_n_neg, post_min = _stats_3d(phi_out)
    if (
        final_polish_fn is not None
        and (post_n_neg > 0 or post_min < threshold + 1e-5)
        and time.time() - t0 < time_budget_s
    ):
        if verbose:
            print(
                f'[final polish] min_V={post_min:+.5f}; running global polish',
                flush=True,
            )
        t_polish = time.time()
        phi_out = final_polish_fn(phi_out)
        history['final_polish_fired'] = True
        history['final_polish_wall'] = time.time() - t_polish

    final_n_neg, final_min = _stats_3d(phi_out)
    history['final'] = dict(n_neg=final_n_neg, min_T=final_min, wall=time.time() - t0)
    return (phi_out, history) if record_history else phi_out


__all__ = [
    'cluster_schwarz_2d_tri',
    'cluster_schwarz_3d_tet',
]
