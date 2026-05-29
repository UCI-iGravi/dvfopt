"""m14-Schwarz prototype: cluster-localized refinement of a folded DVF.

Premise
-------
Global m14 processes every corner of the field even when only ~5% of
cells are folded. On a 320×456 slice with sparse fold clusters this is
mostly wasted compute — most of the field is already feasible and
already at the anchor.

Cluster-localized variant
-------------------------
1. Detect fold clusters via connected components of ``min(T1, T2) <= 0``
   (dilated for grouping).
2. For each cluster, extract its bounding box with ``pad`` cells of
   surrounding context.
3. Run the global :func:`iterative_2d_tri_refine_repair` on each
   crop independently.
4. Splice each crop's result back into the global field.
5. If any new folds appear at crop boundaries (Schwarz artifact),
   repeat for up to ``max_outer_iters`` rounds.

Fallback to global m14 when:
* Cluster covers >70% of the field (no point in cropping).
* Outer-loop fails to reduce ``n_neg`` over two consecutive rounds.

This is an experimental prototype — NOT in the public ``dvfopt`` surface.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.ndimage import (
    label as cc_label, binary_dilation, find_objects,
    generate_binary_structure,
)

from dvfopt.core.wallbreakers import iterative_2d_tri_refine_repair
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _stats(phi: np.ndarray) -> Tuple[int, float]:
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    n_neg = int((np.minimum(T1, T2) <= 0).sum())
    min_T = float(min(T1.min(), T2.min()))
    return n_neg, min_T


def _fold_clusters(phi: np.ndarray, merge_dilation: int = 2):
    """Connected components of folded cells, dilated by ``merge_dilation``
    to merge clusters that are within that many cells of each other."""
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    fold_mask = np.minimum(T1, T2) <= 0
    if not fold_mask.any():
        return [], fold_mask
    grouped = binary_dilation(fold_mask, iterations=merge_dilation)
    labels, n_comp = cc_label(
        grouped, structure=generate_binary_structure(2, 2))
    bboxes = []
    for comp_id in range(1, n_comp + 1):
        comp = (labels == comp_id) & fold_mask
        if not comp.any():
            continue
        cy_idx, cx_idx = np.where(comp)
        bboxes.append(dict(
            comp_id=comp_id,
            cy0=int(cy_idx.min()),
            cy1=int(cy_idx.max()),
            cx0=int(cx_idx.min()),
            cx1=int(cx_idx.max()),
            n_folds=int(comp.sum()),
        ))
    return bboxes, fold_mask


def m14_schwarz(
    phi_in: np.ndarray,
    *,
    threshold: float = 0.01,
    margin: float = 1e-3,
    anchor: str = 'l2',
    pad: int = 4,
    merge_dilation: int = 2,
    max_outer_iters: int = 3,
    fallback_size_ratio: float = 0.7,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    record_history: bool = False,
    **m14_kwargs,
):
    """Cluster-localized m14.

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)``
        Input field with channels ``[dy, dx]``.
    threshold, margin, anchor
        Forwarded to :func:`iterative_2d_tri_refine_repair`.
    pad : int
        Cells of context around each cluster's bounding box.
    merge_dilation : int
        Dilation applied to the fold mask before connected-components
        labeling (merges near-touching clusters).
    max_outer_iters : int
        If splicing introduces new folds at crop boundaries, retry up to
        this many times.
    fallback_size_ratio : float
        If a single cluster's bounding box covers more than this fraction
        of either axis, fall back to global m14.
    time_budget_s : float
        Total wall-clock budget (shared across clusters + retries).
    verbose : int
    record_history : bool
        If True, returns ``(phi, info)``.
    **m14_kwargs : forwarded to m14 per-cluster.

    Returns
    -------
    phi_out : ndarray, shape ``(2, H, W)``
    info : dict (only if ``record_history=True``)
    """
    if phi_in.dtype != np.float64:
        phi_in = phi_in.astype(np.float64)
    H, W = phi_in.shape[1], phi_in.shape[2]
    phi_out = phi_in.copy()
    t0 = time.time()
    init_n_neg, init_min_T = _stats(phi_in)
    history = {
        'cluster_runs': [],
        'outer_rounds': [],
        'fallback_to_global': False,
        'init': dict(n_neg=init_n_neg, min_T=init_min_T),
    }

    if init_n_neg == 0:
        history['reason'] = 'already-feasible'
        return (phi_out, history) if record_history else phi_out

    prev_n_neg = init_n_neg
    no_progress_rounds = 0

    for outer in range(max_outer_iters):
        bboxes, fold_mask = _fold_clusters(phi_out, merge_dilation=merge_dilation)
        if not bboxes:
            break

        n_neg = int(fold_mask.sum())
        if verbose:
            print(f'[outer {outer}] {n_neg} folds in {len(bboxes)} clusters '
                  f'(elapsed {time.time()-t0:.1f}s)', flush=True)

        # Fallback: a single cluster spans most of the field.
        if len(bboxes) == 1:
            b = bboxes[0]
            span_y = b['cy1'] - b['cy0']
            span_x = b['cx1'] - b['cx0']
            if (span_y >= fallback_size_ratio * H
                    or span_x >= fallback_size_ratio * W):
                if verbose:
                    print(f'  single cluster spans {span_y}×{span_x} of '
                          f'{H}×{W}: falling back to global m14', flush=True)
                history['fallback_to_global'] = True
                remaining = max(60.0, time_budget_s - (time.time() - t0))
                phi_out = iterative_2d_tri_refine_repair(
                    phi_out, threshold=threshold, margin=margin,
                    anchor=anchor, time_budget_s=remaining,
                    verbose=0, **m14_kwargs)
                final_n, final_min = _stats(phi_out)
                history['final'] = dict(n_neg=final_n, min_T=final_min,
                                        wall=time.time() - t0)
                return (phi_out, history) if record_history else phi_out

        round_runs = []
        for b in bboxes:
            if time.time() - t0 > time_budget_s:
                if verbose:
                    print('  budget exhausted; stopping', flush=True)
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
            n_before, min_before = _stats(phi_win)
            t_cluster = time.time()
            cluster_budget = max(20.0, (time_budget_s - (time.time() - t0))
                                  / max(1, len(bboxes)))
            try:
                phi_win_out = iterative_2d_tri_refine_repair(
                    phi_win, threshold=threshold, margin=margin,
                    anchor=anchor, time_budget_s=cluster_budget,
                    verbose=0, **m14_kwargs)
            except Exception as exc:
                if verbose:
                    print(f'  cluster {b["comp_id"]} FAILED: '
                          f'{type(exc).__name__}: {exc}', flush=True)
                continue
            wall = time.time() - t_cluster
            n_after, min_after = _stats(phi_win_out)
            phi_out[:, y0:y1, x0:x1] = phi_win_out

            round_runs.append(dict(
                outer=outer, comp_id=b['comp_id'],
                crop=(crop_h, crop_w),
                bbox=(y0, y1, x0, x1),
                n_before=n_before, min_before=min_before,
                n_after=n_after, min_after=min_after,
                wall=wall,
            ))
            if verbose:
                print(f'  cluster {b["comp_id"]} crop=({crop_h}x{crop_w}) '
                      f'@({y0},{x0})  n_neg: {n_before} -> {n_after}  '
                      f'min_T: {min_before:+.3f} -> {min_after:+.4f}  '
                      f'({wall:.1f}s)', flush=True)

        history['cluster_runs'].extend(round_runs)

        # Re-check global state. Splicing may have introduced new folds at
        # crop boundaries that interfere with already-feasible regions.
        post_n_neg, post_min = _stats(phi_out)
        history['outer_rounds'].append(dict(
            outer=outer, before_n_neg=n_neg, after_n_neg=post_n_neg,
            min_T=post_min, wall=time.time() - t0,
        ))
        if verbose:
            print(f'  round done: global n_neg {n_neg} -> {post_n_neg}, '
                  f'min_T {post_min:+.4f}', flush=True)

        if post_n_neg == 0:
            break
        if post_n_neg >= prev_n_neg:
            no_progress_rounds += 1
            if no_progress_rounds >= 2:
                if verbose:
                    print('  no progress for 2 rounds → falling back to '
                          'global m14', flush=True)
                history['fallback_to_global'] = True
                remaining = max(60.0, time_budget_s - (time.time() - t0))
                phi_out = iterative_2d_tri_refine_repair(
                    phi_out, threshold=threshold, margin=margin,
                    anchor=anchor, time_budget_s=remaining,
                    verbose=0, **m14_kwargs)
                break
        else:
            no_progress_rounds = 0
        prev_n_neg = post_n_neg

    final_n, final_min = _stats(phi_out)
    history['final'] = dict(n_neg=final_n, min_T=final_min,
                            wall=time.time() - t0)
    return (phi_out, history) if record_history else phi_out
