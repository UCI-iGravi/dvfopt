"""Cluster-localized m14 (refine-repair) — Schwarz domain decomposition.

Premise
-------
:func:`iterative_2d_tri_refine_repair` (m14) processes every corner of
the field even when only a small fraction of cells are folded. On
slices with sparse fold clusters this means most of the per-iteration
work is on already-feasible already-anchored cells. The remedy is
classical: domain decomposition.

Algorithm
---------
1. Detect connected fold components of ``min(T1, T2) <= 0``, dilated by
   ``merge_dilation`` to group near-touching clusters.
2. For each component, extract a bounding-box crop with ``pad`` cells of
   surrounding context.
3. Run global m14 (:func:`iterative_2d_tri_refine_repair`) on each crop
   independently and splice the result back into the global field.
4. If new folds appear at crop boundaries (Schwarz overlap artifact),
   repeat for up to ``max_outer_iters`` rounds.
5. **Final global polish.** Even when every cluster's m14 lands strictly
   feasible on its own crop, splicing later clusters can nick the
   boundary corners of earlier ones — the composite ``min_T`` may end
   up below ``threshold + margin`` by a small amount. A short global
   log-barrier polish (using :func:`iterative_2d_tri_barrier`)
   walks the iterate strictly back into the interior. This step is
   skipped if the cluster sweep already produced ``min_T >= threshold +
   err_tol``.

Fallback to global m14:

* a single cluster spans more than ``fallback_size_ratio`` of either
  axis (no point cropping),
* the outer loop fails to reduce ``n_neg`` for two consecutive rounds.

Where it helps
--------------
The wall-clock advantage is biggest when fold clusters cover a small
fraction of the field — e.g., on the **full B0039 z=12 slice
(320×456)** the m14-Schwarz prototype reached feasibility ~5× faster
than global m14 with ~11% lower L1. On dense single-cluster crops the
overhead dominates and the wrapper effectively falls back to global
m14.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.ndimage import (
    label as cc_label, binary_dilation, generate_binary_structure,
)

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.core.iterative2d_tri_barrier import iterative_2d_tri_barrier
from dvfopt.core.wallbreakers._refine_repair import (
    iterative_2d_tri_refine_repair,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _stats(phi: np.ndarray) -> Tuple[int, float]:
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    n_neg = int((np.minimum(T1, T2) <= 0).sum())
    min_T = float(min(T1.min(), T2.min()))
    return n_neg, min_T


def _fold_clusters(phi: np.ndarray, merge_dilation: int = 2):
    """Connected components of folded cells, dilated for grouping."""
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
            comp_id=int(comp_id),
            cy0=int(cy_idx.min()),
            cy1=int(cy_idx.max()),
            cx0=int(cx_idx.min()),
            cx1=int(cx_idx.max()),
            n_folds=int(comp.sum()),
        ))
    return bboxes, fold_mask


def iterative_2d_tri_refine_repair_schwarz(
    phi_in: np.ndarray,
    *,
    threshold: float = None,
    margin: float = 1e-3,
    anchor: str = 'l2',
    eps_l1: float = 1e-4,
    pad: int = 4,
    merge_dilation: int = 2,
    max_outer_iters: int = 3,
    fallback_size_ratio: float = 0.7,
    time_budget_s: float = 600.0,
    final_polish: bool = True,
    final_polish_max_iter: int = 200,
    verbose: int = 1,
    record_history: bool = False,
    **m14_kwargs,
):
    """Cluster-localized refine-repair (m14-Schwarz).

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)`` or ``(3, 1, H, W)``
        Input deformation field with channels ``[dy, dx]``.
    threshold : float, optional
        Lower bound for both triangle areas. Defaults to
        ``DEFAULT_PARAMS['threshold']``.
    margin, anchor, eps_l1
        Forwarded to :func:`iterative_2d_tri_refine_repair`.
    pad : int
        Cells of context around each cluster's bounding box.
    merge_dilation : int
        Dilation applied to the fold mask before connected-components
        labeling (merges near-touching clusters into one crop).
    max_outer_iters : int
        Outer-loop budget if splicing introduces new folds at crop
        boundaries.
    fallback_size_ratio : float
        If a single cluster's bounding box covers more than this
        fraction of either axis, fall back immediately to global m14.
    time_budget_s : float
        Total wall-clock budget for the cluster sweep + final polish.
    final_polish : bool
        If True (default), runs a global log-barrier polish at the end
        when ``min_T < threshold + err_tol``. Fast — the field is
        already nearly feasible — and recovers the safety margin global
        m14 produces.
    final_polish_max_iter : int
        ``max_minimize_iter`` for the global polish. The default ``200``
        is plenty since we're already near the optimum.
    verbose : int
    record_history : bool
        If True, returns ``(phi, info)``; ``info`` records per-cluster
        stats, outer-round summaries, fallback flag, and final stats.
    **m14_kwargs
        Forwarded to per-cluster :func:`iterative_2d_tri_refine_repair`
        calls. Use to override e.g. ``polish_mu`` or ``lam_schedule``
        per-cluster.

    Returns
    -------
    phi_out : ndarray, shape ``(2, H, W)`` — channels ``[dy, dx]``.
    info : dict, only if ``record_history=True``.
    """
    # Coerce input shape.
    if phi_in.ndim == 4:
        if phi_in.shape[0] == 3:
            phi_in = np.stack([phi_in[1, 0], phi_in[2, 0]])
        else:
            phi_in = phi_in[:, 0]
    if phi_in.dtype != np.float64:
        phi_in = phi_in.astype(np.float64)

    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    H, W = phi_in.shape[1], phi_in.shape[2]
    phi_out = phi_in.copy()
    t0 = time.time()

    init_n_neg, init_min_T = _stats(phi_in)
    history = {
        'cluster_runs': [],
        'outer_rounds': [],
        'fallback_to_global': False,
        'final_polish_fired': False,
        'init': dict(n_neg=init_n_neg, min_T=init_min_T),
    }

    if init_n_neg == 0:
        # Already feasible — but still polish if min_T < threshold + err_tol.
        if final_polish and init_min_T < threshold + 1e-5:
            phi_out = iterative_2d_tri_barrier(
                phi_out, threshold=threshold, margin=margin,
                max_minimize_iter=final_polish_max_iter,
                anchor=anchor, eps_l1=eps_l1, verbose=0)
            history['final_polish_fired'] = True
        final_n_neg, final_min_T = _stats(phi_out)
        history['final'] = dict(n_neg=final_n_neg, min_T=final_min_T,
                                wall=time.time() - t0)
        history['reason'] = 'already-feasible'
        return (phi_out, history) if record_history else phi_out

    prev_n_neg = init_n_neg
    no_progress_rounds = 0

    for outer in range(max_outer_iters):
        bboxes, fold_mask = _fold_clusters(
            phi_out, merge_dilation=merge_dilation)
        if not bboxes:
            break
        n_neg = int(fold_mask.sum())
        if verbose:
            print(f'[outer {outer}] {n_neg} folds in {len(bboxes)} '
                  f'clusters (elapsed {time.time()-t0:.1f}s)', flush=True)

        # Fallback: single cluster spanning most of the field.
        if len(bboxes) == 1:
            b = bboxes[0]
            span_y = b['cy1'] - b['cy0']
            span_x = b['cx1'] - b['cx0']
            if (span_y >= fallback_size_ratio * H
                    or span_x >= fallback_size_ratio * W):
                if verbose:
                    print(f'  single cluster spans {span_y}x{span_x} of '
                          f'{H}x{W}: falling back to global m14',
                          flush=True)
                history['fallback_to_global'] = True
                remaining = max(60.0, time_budget_s - (time.time() - t0))
                phi_out = iterative_2d_tri_refine_repair(
                    phi_out, threshold=threshold, margin=margin,
                    anchor=anchor, eps_l1=eps_l1,
                    time_budget_s=remaining,
                    verbose=0, **m14_kwargs)
                break

        round_runs = []
        for b in bboxes:
            if time.time() - t0 > time_budget_s:
                if verbose:
                    print('  budget exhausted; stopping cluster sweep',
                          flush=True)
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
            cluster_budget = max(
                20.0,
                (time_budget_s - (time.time() - t0)) / max(1, len(bboxes)))
            try:
                phi_win_out = iterative_2d_tri_refine_repair(
                    phi_win, threshold=threshold, margin=margin,
                    anchor=anchor, eps_l1=eps_l1,
                    time_budget_s=cluster_budget,
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
                crop=(crop_h, crop_w), bbox=(y0, y1, x0, x1),
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

        post_n_neg, post_min = _stats(phi_out)
        history['outer_rounds'].append(dict(
            outer=outer, before_n_neg=n_neg, after_n_neg=post_n_neg,
            min_T=post_min, wall=time.time() - t0,
        ))
        if verbose:
            print(f'  round {outer} done: global n_neg {n_neg} -> '
                  f'{post_n_neg}, min_T {post_min:+.4f}', flush=True)

        if post_n_neg == 0:
            break
        if post_n_neg >= prev_n_neg:
            no_progress_rounds += 1
            if no_progress_rounds >= 2:
                if verbose:
                    print('  no progress for 2 rounds -> falling back to '
                          'global m14', flush=True)
                history['fallback_to_global'] = True
                remaining = max(60.0, time_budget_s - (time.time() - t0))
                phi_out = iterative_2d_tri_refine_repair(
                    phi_out, threshold=threshold, margin=margin,
                    anchor=anchor, eps_l1=eps_l1,
                    time_budget_s=remaining,
                    verbose=0, **m14_kwargs)
                break
        else:
            no_progress_rounds = 0
        prev_n_neg = post_n_neg

    # Final global polish to recover the safety margin if Schwarz overlap
    # produced cells with T just-above-zero-but-below-threshold.
    post_n_neg, post_min = _stats(phi_out)
    if (final_polish
            and (post_n_neg > 0 or post_min < threshold + 1e-5)
            and time.time() - t0 < time_budget_s):
        if verbose:
            print(f'[final polish] min_T={post_min:+.4f} < threshold; '
                  f'running global barrier polish', flush=True)
        t_polish = time.time()
        phi_out = iterative_2d_tri_barrier(
            phi_out, threshold=threshold, margin=margin,
            max_minimize_iter=final_polish_max_iter,
            anchor=anchor, eps_l1=eps_l1, verbose=0)
        history['final_polish_fired'] = True
        history['final_polish_wall'] = time.time() - t_polish
        if verbose:
            final_n, final_m = _stats(phi_out)
            print(f'  polish done: min_T {post_min:+.4f} -> {final_m:+.4f}  '
                  f'n_neg {post_n_neg} -> {final_n}  '
                  f'({time.time()-t_polish:.1f}s)', flush=True)

    final_n, final_min = _stats(phi_out)
    history['final'] = dict(n_neg=final_n, min_T=final_min,
                            wall=time.time() - t0)
    return (phi_out, history) if record_history else phi_out
