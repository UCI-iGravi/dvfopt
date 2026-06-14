"""Cluster-localised LP/SLP for the 2-triangle constraint.

The direct LP at full B0039 slice scale (320x456 -> ~290k decision
variables) is too slow for practical use (>12 min wall on z=12). This
module decomposes the slice into connected fold clusters, runs SLP on
each cluster's padded crop with frozen-edge corners, and splices the
interior back into the slice.

The cluster decomposition pattern matches
``notebooks/manuscript/_run_2d_clusters.py::enumerate_clusters_2d``
but the inner solve is our :func:`slp_iter` instead of the analytic
SLSQP. Each cluster typically yields an LP of <2000 variables -- HiGHS
solves these in tens of milliseconds.

Triggers spec fallback row 5 (``cluster_lp``).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label as cc_label

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from research.strict_feasibility_2d.algorithms.lp_direct_2tri import slp_iter  # noqa: E402

BBOX_PAD = 4  # cell-units of padding around each cluster's bbox
MERGE_DILATION_BASE = 2
MAX_OUTER_ITERS = 3


def _fold_clusters(phi_2hw: np.ndarray, merge_dilation: int = MERGE_DILATION_BASE):
    """Return list of cluster dicts (``y0, y1, x0, x1, n_cells``) for
    every connected fold component, with ``BBOX_PAD`` cells of padding."""
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    cell_min = np.minimum(T1, T2)
    fold_mask = cell_min <= 0
    if not fold_mask.any():
        return []
    merged = binary_dilation(fold_mask, iterations=merge_dilation) if merge_dilation > 0 else fold_mask
    labels, n_comp = cc_label(merged)
    Hc, Wc = fold_mask.shape
    bboxes = find_objects(labels)
    out = []
    for idx, bbox in enumerate(bboxes):
        if bbox is None:
            continue
        cy0, cy1 = bbox[0].start, bbox[0].stop
        cx0, cx1 = bbox[1].start, bbox[1].stop
        y0 = max(0, cy0 - BBOX_PAD)
        y1 = min(Hc, cy1 + BBOX_PAD)
        x0 = max(0, cx0 - BBOX_PAD)
        x1 = min(Wc, cx1 + BBOX_PAD)
        comp_cells = int(((labels[y0:y1, x0:x1] == (idx + 1)) & fold_mask[y0:y1, x0:x1]).sum())
        out.append({
            'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1,
            'crop_cells_y': y1 - y0, 'crop_cells_x': x1 - x0,
            'n_fold_cells': comp_cells,
        })
    # Sort large clusters first.
    out.sort(key=lambda c: -c['n_fold_cells'])
    return out


def _splice_interior(phi_full: np.ndarray, c: dict, phi_crop_corrected: np.ndarray):
    """Splice only the interior corners of the cluster crop back into the
    full slice. The outer one-corner ring is left frozen so neighbouring
    clusters' boundary corners stay consistent."""
    y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
    crop_h = y1 - y0 + 1  # corner-grid height
    crop_w = x1 - x0 + 1
    if crop_h < 3 or crop_w < 3:
        # Too small for an interior — fall back to splicing the whole crop.
        phi_full[:, y0:y0 + crop_h, x0:x0 + crop_w] = phi_crop_corrected
        return
    # Interior corners only.
    phi_full[:, y0 + 1:y0 + crop_h - 1, x0 + 1:x0 + crop_w - 1] = \
        phi_crop_corrected[:, 1:-1, 1:-1]


def cluster_slp_iter(
    phi_in_2hw: np.ndarray,
    *,
    threshold: float = 0.01,
    inner_seed: str = 'm14',
    inner_max_iter: int = 10,
    inner_trust_radius_0: float = 0.5,
    max_outer_iters: int = MAX_OUTER_ITERS,
    merge_dilation: int = MERGE_DILATION_BASE,
    final_global_polish: bool = True,
    verbose: int = 0,
):
    """Per-cluster SLP with frozen-edge splice.

    Repeatedly: enumerate fold clusters, solve SLP on each cluster's
    padded crop (with the inner ``slp_iter``), splice interior back into
    the slice. Outer loop terminates when no folds remain OR no progress
    after one round.

    Parameters
    ----------
    phi_in_2hw : (2, H, W) float64
    threshold : float
    inner_seed : {'m10', 'm14', 'harmonic'}
        Seed kind passed to ``slp_iter`` per cluster.
    inner_max_iter : int
        SLP iteration cap per cluster.
    inner_trust_radius_0 : float
    max_outer_iters : int
    merge_dilation : int
        Cluster CCL merge-dilation in cell units.

    Returns
    -------
    phi_out_2hw : (2, H, W) float64
    info : dict with per-round + per-cluster bookkeeping.
    """
    t0 = time.time()
    phi_out = phi_in_2hw.astype(np.float64).copy()
    info = {
        'rounds': [],
        'inner_seed': inner_seed,
        'total_cluster_solves': 0,
    }

    for outer_it in range(max_outer_iters):
        clusters = _fold_clusters(phi_out, merge_dilation=merge_dilation)
        if not clusters:
            info['rounds'].append({'outer': outer_it, 'n_clusters': 0, 'reason': 'feasible'})
            break

        T1, T2 = _triangle_areas_2d(phi_out[0], phi_out[1])
        pre_n_neg = int((np.minimum(T1, T2) <= 0).sum())
        round_runs = []
        for c in clusters:
            y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
            # Corner-grid crop: (2, y1-y0+1, x1-x0+1).
            phi_crop = phi_out[:, y0:y1 + 1, x0:x1 + 1].copy()
            t_c = time.time()
            try:
                phi_corr, _inner_info = slp_iter(
                    phi_crop,
                    threshold=threshold,
                    trust_radius_0=inner_trust_radius_0,
                    max_iter=inner_max_iter,
                    seed=inner_seed,
                )
            except Exception as exc:
                round_runs.append({**c, 'error': f'{type(exc).__name__}: {exc}'})
                continue
            _splice_interior(phi_out, c, phi_corr)
            info['total_cluster_solves'] += 1
            round_runs.append({**c, 'wall': time.time() - t_c})

        T1, T2 = _triangle_areas_2d(phi_out[0], phi_out[1])
        post_n_neg = int((np.minimum(T1, T2) <= 0).sum())
        info['rounds'].append({
            'outer': outer_it,
            'n_clusters': len(clusters),
            'pre_n_neg': pre_n_neg,
            'post_n_neg': post_n_neg,
            'wall': time.time() - t0,
            'cluster_runs': round_runs if verbose else None,
        })
        if verbose:
            print(
                f'[outer {outer_it}] {len(clusters)} clusters: '
                f'n_neg {pre_n_neg} -> {post_n_neg}  '
                f'({time.time() - t0:.1f}s)',
                flush=True,
            )
        if post_n_neg == 0:
            break
        if post_n_neg >= pre_n_neg:
            # No progress — expand merge_dilation and retry once.
            merge_dilation += 1
            if merge_dilation > MERGE_DILATION_BASE + 3:
                break

    # Final global polish: if cluster passes leave residual folds (the
    # splice boundary can introduce new infeasibility), run global M14
    # **anchored to the cluster output** (not the original input). M14
    # guarantees feasibility, and its L2-refine stays close to its
    # anchor — so the total L1 vs the original input stays near the
    # cluster_slp L1 (which is much lower than global M14 on hard
    # cases), plus a small fix-up term.
    T1f, T2f = _triangle_areas_2d(phi_out[0], phi_out[1])
    cluster_min_T = float(np.minimum(T1f, T2f).min())
    if final_global_polish and cluster_min_T < threshold - 1e-5:
        if verbose:
            print(
                f'[polish] cluster min_T={cluster_min_T:+.4f} < threshold; '
                f'running global M14 anchored to cluster output…',
                flush=True,
            )
        t_p = time.time()
        from dvfopt import (
            HarmonicALMRefineRepairStrategy,
            L1Objective,
            Solver,
            TriConstraint2DFullCoverage,
        )

        H, W = phi_out.shape[1:]
        solver = Solver(
            constraint=TriConstraint2DFullCoverage(shape=(H, W)),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMRefineRepairStrategy(),
            threshold=threshold,
        )
        phi_out = solver.fit(phi_out).corrected
        info['polish_wall'] = time.time() - t_p
        info['polish_fired'] = True
    else:
        info['polish_fired'] = False

    info['final_min_T_exact'] = float(np.minimum(
        *_triangle_areas_2d(phi_out[0], phi_out[1])
    ).min())
    info['L1_dev'] = float(np.abs(phi_out - phi_in_2hw).sum())
    info['wall_s'] = time.time() - t0
    return phi_out, info
