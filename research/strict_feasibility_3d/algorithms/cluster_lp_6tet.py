"""Cluster-localised LP/SLP for the 6-tetrahedron constraint (3D
analog of ``research/strict_feasibility_2d/algorithms/cluster_lp_2tri.py``).

The direct LP at any meaningful 3D scale is intractable — a single
24x24x24 subvolume from B0039 hung HiGHS for >12 hours when both m10
and m14 left residual folds. This module decomposes the volume into
connected fold clusters via 3D connected-components, runs SLP on each
cluster's padded crop with frozen-edge corners, and splices the
interior back into the full field.

Same architecture as the 2D version, but with 3D bboxes, 3D dilation,
6-tet constraint, 3D phi-pack convention (DX_FIRST: [dx, dy, dz]).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.algorithms.lp_direct_6tet import slp_iter

BBOX_PAD = 3  # voxel-units of padding around each cluster's bbox (3D)
MERGE_DILATION_BASE = 2
MAX_OUTER_ITERS = 3
DEFAULT_PARALLEL_WORKERS = 1


def _solve_cluster_worker(args):
    """Top-level (picklable) worker for ProcessPoolExecutor parallelism."""
    phi_crop, inner_threshold, inner_trust_radius_0, inner_max_iter, inner_seed = args
    phi_corr, _ = slp_iter(
        phi_crop,
        threshold=inner_threshold,
        trust_radius_0=inner_trust_radius_0,
        max_iter=inner_max_iter,
        seed=inner_seed,
    )
    return phi_corr


def _partition_clusters_nonoverlapping(clusters):
    """Greedy graph-colour: pack clusters into rounds where no two in
    the same round have overlapping or adjacent bboxes. 3D version of
    the 2D non-overlap check."""
    rounds = []
    for c in clusters:
        z0, z1, y0, y1, x0, x1 = c['z0'], c['z1'], c['y0'], c['y1'], c['x0'], c['x1']
        placed = False
        for r in rounds:
            ok = True
            for c2 in r:
                if not (
                    z1 < c2['z0'] or c2['z1'] < z0
                    or y1 < c2['y0'] or c2['y1'] < y0
                    or x1 < c2['x0'] or c2['x1'] < x0
                ):
                    ok = False
                    break
            if ok:
                r.append(c)
                placed = True
                break
        if not placed:
            rounds.append([c])
    return rounds


def _fold_clusters_3d(
    phi_3dhw: np.ndarray,
    merge_dilation: int = MERGE_DILATION_BASE,
    target_threshold: float = 0.0,
):
    """Return list of 3D cluster dicts for every connected component of
    cells where ``min_tet(T_k) < target_threshold``."""
    V = six_tet_volumes_3d(phi_3dhw)  # (6, D-1, H-1, W-1)
    cell_min = V.min(axis=0)
    fold_mask = cell_min < target_threshold
    if not fold_mask.any():
        return []
    merged = (
        binary_dilation(fold_mask, iterations=merge_dilation)
        if merge_dilation > 0 else fold_mask
    )
    labels, n_comp = cc_label(merged)
    Dc, Hc, Wc = fold_mask.shape
    bboxes = find_objects(labels)
    out = []
    for idx, bbox in enumerate(bboxes):
        if bbox is None:
            continue
        cz0, cz1 = bbox[0].start, bbox[0].stop
        cy0, cy1 = bbox[1].start, bbox[1].stop
        cx0, cx1 = bbox[2].start, bbox[2].stop
        z0 = max(0, cz0 - BBOX_PAD)
        z1 = min(Dc, cz1 + BBOX_PAD)
        y0 = max(0, cy0 - BBOX_PAD)
        y1 = min(Hc, cy1 + BBOX_PAD)
        x0 = max(0, cx0 - BBOX_PAD)
        x1 = min(Wc, cx1 + BBOX_PAD)
        comp_cells = int(
            ((labels[z0:z1, y0:y1, x0:x1] == (idx + 1))
             & fold_mask[z0:z1, y0:y1, x0:x1]).sum()
        )
        out.append({
            'z0': z0, 'z1': z1,
            'y0': y0, 'y1': y1,
            'x0': x0, 'x1': x1,
            'crop_cells_z': z1 - z0,
            'crop_cells_y': y1 - y0,
            'crop_cells_x': x1 - x0,
            'n_fold_cells': comp_cells,
        })
    # Largest clusters first — they tend to dominate the wall time on
    # parallel runs, so dispatching them early keeps cores busy.
    out.sort(key=lambda c: -c['n_fold_cells'])
    return out


def _splice_interior_3d(
    phi_full: np.ndarray, c: dict, phi_crop_corrected: np.ndarray, full_shape=None,
):
    """Splice the cluster crop back into the full volume, freezing only
    the corner ring that touches OTHER clusters.

    The outer corner ring on each side is normally left frozen so
    neighbouring clusters' boundary corners stay consistent. But when a
    side of the cluster bbox is at the volume boundary, there is no
    neighbour to coordinate with — freezing that side discards solver
    work for no benefit. This matters for small volumes where one
    cluster spans most of the field.
    """
    z0, z1 = c['z0'], c['z1']
    y0, y1 = c['y0'], c['y1']
    x0, x1 = c['x0'], c['x1']
    crop_d = z1 - z0 + 1
    crop_h = y1 - y0 + 1
    crop_w = x1 - x0 + 1
    D_full, H_full, W_full = (full_shape if full_shape is not None
                              else phi_full.shape[1:])
    # Per-side: shrink by 1 if NOT at volume boundary, else keep flush.
    sd0 = 1 if z0 > 0 else 0
    sd1 = 1 if (z0 + crop_d) < D_full else 0
    sh0 = 1 if y0 > 0 else 0
    sh1 = 1 if (y0 + crop_h) < H_full else 0
    sw0 = 1 if x0 > 0 else 0
    sw1 = 1 if (x0 + crop_w) < W_full else 0
    # Degenerate-crop fallback: splice everything.
    if crop_d - sd0 - sd1 <= 0 or crop_h - sh0 - sh1 <= 0 or crop_w - sw0 - sw1 <= 0:
        phi_full[:, z0:z0 + crop_d, y0:y0 + crop_h, x0:x0 + crop_w] = phi_crop_corrected
        return
    phi_full[:,
             z0 + sd0:z0 + crop_d - sd1,
             y0 + sh0:y0 + crop_h - sh1,
             x0 + sw0:x0 + crop_w - sw1] = phi_crop_corrected[
                 :,
                 sd0:crop_d - sd1,
                 sh0:crop_h - sh1,
                 sw0:crop_w - sw1,
             ]


def cluster_slp_iter_3d(
    phi_in_3dhw: np.ndarray,
    *,
    threshold: float = 0.01,
    inner_seed: str = 'm10',
    inner_max_iter: int = 10,
    inner_trust_radius_0: float = 0.5,
    max_outer_iters: int = MAX_OUTER_ITERS,
    merge_dilation: int = MERGE_DILATION_BASE,
    final_global_polish: bool = False,
    n_workers: int = DEFAULT_PARALLEL_WORKERS,
    polish_below_threshold: bool = False,
    verbose: int = 0,
):
    """Per-cluster SLP with frozen-edge splice (3D).

    Repeatedly: enumerate fold clusters, solve SLP on each cluster's
    padded crop, splice interior back. Outer loop terminates when no
    folds remain or no progress.

    Default inner_seed is 'm10' (not 'm14' as in 2D): on 3D B0039 m14
    catastrophically overshoots on dense folds, while m10 reaches
    feasibility or near-feasibility (see strict_feasibility_3d README).

    Returns
    -------
    phi_out_3dhw : (3, D, H, W) float64
    info : dict with per-round + per-cluster bookkeeping.
    """
    t0 = time.time()
    phi_out = phi_in_3dhw.astype(np.float64).copy()
    info = {
        'rounds': [],
        'inner_seed': inner_seed,
        'total_cluster_solves': 0,
    }
    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None

    for outer_it in range(max_outer_iters):
        # Round 0: by default target only actual folds. With
        # `polish_below_threshold=True`, target below-threshold cells
        # from the start (useful when the input is already fold-free
        # but doesn't meet the strict threshold — typical of an M10Tet
        # post-pass that left some cells between 0 and threshold).
        if polish_below_threshold:
            target = threshold - 1e-5
        else:
            target = 0.0 if outer_it == 0 else (threshold - 1e-5)
        clusters = _fold_clusters_3d(
            phi_out, merge_dilation=merge_dilation, target_threshold=target
        )
        if not clusters:
            info['rounds'].append({'outer': outer_it, 'n_clusters': 0, 'reason': 'feasible'})
            break

        V = six_tet_volumes_3d(phi_out)
        pre_n_neg = int((V <= 0).sum())
        round_runs = []

        inner_threshold = threshold + 1e-4

        if n_workers > 1:
            sub_rounds = _partition_clusters_nonoverlapping(clusters)
            for sub_round in sub_rounds:
                arg_list = []
                for c in sub_round:
                    phi_crop = phi_out[
                        :,
                        c['z0']:c['z1'] + 1,
                        c['y0']:c['y1'] + 1,
                        c['x0']:c['x1'] + 1,
                    ].copy()
                    arg_list.append((
                        phi_crop, inner_threshold,
                        inner_trust_radius_0, inner_max_iter, inner_seed,
                    ))
                if len(sub_round) > 1:
                    results = list(pool.map(_solve_cluster_worker, arg_list))
                else:
                    results = [_solve_cluster_worker(arg_list[0])]
                for c, phi_corr in zip(sub_round, results):
                    _splice_interior_3d(phi_out, c, phi_corr)
                    info['total_cluster_solves'] += 1
                    round_runs.append({**c})
        else:
            for c in clusters:
                phi_crop = phi_out[
                    :,
                    c['z0']:c['z1'] + 1,
                    c['y0']:c['y1'] + 1,
                    c['x0']:c['x1'] + 1,
                ].copy()
                t_c = time.time()
                try:
                    phi_corr, _ = slp_iter(
                        phi_crop,
                        threshold=inner_threshold,
                        trust_radius_0=inner_trust_radius_0,
                        max_iter=inner_max_iter,
                        seed=inner_seed,
                    )
                except Exception as exc:
                    round_runs.append({**c, 'error': f'{type(exc).__name__}: {exc}'})
                    continue
                _splice_interior_3d(phi_out, c, phi_corr)
                info['total_cluster_solves'] += 1
                round_runs.append({**c, 'wall': time.time() - t_c})

        V = six_tet_volumes_3d(phi_out)
        post_n_neg = int((V <= 0).sum())
        post_n_below_threshold = int((threshold - 1e-5 > V).sum())
        info['rounds'].append({
            'outer': outer_it,
            'n_clusters': len(clusters),
            'pre_n_neg': pre_n_neg,
            'post_n_neg': post_n_neg,
            'post_n_below_threshold': post_n_below_threshold,
            'wall': time.time() - t0,
            'cluster_runs': round_runs if verbose else None,
        })
        if verbose:
            print(
                f'[outer {outer_it}] {len(clusters)} clusters: '
                f'n_neg {pre_n_neg} -> {post_n_neg}  '
                f'n<threshold={post_n_below_threshold}  '
                f'({time.time() - t0:.1f}s)',
                flush=True,
            )
        if post_n_neg == 0 and post_n_below_threshold == 0:
            break
        # No-progress fallback: widen merge_dilation once.
        if outer_it == 0 and post_n_neg >= pre_n_neg:
            merge_dilation += 1
            if merge_dilation > MERGE_DILATION_BASE + 3:
                break

    if pool is not None:
        pool.shutdown(wait=True)

    info['final_min_T_exact'] = float(six_tet_volumes_3d(phi_out).min())
    info['L1_dev'] = float(np.abs(phi_out - phi_in_3dhw).sum())
    info['wall_s'] = time.time() - t0
    return phi_out, info
