"""Cluster-localised LP/SLP for the 6-tetrahedron constraint (3D
analog of :mod:`dvfopt.core.slp.cluster_lp_2tri`), promoted from
``research/strict_feasibility_3d``.

The direct LP at any meaningful 3D scale is intractable — a single
24x24x24 subvolume from B0039 hung HiGHS for >12 hours when both m10
and m14 left residual folds. This module decomposes the volume into
connected fold clusters via 3D connected-components, runs SLP on each
cluster's padded crop with frozen-edge corners, and splices the
interior back into the full field.

Same architecture as the 2D version, but with 3D bboxes, 3D dilation,
simplex (3D) constraint, 3D phi-pack convention (DX_FIRST: [dx, dy, dz]).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label

from dvfopt._logging import log_info, log_warning
from dvfopt.core.slp.lp_direct_6tet import slp_iter
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d, six_tet_volumes_3d

BBOX_PAD = 3  # voxel-units of padding around each cluster's bbox (3D)
MERGE_DILATION_BASE = 2
MAX_OUTER_ITERS = 3


def _fold_clusters_3d(
    phi_3dhw: np.ndarray,
    merge_dilation: int = MERGE_DILATION_BASE,
    target_threshold: float = 0.0,
):
    """Return list of 3D cluster dicts for every connected component of
    cells where ``min_tet(T_k) < target_threshold``."""
    # Fused per-cell min kernel — never materialises the (6, ...) array
    # (~32x faster than six_tet_volumes_3d(...).min(axis=0)).
    cell_min = six_tet_min_volume_3d(phi_3dhw)  # (D-1, H-1, W-1)
    fold_mask = cell_min < target_threshold
    if not fold_mask.any():
        return []
    merged = (
        binary_dilation(fold_mask, iterations=merge_dilation) if merge_dilation > 0 else fold_mask
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
            ((labels[z0:z1, y0:y1, x0:x1] == (idx + 1)) & fold_mask[z0:z1, y0:y1, x0:x1]).sum()
        )
        out.append(
            {
                'z0': z0,
                'z1': z1,
                'y0': y0,
                'y1': y1,
                'x0': x0,
                'x1': x1,
                'crop_cells_z': z1 - z0,
                'crop_cells_y': y1 - y0,
                'crop_cells_x': x1 - x0,
                'n_fold_cells': comp_cells,
            }
        )
    # Largest clusters first — they tend to dominate the wall time on
    # parallel runs, so dispatching them early keeps cores busy.
    out.sort(key=lambda c: -c['n_fold_cells'])
    return out


def _splice_interior_3d(
    phi_full: np.ndarray,
    c: dict,
    phi_crop_corrected: np.ndarray,
    full_shape=None,
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
    D_full, H_full, W_full = full_shape if full_shape is not None else phi_full.shape[1:]
    # Per-side: shrink by 1 if NOT at volume boundary, else keep flush.
    sd0 = 1 if z0 > 0 else 0
    sd1 = 1 if (z0 + crop_d) < D_full else 0
    sh0 = 1 if y0 > 0 else 0
    sh1 = 1 if (y0 + crop_h) < H_full else 0
    sw0 = 1 if x0 > 0 else 0
    sw1 = 1 if (x0 + crop_w) < W_full else 0
    # Degenerate-crop fallback: splice everything.
    if crop_d - sd0 - sd1 <= 0 or crop_h - sh0 - sh1 <= 0 or crop_w - sw0 - sw1 <= 0:
        phi_full[:, z0 : z0 + crop_d, y0 : y0 + crop_h, x0 : x0 + crop_w] = phi_crop_corrected
        return
    phi_full[
        :, z0 + sd0 : z0 + crop_d - sd1, y0 + sh0 : y0 + crop_h - sh1, x0 + sw0 : x0 + crop_w - sw1
    ] = phi_crop_corrected[
        :,
        sd0 : crop_d - sd1,
        sh0 : crop_h - sh1,
        sw0 : crop_w - sw1,
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
    final_global_polish: bool = True,
    polish_below_threshold: bool = False,
    verbose: int = 0,
):
    """Per-cluster SLP with frozen-edge splice (3D).

    Repeatedly: enumerate fold clusters, solve SLP on each cluster's
    padded crop, splice interior back. Outer loop terminates when no
    folds remain or no progress. When *final_global_polish* is true
    (default, mirroring the 2D cluster solver) and the outer rounds
    stall below threshold, a global RefineRepair-3D (m14-3D) polish
    runs as the feasibility safety net.

    Serial only: the promoted parallel branch (per-call process pool)
    was removed — its Windows spawn + JIT warmup tax outweighed the
    parallelism on every measured configuration. The coarse-grained
    win belongs in the orchestrator (parallel z-bands), not here.

    Default inner_seed is 'm10' (not 'm14' as in 2D): on 3D B0039 m14
    catastrophically overshoots on dense folds, while m10 reaches
    feasibility or near-feasibility (see strict_feasibility_3d README).

    Returns
    -------
    phi_out_3dhw : (3, D, H, W) float64
    info : dict with per-round + per-cluster bookkeeping.
    """
    t0 = time.time()
    phi_out = phi_in_3dhw.astype(np.float64)  # astype always copies
    info = {
        'rounds': [],
        'inner_seed': inner_seed,
        'total_cluster_solves': 0,
    }

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

        for c in clusters:
            phi_crop = phi_out[
                :,
                c['z0'] : c['z1'] + 1,
                c['y0'] : c['y1'] + 1,
                c['x0'] : c['x1'] + 1,
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
                log_warning(f'3D cluster solve FAILED: {type(exc).__name__}: {exc}')
                round_runs.append({**c, 'error': f'{type(exc).__name__}: {exc}'})
                continue
            _splice_interior_3d(phi_out, c, phi_corr)
            info['total_cluster_solves'] += 1
            round_runs.append({**c, 'wall': time.time() - t_c})

        V = six_tet_volumes_3d(phi_out)
        post_n_neg = int((V <= 0).sum())
        post_n_below_threshold = int((threshold - 1e-5 > V).sum())
        info['rounds'].append(
            {
                'outer': outer_it,
                'n_clusters': len(clusters),
                'pre_n_neg': pre_n_neg,
                'post_n_neg': post_n_neg,
                'post_n_below_threshold': post_n_below_threshold,
                'wall': time.time() - t0,
                'cluster_runs': round_runs if verbose else None,
            }
        )
        if verbose:
            log_info(
                f'[outer {outer_it}] {len(clusters)} clusters: '
                f'n_neg {pre_n_neg} -> {post_n_neg}  '
                f'n<threshold={post_n_below_threshold}  '
                f'({time.time() - t0:.1f}s)'
            )
        if post_n_neg == 0 and post_n_below_threshold == 0:
            break
        # No-progress fallback: widen merge_dilation once.
        if outer_it == 0 and post_n_neg >= pre_n_neg:
            merge_dilation += 1
            if merge_dilation > MERGE_DILATION_BASE + 3:
                break

    # Feasibility safety net (2D parity): a global RefineRepair-3D pass
    # when the cluster rounds stalled below the strict threshold.
    final_min = float(six_tet_min_volume_3d(phi_out).min())
    if final_global_polish and final_min < threshold - 1e-5:
        if verbose:
            log_info(f'[final-polish] min_T={final_min:+.5f} < threshold — running global m14-3D')
        from dvfopt.core.wallbreakers._refine_repair_3d import (
            iterative_3d_tet_refine_repair,
        )

        t_p = time.time()
        phi_out = iterative_3d_tet_refine_repair(
            phi_out, threshold=threshold, verbose=max(0, verbose - 1)
        )
        info['final_polish'] = {
            'fired': True,
            'pre_min_T': final_min,
            'wall_s': time.time() - t_p,
        }

    info['final_min_T_exact'] = float(six_tet_min_volume_3d(phi_out).min())
    info['L1_dev'] = float(np.abs(phi_out - phi_in_3dhw).sum())
    info['wall_s'] = time.time() - t0
    return phi_out, info
