"""Cluster-localized m14 for the 3D 6-tet constraint (Schwarz domain decomposition).

3D analog of :mod:`dvfopt.core.wallbreakers._m14_schwarz` (the 2D
m14-Schwarz wallbreaker). Wraps :func:`iterative_3d_tet_refine_repair`
with the same per-cluster decomposition pattern.

Algorithm
---------
1. Detect connected fold components via 3D CCL (26-connectivity, dilated
   by ``merge_dilation`` to group near-touching clusters).
2. For each component, extract a padded corner-bbox crop.
3. Run global m14-3D on each crop independently; splice back.
4. If new folds appear at crop boundaries (Schwarz overlap artifact),
   repeat for up to ``max_outer_iters`` rounds.

Fallback to global m14-3D:
* a single cluster spans more than ``fallback_size_ratio`` of any axis;
* the outer loop fails to reduce ``n_neg`` for two consecutive rounds.

When folds cover a small fraction of a large 3D volume the wall-clock
advantage is significant — same shape of speedup as the 2D version,
just applied per-volume.

For now no final global polish (the 2D version has one) — the
per-cluster m14-3D already runs its own log-barrier polish stage on
each crop. A composite global polish can be added later if cluster
overlap consistently degrades the global min_V.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as cc_label

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.core.wallbreakers._refine_repair_3d import iterative_3d_tet_refine_repair
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d


def _stats_3d(phi: np.ndarray) -> tuple[int, float]:
    """Return (n_neg per-tet, min per-tet volume) for the whole field."""
    V = six_tet_volumes_3d(phi)
    return int((V <= 0).sum()), float(V.min())


def _fold_clusters_3d(phi: np.ndarray, threshold: float, merge_dilation: int = 2):
    """Connected components of folded VOXEL cells, dilated for grouping.

    Returns
    -------
    bboxes : list of dict with keys 'comp_id', 'cz0', 'cz1', 'cy0', 'cy1',
             'cx0', 'cx1', 'n_folds' (all cell-space indices, inclusive).
    fold_cells : ndarray, shape ``(D-1, H-1, W-1)``, bool — the underlying
        cells with any tet below threshold.
    """
    V = six_tet_volumes_3d(phi)
    fold_cells = V.min(axis=0) < threshold
    if not fold_cells.any():
        return [], fold_cells
    grouped = binary_dilation(fold_cells, iterations=merge_dilation)
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


def iterative_3d_tet_refine_repair_schwarz(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    margin: float = 1e-3,
    anchor: str = 'l2',
    eps_l1: float = 1e-4,
    pad: int = 4,
    merge_dilation: int = 2,
    max_outer_iters: int = 3,
    fallback_size_ratio: float = 0.7,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    record_history: bool = False,
    **m14_kwargs,
):
    """Cluster-localized 3D refine-repair (m14-Schwarz-3D).

    Parameters
    ----------
    phi_in : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
    threshold, margin, anchor, eps_l1
        Forwarded to per-cluster ``iterative_3d_tet_refine_repair``.
    pad : int
        Voxels of corner-space expansion around each cluster's bounding
        box. Picks up the per-cluster m14's "frozen ring" context.
    merge_dilation : int
        3D dilation iterations on the fold-cell mask before CCL.
    max_outer_iters : int
        Outer-loop budget if splicing introduces new folds at crop
        boundaries.
    fallback_size_ratio : float
        If a single cluster's bounding box covers more than this fraction
        of any axis, fall back immediately to global m14-3D.
    time_budget_s, verbose, record_history : as elsewhere.
    **m14_kwargs
        Forwarded to per-cluster ``iterative_3d_tet_refine_repair``.

    Returns
    -------
    phi_out : ndarray, shape ``(3, D, H, W)``.
    info : dict, only if ``record_history=True``.
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    phi_in = np.asarray(phi_in, dtype=np.float64)
    if phi_in.ndim != 4 or phi_in.shape[0] != 3:
        raise ValueError(f'expected (3, D, H, W) input; got shape {phi_in.shape}')
    _, D, H, W = phi_in.shape
    phi_out = phi_in.copy()
    t0 = time.time()

    init_n_neg, init_min = _stats_3d(phi_in)
    history = {
        'init': dict(n_neg=init_n_neg, min_T=init_min),
        'cluster_runs': [],
        'outer_rounds': [],
        'fallback_to_global': False,
    }

    # Already feasible? early-out.
    if init_n_neg == 0:
        history['final'] = dict(n_neg=0, min_T=init_min, mode='no-folds')
        return (phi_out, history) if record_history else phi_out

    last_n_neg = init_n_neg
    last_round_no_progress = 0

    for outer_round in range(max_outer_iters):
        if time.time() - t0 > time_budget_s:
            break
        bboxes, _fold_cells = _fold_clusters_3d(phi_out, threshold, merge_dilation)
        if not bboxes:
            break

        # Cell-space dimensions: (D-1, H-1, W-1).
        Dc, Hc, Wc = D - 1, H - 1, W - 1
        # Fallback check: does any single cluster span > ratio of any axis?
        for b in bboxes:
            span_z = (b['cz1'] - b['cz0'] + 1) / max(1, Dc)
            span_y = (b['cy1'] - b['cy0'] + 1) / max(1, Hc)
            span_x = (b['cx1'] - b['cx0'] + 1) / max(1, Wc)
            if max(span_z, span_y, span_x) > fallback_size_ratio:
                history['fallback_to_global'] = True
                if verbose:
                    print(
                        f'  [schwarz outer {outer_round}] cluster {b["comp_id"]} '
                        f'spans {max(span_z, span_y, span_x):.2f} > '
                        f'{fallback_size_ratio} — falling back to global m14-3D',
                        flush=True,
                    )
                phi_out = iterative_3d_tet_refine_repair(
                    phi_out,
                    threshold=threshold,
                    margin=margin,
                    anchor=anchor,
                    eps_l1=eps_l1,
                    time_budget_s=max(60.0, time_budget_s - (time.time() - t0)),
                    verbose=max(0, verbose - 1),
                    **m14_kwargs,
                )
                final_n_neg, final_min = _stats_3d(phi_out)
                history['final'] = dict(n_neg=final_n_neg, min_T=final_min, mode='fallback-global')
                return (phi_out, history) if record_history else phi_out

        # Per-cluster m14-3D crops.
        if verbose:
            print(
                f'  [schwarz outer {outer_round}] {len(bboxes)} cluster(s); n_neg={last_n_neg}',
                flush=True,
            )
        for b in bboxes:
            # Cells (cz0..cz1) use corners (cz0..cz1+1). Add pad on each side.
            z0 = max(0, b['cz0'] - pad)
            z1 = min(D - 1, b['cz1'] + 1 + pad)
            y0 = max(0, b['cy0'] - pad)
            y1 = min(H - 1, b['cy1'] + 1 + pad)
            x0 = max(0, b['cx0'] - pad)
            x1 = min(W - 1, b['cx1'] + 1 + pad)
            if z1 - z0 < 2 or y1 - y0 < 2 or x1 - x0 < 2:
                continue  # degenerate crop
            crop = phi_out[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1].copy()
            crop_out = iterative_3d_tet_refine_repair(
                crop,
                threshold=threshold,
                margin=margin,
                anchor=anchor,
                eps_l1=eps_l1,
                time_budget_s=max(
                    30.0,
                    (time_budget_s - (time.time() - t0)) / max(1, len(bboxes)),
                ),
                verbose=max(0, verbose - 1),
                **m14_kwargs,
            )
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
        # Progress check (Schwarz overlap artifact: did n_neg actually decrease?)
        if cur_n_neg >= last_n_neg:
            last_round_no_progress += 1
            if last_round_no_progress >= 2:
                # Fall back to global on stall.
                history['fallback_to_global'] = True
                if verbose:
                    print(
                        '  [schwarz] no progress for 2 rounds — falling back to global',
                        flush=True,
                    )
                phi_out = iterative_3d_tet_refine_repair(
                    phi_out,
                    threshold=threshold,
                    margin=margin,
                    anchor=anchor,
                    eps_l1=eps_l1,
                    time_budget_s=max(60.0, time_budget_s - (time.time() - t0)),
                    verbose=max(0, verbose - 1),
                    **m14_kwargs,
                )
                break
        else:
            last_round_no_progress = 0
        last_n_neg = cur_n_neg

    final_n_neg, final_min = _stats_3d(phi_out)
    history['final'] = dict(n_neg=final_n_neg, min_T=final_min)
    return (phi_out, history) if record_history else phi_out


__all__ = ['iterative_3d_tet_refine_repair_schwarz']
