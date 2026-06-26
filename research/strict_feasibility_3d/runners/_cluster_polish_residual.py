"""Cluster-LP polish on the residual folds left by M10Tet stall.

Loads the chunked z=0..15 result (173 folds after Stage 2 + 1x
Stage 3) and runs cluster_slp_iter_3d. The folds should now be
small isolated specks rather than the 16-slice fold columns that
the earlier 16^3 dense-cluster test hit, so cluster decomposition
should actually decompose this time.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

from research.strict_feasibility_3d.algorithms.cluster_lp_6tet import (
    _fold_clusters_3d,
    cluster_slp_iter_3d,
)


OUTPUT = _HERE / 'output'


def _stats(phi, label):
    V = six_tet_volumes_3d(phi)
    return (
        f'{label}  n_neg={int((V<=0).sum()):>6d}  '
        f'n<0.01={int((V<0.01-1e-5).sum()):>6d}  '
        f'min_T={float(V.min()):+.6f}'
    )


def main():
    cache = OUTPUT / 'b0039_FULL_stage3_z000_016.npy'
    phi = np.load(cache)
    print(_stats(phi, 'Loaded input:    '), flush=True)
    print(f'  shape: {phi.shape}', flush=True)

    # Inspect cluster structure first.
    V = six_tet_volumes_3d(phi)
    fold_mask = (V.min(axis=0) <= 0)
    print(f'\n  fold mask shape: {fold_mask.shape}  fold cells: {int(fold_mask.sum())}', flush=True)
    for md in [0, 1, 2]:
        merged = binary_dilation(fold_mask, iterations=md) if md > 0 else fold_mask
        labels, n_comp = cc_label(merged)
        if n_comp > 0:
            sizes = sorted(np.bincount(labels.ravel())[1:].tolist(), reverse=True)[:5]
        else:
            sizes = []
        print(f'  merge_dilation={md}: n_components={n_comp}  top sizes: {sizes}', flush=True)

    # cluster_slp_3d with m10_fast inner seed (skips barrier polish).
    # Smaller per-cluster cost — the outer SLP polishes L1 anyway.
    # Also tighten merge_dilation to keep clusters smaller (current
    # default md=2 produced a 536-cell mega-cluster; md=1 gives 20
    # smaller pieces).
    print('\n=== cluster_slp_3d polish (m10_fast inner, md=1) ===', flush=True)
    t0 = time.time()
    phi_out, info = cluster_slp_iter_3d(
        phi,
        threshold=0.01,
        inner_seed='m10_fast',
        merge_dilation=1,
        max_outer_iters=4,
        polish_below_threshold=True,
        verbose=1,
    )
    wall = time.time() - t0
    print(_stats(phi_out, '\nResult:          '), flush=True)
    print(
        f'  wall: {wall:.1f}s  cluster solves: {info["total_cluster_solves"]}',
        flush=True,
    )
    V_out = six_tet_volumes_3d(phi_out)
    n_neg = int((V_out <= 0).sum())
    n_below = int((V_out < 0.01 - 1e-5).sum())
    if n_neg == 0 and n_below == 0:
        print('\n*** STRICT 100% FEASIBLE on z=0..15 chunk ***', flush=True)
        np.save(OUTPUT / 'b0039_FULL_strict_feas_z000_016.npy', phi_out)


if __name__ == '__main__':
    main()
