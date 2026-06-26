"""Trust-constr with SUBDIVIDED hard clusters.

For each connected cluster of unfixable cells, split into small
sub-clusters of at most MAX_TARGET_PER_SUB cells each (using a
spatial partition). Solve each sub-cluster with 2-ring buffer.

Hypothesis: cluster 1 (21 cells) and cluster 2 (16 cells) failed
because they're too big for trust-constr at maxiter=1000. Splitting
them into ~5-cell sub-clusters lets each converge reliably (as
demonstrated by 100% success on the 5-cell clusters in the 1-ring
test).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as cc_label

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._trust_constr_cluster import solve_cluster_nlp
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
MAX_TARGET_PER_SUB = 5  # max unfixable cells per sub-cluster


def subdivide_cluster(target_cells, max_size=MAX_TARGET_PER_SUB):
    """Split a list of target cells into sub-groups of at most
    max_size cells each. Greedy spatial grouping: pick a seed cell,
    add nearest neighbours until size limit, repeat."""
    if len(target_cells) <= max_size:
        return [target_cells]
    remaining = list(target_cells)
    sub_clusters = []
    while remaining:
        seed = remaining[0]
        # Sort remaining by distance to seed.
        sz, sy, sx = seed
        dists = [(c, (c[0] - sz) ** 2 + (c[1] - sy) ** 2 + (c[2] - sx) ** 2) for c in remaining]
        dists.sort(key=lambda t: t[1])
        sub = [c for c, _ in dists[:max_size]]
        sub_clusters.append(sub)
        remaining = [c for c in remaining if c not in set(sub)]
    return sub_clusters


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    D, H, W = phi.shape[1:]
    cube_shape = (D - 1, H - 1, W - 1)
    V0 = six_tet_volumes_3d(phi)
    best_min0 = _best_min_per_cell(phi)
    print(
        f'Start: n_neg={int((V0 <= 0).sum())}  unfixable={int((best_min0 <= 0).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    unfix_mask = best_min0 <= 0
    grid = binary_dilation(unfix_mask, iterations=1)
    labels, n_comp = cc_label(grid)
    clusters = []
    for cid in range(1, n_comp + 1):
        in_comp = (labels == cid) & unfix_mask
        cz, cy, cx = np.where(in_comp)
        cells = list(zip(cz.tolist(), cy.tolist(), cx.tolist()))
        if cells:
            clusters.append(cells)
    clusters.sort(key=lambda c: -len(c))
    print(f'{len(clusters)} clusters, sizes: {[len(c) for c in clusters]}', flush=True)

    phi_new = phi.astype(np.float64).copy()
    total_L1 = 0.0
    total_subs = 0
    n_fully_feas = 0
    for ci, cluster in enumerate(clusters):
        sub_clusters = subdivide_cluster(cluster, max_size=MAX_TARGET_PER_SUB)
        print(
            f'\n--- Cluster {ci + 1}/{len(clusters)} ({len(cluster)} cells) split into '
            f'{len(sub_clusters)} sub-clusters ---',
            flush=True,
        )
        for si, sub in enumerate(sub_clusters):
            total_subs += 1
            # Build 2-ring around this sub-cluster.
            sub_mask = np.zeros(cube_shape, dtype=bool)
            for z, y, x in sub:
                sub_mask[z, y, x] = True
            ring2 = binary_dilation(sub_mask, iterations=2)
            ring2_cells = list(zip(*np.where(ring2)))
            ring2_cells = [(int(z), int(y), int(x)) for z, y, x in ring2_cells]
            t0 = time.time()
            phi_new, info = solve_cluster_nlp(
                phi_new,
                ring2_cells,
                threshold=THRESHOLD,
                max_iter=500,
                verbose=False,
            )
            wall = time.time() - t0
            n_cubes = info['n_cubes']
            cf = info['n_cubes_feasible']
            print(
                f'  sub {si + 1}/{len(sub_clusters)} ({len(sub)} target + '
                f'{n_cubes - len(sub)} ring): {cf}/{n_cubes}  '
                f'min_V={info["min_V"]:+.4f}  L1+={info["L1_added"]:.1f}  '
                f'wall={wall:.0f}s',
                flush=True,
            )
            total_L1 += info['L1_added']
            if cf == n_cubes:
                n_fully_feas += 1

    V_final = six_tet_volumes_3d(phi_new)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1_final = float(np.abs(phi_new - phi).sum())
    print(
        f'\n=== Final ===\n'
        f'  sub-clusters fully feasible: {n_fully_feas}/{total_subs}\n'
        f'  intra-NLP L1 cost: {total_L1:.1f}\n'
        f'  global n_neg: {n_neg}\n'
        f'  global n<0.01: {n_below}\n'
        f'  global min_T: {float(V_final.min()):+.6f}\n'
        f'  global L1 from input: {L1_final:.1f}\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_trust_constr_subdivide.npy', phi_new)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
