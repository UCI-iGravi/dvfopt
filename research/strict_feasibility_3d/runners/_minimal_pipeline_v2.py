"""Minimal pipeline D' (fixed): cluster SLSQP directly on raw input
+ M10Tet recovery. Fixes Variant D bugs:

  1. Don't skip boundary clusters — build_coupled_problem already
     handles boundary trimming; just clamp k_ring conservatively.
  2. For big clusters (radius > 4), split into sub-clusters via
     a denser radius=1 re-clustering.
  3. Use larger k_ring (up to K_RING_MAX=5).

Pipeline:
  1. Cluster raw-input fold cubes (radius=3 → ~13 clusters).
  2. Re-cluster big clusters (radius>4) at finer radius=1.
  3. For each (sub-)cluster, run coupled SLSQP @ thr=1e-3
     with k_ring = max(2, min(K_RING_MAX, radius + 2)).
  4. M10Tet @ 0.012 recovery.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._coupled_kring import (
    build_coupled_problem, make_apply_x, make_constraint_fn,
    make_objective, report,
)
import research.strict_feasibility_3d.runners._coupled_kring as ck
from research.strict_feasibility_3d.runners._cluster_pipeline import (
    cluster_fold_cubes, run_slsqp_around, m10tet,
)


OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
K_RING_MAX = 5


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'INPUT (raw)', None)

    # === Step 1: Coarse cluster (radius=3). ===
    V = six_tet_volumes_3d(phi_input)
    min_per_cube = V.min(axis=0)
    fold_mask = (min_per_cube <= 0)
    fold_cells = [tuple(int(c) for c in p) for p in zip(*np.where(fold_mask))]
    print(f'\n=== Cluster raw-input fold cubes (radius=3) ===', flush=True)
    print(f'  # fold cubes: {len(fold_cells)}', flush=True)
    _, centroids, members, radii = cluster_fold_cubes(fold_cells, radius=3)
    n_clusters = len(centroids)
    print(f'  # clusters: {n_clusters}', flush=True)

    # Split big clusters via finer re-clustering (radius=1).
    print('\n=== Re-cluster big clusters (radius>4) at radius=1 ===',
          flush=True)
    refined_clusters = []  # list of (centroid, members, radius)
    for i, (c, mem, r) in enumerate(zip(centroids, members, radii)):
        if r <= 4:
            refined_clusters.append((c, mem, r))
            continue
        # Re-cluster.
        _, sub_centroids, sub_members, sub_radii = cluster_fold_cubes(mem, radius=1)
        print(f'  big cluster {i} (size={len(mem)}, r={r}) -> {len(sub_centroids)} sub-clusters',
              flush=True)
        for j, (sc, sm, sr) in enumerate(zip(sub_centroids, sub_members, sub_radii)):
            print(f'    sub {j}: centroid={sc}, size={len(sm)}, r={sr}',
                  flush=True)
            refined_clusters.append((sc, sm, sr))
    print(f'\n  total refined (sub-)clusters: {len(refined_clusters)}',
          flush=True)

    # === Step 2: Per-cluster SLSQP, with conservative k_ring clipping. ===
    print('\n=== Step 2: Per-cluster coupled SLSQP @ thr=1e-3 ===',
          flush=True)
    cur = phi_input.copy()
    total_wall = 0.0
    accepted = 0
    rejected = 0
    skipped = 0
    for i, (c, mem, r) in enumerate(zip(
            [c for c, _, _ in refined_clusters],
            [m for _, m, _ in refined_clusters],
            [r for _, _, r in refined_clusters])):
        cz, cy, cx = c
        k_ring = max(2, min(K_RING_MAX, r + 2))
        # Build trial. Let build_coupled_problem handle boundary trimming.
        try:
            new, res, wall, n_cubes, n_dof = run_slsqp_around(cur, cz, cy, cx, k_ring)
        except Exception as e:
            print(f'  cluster {i} ({c}): error {e}', flush=True)
            skipped += 1
            continue
        total_wall += wall
        V_before = six_tet_volumes_3d(cur)
        n_before = int((V_before <= 0).sum())
        if res is None or not res.success:
            print(f'  cluster {i} ({c}, size={len(mem)}, r={r}, k={k_ring}): '
                  f'SLSQP NOT converged (msg={res.message if res else None}) wall={wall:.1f}s',
                  flush=True)
            skipped += 1
            continue
        V_new = six_tet_volumes_3d(new)
        n_new = int((V_new <= 0).sum())
        delta = n_new - n_before
        marker = 'ACCEPT' if delta <= 0 else 'REJECT'
        print(f'  cluster {i} ({c}, size={len(mem)}, r={r}, k={k_ring}, '
              f'cubes={n_cubes}, DOF={n_dof}): '
              f'n_neg {n_before} -> {n_new} (delta={delta:+d}) wall={wall:.1f}s [{marker}]',
              flush=True)
        if delta <= 0:
            cur = new
            accepted += 1
        else:
            rejected += 1
    print(f'\n  Accepted: {accepted}, rejected: {rejected}, skipped: {skipped}',
          flush=True)
    print(f'  total cluster-SLSQP wall = {total_wall:.1f}s ({total_wall/60:.1f} min)',
          flush=True)
    n_after, b_after, _ = report(cur, '  after all cluster SLSQPs', phi_input)

    # === Step 3: M10Tet @ 0.012 recovery. ===
    print('\n=== Step 3: M10Tet @ 0.012 recovery ===', flush=True)
    t_rec = time.time()
    final = m10tet(cur, 0.012)
    wall_rec = time.time() - t_rec
    print(f'  recovery wall={wall_rec:.1f}s ({wall_rec/60:.1f} min)', flush=True)
    n, b, mn = report(final, '  FINAL', phi_input)

    total_pipeline = total_wall + wall_rec
    print(f'\n=== VARIANT D\' FINAL ===\n'
          f'  n_neg={n}, n<0.01={b}\n'
          f'  total wall = {total_pipeline:.1f}s ({total_pipeline/60:.1f} min)',
          flush=True)
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_simple_Dprime.npy', final)
        print(f'  *** STRICT FEASIBLE via VARIANT D\' ***', flush=True)


if __name__ == '__main__':
    main()
