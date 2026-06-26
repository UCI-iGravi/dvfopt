"""Minimal pipeline (Variant D): no pre-pass. Directly cluster the
raw input's fold cubes and run coupled SLSQP per cluster, then
M10Tet recovery.

This is the simplest possible 2-stage pipeline:
  1. Cluster raw input's 173 fold cubes into ~13 spatial clusters.
  2. For each cluster, run coupled SLSQP @ thr=1e-3 with adaptive
     k_ring = max(2, cluster_radius + 2), clipped at k_max=4 to keep
     DOF tractable.
  3. After all clusters, run M10Tet @ 0.012 recovery to handle any
     residual + tighten cells below 0.01 threshold.

Hypothesis: even with 173 folds spread across 13 clusters, per-cluster
SLSQP can fix each independently (they don't share corners across
clusters since clusters are >=3 cells apart in Chebyshev distance).

Estimated wall: ~2 minutes per-cluster SLSQP + ~30-45 min recovery
= ~35-50 minutes total.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import minimize

import research.strict_feasibility_3d.runners._coupled_kring as ck
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._cluster_pipeline import (
    cluster_fold_cubes,
    m10tet,
    run_slsqp_around,
)
from research.strict_feasibility_3d.runners._coupled_kring import (
    build_coupled_problem,
    make_apply_x,
    make_constraint_fn,
    make_objective,
    report,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
K_RING_MAX = 4  # cap to keep DOF tractable


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'INPUT (raw)', None)

    # === Step 1: Identify clusters in RAW input. ===
    print('\n=== Step 1: Cluster raw-input fold cubes ===', flush=True)
    V = six_tet_volumes_3d(phi_input)
    min_per_cube = V.min(axis=0)
    fold_mask = (min_per_cube <= 0)
    fold_cells = [tuple(int(c) for c in p) for p in zip(*np.where(fold_mask))]
    print(f'  raw # fold cubes: {len(fold_cells)}', flush=True)
    _, centroids, members, radii = cluster_fold_cubes(fold_cells, radius=3)
    n_clusters = len(centroids)
    print(f'  # clusters: {n_clusters}', flush=True)
    for i, (c, mem, r) in enumerate(zip(centroids, members, radii)):
        chosen_k = max(2, min(K_RING_MAX, r + 2))
        print(f'    cluster {i}: centroid={c}, size={len(mem)}, '
              f'radius={r}, chosen_k_ring={chosen_k}',
              flush=True)

    # === Step 2: Per-cluster SLSQP. ===
    print('\n=== Step 2: Per-cluster coupled SLSQP @ thr=1e-3 (raw input) ===',
          flush=True)
    cur = phi_input.copy()
    total_wall = 0
    for i, (c, mem, r) in enumerate(zip(centroids, members, radii)):
        cz, cy, cx = c
        k_ring = max(2, min(K_RING_MAX, r + 2))
        D, H, W = cur.shape[1:]
        safe_k_z = min(cz, D - 1 - cz - 1)
        safe_k_y = min(cy, H - 1 - cy - 1)
        safe_k_x = min(cx, W - 1 - cx - 1)
        safe_k = min(safe_k_z, safe_k_y, safe_k_x)
        if safe_k < 1:
            print(f'  cluster {i}: too close to boundary (safe_k={safe_k}); skipping',
                  flush=True)
            continue
        k_ring = max(1, min(k_ring, safe_k))

        # Check global n_neg before & after.
        V_before = six_tet_volumes_3d(cur)
        n_before = int((V_before <= 0).sum())

        new, res, wall, n_cubes, n_dof = run_slsqp_around(cur, cz, cy, cx, k_ring)
        total_wall += wall
        if res is None or not res.success:
            print(f'  cluster {i}: SLSQP NOT converged '
                  f'(success={res.success if res else None}, '
                  f'msg={res.message if res else None})',
                  flush=True)
            continue
        V_new = six_tet_volumes_3d(new)
        n_new = int((V_new <= 0).sum())
        print(f'  cluster {i}: k={k_ring} cubes={n_cubes} DOF={n_dof} '
              f'wall={wall:.1f}s  n_neg {n_before}->{n_new}',
              flush=True)
        # Accept if global n_neg dropped substantially or stayed close.
        if n_new <= n_before:
            cur = new
        else:
            print('    rejected (n_neg increased)', flush=True)
    print(f'\n  total cluster-SLSQP wall={total_wall:.1f}s '
          f'({total_wall/60:.2f} min)', flush=True)
    n_after, b_after, _ = report(cur, '  after all cluster SLSQPs', phi_input)

    # === Step 3: M10Tet @ 0.012 recovery. ===
    print('\n=== Step 3: M10Tet @ 0.012 recovery ===', flush=True)
    t_rec = time.time()
    final = m10tet(cur, 0.012)
    wall_rec = time.time() - t_rec
    print(f'  recovery wall={wall_rec:.1f}s', flush=True)
    n, b, mn = report(final, '  FINAL', phi_input)

    total_pipeline_wall = total_wall + wall_rec
    print(f'\n=== VARIANT D FINAL ===\n'
          f'  n_neg={n}, n<0.01={b}\n'
          f'  total wall = {total_pipeline_wall:.1f}s = '
          f'{total_pipeline_wall/60:.1f} min',
          flush=True)
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_simple_D.npy', final)
        print('  *** STRICT FEASIBLE via VARIANT D ***', flush=True)


if __name__ == '__main__':
    main()
