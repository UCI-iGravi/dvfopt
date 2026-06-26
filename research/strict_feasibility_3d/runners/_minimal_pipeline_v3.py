"""Minimal pipeline D'' (v3): finer clustering + smaller k_ring cap.

Lessons from D' (stuck on big clusters with k=5):
  - k=5 SLSQP at boundary = ~700 cubes, ~3000+ DOF, ~5000 constraints —
    too slow.
  - Need to cap k_ring at 3 and split clusters more aggressively.

This variant:
  1. Cluster at radius=2 (tighter; more clusters but each is smaller).
  2. Re-cluster big clusters (radius>2) at radius=1.
  3. K_RING_MAX = 3 (caps DOF at ~7^3*8*3 = 8232 raw; typical 1000-2000).
  4. SLSQP @ thr=1e-3 per (sub-)cluster.
  5. M10Tet @ 0.012 recovery.
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
from research.strict_feasibility_3d.runners._coupled_kring import report
from research.strict_feasibility_3d.runners._cluster_pipeline import (
    cluster_fold_cubes, run_slsqp_around, m10tet,
)


OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
K_RING_MAX = 3
INIT_CLUSTER_RADIUS = 2
SUB_CLUSTER_RADIUS = 1


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'INPUT (raw)', None)

    V = six_tet_volumes_3d(phi_input)
    min_per_cube = V.min(axis=0)
    fold_mask = (min_per_cube <= 0)
    fold_cells = [tuple(int(c) for c in p) for p in zip(*np.where(fold_mask))]
    print(f'\n=== Cluster (radius={INIT_CLUSTER_RADIUS}) ===', flush=True)
    print(f'  # fold cubes: {len(fold_cells)}', flush=True)
    _, centroids, members, radii = cluster_fold_cubes(
        fold_cells, radius=INIT_CLUSTER_RADIUS)
    print(f'  # clusters: {len(centroids)}', flush=True)

    # Refine: split big clusters at sub_cluster_radius.
    refined = []
    for c, mem, r in zip(centroids, members, radii):
        if r <= K_RING_MAX - 1:
            refined.append((c, mem, r))
            continue
        _, sub_cents, sub_mems, sub_rs = cluster_fold_cubes(
            mem, radius=SUB_CLUSTER_RADIUS)
        for sc, sm, sr in zip(sub_cents, sub_mems, sub_rs):
            refined.append((sc, sm, sr))
    print(f'  total refined: {len(refined)} (sub-)clusters', flush=True)

    # Step 2: Per-cluster SLSQP.
    print('\n=== Step 2: Per-cluster coupled SLSQP @ thr=1e-3 ===', flush=True)
    cur = phi_input.copy()
    total_wall = 0.0
    accepted = 0
    for i, (c, mem, r) in enumerate(refined):
        cz, cy, cx = c
        k_ring = max(2, min(K_RING_MAX, r + 2))
        try:
            new, res, wall, n_cubes, n_dof = run_slsqp_around(cur, cz, cy, cx, k_ring)
        except Exception as e:
            print(f'  cluster {i} ({c}): error {e}', flush=True)
            continue
        total_wall += wall
        V_before = six_tet_volumes_3d(cur)
        n_before = int((V_before <= 0).sum())
        if res is None or not res.success:
            print(f'  cluster {i} ({c}, size={len(mem)}, r={r}, k={k_ring}, '
                  f'cubes={n_cubes}, DOF={n_dof}): NOT CONVERGED wall={wall:.1f}s',
                  flush=True)
            continue
        V_new = six_tet_volumes_3d(new)
        n_new = int((V_new <= 0).sum())
        delta = n_new - n_before
        marker = 'ACCEPT' if delta <= 0 else 'REJECT'
        print(f'  cluster {i} ({c}, size={len(mem)}, r={r}, k={k_ring}, '
              f'cubes={n_cubes}, DOF={n_dof}): n_neg {n_before}->{n_new} '
              f'(delta={delta:+d}) wall={wall:.1f}s [{marker}]',
              flush=True)
        if delta <= 0:
            cur = new
            accepted += 1
    print(f'\n  Accepted: {accepted}/{len(refined)}', flush=True)
    print(f'  total cluster-SLSQP wall = {total_wall:.1f}s ({total_wall/60:.1f} min)',
          flush=True)
    n_after, b_after, _ = report(cur, '  after cluster SLSQPs', phi_input)

    # Step 3: M10Tet recovery.
    print('\n=== Step 3: M10Tet @ 0.012 recovery ===', flush=True)
    t_rec = time.time()
    final = m10tet(cur, 0.012)
    wall_rec = time.time() - t_rec
    print(f'  recovery wall={wall_rec:.1f}s ({wall_rec/60:.1f} min)', flush=True)
    n, b, mn = report(final, '  FINAL', phi_input)

    total_pipeline = total_wall + wall_rec
    print(f'\n=== VARIANT D\'\' FINAL ===\n'
          f'  n_neg={n}, n<0.01={b}\n'
          f'  total wall = {total_pipeline:.1f}s ({total_pipeline/60:.1f} min)',
          flush=True)
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_simple_Dpp.npy', final)
        print(f'  *** STRICT FEASIBLE via VARIANT D\'\' ***', flush=True)


if __name__ == '__main__':
    main()
