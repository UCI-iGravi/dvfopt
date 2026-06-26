"""Iterative simple pipeline: alternate cluster-SLSQP and M10Tet
recovery until n_neg=0 or convergence stalls.

Findings from Variants A-D'':
  - M10Tet pre-pass on raw input reaches 17-19 fold plateau.
  - Cluster SLSQP from this plateau reduces folds but often
    can't reach 0 in one pass (Variant C reached 7).
  - M10Tet recovery on the SLSQP result further reduces folds.
  - Iterating SLSQP + recovery should converge to 0.

Pipeline:
  Stage 1: M10Tet @ 0.015 on raw input → ~17-19 folds.
  Loop (up to 5 iters):
    Stage 2.1: cluster-SLSQP (radius=2, k_max=3, thr=1e-3, STRICT accept)
    Stage 2.2: M10Tet @ 0.012 recovery
    Check: stop if n_neg=0 and n<0.01=0.

Estimated wall: 30 min (Stage 1) + N * (1 min + 30 min)
              = 30 + 31N min.
For N=3 iters: ~2 hours total.
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
MAX_OUTER_ITERS = 6


def cluster_slsqp_pass(phi, phi_input):
    """One pass of per-cluster SLSQP @ thr=1e-3, STRICT acceptance
    (only accept moves that decrease n_neg or keep it the same)."""
    V = six_tet_volumes_3d(phi)
    min_per_cube = V.min(axis=0)
    fold_mask = (min_per_cube <= 0)
    fold_cells = [tuple(int(c) for c in p) for p in zip(*np.where(fold_mask))]
    if not fold_cells:
        return phi, 0.0, 0
    _, centroids, members, radii = cluster_fold_cubes(fold_cells, radius=2)
    # Refine big clusters.
    refined = []
    for c, mem, r in zip(centroids, members, radii):
        if r <= K_RING_MAX - 1:
            refined.append((c, mem, r))
            continue
        _, sc, sm, sr = cluster_fold_cubes(mem, radius=1)
        for cc, cm, cr in zip(sc, sm, sr):
            refined.append((cc, cm, cr))
    cur = phi.copy()
    total_wall = 0.0
    accepted = 0
    print(f'  -> {len(refined)} (sub-)clusters', flush=True)
    for i, (c, mem, r) in enumerate(refined):
        cz, cy, cx = c
        k_ring = max(2, min(K_RING_MAX, r + 2))
        try:
            new, res, wall, n_cubes, n_dof = run_slsqp_around(
                cur, cz, cy, cx, k_ring)
        except Exception as e:
            print(f'    cluster {i}: error {e}', flush=True)
            continue
        total_wall += wall
        if res is None or not res.success:
            continue
        V_before = six_tet_volumes_3d(cur)
        n_before = int((V_before <= 0).sum())
        V_new = six_tet_volumes_3d(new)
        n_new = int((V_new <= 0).sum())
        delta = n_new - n_before
        # STRICT: only accept if delta <= 0.
        if delta <= 0:
            cur = new
            accepted += 1
            print(f'    cluster {i}/{len(refined)}: n_neg {n_before}->{n_new} '
                  f'wall={wall:.1f}s [ACCEPT]', flush=True)
    return cur, total_wall, accepted


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'INPUT (raw)', None)

    # Stage 1: M10Tet pre-pass.
    print('\n=== Stage 1: M10Tet @ 0.015 on raw input ===', flush=True)
    t0 = time.time()
    cur = m10tet(phi_input, 0.015)
    wall1 = time.time() - t0
    print(f'  wall={wall1:.1f}s ({wall1/60:.1f} min)', flush=True)
    n, b, mn = report(cur, '  after M10Tet @ 0.015', phi_input)
    if n == 0 and b == 0:
        print('  Already strict feasible.', flush=True)
        return

    # Loop.
    total_wall = wall1
    for outer in range(MAX_OUTER_ITERS):
        print(f'\n=== Outer iter {outer + 1}/{MAX_OUTER_ITERS} ===', flush=True)

        # Stage 2.1: cluster SLSQP.
        print(f'  -- cluster-SLSQP pass --', flush=True)
        cur, slsqp_wall, accepted = cluster_slsqp_pass(cur, phi_input)
        total_wall += slsqp_wall
        print(f'  cluster-SLSQP wall={slsqp_wall:.1f}s, accepted={accepted}',
              flush=True)
        n, b, _ = report(cur, f'  after iter {outer+1} SLSQP', phi_input)
        if n == 0 and b == 0:
            print(f'  *** STRICT FEASIBLE after iter {outer+1} SLSQP ***',
                  flush=True)
            break

        # Stage 2.2: M10Tet recovery.
        print(f'  -- M10Tet @ 0.012 recovery --', flush=True)
        t0 = time.time()
        cur = m10tet(cur, 0.012)
        rec_wall = time.time() - t0
        total_wall += rec_wall
        print(f'  recovery wall={rec_wall:.1f}s ({rec_wall/60:.1f} min)',
              flush=True)
        n, b, _ = report(cur, f'  after iter {outer+1} M10Tet recovery',
                          phi_input)
        if n == 0 and b == 0:
            print(f'  *** STRICT FEASIBLE after iter {outer+1} M10Tet ***',
                  flush=True)
            break

    print(f'\n=== Iter pipeline FINAL ===', flush=True)
    n, b, mn = report(cur, '  FINAL', phi_input)
    print(f'  total wall = {total_wall:.1f}s ({total_wall/60:.1f} min = {total_wall/3600:.1f} hours)',
          flush=True)
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_iter.npy', cur)
        print('  *** STRICT FEASIBLE via ITER PIPELINE ***', flush=True)


if __name__ == '__main__':
    main()
