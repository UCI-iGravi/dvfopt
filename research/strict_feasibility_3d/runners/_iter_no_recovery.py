"""Iter pipeline WITHOUT M10Tet recovery between iterations.

Hypothesis: maybe cluster SLSQP alone (without M10Tet recovery in
between) can reduce folds to 1 by repeated application. If so, we
save 5-6 × 60 min = ~5-6 hours from the iter pipeline.

Pipeline:
  Stage 1: M10Tet @ 0.015 on raw input → ~17-19 folds.
  Stage 2: LOOP up to 15 times:
    - cluster-SLSQP @ thr=1e-3 (STRICT accept)
    - check: stop if n_neg=0 or no improvement in 2 consecutive iters
  Stage 3: M10Tet @ 0.012 recovery (final, once)
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
MAX_OUTER_ITERS = 15
STALL_PATIENCE = 2  # stop if no improvement in this many consecutive iters


def cluster_slsqp_pass(phi, phi_input):
    V = six_tet_volumes_3d(phi)
    min_per_cube = V.min(axis=0)
    fold_mask = (min_per_cube <= 0)
    fold_cells = [tuple(int(c) for c in p) for p in zip(*np.where(fold_mask))]
    if not fold_cells:
        return phi, 0.0, 0
    _, centroids, members, radii = cluster_fold_cubes(fold_cells, radius=2)
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
            continue
        total_wall += wall
        if res is None or not res.success:
            continue
        V_before = six_tet_volumes_3d(cur)
        n_before = int((V_before <= 0).sum())
        V_new = six_tet_volumes_3d(new)
        n_new = int((V_new <= 0).sum())
        delta = n_new - n_before
        if delta <= 0:
            cur = new
            accepted += 1
            print(f'    cluster {i+1}/{len(refined)}: n_neg {n_before}->{n_new} '
                  f'wall={wall:.1f}s [ACCEPT]', flush=True)
    return cur, total_wall, accepted


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'INPUT (raw)', None)

    print('\n=== Stage 1: M10Tet @ 0.015 on raw input ===', flush=True)
    t0 = time.time()
    cur = m10tet(phi_input, 0.015)
    wall1 = time.time() - t0
    print(f'  wall={wall1:.1f}s ({wall1/60:.1f} min)', flush=True)
    n, b, _ = report(cur, '  after M10Tet @ 0.015', phi_input)
    if n == 0 and b == 0:
        return

    print('\n=== Stage 2: ITERATE cluster-SLSQP only (no recovery) ===',
          flush=True)
    total_wall = wall1
    last_n = n
    stall_count = 0
    for outer in range(MAX_OUTER_ITERS):
        print(f'\n-- Outer iter {outer+1}/{MAX_OUTER_ITERS}, current n_neg={last_n} --',
              flush=True)
        cur, slsqp_wall, accepted = cluster_slsqp_pass(cur, phi_input)
        total_wall += slsqp_wall
        n, b, mn = report(cur, f'  after SLSQP iter {outer+1}', phi_input)
        print(f'  iter wall={slsqp_wall:.1f}s, accepted={accepted}, '
              f'cumulative={total_wall/60:.1f}min', flush=True)
        if n == 0:
            print('  *** REACHED n_neg=0 ***', flush=True)
            break
        if n >= last_n:
            stall_count += 1
            print(f'  no improvement (stall {stall_count}/{STALL_PATIENCE})',
                  flush=True)
            if stall_count >= STALL_PATIENCE:
                print('  STALL: no progress in 2 consecutive iters, stopping',
                      flush=True)
                break
        else:
            stall_count = 0
        last_n = n

    print('\n=== Stage 3: final M10Tet @ 0.012 recovery ===', flush=True)
    t0 = time.time()
    final = m10tet(cur, 0.012)
    wall_rec = time.time() - t0
    total_wall += wall_rec
    print(f'  recovery wall={wall_rec:.1f}s ({wall_rec/60:.1f} min)',
          flush=True)
    n, b, mn = report(final, '  FINAL', phi_input)
    print(f'  total wall = {total_wall:.1f}s ({total_wall/60:.1f} min)',
          flush=True)
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_iter_norec.npy', final)
        print('  *** STRICT FEASIBLE via ITER-NO-RECOVERY ***', flush=True)


if __name__ == '__main__':
    main()
