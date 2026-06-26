"""GLOBAL SLSQP: one big SLSQP covering ALL fold cubes simultaneously.

If we can do this in reasonable time, it skips the iter-loop entirely.

Pipeline:
  Stage 1: M10Tet @ 0.015 on raw input → ~17 folds.
  Stage 2: Build ONE SLSQP problem covering ALL fold cubes + k-ring halo
           (union of k=2 halos around each fold cube).
  Stage 3: M10Tet @ 0.012 recovery.

The decision DOF will be larger (e.g., maybe 1500-3000) and constraints
will be ~5000-10000. SLSQP at this scale is at the edge of practicality
but may still work.
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

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._coupled_kring import (
    make_apply_x, make_constraint_fn, make_objective, report,
)
import research.strict_feasibility_3d.runners._coupled_kring as ck
from research.strict_feasibility_3d.runners._cluster_pipeline import m10tet


OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
SLSQP_THR = 1e-3


def build_global_slsqp_problem(phi, k_ring=2):
    """Build one SLSQP problem covering ALL fold cubes + k-ring halos.

    Returns:
      cubes : list of (cz, cy, cx) — every cube whose corners must be
              feasible (union of k-ring around each fold cube).
      free_corners : sorted list of corners with free DOF.
      x0 : initial values.
    """
    V = six_tet_volumes_3d(phi)
    min_per_cube = V.min(axis=0)
    fold_mask = (min_per_cube <= 0)
    fold_cells = [tuple(int(c) for c in p) for p in zip(*np.where(fold_mask))]
    D, H, W = phi.shape[1:]
    cells_max = (D - 1, H - 1, W - 1)
    cubes_set = set()
    for (cz, cy, cx) in fold_cells:
        for dz in range(-k_ring, k_ring + 1):
            for dy in range(-k_ring, k_ring + 1):
                for dx in range(-k_ring, k_ring + 1):
                    nz, ny, nx = cz + dz, cy + dy, cx + dx
                    if (0 <= nz < cells_max[0]
                            and 0 <= ny < cells_max[1]
                            and 0 <= nx < cells_max[2]):
                        cubes_set.add((nz, ny, nx))
    cubes = sorted(cubes_set)
    corner_set = set()
    for (cz, cy, cx) in cubes:
        for i in range(8):
            iz = (i >> 2) & 1; iy = (i >> 1) & 1; ix = i & 1
            corner_set.add((cz + iz, cy + iy, cx + ix))
    free_corners = sorted(corner_set)
    x0 = np.zeros(3 * len(free_corners))
    for ci, (z, y, x) in enumerate(free_corners):
        x0[3*ci+0] = phi[0, z, y, x]
        x0[3*ci+1] = phi[1, z, y, x]
        x0[3*ci+2] = phi[2, z, y, x]
    return cubes, free_corners, x0


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'INPUT (raw)', None)

    # Check if we have a saved M10Tet plateau.
    plateau_path = OUTPUT / 'b0039_z0_15_m10tet_plateau.npy'
    if plateau_path.exists():
        print(f'\n=== Loading saved M10Tet plateau from {plateau_path} ===',
              flush=True)
        phi_after_m10 = np.load(plateau_path).astype(np.float64)
    else:
        print('\n=== Stage 1: M10Tet @ 0.015 on raw input ===', flush=True)
        t0 = time.time()
        phi_after_m10 = m10tet(phi_input, 0.015)
        wall = time.time() - t0
        print(f'  M10Tet wall={wall:.1f}s ({wall/60:.1f} min)', flush=True)
        np.save(plateau_path, phi_after_m10)
        print(f'  Saved plateau to {plateau_path}', flush=True)
    report(phi_after_m10, '  after M10Tet @ 0.015', phi_input)

    # Stage 2: Global SLSQP covering all fold cubes.
    print('\n=== Stage 2: GLOBAL SLSQP @ thr=1e-3 ===', flush=True)
    for k_ring in [2, 3]:
        print(f'\n--- Trying k_ring={k_ring} ---', flush=True)
        cubes, free_corners, x0 = build_global_slsqp_problem(
            phi_after_m10, k_ring=k_ring)
        print(f'  Problem size: cubes={len(cubes)}, '
              f'free_corners={len(free_corners)}, DOF={3*len(free_corners)}, '
              f'constraints={6*len(cubes)}', flush=True)

        apply_x, corner_idx_map = make_apply_x(cubes, free_corners, phi_after_m10)
        ck.FEASIBILITY_THR = SLSQP_THR
        constraint_fn, _ = make_constraint_fn(cubes, corner_idx_map)
        obj, obj_grad = make_objective(x0.copy())
        cons = [{'type': 'ineq', 'fun': constraint_fn}]
        t0 = time.time()
        res = minimize(
            obj, x0, jac=obj_grad,
            constraints=cons, method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': True},
        )
        wall = time.time() - t0
        print(f'  SLSQP wall={wall:.1f}s ({wall/60:.1f} min)', flush=True)
        print(f'  SLSQP success={res.success}, fun={res.fun:.4f}, iter={res.nit}',
              flush=True)
        print(f'  message: {res.message}', flush=True)
        if res is not None and res.success:
            phi_after_slsqp = apply_x(res.x)
            n, b, _ = report(phi_after_slsqp, f'  after global SLSQP k={k_ring}',
                              phi_input)
            np.save(OUTPUT / f'b0039_z0_15_global_k{k_ring}.npy', phi_after_slsqp)
            if n == 0 and b == 0:
                print(f'  *** STRICT FEASIBLE via global k={k_ring} ***', flush=True)
                return
            if n <= 2:
                # Worth running recovery.
                print(f'\n  M10Tet @ 0.012 recovery on k={k_ring} result...',
                      flush=True)
                t0 = time.time()
                final = m10tet(phi_after_slsqp, 0.012)
                wall_rec = time.time() - t0
                print(f'  recovery wall={wall_rec:.1f}s ({wall_rec/60:.1f} min)',
                      flush=True)
                n, b, mn = report(final, f'  FINAL after k={k_ring}+recovery',
                                   phi_input)
                if n == 0 and b == 0:
                    np.save(OUTPUT / 'b0039_z0_15_strict_via_global.npy', final)
                    print(f'  *** STRICT FEASIBLE via global k={k_ring}+recovery ***',
                          flush=True)
                    return
        else:
            print('  Skipping recovery; SLSQP did not converge', flush=True)


if __name__ == '__main__':
    main()
