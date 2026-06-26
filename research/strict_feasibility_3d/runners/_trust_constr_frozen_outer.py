"""Trust-constr cluster NLP with FROZEN outer ring corners.

Improvement over _trust_constr_with_ring.py: the outermost corners
of the joint NLP scope (corners shared with cubes OUTSIDE the
constraint set) are frozen at their input values. This prevents
modifications from propagating to cubes the NLP doesn't see.

Concretely:
  - target_cells = unfixable cubes in the cluster
  - ring_cells = 1-ring of target cells (cubes within distance 1)
  - frozen_corners = corners touched by cubes OUTSIDE (target+ring)
                     that ALSO touch ring cells (the outermost face)
  - free_corners = corners touched only by (target + ring) and not
                   by outer cubes

The NLP varies only free_corners while constraining all (target +
ring) cubes' 6-tet volumes >= threshold.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.ndimage import label as cc_label, binary_dilation

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_VERTICES,
    _TET_SIGN,
    six_tet_volumes_3d,
)

from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell


OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def _cube_six_tet_signed(corner_pos):
    out = np.empty(6)
    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        A, B, C, Dv = corner_pos[i0], corner_pos[i1], corner_pos[i2], corner_pos[i3]
        AB = B - A
        AC = C - A
        AD = Dv - A
        det = (AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
               - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
               + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0]))
        out[k] = float(_TET_SIGN[k]) * det / 6.0
    return out


def _corners_of_cube(cz, cy, cx):
    out = []
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        out.append((cz + iz, cy + iy, cx + ix))
    return out


def solve_cluster_frozen(phi, target_cells, ring_cells, outer_cells,
                          threshold=THRESHOLD, max_iter=1000, verbose=False):
    """outer_cells: cubes ADJACENT to ring cells that are NOT in target+ring.
    Their corners (shared with ring) get FROZEN."""
    cluster_cubes = list(target_cells) + [c for c in ring_cells if c not in set(target_cells)]

    # All corners touched by ANY cluster cube.
    cluster_corner_set = set()
    cube_corner_ids = []
    for (cz, cy, cx) in cluster_cubes:
        ids = _corners_of_cube(cz, cy, cx)
        cluster_corner_set.update(ids)
        cube_corner_ids.append(ids)

    # Corners touched by an outer cube.
    outer_corner_set = set()
    for (cz, cy, cx) in outer_cells:
        outer_corner_set.update(_corners_of_cube(cz, cy, cx))

    # Frozen corners: cluster corners that are ALSO touched by outer cubes.
    frozen_corner_set = cluster_corner_set & outer_corner_set
    free_corner_set = cluster_corner_set - frozen_corner_set

    # Order corners: free first, then frozen.
    free_corners = sorted(free_corner_set)
    frozen_corners = sorted(frozen_corner_set)
    all_corners = free_corners + frozen_corners
    corner_index = {c: i for i, c in enumerate(all_corners)}
    n_free = len(free_corners)
    n_frozen = len(frozen_corners)
    n_total = n_free + n_frozen
    n_free_vars = 3 * n_free

    # Initial values for ALL corners.
    x_in_all = np.zeros(3 * n_total)
    ref_pos = np.zeros((n_total, 3))
    for i, (cz, cy, cx) in enumerate(all_corners):
        x_in_all[i * 3 + 0] = phi[0, cz, cy, cx]
        x_in_all[i * 3 + 1] = phi[1, cz, cy, cx]
        x_in_all[i * 3 + 2] = phi[2, cz, cy, cx]
        ref_pos[i] = (cz, cy, cx)

    # Frozen part is held at x_in_all[n_free*3:].
    x_in_free = x_in_all[:n_free_vars].copy()
    x_frozen = x_in_all[n_free_vars:].copy()

    # Map each cube to its 8 corner-indices (in all_corners ordering).
    cube_var_idx = np.stack([
        np.array([corner_index[c] for c in ids], dtype=np.int64)
        for ids in cube_corner_ids
    ])

    n_cubes = len(cluster_cubes)

    def constraint_values(x_free):
        # Combine free and frozen.
        x_all = np.empty(3 * n_total)
        x_all[:n_free_vars] = x_free
        x_all[n_free_vars:] = x_frozen
        disp = x_all.reshape(n_total, 3)
        out = np.empty(6 * n_cubes)
        for cube_i, var_idx in enumerate(cube_var_idx):
            pos = ref_pos[var_idx] + disp[var_idx]
            V = _cube_six_tet_signed(pos)
            out[6 * cube_i:6 * (cube_i + 1)] = V
        return out

    def objective(x_free):
        d = x_free - x_in_free
        return 0.5 * float(d @ d)

    def grad(x_free):
        return x_free - x_in_free

    nlc = NonlinearConstraint(constraint_values, lb=threshold, ub=np.inf)

    t0 = time.time()
    res = minimize(
        objective, x_in_free, jac=grad,
        method='trust-constr',
        constraints=[nlc],
        options={'maxiter': max_iter, 'verbose': 0, 'gtol': 1e-8, 'xtol': 1e-12},
    )
    wall = time.time() - t0

    # Apply to global phi (only free corners change).
    phi_new = phi.astype(np.float64).copy()
    final_free = res.x.reshape(n_free, 3)
    for i, (cz, cy, cx) in enumerate(free_corners):
        phi_new[0, cz, cy, cx] = final_free[i, 0]
        phi_new[1, cz, cy, cx] = final_free[i, 1]
        phi_new[2, cz, cy, cx] = final_free[i, 2]
    # Frozen corners are unchanged.

    V_final = constraint_values(res.x).reshape(n_cubes, 6)
    n_cube_feas = int((V_final.min(axis=1) >= threshold - 1e-7).sum())
    if verbose:
        print(
            f'  ({len(target_cells)} target + {n_cubes - len(target_cells)} ring), '
            f'free={n_free} corners ({n_free_vars} vars), frozen={n_frozen} corners: '
            f'cube_feas={n_cube_feas}/{n_cubes}  '
            f'min_V={float(V_final.min()):+.6f}  '
            f'L1_added={float(np.abs(res.x - x_in_free).sum()):.1f}  '
            f'status={res.status}  wall={wall:.1f}s',
            flush=True,
        )
    return phi_new, {
        'n_cubes_feasible': n_cube_feas,
        'n_cubes': n_cubes,
        'min_V': float(V_final.min()),
        'L1_added': float(np.abs(res.x - x_in_free).sum()),
        'status': res.status,
        'wall_s': wall,
    }


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    D, H, W = phi.shape[1:]
    cube_shape = (D - 1, H - 1, W - 1)
    print(f'Loaded, shape={phi.shape}', flush=True)
    V0 = six_tet_volumes_3d(phi)
    best_min0 = _best_min_per_cell(phi)
    print(
        f'Start: n_neg={int((V0 <= 0).sum())}  '
        f'unfixable={int((best_min0 <= 0).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    unfix_mask = (best_min0 <= 0)
    grid = unfix_mask.copy()
    grid_dilated1 = binary_dilation(grid, iterations=1)
    labels, n_comp = cc_label(grid_dilated1)
    clusters = []
    for cid in range(1, n_comp + 1):
        in_comp = (labels == cid) & unfix_mask
        cz, cy, cx = np.where(in_comp)
        cells = list(zip(cz.tolist(), cy.tolist(), cx.tolist()))
        if cells:
            clusters.append(cells)
    clusters.sort(key=lambda c: -len(c))

    phi_new = phi.astype(np.float64).copy()
    total_L1 = 0.0
    for i, target_cells in enumerate(clusters):
        # Build ring (1-cell expansion of target).
        target_mask = np.zeros(cube_shape, dtype=bool)
        for (z, y, x) in target_cells:
            target_mask[z, y, x] = True
        ring_mask = binary_dilation(target_mask, iterations=1)
        ring_cells = list(zip(*np.where(ring_mask)))
        ring_cells = [(int(z), int(y), int(x)) for z, y, x in ring_cells]
        # Build outer (2-ring): cubes ADJACENT to ring that aren't in ring.
        outer_mask = binary_dilation(ring_mask, iterations=1) & ~ring_mask
        outer_cells = list(zip(*np.where(outer_mask)))
        outer_cells = [(int(z), int(y), int(x)) for z, y, x in outer_cells]
        print(
            f'\n--- Cluster {i+1}/{len(clusters)}: {len(target_cells)} target + '
            f'{len(ring_cells) - len(target_cells)} ring + {len(outer_cells)} outer-frozen ---',
            flush=True,
        )
        phi_new, info = solve_cluster_frozen(
            phi_new, target_cells, ring_cells, outer_cells,
            threshold=THRESHOLD, max_iter=1000, verbose=True,
        )
        total_L1 += info['L1_added']

    V_final = six_tet_volumes_3d(phi_new)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1_final = float(np.abs(phi_new - phi).sum())
    print(
        f'\n=== Final ===\n'
        f'  total intra-cluster L1: {total_L1:.1f}\n'
        f'  global n_neg: {n_neg}\n'
        f'  global n<0.01: {n_below}\n'
        f'  global min_T: {float(V_final.min()):+.6f}\n'
        f'  global L1 from input: {L1_final:.1f}\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_trust_constr_frozen.npy', phi_new)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
