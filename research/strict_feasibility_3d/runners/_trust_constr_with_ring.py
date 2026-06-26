"""Trust-constr cluster NLP with NEIGHBOUR-RING constraint.

Expansion of `_trust_constr_cluster.py`: each cluster's joint NLP
includes not just the unfixable cubes, but also the 1-cell ring of
neighbour cubes. All cubes (unfixable + neighbours) must satisfy
the constraint at the optimum. This prevents the modification of
unfixable cubes from breaking external neighbours.

Increased maxiter to handle the larger problem.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
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


def expand_cluster_with_ring(unfix_cells, cube_shape):
    """Given unfixable cells (cube indices), expand to include all
    cells within distance 1 in cube space."""
    D, H, W = cube_shape
    mask = np.zeros(cube_shape, dtype=bool)
    for (z, y, x) in unfix_cells:
        mask[z, y, x] = True
    expanded = binary_dilation(mask, iterations=1)
    cz, cy, cx = np.where(expanded)
    return list(zip(cz.tolist(), cy.tolist(), cx.tolist()))


def solve_cluster_nlp(phi, target_cells, ring_cells, threshold=THRESHOLD,
                      max_iter=500, verbose=False):
    """target_cells must be satisfied (unfixable); ring_cells must ALSO
    be satisfied (they're neighbours). Both are constrained.
    Variables: union of phi vars touching any (target + ring) cube.
    """
    all_cubes = list(target_cells) + [c for c in ring_cells if c not in set(target_cells)]
    corner_set = set()
    cube_corner_ids = []
    for (cz, cy, cx) in all_cubes:
        ids_for_cube = []
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            corner_set.add((cz + iz, cy + iy, cx + ix))
            ids_for_cube.append((cz + iz, cy + iy, cx + ix))
        cube_corner_ids.append(ids_for_cube)
    corners = sorted(corner_set)
    corner_index = {c: i for i, c in enumerate(corners)}
    n_corners = len(corners)
    n_vars = 3 * n_corners

    x_in = np.zeros(n_vars)
    ref_pos = np.zeros((n_corners, 3))
    for i, (cz, cy, cx) in enumerate(corners):
        x_in[i * 3 + 0] = phi[0, cz, cy, cx]
        x_in[i * 3 + 1] = phi[1, cz, cy, cx]
        x_in[i * 3 + 2] = phi[2, cz, cy, cx]
        ref_pos[i] = (cz, cy, cx)

    cube_var_idx = np.stack([
        np.array([corner_index[c] for c in ids], dtype=np.int64)
        for ids in cube_corner_ids
    ])

    n_cubes = len(all_cubes)

    def constraint_values(x):
        disp = x.reshape(n_corners, 3)
        out = np.empty(6 * n_cubes)
        for cube_i, var_idx in enumerate(cube_var_idx):
            pos = ref_pos[var_idx] + disp[var_idx]
            V = _cube_six_tet_signed(pos)
            out[6 * cube_i:6 * (cube_i + 1)] = V
        return out

    def objective(x):
        d = x - x_in
        return 0.5 * float(d @ d)

    def grad(x):
        return x - x_in

    nlc = NonlinearConstraint(constraint_values, lb=threshold, ub=np.inf)

    t0 = time.time()
    res = minimize(
        objective, x_in, jac=grad,
        method='trust-constr',
        constraints=[nlc],
        options={'maxiter': max_iter, 'verbose': 0, 'gtol': 1e-8, 'xtol': 1e-12},
    )
    wall = time.time() - t0

    phi_new = phi.astype(np.float64).copy()
    final_disp = res.x.reshape(n_corners, 3)
    for i, (cz, cy, cx) in enumerate(corners):
        phi_new[0, cz, cy, cx] = final_disp[i, 0]
        phi_new[1, cz, cy, cx] = final_disp[i, 1]
        phi_new[2, cz, cy, cx] = final_disp[i, 2]

    V_final = constraint_values(res.x).reshape(n_cubes, 6)
    n_cube_feas = int((V_final.min(axis=1) >= threshold - 1e-7).sum())
    if verbose:
        print(
            f'  ({len(target_cells)} target + {n_cubes - len(target_cells)} ring) '
            f'cubes, {n_vars} vars: cube_feas={n_cube_feas}/{n_cubes}  '
            f'min_V={float(V_final.min()):+.6f}  '
            f'L1_added={float(np.abs(res.x - x_in).sum()):.1f}  '
            f'status={res.status}  wall={wall:.1f}s',
            flush=True,
        )
    return phi_new, {
        'n_cubes_feasible': n_cube_feas,
        'n_cubes': n_cubes,
        'min_V': float(V_final.min()),
        'L1_added': float(np.abs(res.x - x_in).sum()),
        'status': res.status,
        'wall_s': wall,
    }


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    print(f'Loaded, shape={phi.shape}', flush=True)
    V0 = six_tet_volumes_3d(phi)
    best_min0 = _best_min_per_cell(phi)
    cube_shape = best_min0.shape
    print(
        f'Start: n_neg={int((V0 <= 0).sum())}  '
        f'unfixable={int((best_min0 <= 0).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    unfix_mask = (best_min0 <= 0)
    nz, ny, nx = np.where(unfix_mask)
    # Cluster unfix cells.
    grid = unfix_mask.copy()
    grid = binary_dilation(grid, iterations=1)
    labels, n_comp = cc_label(grid)
    clusters = []
    for cid in range(1, n_comp + 1):
        in_comp = (labels == cid) & unfix_mask
        cz, cy, cx = np.where(in_comp)
        cells = list(zip(cz.tolist(), cy.tolist(), cx.tolist()))
        if cells:
            clusters.append(cells)
    clusters.sort(key=lambda c: -len(c))
    print(f'{len(clusters)} clusters, sizes: {[len(c) for c in clusters[:10]]}', flush=True)

    phi_new = phi.astype(np.float64).copy()
    total_L1 = 0.0
    for i, target_cells in enumerate(clusters):
        ring_cells = expand_cluster_with_ring(target_cells, cube_shape)
        print(f'\n--- Cluster {i+1}/{len(clusters)}: {len(target_cells)} target + '
              f'{len(ring_cells) - len(target_cells)} ring ---', flush=True)
        phi_new, info = solve_cluster_nlp(phi_new, target_cells, ring_cells,
                                           threshold=THRESHOLD, verbose=True)
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
        np.save(OUTPUT / 'b0039_z0_15_strict_via_trust_constr_with_ring.npy', phi_new)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
