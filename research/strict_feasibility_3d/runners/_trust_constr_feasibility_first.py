"""Trust-constr cluster NLP with FEASIBILITY-FIRST formulation.

The L1-anchor variant (_trust_constr_cluster.py) plateaus because
minimizing ||phi - phi_in||^2 trades off feasibility for L1 cost.
This variant inverts the priority:

  Objective: maximize the minimum tet volume (i.e., feasibility margin)
  Constraint: ||phi - phi_in||_inf <= L (large trust)

Equivalently, we can formulate as:
  Minimize max_k (threshold - V_k(phi))   (this is concave)
  Subject to L1 bounded movement

To make the objective smooth: use log-sum-exp soft-max approximation
of max_k:
  Min log(sum_k exp(beta * (threshold - V_k)))  /  beta   (for large beta)

Or just: minimize sum_k max(0, threshold + margin - V_k)^2 — pure
penalty, no L1 anchor.
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


def solve_cluster_feasibility(phi, cluster_cells, threshold=THRESHOLD,
                              trust_radius=5.0, max_iter=300, verbose=False):
    """Solve for feasibility via penalty minimization with bounded trust."""
    corner_set = set()
    cube_corner_ids = []
    for (cz, cy, cx) in cluster_cells:
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

    n_cubes = len(cluster_cells)

    def constraint_values(x):
        disp = x.reshape(n_corners, 3)
        out = np.empty(6 * n_cubes)
        for cube_i, var_idx in enumerate(cube_var_idx):
            pos = ref_pos[var_idx] + disp[var_idx]
            V = _cube_six_tet_signed(pos)
            out[6 * cube_i:6 * (cube_i + 1)] = V
        return out

    # Pure penalty objective — strongly penalize infeasibility, tiny L1 anchor.
    eps_l1 = 1e-6  # tiny — basically just to break ties

    def objective(x):
        V = constraint_values(x)
        viol = np.maximum(0.0, threshold - V + 1e-4)  # small margin to push above threshold
        d = x - x_in
        return float((viol * viol).sum()) + eps_l1 * 0.5 * float(d @ d)

    # Bounds: x within trust_radius of x_in.
    bounds = Bounds(x_in - trust_radius, x_in + trust_radius)

    t0 = time.time()
    res = minimize(
        objective, x_in,
        method='trust-constr',
        bounds=bounds,
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
            f'  cluster ({n_cubes} cubes, {n_vars} vars): '
            f'cube_feas={n_cube_feas}/{n_cubes}  '
            f'min_V={float(V_final.min()):+.6f}  '
            f'L1_added={float(np.abs(res.x - x_in).sum()):.1f}  '
            f'wall={wall:.1f}s',
            flush=True,
        )
    return phi_new, {
        'n_cubes_feasible': n_cube_feas,
        'n_cubes': n_cubes,
        'min_V': float(V_final.min()),
        'L1_added': float(np.abs(res.x - x_in).sum()),
        'wall_s': wall,
    }


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
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
    nz, ny, nx = np.where(unfix_mask)
    grid = np.zeros(unfix_mask.shape, dtype=bool)
    grid[nz, ny, nx] = True
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
    n_total_cubes_fixed = 0
    n_total_cubes = 0
    total_L1 = 0.0
    for i, cells in enumerate(clusters):
        print(f'\n--- Cluster {i+1}/{len(clusters)}: {len(cells)} cubes ---', flush=True)
        phi_new, info = solve_cluster_feasibility(phi_new, cells, threshold=THRESHOLD,
                                                   trust_radius=5.0, verbose=True)
        n_total_cubes_fixed += info['n_cubes_feasible']
        n_total_cubes += info['n_cubes']
        total_L1 += info['L1_added']

    V_final = six_tet_volumes_3d(phi_new)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1_final = float(np.abs(phi_new - phi).sum())
    print(
        f'\n=== Final ===\n'
        f'  cluster cubes fixed: {n_total_cubes_fixed}/{n_total_cubes}\n'
        f'  total intra-cluster L1: {total_L1:.1f}\n'
        f'  global n_neg: {n_neg}\n'
        f'  global n<0.01: {n_below}\n'
        f'  global min_T: {float(V_final.min()):+.6f}\n'
        f'  global L1 from input: {L1_final:.1f}\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_feasibility_first.npy', phi_new)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
