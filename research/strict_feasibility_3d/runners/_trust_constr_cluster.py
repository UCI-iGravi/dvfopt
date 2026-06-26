"""Joint cluster NLP via scipy trust-constr.

For each connected cluster of unfixable cells, solve a single
nonlinear NLP that simultaneously satisfies all cube constraints
in the cluster. Frozen exterior boundary (corners NOT shared with
any cluster cube are held at their input values).

Trust-constr handles cubic-curvature non-linear constraints
directly (unlike SLP's linearization or M10Tet's smoothed
barrier). Trust region bounds the step size, so deep-fold
constraints become tractable.
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
from scipy.optimize import NonlinearConstraint, minimize

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    six_tet_volumes_3d,
)
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def _cube_six_tet_signed(corner_pos):
    """Signed volume of each of 6 tets for one cube (8x3 array of
    deformed corner positions). Returns (6,) vector."""
    out = np.empty(6)
    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        A, B, C, Dv = corner_pos[i0], corner_pos[i1], corner_pos[i2], corner_pos[i3]
        AB = B - A
        AC = C - A
        AD = Dv - A
        det = (
            AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
            - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
            + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])
        )
        out[k] = float(_TET_SIGN[k]) * det / 6.0
    return out


def solve_cluster_nlp(phi, cluster_cells, threshold=THRESHOLD, max_iter=200, verbose=False):
    """Solve joint NLP for a cluster of cubes.

    Variables: phi values at all CORNERS touched by any cube in the
    cluster. (Each corner is identified by its (z, y, x) integer grid
    index.) Each variable has 3 components (dz, dy, dx).

    Constraints: 6 tet volumes >= threshold for each cube in cluster.

    Objective: 0.5 * sum_i (x_i - x_in_i)^2.

    Returns updated phi (only cluster corners changed) and info.
    """
    # Collect unique corners touched by any cluster cube.
    corner_set = set()
    cube_corner_ids = []  # for each cube in cluster, the 8 corner ids (indices into corner list)
    for cz, cy, cx in cluster_cells:
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

    # Build initial x (current state) and reference identity positions.
    x_in = np.zeros(n_vars)
    ref_pos = np.zeros((n_corners, 3))
    for i, (cz, cy, cx) in enumerate(corners):
        x_in[i * 3 + 0] = phi[0, cz, cy, cx]
        x_in[i * 3 + 1] = phi[1, cz, cy, cx]
        x_in[i * 3 + 2] = phi[2, cz, cy, cx]
        ref_pos[i] = (cz, cy, cx)

    # Map each cube to its 8 corner-indices in the corner list.
    cube_var_idx = []  # list of (8,) arrays of indices into corners
    for ids in cube_corner_ids:
        cube_var_idx.append(np.array([corner_index[c] for c in ids], dtype=np.int64))
    cube_var_idx = np.stack(cube_var_idx)  # (n_cubes, 8)

    # Identity offsets for ordering corners within a cube.
    id_offsets = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], dtype=np.float64)

    n_cubes = len(cluster_cells)

    def constraint_values(x):
        """Compute all 6 * n_cubes tet volumes from flat x."""
        # Reshape x to (n_corners, 3).
        disp = x.reshape(n_corners, 3)
        out = np.empty(6 * n_cubes)
        for cube_i, var_idx in enumerate(cube_var_idx):
            # 8 corners' deformed positions for this cube.
            pos = ref_pos[var_idx] + disp[var_idx]  # (8, 3)
            V = _cube_six_tet_signed(pos)
            out[6 * cube_i : 6 * (cube_i + 1)] = V
        return out

    def objective(x):
        d = x - x_in
        return 0.5 * float(d @ d)

    def grad(x):
        return x - x_in

    # NonlinearConstraint with V >= threshold ↔ V - threshold ∈ [0, ∞).
    nlc = NonlinearConstraint(
        constraint_values,
        lb=threshold,
        ub=np.inf,
    )

    t0 = time.time()
    res = minimize(
        objective,
        x_in,
        jac=grad,
        method='trust-constr',
        constraints=[nlc],
        options={'maxiter': max_iter, 'verbose': 0, 'gtol': 1e-6, 'xtol': 1e-10},
    )
    wall = time.time() - t0

    # Extract final phi.
    phi_new = phi.astype(np.float64).copy()
    final_disp = res.x.reshape(n_corners, 3)
    for i, (cz, cy, cx) in enumerate(corners):
        phi_new[0, cz, cy, cx] = final_disp[i, 0]
        phi_new[1, cz, cy, cx] = final_disp[i, 1]
        phi_new[2, cz, cy, cx] = final_disp[i, 2]

    # Check post-feasibility on cluster cubes.
    V_final = constraint_values(res.x).reshape(n_cubes, 6)
    n_cube_feas = int((V_final.min(axis=1) >= threshold - 1e-7).sum())
    if verbose:
        print(
            f'  cluster ({n_cubes} cubes, {n_vars} vars): '
            f'cube_feas={n_cube_feas}/{n_cubes}  '
            f'min_V={float(V_final.min()):+.6f}  '
            f'wall={wall:.1f}s  '
            f'status={res.status}',
            flush=True,
        )
    return phi_new, {
        'n_cubes_feasible': n_cube_feas,
        'n_cubes': n_cubes,
        'min_V': float(V_final.min()),
        'status': res.status,
        'wall_s': wall,
        'L1_added': float(np.abs(res.x - x_in).sum()),
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

    # Cluster unfixable cells.
    unfix_mask = best_min0 <= 0
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
    clusters.sort(key=lambda c: -len(c))  # largest first
    print(f'{len(clusters)} clusters, sizes: {[len(c) for c in clusters[:10]]}', flush=True)

    # Solve each cluster sequentially.
    phi_new = phi.astype(np.float64).copy()
    n_total_cubes_fixed = 0
    n_total_cubes = 0
    total_L1 = 0.0
    for i, cells in enumerate(clusters):
        print(f'\n--- Cluster {i + 1}/{len(clusters)}: {len(cells)} cubes ---', flush=True)
        phi_new, info = solve_cluster_nlp(phi_new, cells, threshold=THRESHOLD, verbose=True)
        n_total_cubes_fixed += info['n_cubes_feasible']
        n_total_cubes += info['n_cubes']
        total_L1 += info['L1_added']

    # Global recheck.
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
        f'  L1 from input: {L1_final:.1f}\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_trust_constr.npy', phi_new)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
