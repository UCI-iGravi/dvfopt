"""Uncrush v2: process clusters of unfixable cells jointly with
shared-corner consistency. Each "fold column" (contiguous z-stack of
unfixable cells) gets a SINGLE uncrush direction so the geometric
expansion is coherent through z.

Hypothesis: v1 failed because cells sharing corners had different
intended deltas, which averaged out destructively. v2 processes
contiguous clusters as one geometric unit.
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

from dvfopt.jacobian.tetrahedron_sign import (
    _tet_volume_from_vertices,
    _voxel_corner_positions,
    six_tet_volumes_3d,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01

_DIAGONALS = [(0, 7), (1, 6), (2, 5), (3, 4)]


def _six_tets_for_diagonal(start, end):
    all_edges = [(v, w) for v in range(8) for w in range(v + 1, 8)
                 if (v ^ w) in (1, 2, 4)]
    perimeter = [e for e in all_edges if start not in e and end not in e]
    return [(start, a, b, end) for (a, b) in perimeter]


def _best_min_per_cell(phi):
    V_default = six_tet_volumes_3d(phi)
    dz, dy, dx = phi[0], phi[1], phi[2]
    pos_all = _voxel_corner_positions(dz, dy, dx)
    min_per_diag = np.empty((4, *V_default.shape[1:]))
    min_per_diag[0] = V_default.min(axis=0)
    z0 = np.zeros_like(dz)
    pos_id = _voxel_corner_positions(z0, z0, z0)
    for di in range(1, 4):
        s, e = _DIAGONALS[di]
        tets = _six_tets_for_diagonal(s, e)
        V_d = np.empty((6, *V_default.shape[1:]))
        for k, (i0, i1, i2, i3) in enumerate(tets):
            v_id = float(_tet_volume_from_vertices(pos_id[i0], pos_id[i1],
                                                    pos_id[i2], pos_id[i3])[0, 0, 0])
            sgn = +1.0 if v_id > 0 else -1.0
            V_d[k] = sgn * _tet_volume_from_vertices(pos_all[i0], pos_all[i1],
                                                      pos_all[i2], pos_all[i3])
        min_per_diag[di] = V_d.min(axis=0)
    return min_per_diag.max(axis=0)


def _identity_offsets():
    return np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], dtype=np.float64)


def _gather_cluster_corners(phi, cells):
    """Collect unique (z, y, x) corner indices touched by any cell in
    the cluster."""
    corners = set()
    for (z, y, x) in cells:
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            corners.add((z + iz, y + iy, x + ix))
    return sorted(corners)


def uncrush_cluster_coherent(phi, cells, target_extent=1.2):
    """Uncrush a cluster of contiguous unfixable cells with a SINGLE
    coherent geometric direction.

    Steps:
      1. Gather all unique corners (z, y, x) touched by cluster cells.
      2. Form (N, 3) matrix of deformed positions of those corners.
      3. SVD → smallest singular direction v_min for the WHOLE cluster.
      4. For each corner, compute its identity offset along v_min.
      5. Push by sign(proj) * push_magnitude * v_min.

    Returns modified phi (only cluster corners updated)."""
    corners = _gather_cluster_corners(phi, cells)
    n_corners = len(corners)
    if n_corners < 4:
        return phi, 0.0
    # Deformed positions of corners.
    pos = np.zeros((n_corners, 3))
    id_pos = np.zeros((n_corners, 3))
    for k, (cz, cy, cx) in enumerate(corners):
        id_pos[k] = (cz, cy, cx)
        pos[k, 0] = cz + phi[0, cz, cy, cx]
        pos[k, 1] = cy + phi[1, cz, cy, cx]
        pos[k, 2] = cx + phi[2, cz, cy, cx]
    centroid = pos.mean(axis=0)
    P = pos - centroid
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    sigma_min = float(S[-1])
    v_min = Vt[-1, :]
    # Identity centroid.
    id_centroid = id_pos.mean(axis=0)
    id_centered = id_pos - id_centroid
    proj = id_centered @ v_min
    # Push magnitude: enough to bring smallest cluster extent to target_extent.
    push = (target_extent - sigma_min) / 2.0
    if push <= 0:
        return phi, 0.0
    # Per-corner delta.
    delta_pos = np.outer(np.sign(proj), v_min) * push
    # Apply: phi[c, corner] += delta_pos[corner, c].
    phi_new = phi.astype(np.float64).copy()
    l1_cost = 0.0
    for k, (cz, cy, cx) in enumerate(corners):
        for c in range(3):
            phi_new[c, cz, cy, cx] += delta_pos[k, c]
            l1_cost += abs(delta_pos[k, c])
    return phi_new, l1_cost


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    print(f'Loaded, shape={phi.shape}', flush=True)

    best_min = _best_min_per_cell(phi)
    unfix_mask = (best_min <= 0)
    nz, ny, nx = np.where(unfix_mask)
    print(f'Start: {len(nz)} unfixable cells', flush=True)

    # Cluster the unfixable cells. Dilate by 1 to merge nearby.
    cluster_grid = np.zeros(unfix_mask.shape, dtype=bool)
    cluster_grid[nz, ny, nx] = True
    cluster_grid = binary_dilation(cluster_grid, iterations=1)
    labels, n_comp = cc_label(cluster_grid)
    # For each component, gather the ORIGINAL unfixable cells (not dilated).
    clusters = []
    for cid in range(1, n_comp + 1):
        in_comp = (labels == cid) & unfix_mask
        cz, cy, cx = np.where(in_comp)
        cells = list(zip(cz.tolist(), cy.tolist(), cx.tolist()))
        if cells:
            clusters.append(cells)
    print(f'  partitioned into {len(clusters)} clusters', flush=True)
    sizes = sorted([len(c) for c in clusters], reverse=True)
    print(f'  cluster sizes (top 10): {sizes[:10]}', flush=True)

    print('\n=== Coherent cluster uncrush, target_extent=1.2 ===', flush=True)
    phi_new = phi.astype(np.float64).copy()
    total_l1 = 0.0
    for cells in clusters:
        phi_new, cost = uncrush_cluster_coherent(phi_new, cells, target_extent=1.2)
        total_l1 += cost
    print(f'  total L1 cost: {total_l1:.1f}', flush=True)
    V_after = six_tet_volumes_3d(phi_new)
    print(
        f'After uncrush:  n_neg={int((V_after <= 0).sum())}  '
        f'n<0.01={int((V_after < THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V_after.min()):+.6f}  '
        f'L1_from_orig={float(np.abs(phi_new - phi).sum()):.1f}',
        flush=True,
    )

    # Polish with M10Tet @ 0.015.
    print('\n=== M10Tet polish @ threshold=0.015 ===', flush=True)
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )
    t0 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi_new.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    phi_polished = solver.fit(phi_new).corrected
    wall = time.time() - t0
    V_final = six_tet_volumes_3d(phi_polished)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1_final = float(np.abs(phi_polished - phi).sum())
    print(
        f'\nFinal:  n_neg={n_neg}  n<0.01={n_below}  '
        f'min_T={float(V_final.min()):+.6f}  L1_from_orig={L1_final:.1f}  '
        f'polish_wall={wall:.1f}s\n'
        f'  STRICT 100% feasible: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_uncrush_v2.npy', phi_polished)
        print('  *** Saved strict-feasible result via v2. ***', flush=True)


if __name__ == '__main__':
    main()
