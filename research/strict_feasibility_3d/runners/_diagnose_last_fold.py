"""Diagnose the single remaining fold cell and try to push it
above 0 via targeted local methods.

Inputs:
  final_push_best.npy : n_neg=1, min_T=-0.000146

Steps:
  1. Locate the (cz, cy, cx) of the folded cell.
  2. Print the 6 tet volumes under the default decomposition.
  3. Check all 4 main-diagonal decompositions of that one cube:
     does any give all-positive tets? If yes, this fold is
     "removable via triangulation choice" - a soft feasibility.
  4. Try a tight local SLSQP solve on just that cell's 8 corners
     with the surrounding ring frozen.
  5. Save result.
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

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    _tet_volume_from_vertices,
    _voxel_corner_positions,
    six_tet_volumes_3d,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01

_DIAGONALS = [(0, 7), (1, 6), (2, 5), (3, 4)]


def _six_tets_for_diagonal(start, end):
    all_edges = [(v, w) for v in range(8) for w in range(v + 1, 8) if (v ^ w) in (1, 2, 4)]
    perimeter = [e for e in all_edges if start not in e and end not in e]
    return [(start, a, b, end) for (a, b) in perimeter]


def cube_tet_vols(phi, cz, cy, cx, diagonal_idx=0):
    pos = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        pos[i, 0] = (cz + iz) + phi[0, cz + iz, cy + iy, cx + ix]
        pos[i, 1] = (cy + iy) + phi[1, cz + iz, cy + iy, cx + ix]
        pos[i, 2] = (cx + ix) + phi[2, cz + iz, cy + iy, cx + ix]
    s, e = _DIAGONALS[diagonal_idx]
    if diagonal_idx == 0:
        tets = _TET_VERTICES
        signs = _TET_SIGN
    else:
        tets = _six_tets_for_diagonal(s, e)
        # Compute identity volumes to get signs.
        pos_id = np.zeros((8, 3))
        for i in range(8):
            pos_id[i, 0] = (i >> 2) & 1
            pos_id[i, 1] = (i >> 1) & 1
            pos_id[i, 2] = i & 1
        signs = []
        for i0, i1, i2, i3 in tets:
            v_id = _signed_vol(pos_id[i0], pos_id[i1], pos_id[i2], pos_id[i3])
            signs.append(+1.0 if v_id > 0 else -1.0)
    vols = np.empty(6)
    for k, (i0, i1, i2, i3) in enumerate(tets):
        vols[k] = signs[k] * _signed_vol(pos[i0], pos[i1], pos[i2], pos[i3])
    return vols, pos


def _signed_vol(A, B, C, D):
    AB = B - A
    AC = C - A
    AD = D - A
    return (
        AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
        - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
        + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])
    ) / 6.0


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    cur = np.load(OUTPUT / 'b0039_z0_15_final_push_best.npy').astype(np.float64)
    V = six_tet_volumes_3d(cur)
    print(
        f'Loaded: n_neg={int((V <= 0).sum())}  '
        f'n<0.01={int((V < 0.01 - 1e-5).sum())}  '
        f'min_T={float(V.min()):+.6f}',
        flush=True,
    )

    # Find the fold cell.
    min_per_cell = V.min(axis=0)
    folded = min_per_cell <= 0
    cells = list(zip(*np.where(folded)))
    print(f'\nFolded cells: {len(cells)}', flush=True)
    for ci, (cz, cy, cx) in enumerate(cells):
        print(
            f'  cell {ci}: (z, y, x) = ({cz}, {cy}, {cx})  min_T = {min_per_cell[cz, cy, cx]:+.6f}',
            flush=True,
        )

    if not cells:
        print('No folded cells.', flush=True)
        return

    cz, cy, cx = cells[0]

    # Show 6 tet vols under each of 4 diagonal decompositions.
    print(f'\n=== Cube ({cz}, {cy}, {cx}) under 4 diagonal triangulations ===', flush=True)
    best_diag = None
    best_min = -float('inf')
    for di in range(4):
        vols, pos = cube_tet_vols(cur, cz, cy, cx, di)
        min_v = float(vols.min())
        print(
            f'  diagonal {_DIAGONALS[di]}: tet vols = {vols.round(6).tolist()}  min = {min_v:+.6f}',
            flush=True,
        )
        if min_v > best_min:
            best_min = min_v
            best_diag = di
    print(f'\n  BEST diagonal: {_DIAGONALS[best_diag]}  min_T = {best_min:+.6f}', flush=True)
    if best_min > 0:
        print(f'  *** Cell IS feasible under diagonal {_DIAGONALS[best_diag]} ***', flush=True)
    else:
        print('  Cell INFEASIBLE under all 4 diagonals — combinatorial obstruction.', flush=True)

    # Print the 8 corner positions to understand geometry.
    print(f'\n=== Cell ({cz}, {cy}, {cx}) corner positions ===', flush=True)
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        z, y, x = cz + iz, cy + iy, cx + ix
        dz = cur[0, z, y, x]
        dy = cur[1, z, y, x]
        dx = cur[2, z, y, x]
        print(
            f'  corner {i} (lattice {z}, {y}, {x}): displacement '
            f'(dz, dy, dx) = ({dz:+.4f}, {dy:+.4f}, {dx:+.4f})  '
            f'position = ({z + dz:+.4f}, {y + dy:+.4f}, {x + dx:+.4f})',
            flush=True,
        )

    # Tight local solve: optimize this cell's 8 corner displacements
    # to maximize min(tet vols) while staying close to current values.
    print(f'\n=== Local SLSQP on cell ({cz}, {cy}, {cx}) ===', flush=True)
    # Decision vars: 24 (8 corners * 3 channels). Keep all neighbors frozen.
    corner_idx = []
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        corner_idx.append((cz + iz, cy + iy, cx + ix))

    x0 = np.zeros(24)
    for ci, (z, y, x) in enumerate(corner_idx):
        x0[3 * ci + 0] = cur[0, z, y, x]
        x0[3 * ci + 1] = cur[1, z, y, x]
        x0[3 * ci + 2] = cur[2, z, y, x]

    def get_pos(x):
        pos = np.zeros((8, 3))
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            pos[i, 0] = (cz + iz) + x[3 * i + 0]
            pos[i, 1] = (cy + iy) + x[3 * i + 1]
            pos[i, 2] = (cx + ix) + x[3 * i + 2]
        return pos

    def get_vols(x, di=0):
        pos = get_pos(x)
        s, e = _DIAGONALS[di]
        if di == 0:
            tets = _TET_VERTICES
            signs = _TET_SIGN
        else:
            tets = _six_tets_for_diagonal(s, e)
            pos_id = np.zeros((8, 3))
            for i in range(8):
                pos_id[i, 0] = (i >> 2) & 1
                pos_id[i, 1] = (i >> 1) & 1
                pos_id[i, 2] = i & 1
            signs = [
                (+1.0 if _signed_vol(pos_id[i0], pos_id[i1], pos_id[i2], pos_id[i3]) > 0 else -1.0)
                for (i0, i1, i2, i3) in tets
            ]
        vols = np.empty(6)
        for k, (i0, i1, i2, i3) in enumerate(tets):
            vols[k] = signs[k] * _signed_vol(pos[i0], pos[i1], pos[i2], pos[i3])
        return vols

    def objective(x):
        return float(np.sum((x - x0) ** 2))

    def all_diag_constraints(x):
        # Each diagonal gives 6 tet vols; require best diagonal to be all > 0.011.
        # But we can use the "any diagonal works" approach by maximizing
        # min over the 4 diagonals separately. For simplicity, just enforce
        # diagonal 0 (default) to be all positive.
        v0 = get_vols(x, 0)
        return v0 - 0.011  # Each tet >= 0.011

    cons = [{'type': 'ineq', 'fun': all_diag_constraints}]
    print('  Initial diagonal-0 tet volumes:', get_vols(x0, 0).round(6).tolist(), flush=True)
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=cons,
        options={'maxiter': 200, 'ftol': 1e-9, 'disp': True},
    )
    print(f'  SLSQP success={res.success}  fun={res.fun:.6f}  message={res.message}', flush=True)
    print(f'  Final diagonal-0 tet volumes: {get_vols(res.x, 0).round(6).tolist()}', flush=True)

    # Apply the result and re-check globally.
    out = cur.copy()
    for ci, (z, y, x) in enumerate(corner_idx):
        out[0, z, y, x] = res.x[3 * ci + 0]
        out[1, z, y, x] = res.x[3 * ci + 1]
        out[2, z, y, x] = res.x[3 * ci + 2]
    V_out = six_tet_volumes_3d(out)
    n_neg = int((V_out <= 0).sum())
    n_below = int((V_out < THRESHOLD - 1e-5).sum())
    print('\n=== After local solve, global check ===', flush=True)
    print(f'  n_neg={n_neg}  n<0.01={n_below}  min_T={float(V_out.min()):+.6f}', flush=True)
    if n_neg == 0 and n_below == 0:
        print('  *** STRICT 100% FEASIBLE ***', flush=True)
        np.save(OUTPUT / 'b0039_z0_15_strict_via_local.npy', out)


if __name__ == '__main__':
    main()
