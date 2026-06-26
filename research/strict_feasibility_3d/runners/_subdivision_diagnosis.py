"""Subdivision diagnostic: for each unfixable cell, subdivide into
K^3 sub-cubes (trilinear-interp sub-corners from original 8) and
check sub-cube 6-tet feasibility.

Two questions answered:

  1. Does the fold occupy the WHOLE cube, or is it localized?
     If most sub-cubes are feasible, the fold is a small region —
     subdivision + local modification could remove it.

  2. Does increasing K help? At K=2: 8 sub-cubes per cell. At K=4:
     64. If feasibility rate grows with K, subdivision is a real
     path forward. If it doesn't (fold pervades), subdivision
     can't resolve these cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    _tet_volume_from_vertices,
    _voxel_corner_positions,
    six_tet_volumes_3d,
)
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell

OUTPUT = _HERE / 'output'


def _trilinear_interp_corners(disp_8, K):
    """Given 8 corner displacements (8, 3), return (K+1)^3 sub-corner
    displacements via trilinear interpolation.

    Indexing: sub_disp[i, j, k] = disp at sub-corner (i/K, j/K, k/K)
    in cube-local coords. Sub-corners (0, *, *) lie on the original
    z=0 face, (K, *, *) on the original z=1 face, etc.
    """
    out = np.zeros((K + 1, K + 1, K + 1, 3))
    for i in range(K + 1):
        u = i / K
        for j in range(K + 1):
            v = j / K
            for k in range(K + 1):
                w = k / K
                # Trilinear basis weights for 8 corners.
                for c in range(8):
                    iz = (c >> 2) & 1
                    iy = (c >> 1) & 1
                    ix = c & 1
                    wz = u if iz else (1 - u)
                    wy = v if iy else (1 - v)
                    wx = w if ix else (1 - w)
                    out[i, j, k] += wz * wy * wx * disp_8[c]
    return out


def _sub_cube_six_tet(sub_disp, sub_ref):
    """Given 8 corner displacements + identity positions for a SINGLE
    sub-cube (both (8, 3)), compute the 6 tet volumes."""
    # Deformed positions.
    pos = sub_ref + sub_disp
    out = np.empty(6)
    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        A, B, C, Dv = pos[i0], pos[i1], pos[i2], pos[i3]
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


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    best_min = _best_min_per_cell(phi)
    unfix_mask = best_min <= 0
    nz, ny, nx = np.where(unfix_mask)
    print(f'{len(nz)} unfixable cells\n', flush=True)

    for K in [2, 4, 8]:
        print(f'--- K={K} ({K**3} sub-cubes per cell) ---', flush=True)
        n_cells_fully_feasible = 0
        total_sub = 0
        n_sub_feasible = 0
        per_cell_sub_feas_frac = []
        for ci in range(len(nz)):
            z, y, x = int(nz[ci]), int(ny[ci]), int(nx[ci])
            # Gather 8 original corner displacements.
            disp_8 = np.zeros((8, 3))
            for i in range(8):
                iz = (i >> 2) & 1
                iy = (i >> 1) & 1
                ix = i & 1
                disp_8[i, 0] = phi[0, z + iz, y + iy, x + ix]
                disp_8[i, 1] = phi[1, z + iz, y + iy, x + ix]
                disp_8[i, 2] = phi[2, z + iz, y + iy, x + ix]
            # Trilinear-interpolate to (K+1)^3 sub-corner displacements.
            sub_disp_grid = _trilinear_interp_corners(disp_8, K)  # (K+1, K+1, K+1, 3)
            # For each sub-cube, compute 6-tet volumes.
            n_cell_sub_feasible = 0
            for sz in range(K):
                for sy in range(K):
                    for sx in range(K):
                        # 8 sub-corners' displacements + identity positions.
                        sub_disp = np.zeros((8, 3))
                        sub_ref = np.zeros((8, 3))
                        for i in range(8):
                            iz = (i >> 2) & 1
                            iy = (i >> 1) & 1
                            ix = i & 1
                            sub_disp[i] = sub_disp_grid[sz + iz, sy + iy, sx + ix]
                            sub_ref[i] = (
                                z + (sz + iz) / K,
                                y + (sy + iy) / K,
                                x + (sx + ix) / K,
                            )
                        V = _sub_cube_six_tet(sub_disp, sub_ref)
                        if V.min() > 0:
                            n_cell_sub_feasible += 1
            per_cell_sub_feas_frac.append(n_cell_sub_feasible / (K**3))
            n_sub_feasible += n_cell_sub_feasible
            total_sub += K**3
            if n_cell_sub_feasible == K**3:
                n_cells_fully_feasible += 1
        per_cell_arr = np.array(per_cell_sub_feas_frac)
        print(
            f'  cells fully sub-feasible:                  {n_cells_fully_feasible}/{len(nz)}\n'
            f'  total sub-cubes feasible:                  {n_sub_feasible}/{total_sub}  '
            f'({n_sub_feasible / total_sub * 100:.1f}%)\n'
            f'  per-cell sub-feasibility fraction stats:\n'
            f'    mean   = {per_cell_arr.mean() * 100:.1f}%\n'
            f'    median = {float(np.median(per_cell_arr)) * 100:.1f}%\n'
            f'    worst  = {per_cell_arr.min() * 100:.1f}%\n'
            f'    best   = {per_cell_arr.max() * 100:.1f}%',
            flush=True,
        )


if __name__ == '__main__':
    main()
