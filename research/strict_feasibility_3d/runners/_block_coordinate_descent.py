"""Block coordinate descent: search each corner's (dz, dy, dx)
to maximize minimum tet volume of its 8 surrounding cubes.

Unlike Strategy D (per-cell SLSQP with L1 objective), this:
  - Operates on CORNERS not cells
  - Uses FEASIBILITY-FIRST objective (max min_V)
  - Searches in a local neighborhood of the current corner value
  - Cycles through fold-zone corners in Gauss-Seidel order
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    six_tet_volumes_3d,
)
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def _eight_cubes_touching_corner(z, y, x, D_c, H_c, W_c):
    """Return list of cube (cz, cy, cx) indices that have (z, y, x) as one
    of their 8 corners. Cubes are indexed by their (0,0,0) corner."""
    cubes = []
    for dz in (-1, 0):
        for dy in (-1, 0):
            for dx in (-1, 0):
                cz, cy, cx = z + dz, y + dy, x + dx
                if 0 <= cz < D_c and 0 <= cy < H_c and 0 <= cx < W_c:
                    cubes.append((cz, cy, cx))
    return cubes


def _cube_min_V(phi, cz, cy, cx):
    """Min of 6 tet volumes for cube (cz, cy, cx). Used as objective."""
    pos = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        pos[i, 0] = (cz + iz) + phi[0, cz + iz, cy + iy, cx + ix]
        pos[i, 1] = (cy + iy) + phi[1, cz + iz, cy + iy, cx + ix]
        pos[i, 2] = (cx + ix) + phi[2, cz + iz, cy + iy, cx + ix]
    out = np.empty(6)
    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        A, B, C, Dv = pos[i0], pos[i1], pos[i2], pos[i3]
        AB = B - A
        AC = C - A
        AD = Dv - A
        det = (AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
               - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
               + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0]))
        out[k] = float(_TET_SIGN[k]) * det / 6.0
    return float(out.min())


def search_corner_3d(phi, z, y, x, search_radius=2.0):
    """For corner (z, y, x), find (dz, dy, dx) that maximizes the min
    tet volume across its 8 surrounding cubes. Returns new (dz, dy, dx).
    """
    D_c, H_c, W_c = phi.shape[1] - 1, phi.shape[2] - 1, phi.shape[3] - 1
    cubes = _eight_cubes_touching_corner(z, y, x, D_c, H_c, W_c)
    if not cubes:
        return phi[:, z, y, x].copy()
    # Current value.
    cur = phi[:, z, y, x].copy()

    def neg_min_V(disp_3):
        # Set this corner to disp_3 and evaluate min_V across cubes.
        phi_tmp = phi  # Reference (we'll restore)
        old = phi[:, z, y, x].copy()
        phi_tmp[:, z, y, x] = disp_3
        try:
            min_V = min(_cube_min_V(phi_tmp, cz, cy, cx) for (cz, cy, cx) in cubes)
        finally:
            phi_tmp[:, z, y, x] = old
        # Maximize min_V → minimize -min_V. Add small L1 penalty for stability.
        l1 = float(np.abs(disp_3 - cur).sum())
        return -min_V + 1e-6 * l1

    # Use scipy minimize starting from current value, bounded search.
    bounds = [(cur[c] - search_radius, cur[c] + search_radius) for c in range(3)]
    res = minimize(
        neg_min_V, cur,
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-7},
    )
    return res.x.copy()


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    V0 = six_tet_volumes_3d(phi)
    print(
        f'Start: n_neg={int((V0 <= 0).sum())}  '
        f'n<0.01={int((V0 < THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    best_min = _best_min_per_cell(phi)
    unfix_mask = (best_min <= 0)
    # Get all unique CORNERS touched by unfixable cubes.
    corners = set()
    for (z, y, x) in zip(*np.where(unfix_mask)):
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            corners.add((int(z + iz), int(y + iy), int(x + ix)))
    corners = sorted(corners)
    print(f'{len(corners)} corners in fold zone to optimize', flush=True)

    # Iterate Gauss-Seidel sweeps.
    phi_cur = phi.copy()
    prev_n_neg = int((V0 <= 0).sum())
    for sweep in range(10):
        t0 = time.time()
        for ci, (z, y, x) in enumerate(corners):
            new_val = search_corner_3d(phi_cur, z, y, x)
            phi_cur[:, z, y, x] = new_val
        V = six_tet_volumes_3d(phi_cur)
        n_neg = int((V <= 0).sum())
        n_below = int((V < THRESHOLD - 1e-5).sum())
        L1 = float(np.abs(phi_cur - phi).sum())
        print(
            f'Sweep {sweep+1}: n_neg={n_neg}  n<0.01={n_below}  '
            f'min_T={float(V.min()):+.6f}  L1+={L1:.1f}  wall={time.time()-t0:.1f}s',
            flush=True,
        )
        if n_neg == 0 and n_below == 0:
            print('*** STRICT 100% feasible ***', flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_block_coord.npy', phi_cur)
            break
        if n_neg >= prev_n_neg:
            print('No progress, stopping.', flush=True)
            break
        prev_n_neg = n_neg


if __name__ == '__main__':
    main()
