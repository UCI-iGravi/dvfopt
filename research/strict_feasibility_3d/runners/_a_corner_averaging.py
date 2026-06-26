"""Strategy A: Local corner averaging (cube-flattening) on unfixable
3D fold cells.

For each cell whose 6-tet check fails under every diagonal choice,
replace the 8 corner displacements with their average. The local
cube becomes an exact identity unit cube (a pure translation),
so all 6 tets equal +1/6.

Iterate to handle neighbour conflicts: modifying a shared corner
can break a neighbour cube. After each pass, recompute folds and
re-target the still-unfixable cells.

Reports L1 cost and convergence per pass.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
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
    """Return min_T across all 4 cube diagonals, per cell."""
    V_default = six_tet_volumes_3d(phi)
    dz, dy, dx = phi[0], phi[1], phi[2]
    pos_all = _voxel_corner_positions(dz, dy, dx)
    min_per_diag = np.empty((4, *V_default.shape[1:]))
    min_per_diag[0] = V_default.min(axis=0)
    D, H, W = dz.shape
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


def corner_avg_fix(phi, target_threshold=THRESHOLD, max_passes=5, mode='unfixable_only'):
    """Apply local corner averaging at unfixable cells.

    mode: 'unfixable_only' — only target cells where best-diagonal
                              min_T <= 0
          'all_folds' — target all cells with default-diagonal min_T <= 0
    """
    phi_out = phi.astype(np.float64).copy()
    phi_orig = phi.astype(np.float64)
    D_phi, H_phi, W_phi = phi.shape[1:]

    for pass_idx in range(max_passes):
        if mode == 'unfixable_only':
            best_min = _best_min_per_cell(phi_out)
            target_mask = (best_min <= 0)
        else:
            V = six_tet_volumes_3d(phi_out)
            target_mask = (V.min(axis=0) <= 0)
        n_target = int(target_mask.sum())
        if n_target == 0:
            print(f'  pass {pass_idx}: 0 target cells, done.', flush=True)
            break
        print(f'  pass {pass_idx}: {n_target} target cells', flush=True)

        # Average corner displacements per target cell. Use accumulators
        # for conflicts (multiple target cells writing the same corner).
        sum_buf = np.zeros_like(phi_out)
        count_buf = np.zeros(phi_out.shape[1:], dtype=np.int32)
        nz, ny, nx = np.where(target_mask)
        for k in range(len(nz)):
            z, y, x = int(nz[k]), int(ny[k]), int(nx[k])
            # 8 corners of cell (z, y, x).
            corners = [(z + (i >> 2) & 1, y + (i >> 1) & 1, x + (i & 1)) for i in range(8)]
            # Actually fix the indexing:
            corners = []
            for i in range(8):
                oz = (i >> 2) & 1
                oy = (i >> 1) & 1
                ox = i & 1
                corners.append((z + oz, y + oy, x + ox))
            # Mean displacement across the 8 corners.
            mean_disp = np.zeros(3)
            for cz, cy, cx in corners:
                mean_disp += phi_out[:, cz, cy, cx]
            mean_disp /= 8.0
            # Write mean to each corner (accumulate).
            for cz, cy, cx in corners:
                sum_buf[:, cz, cy, cx] += mean_disp
                count_buf[cz, cy, cx] += 1
        # Apply averaged means where count > 0.
        mask = count_buf > 0
        new_disp = sum_buf[:, mask] / count_buf[mask]
        for c in range(3):
            phi_out[c][mask] = new_disp[c]

        # Recompute stats.
        V = six_tet_volumes_3d(phi_out)
        n_neg = int((V <= 0).sum())
        n_below = int((V < target_threshold - 1e-5).sum())
        L1 = float(np.abs(phi_out - phi_orig).sum())
        best_min_after = _best_min_per_cell(phi_out)
        n_unfixable_after = int((best_min_after <= 0).sum())
        print(
            f'    after pass: n_neg={n_neg}  n<{target_threshold}={n_below}  '
            f'unfixable={n_unfixable_after}  min_T={float(V.min()):+.6f}  '
            f'L1_from_orig={L1:.1f}',
            flush=True,
        )
    return phi_out


def main():
    cache = OUTPUT / 'b0039_FULL_stage3_z000_016.npy'
    phi = np.load(cache)
    print(f'Loaded {cache}  shape={phi.shape}', flush=True)
    V = six_tet_volumes_3d(phi)
    best_min = _best_min_per_cell(phi)
    print(
        f'Start:  default n_neg={int((V<=0).sum())}  '
        f'unfixable (any-diag-fails)={int((best_min<=0).sum())}  '
        f'min_T={float(V.min()):+.6f}',
        flush=True,
    )

    print('\n=== Strategy A: corner averaging on UNFIXABLE cells only ===', flush=True)
    phi_unfix = corner_avg_fix(phi, mode='unfixable_only', max_passes=5)
    V_unfix = six_tet_volumes_3d(phi_unfix)
    print(
        f'\nUnfixable-only result:\n'
        f'  n_neg={int((V_unfix<=0).sum())}\n'
        f'  n<0.01={int((V_unfix<THRESHOLD-1e-5).sum())}\n'
        f'  min_T={float(V_unfix.min()):+.6f}\n'
        f'  L1_from_orig={float(np.abs(phi_unfix - phi).sum()):.1f}',
        flush=True,
    )

    print('\n=== Strategy A variant: corner averaging on ALL fold cells ===', flush=True)
    phi_all = corner_avg_fix(phi, mode='all_folds', max_passes=8)
    V_all = six_tet_volumes_3d(phi_all)
    print(
        f'\nAll-folds result:\n'
        f'  n_neg={int((V_all<=0).sum())}\n'
        f'  n<0.01={int((V_all<THRESHOLD-1e-5).sum())}\n'
        f'  min_T={float(V_all.min()):+.6f}\n'
        f'  L1_from_orig={float(np.abs(phi_all - phi).sum()):.1f}',
        flush=True,
    )


if __name__ == '__main__':
    main()
