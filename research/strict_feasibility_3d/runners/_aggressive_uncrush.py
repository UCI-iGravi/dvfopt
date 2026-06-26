"""Aggressive uncrush: try target_extent in {2, 3, 5} to find the
smallest expansion that flips the continuous Jacobian positive at
all cells.

Different from v1/v2: instead of relying on M10Tet to polish after,
this directly checks det(J) at sampled internal points and reports
how many cells are GENUINELY uncrushed (det(J) > 0 everywhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.ndimage import label as cc_label, binary_dilation

from dvfopt.jacobian.tetrahedron_sign import (
    _voxel_corner_positions,
    _tet_volume_from_vertices,
    six_tet_volumes_3d,
)

# Reuse v2's logic.
from research.strict_feasibility_3d.runners._uncrush_v2 import (
    uncrush_cluster_coherent,
    _best_min_per_cell,
)
from research.strict_feasibility_3d.runners._jacobian_diagnosis import (
    _trilinear_jacobian_det,
)


OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def _check_cell_continuous(phi, z, y, x, samples=5):
    """Return min det(J) sampled inside cell (z, y, x)."""
    ref = np.zeros((8, 3))
    disp = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        ref[i] = (z + iz, y + iy, x + ix)
        disp[i, 0] = phi[0, z + iz, y + iy, x + ix]
        disp[i, 1] = phi[1, z + iz, y + iy, x + ix]
        disp[i, 2] = phi[2, z + iz, y + iy, x + ix]
    det_grid = _trilinear_jacobian_det(disp, ref, samples_per_dim=samples)
    return float(det_grid.min())


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    best_min = _best_min_per_cell(phi)
    unfix_mask = (best_min <= 0)
    nz, ny, nx = np.where(unfix_mask)
    cells = list(zip(nz.tolist(), ny.tolist(), nx.tolist()))
    print(f'{len(cells)} unfixable cells', flush=True)

    # Cluster.
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
    print(f'{len(clusters)} clusters', flush=True)

    # Sweep target_extent.
    for tx in [1.5, 2.0, 3.0, 5.0]:
        print(f'\n--- target_extent={tx} ---', flush=True)
        phi_new = phi.astype(np.float64).copy()
        total_l1 = 0.0
        for cells in clusters:
            phi_new, cost = uncrush_cluster_coherent(phi_new, cells, target_extent=tx)
            total_l1 += cost

        # Check discrete 6-tet.
        V = six_tet_volumes_3d(phi_new)
        n_neg = int((V <= 0).sum())
        n_below = int((V < THRESHOLD - 1e-5).sum())
        best_min_new = _best_min_per_cell(phi_new)
        n_unfix_new = int((best_min_new <= 0).sum())

        # Check continuous det(J) at originally-unfixable cells.
        n_continuously_fixed = 0
        cells_still_negdet = []
        for (cz, cy, cx) in zip(nz, ny, nx):
            mn = _check_cell_continuous(phi_new, int(cz), int(cy), int(cx))
            if mn > 0:
                n_continuously_fixed += 1
            else:
                cells_still_negdet.append((int(cz), int(cy), int(cx), mn))

        L1 = float(np.abs(phi_new - phi).sum())
        print(
            f'  uncrush total L1: {total_l1:.1f}  applied L1: {L1:.1f}\n'
            f'  discrete: n_neg={n_neg}  n<0.01={n_below}  '
            f'unfixable={n_unfix_new}  min_T={float(V.min()):+.6f}\n'
            f'  continuous: {n_continuously_fixed}/{len(nz)} of original unfixable '
            f'cells now have det(J) > 0 everywhere',
            flush=True,
        )
        if cells_still_negdet:
            min_mns = sorted([c[3] for c in cells_still_negdet])
            print(f'  remaining-bad-det stats: worst={min_mns[0]:.4f}  median={min_mns[len(min_mns)//2]:.4f}', flush=True)
        else:
            print(f'  *** ALL originally-unfixable cells now have positive continuous Jacobian ***', flush=True)


if __name__ == '__main__':
    main()
