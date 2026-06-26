"""H7: Test whether local Gaussian smoothing fixes the 94 "unavoidable" folds.

Hypothesis: If a cube has all 6 tets folded but a smoothed version
of its (dy, dx, dz) corners is feasible, the fold is a
discretization artifact (case B), not a genuine continuous fold
(case A). Smoothing locally can fix case B at a small L1 cost.

For each fold cell:
  1. Apply a 3x3x3 average over the (dz, dy, dx) at its 8 corners
  2. Check the 6 tets of the smoothed cube
  3. If feasible, smoothing fixes this cell

Reports the case-A / case-B split. If many cells are case B,
local smoothing pass should resolve them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.ndimage import uniform_filter

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    V = six_tet_volumes_3d(phi)
    fold_mask = V.min(axis=0) <= 0
    n_folds = int(fold_mask.sum())
    print(f'Input: {n_folds} fold cells', flush=True)

    # Apply 3x3x3 box-filter smoothing to each channel of phi.
    phi_smoothed = np.empty_like(phi)
    for c in range(3):
        phi_smoothed[c] = uniform_filter(phi[c], size=3, mode='nearest')

    V_sm = six_tet_volumes_3d(phi_smoothed)
    fold_mask_sm = V_sm.min(axis=0) <= 0
    n_folds_sm = int(fold_mask_sm.sum())
    L1_sm = float(np.abs(phi_smoothed - phi).sum())
    print(
        f'After 3x3x3 smoothing: {n_folds_sm} fold cells  (L1_from_input={L1_sm:.1f})',
        flush=True,
    )

    # Test stronger smoothing: 5x5x5.
    phi_smoothed5 = np.empty_like(phi)
    for c in range(3):
        phi_smoothed5[c] = uniform_filter(phi[c], size=5, mode='nearest')
    V_sm5 = six_tet_volumes_3d(phi_smoothed5)
    n_folds_sm5 = int((V_sm5.min(axis=0) <= 0).sum())
    L1_sm5 = float(np.abs(phi_smoothed5 - phi).sum())
    print(
        f'After 5x5x5 smoothing: {n_folds_sm5} fold cells  (L1_from_input={L1_sm5:.1f})',
        flush=True,
    )

    # 7x7x7
    phi_smoothed7 = np.empty_like(phi)
    for c in range(3):
        phi_smoothed7[c] = uniform_filter(phi[c], size=7, mode='nearest')
    V_sm7 = six_tet_volumes_3d(phi_smoothed7)
    n_folds_sm7 = int((V_sm7.min(axis=0) <= 0).sum())
    L1_sm7 = float(np.abs(phi_smoothed7 - phi).sum())
    print(
        f'After 7x7x7 smoothing: {n_folds_sm7} fold cells  (L1_from_input={L1_sm7:.1f})',
        flush=True,
    )

    # Selective smoothing: only at fold cells. Replace each fold cube's
    # 8 corners with the local 5x5x5 average of those corners.
    print('\nSelective smoothing (only fold cells, 5x5x5 average):', flush=True)
    phi_selective = phi.copy()
    fold_z, fold_y, fold_x = np.where(fold_mask)
    for k in range(len(fold_z)):
        z, y, x = int(fold_z[k]), int(fold_y[k]), int(fold_x[k])
        # Average 8 corners of this fold cube.
        for cz, cy, cx in [
            (z, y, x),
            (z, y, x + 1),
            (z, y + 1, x),
            (z, y + 1, x + 1),
            (z + 1, y, x),
            (z + 1, y, x + 1),
            (z + 1, y + 1, x),
            (z + 1, y + 1, x + 1),
        ]:
            # Skip if out of bounds.
            if (
                cz < 0
                or cy < 0
                or cx < 0
                or cz >= phi.shape[1]
                or cy >= phi.shape[2]
                or cx >= phi.shape[3]
            ):
                continue
            for c in range(3):
                z0 = max(0, cz - 2)
                z1 = min(phi.shape[1], cz + 3)
                y0 = max(0, cy - 2)
                y1 = min(phi.shape[2], cy + 3)
                x0 = max(0, cx - 2)
                x1 = min(phi.shape[3], cx + 3)
                phi_selective[c, cz, cy, cx] = float(phi[c, z0:z1, y0:y1, x0:x1].mean())
    V_sel = six_tet_volumes_3d(phi_selective)
    n_folds_sel = int((V_sel.min(axis=0) <= 0).sum())
    L1_sel = float(np.abs(phi_selective - phi).sum())
    print(
        f'  result: {n_folds_sel} folds  L1_from_input={L1_sel:.1f}',
        flush=True,
    )

    # And: smoothing creates folds elsewhere? Compare counts.
    print('\nNote: smoothing may CREATE new folds at high-shear regions.', flush=True)


if __name__ == '__main__':
    main()
