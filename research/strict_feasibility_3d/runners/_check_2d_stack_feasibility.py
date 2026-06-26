"""Check whether 2D auto_slp per-slice → stack → 3D is strictly
feasible under the 6-tetrahedron constraint.

B0039 has dz=0 throughout (Laplacian extension of 2D registration
data, stacked along z). The 2D auto_slp achieves 100% per-slice
2-triangle feasibility. Open question: does that imply 6-tet
feasibility on the stacked 3D field?

Mathematically: pure z-layer tets (3 vertices at z=k) match the 2D
T1/T2 of that slice — so per-slice feasibility implies feasibility
of 2 of the 6 tets per cube. The remaining 4 are straddling tets
whose sign depends on (dy(z+1) - dy(z)) and (dx(z+1) - dx(z)). If
adjacent slices have similar corrections, the straddling tets are
small perturbations of the layer tets and stay positive. If
adjacent slices have very different corrections, they may flip.

This script runs the test on z=0..20 (the densest part of B0039).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from research.strict_feasibility_2d.runners._compare import run_method as run_2d


def main():
    print('Loading B0039...', flush=True)
    arr = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr'].astype(np.float64)
    print(f'  full shape: {arr.shape}', flush=True)

    # Process a smaller range of representative slices.
    # z=10..14 covers the dense fold region without burning ~70 min on
    # the full z=0..20 chunk.
    z_range = list(range(10, 15))
    print(f'\nProcessing z={z_range[0]}..{z_range[-1]} with 2D auto_slp...', flush=True)
    corrected_slices = []
    t_total = time.time()
    for z in z_range:
        phi_2d = arr[1:3, z]  # (2, H, W) — [dy, dx]
        t0 = time.time()
        rec = run_2d('auto_slp', phi_2d)
        wall = time.time() - t0
        T1, T2 = _triangle_areas_2d(rec['phi_out'][0], rec['phi_out'][1])
        Tmin = float(np.minimum(T1, T2).min())
        print(
            f'  z={z:3d}  feas={rec["feasible"]}  min_T={Tmin:+.4f}  '
            f'init_n_neg={rec["init_n_neg_2tri"]}  L1={rec["L1_dev"]:.1f}  '
            f'({wall:.1f}s)',
            flush=True,
        )
        corrected_slices.append(rec['phi_out'])
    print(f'\n2D processing total: {time.time() - t_total:.1f}s', flush=True)

    # Stack back into 3D. Reconstruct dz=0 and the corrected dy/dx.
    n_slices = len(z_range)
    H, W = arr.shape[2:]
    field_3d = np.zeros((3, n_slices, H, W), dtype=np.float64)
    field_3d[0] = 0.0  # dz unchanged (was 0)
    for i, phi_2d in enumerate(corrected_slices):
        field_3d[1, i] = phi_2d[0]  # dy
        field_3d[2, i] = phi_2d[1]  # dx

    # Check 6-tet feasibility on stacked field.
    print('\nChecking 6-tet feasibility on stacked 3D field...', flush=True)
    V = six_tet_volumes_3d(field_3d)
    n_tets = V.size
    n_neg = int((V <= 0).sum())
    n_below = int((V < 0.01 - 1e-5).sum())
    print(
        f'  shape: {field_3d.shape}  total tets: {n_tets:,}',
        flush=True,
    )
    print(
        f'  n_neg (V<=0):         {n_neg:>8d}  ({n_neg/n_tets*100:.4f}%)',
        flush=True,
    )
    print(
        f'  n_below_thresh:       {n_below:>8d}  ({n_below/n_tets*100:.4f}%)',
        flush=True,
    )
    print(f'  min_T: {float(V.min()):+.4f}  max_T: {float(V.max()):+.4f}', flush=True)

    # Per-z analysis to see WHERE the residual 3D folds are.
    if n_neg > 0:
        fold_per_z = (V.min(axis=0) <= 0).sum(axis=(1, 2))
        print('\nPer-z fold cell count (after 2D processing, stacked):', flush=True)
        for i in range(n_slices - 1):
            if fold_per_z[i] > 0:
                print(f'  z={z_range[i]:3d} (cube z={i}): {int(fold_per_z[i])} fold cells', flush=True)


if __name__ == '__main__':
    main()
