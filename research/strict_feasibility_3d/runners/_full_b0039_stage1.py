"""Stage 1 only on the full 528-slice B0039 3D DVF.

Processes every z-slice with 2D auto_slp (independent per-slice work,
already parallelised inside auto_slp via n_workers=16). After all
slices, stacks into 3D and reports the residual 3D fold structure —
this tells us how much stage 2+3 work is needed for strict 3D
feasibility on the full volume.

Caches the per-slice corrected fields to a single .npz so reruns
(e.g. to test different stage 2+3 strategies) don't have to redo
the 2-hour Stage 1.
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
from research.strict_feasibility_2d.runners._compare import run_method as run_2d

OUTPUT = _HERE / 'output'
FULL_STAGE1_CACHE = OUTPUT / 'b0039_FULL_stage1.npy'


def main():
    if FULL_STAGE1_CACHE.exists():
        print(f'Loading existing Stage 1 cache: {FULL_STAGE1_CACHE}', flush=True)
        stack = np.load(FULL_STAGE1_CACHE)
    else:
        print('Loading raw B0039...', flush=True)
        arr = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr'].astype(
            np.float64
        )
        D, H, W = arr.shape[1:]
        print(f'  full shape: {arr.shape}', flush=True)

        # Stage 1: 2D auto_slp on each of 528 slices.
        print('\n=== Stage 1: 2D auto_slp on every z-slice ===', flush=True)
        stack = np.zeros((3, D, H, W), dtype=np.float64)
        t_start = time.time()
        for z in range(D):
            t0 = time.time()
            rec = run_2d('auto_slp', arr[1:3, z])
            stack[1, z] = rec['phi_out'][0]
            stack[2, z] = rec['phi_out'][1]
            if rec['init_n_neg_2tri'] == 0:
                flag = 'NOFOLDS'
            elif rec['feasible']:
                flag = 'feas'
            else:
                flag = 'INF'
            print(
                f'  z={z:3d}: {flag:8s}  init_n_neg={rec["init_n_neg_2tri"]:5d}  '
                f'wall={rec["wall_s"]:5.1f}s  '
                f'elapsed={time.time() - t_start:.0f}s',
                flush=True,
            )
        np.save(FULL_STAGE1_CACHE, stack)
        print(
            f'\n[stage 1] cached to {FULL_STAGE1_CACHE}  ({time.time() - t_start:.1f}s)', flush=True
        )

    # Check 3D feasibility on stacked field.
    print('\n=== Checking 3D 6-tet feasibility on full stacked field ===', flush=True)
    t0 = time.time()
    V = six_tet_volumes_3d(stack)
    print(f'  six_tet_volumes_3d wall: {time.time() - t0:.1f}s', flush=True)
    n_tets = V.size
    n_neg = int((V <= 0).sum())
    n_below = int((V < 0.01 - 1e-5).sum())
    print(
        f'\n  total tets: {n_tets:,}\n'
        f'  n_neg (V<=0):         {n_neg:>10d}  ({n_neg / n_tets * 100:.6f}%)\n'
        f'  n_below_threshold:    {n_below:>10d}  ({n_below / n_tets * 100:.6f}%)\n'
        f'  min_T:                {float(V.min()):+.6f}\n'
        f'  max_T:                {float(V.max()):+.6f}',
        flush=True,
    )

    # Where are the residual 3D folds? Per-z fold density.
    fold_per_z = (V.min(axis=0) <= 0).sum(axis=(1, 2))
    nz_with_folds = np.where(fold_per_z > 0)[0]
    print(
        f'\n  z-layers with at least one 3D fold: {len(nz_with_folds)} of {len(fold_per_z)}',
        flush=True,
    )
    if len(nz_with_folds) > 0:
        print('  Top 10 z-layers by 3D fold count:', flush=True)
        top = sorted(nz_with_folds, key=lambda z: -fold_per_z[z])[:10]
        for z in top:
            print(f'    cube z={z}: {int(fold_per_z[z])} fold cells', flush=True)


if __name__ == '__main__':
    main()
