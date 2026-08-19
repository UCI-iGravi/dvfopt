"""Bench the hybrid Adam->L-BFGS GPU untangler for FULL feasibility.

Goal: reach 0 folds on the GPU alone (no SLP mop-up), keeping the dense
L1 win (global basin ~2x below the local champion) at competitive wall.
Reports GPU-only feasibility/L1/wall vs champion on mild + dense slices.
Verifies feasibility with the numpy reference tri_areas_flat.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _areas_np(p):
    from dvfopt.core.tri_primitives import tri_areas_flat
    H, W = p.shape[1:]
    return tri_areas_flat(np.concatenate([p[0].ravel(), p[1].ravel()]), H, W)


def _nneg(p, thr=0.01):
    return int((_areas_np(p) < thr).sum())


def main():
    from dvfopt.core.slp import cluster_slp_iter
    from research.strict_feasibility_2d.algorithms._gpu_untangle import (
        gpu_untangle_full_2d,
    )

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    for z, tag in ((300, 'mild'), (13, 'dense'), (12, 'worst')):
        sl = raw[1:3, z].astype(np.float64)
        print(f'\n=== z={z} ({tag})  raw folds={_nneg(sl)} ===', flush=True)

        t0 = time.time()
        g, info = gpu_untangle_full_2d(sl, threshold=THR, verbose=1)
        wg = time.time() - t0
        nn = _nneg(g)
        print(f'  GPU-full (no mop) : wall={wg:6.1f}s  folds(np)={nn}  '
              f'folds(torch)={info["n_neg"]}  L1={np.abs(g - sl).sum():10.1f}',
              flush=True)

        # safety mop only if a few slivers survived
        if 0 < nn <= 300:
            from dvfopt.core.slp import slp_iter
            t0 = time.time()
            gr, _ = cluster_slp_iter(g, threshold=THR, max_outer_iters=6,
                                     n_workers=8, scheduler='continuous')
            print(f'  +tiny mop         : wall={time.time() - t0:6.1f}s  '
                  f'folds={_nneg(gr)}  L1={np.abs(gr - sl).sum():10.1f}',
                  flush=True)

        t0 = time.time()
        base, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                   n_workers=8, scheduler='continuous')
        print(f'  champion          : wall={time.time() - t0:6.1f}s  '
              f'folds={_nneg(base)}  L1={np.abs(base - sl).sum():10.1f}',
              flush=True)


if __name__ == '__main__':
    main()
