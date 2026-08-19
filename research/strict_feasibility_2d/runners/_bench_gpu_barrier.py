"""Bench the interior-point (log-barrier homotopy) GPU untangler.

Feasible by construction: starts from identity, morphs toward phi_in with a
log barrier that forbids crossing the fold boundary. Should reach 0 folds
with NO SLP mop-up. Question: does it, and does it keep the global-basin L1
advantage vs the local champion (esp. dense)?
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
        gpu_barrier_untangle_2d,
    )

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    for z, tag in ((300, 'mild'), (13, 'dense')):
        sl = raw[1:3, z].astype(np.float64)
        print(f'\n=== z={z} ({tag})  raw folds(<thr)={_nneg(sl)} ===', flush=True)

        t0 = time.time()
        g, info = gpu_barrier_untangle_2d(sl, threshold=THR, verbose=1)
        wg = time.time() - t0
        print(f'  GPU-barrier       : wall={wg:6.1f}s  folds(np)={_nneg(g)}  '
              f'min_area={info["min_area"]:+.4f}  '
              f'L1={np.abs(g - sl).sum():10.1f}', flush=True)

        t0 = time.time()
        base, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                   n_workers=8, scheduler='continuous')
        print(f'  champion          : wall={time.time() - t0:6.1f}s  '
              f'folds={_nneg(base)}  L1={np.abs(base - sl).sum():10.1f}',
              flush=True)


if __name__ == '__main__':
    main()
