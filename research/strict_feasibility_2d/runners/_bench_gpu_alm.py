"""Bench the GPU augmented-Lagrangian untangler vs champion.

The plain quadratic-penalty GPU untangler plateaued (min_A stalled short
of feasibility), forcing an expensive SLP cleanup. The PHR-ALM variant
adds per-triangle multipliers to reach feasibility globally. Question:
does it (a) reach 0 folds on its own, (b) keep the dense-slice L1 win
(GPU basin was 16% below champion), (c) beat champion wall?
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


def _nneg(p):
    return int((_areas_np(p) <= 0).sum())


def main():
    from research.strict_feasibility_2d.algorithms._gpu_untangle import (
        gpu_untangle_alm_2d,
    )
    from dvfopt.core.slp import slp_iter, cluster_slp_iter

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    for z, tag in ((300, 'mild'), (13, 'dense')):
        sl = raw[1:3, z].astype(np.float64)
        print(f'\n=== z={z} ({tag})  raw folds={_nneg(sl)} ===', flush=True)

        t0 = time.time()
        g = gpu_untangle_alm_2d(sl, threshold=THR, n_outer=40, n_inner=300,
                                verbose=1)
        wg = time.time() - t0
        nn = _nneg(g)
        print(f'  GPU-ALM           : wall={wg:6.1f}s  folds={nn}  '
              f'L1={np.abs(g - sl).sum():10.1f}', flush=True)

        # tiny SLP mop-up only if a handful of residual slivers remain
        if 0 < nn <= 1500:
            t0 = time.time()
            gr, _ = cluster_slp_iter(g, threshold=THR, max_outer_iters=6,
                                     n_workers=8, scheduler='continuous')
            wr = time.time() - t0
            print(f'  GPU-ALM->SLP mop  : wall={wr:6.1f}s (+{wg:.0f})  '
                  f'folds={_nneg(gr)}  L1={np.abs(gr - sl).sum():10.1f}',
                  flush=True)

        t0 = time.time()
        base, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                   n_workers=8, scheduler='continuous')
        wb = time.time() - t0
        print(f'  champion          : wall={wb:6.1f}s  folds={_nneg(base)}  '
              f'L1={np.abs(base - sl).sum():10.1f}', flush=True)


if __name__ == '__main__':
    main()
