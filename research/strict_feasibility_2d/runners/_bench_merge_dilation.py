"""Verify merge_dilation=1 vs the default (=2) across slices.

On z=450, merge_dilation=1 gave ~8% lower L1 at identical wall. Before
recommending a default change, check whether that holds across the density
spectrum (it could be slice-specific; the default was presumably tuned).
Reports wall + feasibility + L1 for md=1 and md=2 on each slice.

Guarded for spawn.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _n_neg(out):
    from dvfopt.core.tri_primitives import tri_areas_flat
    H, W = out.shape[1:]
    return int((tri_areas_flat(
        np.concatenate([out[0].ravel(), out[1].ravel()]), H, W) <= 0).sum())


def main():
    from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
        cluster_slp_iter,
    )
    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    zs = [100, 200, 300, 400, 450, 500]
    print(f'{"z":>4} | {"md":>2} | {"wall":>6} | {"n_neg":>5} | {"L1":>9}', flush=True)
    for z in zs:
        sl = raw[1:3, z].astype(np.float64)
        for md in [1, 2]:
            t0 = time.time()
            out, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                      n_workers=16, scheduler='continuous',
                                      merge_dilation=md)
            dt = time.time() - t0
            print(f'{z:>4} | {md:>2} | {dt:>6.1f} | {_n_neg(out):>5} | '
                  f'{float(np.abs(out - sl).sum()):>9.1f}', flush=True)


if __name__ == '__main__':
    main()
