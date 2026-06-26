"""Benchmark the continuous (as-completed) scheduler vs the subround
barrier scheduler for within-slice cluster parallelism.

The subround scheduler runs pool.map per non-overlapping sub-round (a
barrier: stragglers idle workers that finished). The continuous scheduler
keeps the pool full by admitting any non-conflicting cluster as soon as a
worker frees. Measures wall + feasibility (n_neg 2-tri) + L1 deviation, on
the same slices, at n_workers=16.

Guarded for Windows spawn.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _n_neg_2tri(out):
    from dvfopt.core.tri_primitives import tri_areas_flat
    H, W = out.shape[1:]
    flat = np.concatenate([out[0].ravel(), out[1].ravel()])
    return int((tri_areas_flat(flat, H, W) <= 0).sum())


def main():
    from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
        cluster_slp_iter,
    )
    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    zs = [300, 200, 12]  # moderate, moderate, densest
    NW = 16

    for z in zs:
        sl = raw[1:3, z].astype(np.float64)
        n0 = _n_neg_2tri(sl)
        print(f'\n=== slice z={z} input n_neg={n0}  (n_workers={NW}) ===', flush=True)
        res = {}
        for sched in ('subround', 'continuous'):
            t0 = time.time()
            out, info = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                         n_workers=NW, scheduler=sched)
            dt = time.time() - t0
            res[sched] = (dt, _n_neg_2tri(out), float(np.abs(out - sl).sum()),
                          info.get('total_cluster_solves'))
            print(f'  {sched:>10}: wall={dt:6.1f}s  n_neg={res[sched][1]:4d}  '
                  f'L1={res[sched][2]:10.1f}  cluster_solves={res[sched][3]}',
                  flush=True)
        sub, con = res['subround'], res['continuous']
        print(f'  -> continuous speedup {sub[0]/con[0]:.2f}x  '
              f'L1 delta {(con[2]-sub[2])/sub[2]*100:+.2f}%  '
              f'feasible both={sub[1] == 0 and con[1] == 0}', flush=True)


if __name__ == '__main__':
    main()
