"""Part XXI 2D combo: CHEAP seed + feasibility-checked L1 polish.

The seed sweep showed cheap seeds are ~2x faster but ~8x worse L1
(basin-determined). Question: can the feasibility-checked overlap polish
(anchored to the ORIGINAL input) recover the cheap seed's L1 penalty
after the fact? If yes -> "fast AND accurate": harmonic-seeded cluster
SLP (fast) + polish (accuracy) beats the m14_fast default on wall at
equal L1. If no -> confirms the basin gap is topological (polish cannot
cross basins) and the m14 seed stays essential.

Compares on z=300 and z=450:
  (a) default:  cluster_slp_iter(inner_seed='m14_fast')
  (b) cheap:    cluster_slp_iter(inner_seed='harmonic')
  (c) cheap+polish: (b) then polish_sweeps(anchored to input)
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from research.strict_feasibility_2d.runners._bench_overlap_polish import (  # noqa: E402
    _areas,
    polish_sweeps,
)


def main():
    from dvfopt.core.slp import cluster_slp_iter

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    for z in (300, 450):
        sl = raw[1:3, z].astype(np.float64)
        print(f'\n=== z={z} ===', flush=True)

        t0 = time.time()
        out_a, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                    n_workers=16, scheduler='continuous',
                                    inner_seed='m14_fast')
        wa = time.time() - t0
        a = _areas(out_a)
        print(f'  (a) m14_fast seed : wall={wa:6.1f}s  '
              f'L1={float(np.abs(out_a - sl).sum()):9.1f}  '
              f'n_neg={int((a <= 0).sum())}', flush=True)

        t0 = time.time()
        out_b, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                    n_workers=16, scheduler='continuous',
                                    inner_seed='harmonic')
        wb = time.time() - t0
        a = _areas(out_b)
        l1b = float(np.abs(out_b - sl).sum())
        print(f'  (b) harmonic seed : wall={wb:6.1f}s  L1={l1b:9.1f}  '
              f'n_neg={int((a <= 0).sum())}', flush=True)

        t0 = time.time()
        out_c = polish_sweeps(out_b, sl, sweeps=3, verbose=0)
        wc = time.time() - t0
        a = _areas(out_c)
        l1c = float(np.abs(out_c - sl).sum())
        print(f'  (c) (b)+polish x3 : wall={wb + wc:6.1f}s (+{wc:.1f})  '
              f'L1={l1c:9.1f}  n_neg={int((a <= 0).sum())}  '
              f'recovered {100 * (l1b - l1c) / max(l1b, 1e-9):.1f}% of (b) L1',
              flush=True)


if __name__ == '__main__':
    main()
