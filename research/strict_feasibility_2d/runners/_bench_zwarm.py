"""2D investigation: z-warm-start for DENSE slices.

Measured: warm_z = raw_z + (corrected_{z-1} - raw_{z-1}) pre-fixes 68-78%
of dense-slice folds (and hurts sparse slices -> dense-only routing).
This bench answers: does that convert to wall savings at comparable L1?

  (a) baseline: cluster_slp_iter(raw_z)
  (b) warm:     cluster_slp_iter(warm_z)   [anchors to warm; L1 reported
                vs RAW so the comparison is honest]

Both at the same n_workers under identical machine load.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _n_neg(p):
    from dvfopt.core.tri_primitives import tri_areas_flat

    H, W = p.shape[1:]
    return int((tri_areas_flat(
        np.concatenate([p[0].ravel(), p[1].ravel()]), H, W) <= 0).sum())


def main():
    from dvfopt.core.slp import cluster_slp_iter

    THR = 0.01
    NW = 8
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    cor = np.load('research/strict_feasibility_2d/runners/output/'
                  'b0039_FULL_stage1_continuous.npy', mmap_mode='r')
    for z in (5, 13):
        r = raw[1:3, z].astype(np.float64)
        warm = r + (np.asarray(cor[1:3, z - 1]).astype(np.float64)
                    - raw[1:3, z - 1].astype(np.float64))
        print(f'\n=== dense z={z}: raw folds={_n_neg(r)}, warm folds={_n_neg(warm)} ===',
              flush=True)

        t0 = time.time()
        out_a, _ = cluster_slp_iter(r, threshold=THR, max_outer_iters=6,
                                    n_workers=NW, scheduler='continuous')
        wa = time.time() - t0
        print(f'  (a) baseline: wall={wa:7.1f}s  n_neg={_n_neg(out_a)}  '
              f'L1vsRAW={float(np.abs(out_a - r).sum()):11.1f}', flush=True)

        t0 = time.time()
        out_b, _ = cluster_slp_iter(warm, threshold=THR, max_outer_iters=6,
                                    n_workers=NW, scheduler='continuous')
        wb = time.time() - t0
        print(f'  (b) z-warm  : wall={wb:7.1f}s  n_neg={_n_neg(out_b)}  '
              f'L1vsRAW={float(np.abs(out_b - r).sum()):11.1f}  '
              f'speedup={wa / max(wb, 1e-9):.2f}x', flush=True)


if __name__ == '__main__':
    main()
