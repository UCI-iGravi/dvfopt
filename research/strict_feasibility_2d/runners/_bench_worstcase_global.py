"""Worst-case wall-time saving from clustering vs the global solve.

z=12 is the densest B0039 slice (~9k folds). Measures clustered auto_slp
(continuous) vs the global non-clustered slp_iter on it, so we can state
the actual end-to-end saving on a worst-case section (vs the 18x measured
on the moderate z=450). Reports wall + feasibility + L1 for both.

The global solve on a ~9k-fold 287k-variable slice is slow (could be
10-40 min); guarded + background-friendly.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _n_neg(out):
    from dvfopt.core.tri_primitives import tri_areas_flat

    H, W = out.shape[1:]
    return int((tri_areas_flat(np.concatenate([out[0].ravel(), out[1].ravel()]), H, W) <= 0).sum())


def main():
    from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
        cluster_slp_iter,
    )
    from research.strict_feasibility_2d.algorithms.lp_direct_2tri import slp_iter

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    sl = raw[1:3, 12].astype(np.float64)  # densest slice
    print(f'worst-case slice z=12 {sl.shape} input n_neg={_n_neg(sl)}', flush=True)

    t0 = time.time()
    out_c, _ = cluster_slp_iter(
        sl, threshold=THR, max_outer_iters=6, n_workers=16, scheduler='continuous'
    )
    wc = time.time() - t0
    l1c = float(np.abs(out_c - sl).sum())
    print(f'[clustered] wall={wc:.1f}s  n_neg={_n_neg(out_c)}  L1={l1c:.1f}', flush=True)

    t0 = time.time()
    out_g, _ = slp_iter(sl, threshold=THR, seed='m14')
    wg = time.time() - t0
    l1g = float(np.abs(out_g - sl).sum())
    print(f'[global]    wall={wg:.1f}s  n_neg={_n_neg(out_g)}  L1={l1g:.1f}', flush=True)

    print(
        f'\nWORST-CASE SAVING: {wg - wc:.0f}s  ({wg / wc:.1f}x faster)  '
        f'L1 gap clustered vs global = {(l1c - l1g) / l1g * 100:+.1f}%',
        flush=True,
    )


if __name__ == '__main__':
    main()
