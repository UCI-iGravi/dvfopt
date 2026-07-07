"""Part XXI option C benchmark: elastic (seedless) SLP vs seeded slp_iter.

Head-to-head on identical cluster-sized fold crops (the actual unit of
work in the champion pipeline). If elastic reaches feasibility with
comparable L1 but without the m14 seed, the profiled per-cluster
bottleneck disappears.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _crop_around_folds(sl, half):
    from dvfopt.core.tri_primitives import tri_areas_flat

    H, W = sl.shape[1:]
    a = tri_areas_flat(np.concatenate([sl[0].ravel(), sl[1].ravel()]), H, W)
    a2 = a.reshape(2, H - 1, W - 1).min(0)
    ys, xs = np.where(a2 <= 0)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = max(0, cy - half), max(0, cx - half)
    return sl[:, y0:y0 + 2 * half, x0:x0 + 2 * half].copy()


def _stats(out, inp, thr):
    from dvfopt.core.tri_primitives import tri_areas_flat

    H, W = out.shape[1:]
    a = tri_areas_flat(np.concatenate([out[0].ravel(), out[1].ravel()]), H, W)
    return int((a <= 0).sum()), float(a.min()), float(np.abs(out - inp).sum())


def main():
    from dvfopt.core.slp import slp_iter
    from research.strict_feasibility_2d.algorithms._elastic_slp import (
        elastic_slp_iter,
    )

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    cases = [
        ('z300 moderate crop', _crop_around_folds(raw[1:3, 300].astype(np.float64), 32)),
        ('z450 sparse crop', _crop_around_folds(raw[1:3, 450].astype(np.float64), 32)),
        ('z12 dense crop', _crop_around_folds(raw[1:3, 12].astype(np.float64), 40)),
    ]
    for label, crop in cases:
        n0, m0, _ = _stats(crop, crop, THR)
        print(f'\n=== {label} {crop.shape[1:]} n_neg={n0} min_T={m0:+.3f} ===',
              flush=True)

        t0 = time.time()
        out_s, _ = slp_iter(crop, threshold=THR, seed='m14_fast')
        ws = time.time() - t0
        nn, mt, l1 = _stats(out_s, crop, THR)
        print(f'  seeded (m14_fast): wall={ws:6.2f}s n_neg={nn} '
              f'min_T={mt:+.5f} L1={l1:9.2f}', flush=True)

        for mu in (100.0, 1000.0):
            t0 = time.time()
            out_e, info = elastic_slp_iter(crop, threshold=THR, mu=mu)
            we = time.time() - t0
            nn, mt, l1 = _stats(out_e, crop, THR)
            print(f'  elastic (mu={mu:g}, seedless): wall={we:6.2f}s n_neg={nn} '
                  f'min_T={mt:+.5f} L1={l1:9.2f}  n_lp={info["n_lp"]}', flush=True)


if __name__ == '__main__':
    main()
