"""Independent-2D lever: shrink the SEED cost without changing its basin.

The single-slice bottleneck is the m14 seed (harmonic -> ALM -> L2-refine,
all L-BFGS-B). L1 is basin-fixed by the seed, but *how many inner
iterations* the seed spends may be reducible without moving the basin.
This probes per-cluster: solve with the full seed vs a budget-capped ALM
(fewer outer/inner), measuring wall + whether the final SLP L1/feasibility
are unchanged. If capped seeds land the same basin, that's a direct
single-slice speedup at zero L1 cost.

Uses slp_iter on real fold-cluster crops with different seeds:
  m14       (full)          m14_fast (drop barrier, current default)
  m14_quick (drop refine)   m10      (harmonic+ALM+barrier)
Reports wall + feasibility + L1 per seed per crop.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _stats(out, inp):
    from dvfopt.core.tri_primitives import tri_areas_flat

    H, W = out.shape[1:]
    a = tri_areas_flat(np.concatenate([out[0].ravel(), out[1].ravel()]), H, W)
    return int((a <= 0).sum()), float(np.abs(out - inp).sum())


def _crop(sl, half):
    from dvfopt.core.tri_primitives import tri_areas_flat

    H, W = sl.shape[1:]
    a = tri_areas_flat(np.concatenate([sl[0].ravel(), sl[1].ravel()]), H, W)
    a2 = a.reshape(2, H - 1, W - 1).min(0)
    ys, xs = np.where(a2 <= 0)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = max(0, cy - half), max(0, cx - half)
    return sl[:, y0:y0 + 2 * half, x0:x0 + 2 * half].copy()


def main():
    from dvfopt.core.slp import slp_iter

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    cases = [('z300 mild crop', _crop(raw[1:3, 300].astype(np.float64), 28)),
             ('z450 medium crop', _crop(raw[1:3, 450].astype(np.float64), 28)),
             ('z12 deep crop', _crop(raw[1:3, 12].astype(np.float64), 36))]
    seeds = ['m14', 'm14_fast', 'm14_quick', 'm10']
    for label, crop in cases:
        print(f'\n=== {label} {crop.shape[1:]} ===', flush=True)
        for seed in seeds:
            t0 = time.time()
            try:
                out, _ = slp_iter(crop, threshold=THR, seed=seed)
                nn, l1 = _stats(out, crop)
                print(f'  {seed:>10}: wall={time.time() - t0:6.2f}s  '
                      f'n_neg={nn}  L1={l1:9.2f}', flush=True)
            except Exception as e:
                print(f'  {seed:>10}: FAILED {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
