"""2D investigation: DEPTH-ROUTED hybrid (elastic vs seeded per cluster).

Our option-C result: elastic (seedless) SLP matches the seeded champion on
mild-fold crops and stalls on deep ones. So route per cluster by fold
depth (min_T of the crop): mild -> elastic direct (skip the m14 seed, the
profiled per-cluster bottleneck); deep -> seeded. This bench solves every
cluster of representative slices BOTH ways (serially, 1 core) and reports
wall + feasibility + L1 per depth bucket, giving the routing threshold
and the projected saving.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def main():
    from dvfopt.core.slp.cluster_lp_2tri import _fold_clusters
    from dvfopt.core.slp import slp_iter
    from dvfopt.core.tri_primitives import tri_areas_flat
    from research.strict_feasibility_2d.algorithms._elastic_slp import (
        elastic_slp_iter,
    )

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    buckets = {}  # depth bucket -> [n, seed_wall, el_wall, seed_l1, el_l1, el_feas]
    for z in (300, 450):
        sl = raw[1:3, z].astype(np.float64)
        clusters = _fold_clusters(sl, merge_dilation=2, target_threshold=0.0)
        print(f'z={z}: {len(clusters)} clusters', flush=True)
        for c in clusters[:60]:          # cap per slice for bench time
            y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
            crop = sl[:, y0:y1 + 1, x0:x1 + 1].copy()
            Hc, Wc = crop.shape[1:]
            a = tri_areas_flat(
                np.concatenate([crop[0].ravel(), crop[1].ravel()]), Hc, Wc)
            depth = float(a.min())
            key = ('mild (>-2)' if depth > -2 else
                   'medium (-2..-8)' if depth > -8 else 'deep (<=-8)')

            t0 = time.time()
            out_s, _ = slp_iter(crop, threshold=THR, seed='m14_fast')
            ws = time.time() - t0
            l1s = float(np.abs(out_s - crop).sum())

            t0 = time.time()
            out_e, info = elastic_slp_iter(crop, threshold=THR, mu=1000.0,
                                           max_iter=25)
            we = time.time() - t0
            l1e = float(np.abs(out_e - crop).sum())
            feas_e = info['n_neg'] == 0

            b = buckets.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0, 0])
            b[0] += 1
            b[1] += ws
            b[2] += we
            b[3] += l1s
            b[4] += l1e
            b[5] += int(feas_e)

    print(f'\n{"bucket":>16} | {"n":>3} | {"seed wall":>9} | {"elast wall":>10} | '
          f'{"el feas":>7} | {"seed L1":>9} | {"elast L1":>9}', flush=True)
    for k, b in sorted(buckets.items()):
        print(f'{k:>16} | {b[0]:>3} | {b[1]:>8.1f}s | {b[2]:>9.1f}s | '
              f'{b[5]:>4}/{b[0]:<2} | {b[3]:>9.1f} | {b[4]:>9.1f}', flush=True)


if __name__ == '__main__':
    main()
