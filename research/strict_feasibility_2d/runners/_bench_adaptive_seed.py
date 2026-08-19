"""Independent-2D lever: DEPTH-ADAPTIVE per-cluster seed selection.

Seed-budget showed m10 is ~2x faster than m14_fast on shallow clusters at
negligible L1 cost, but wrecks L1 on deep ones. This aggregates the effect
over all clusters of a slice: solve each cluster with
  (a) m14_fast  (current uniform default)
  (b) adaptive  (m10 if crop min_T > DEPTH_CUT else m14_fast)
and report total wall + total L1 + feasibility. Answers whether adaptive
seeding is a net single-slice win on arbitrary slices.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

DEPTH_CUT = -2.0


def main():
    from dvfopt.core.slp import slp_iter
    from dvfopt.core.slp.cluster_lp_2tri import _fold_clusters
    from dvfopt.core.tri_primitives import tri_areas_flat

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    for z in (300, 450, 200):
        sl = raw[1:3, z].astype(np.float64)
        clusters = _fold_clusters(sl, merge_dilation=2, target_threshold=0.0)[:80]
        agg = {'fast': [0.0, 0.0, 0], 'adaptive': [0.0, 0.0, 0]}  # wall, L1, nfeas
        n_m10 = 0
        for c in clusters:
            y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
            crop = sl[:, y0:y1 + 1, x0:x1 + 1].copy()
            Hc, Wc = crop.shape[1:]
            a = tri_areas_flat(np.concatenate([crop[0].ravel(), crop[1].ravel()]),
                               Hc, Wc)
            depth = float(a.min())
            ad_seed = 'm10' if depth > DEPTH_CUT else 'm14_fast'
            if ad_seed == 'm10':
                n_m10 += 1
            for name, seed in (('fast', 'm14_fast'), ('adaptive', ad_seed)):
                t0 = time.time()
                out, _ = slp_iter(crop, threshold=THR, seed=seed)
                ao = tri_areas_flat(
                    np.concatenate([out[0].ravel(), out[1].ravel()]), Hc, Wc)
                agg[name][0] += time.time() - t0
                agg[name][1] += float(np.abs(out - crop).sum())
                agg[name][2] += int(ao.min() >= THR)
        n = len(clusters)
        f, ad = agg['fast'], agg['adaptive']
        print(f'\n=== z={z}: {n} clusters ({n_m10} routed to m10) ===', flush=True)
        print(f'  uniform m14_fast: wall={f[0]:6.1f}s  L1={f[1]:9.1f}  feas={f[2]}/{n}',
              flush=True)
        print(f'  depth-adaptive  : wall={ad[0]:6.1f}s  L1={ad[1]:9.1f}  '
              f'feas={ad[2]}/{n}  speedup={f[0] / max(ad[0], 1e-9):.2f}x  '
              f'L1 delta={100 * (ad[1] - f[1]) / max(f[1], 1e-9):+.1f}%', flush=True)


if __name__ == '__main__':
    main()
