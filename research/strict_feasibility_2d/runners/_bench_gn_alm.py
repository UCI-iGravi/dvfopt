"""Benchmark: Gauss-Newton ALM inner step vs scipy L-BFGS-B ALM inner.

Isolates the inner-solver lever the profiling pointed at. Runs both on the
SAME fold-containing 2-tri crop and reports inner-iteration count, wall,
feasibility (min_T), and L1 deviation. The hypothesis: GN converges in
tens of (expensive, sparse-solve) iterations vs L-BFGS-B's thousands of
cheap ones, for a net win — at equal feasibility and L1.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _fold_crop(sl, half=32):
    from dvfopt.core.tri_primitives import tri_areas_flat
    H, W = sl.shape[1:]
    a = tri_areas_flat(np.concatenate([sl[0].ravel(), sl[1].ravel()]), H, W)
    a2 = a.reshape(2, H - 1, W - 1).min(0)
    ys, xs = np.where(a2 <= 0)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = max(0, cy - half), max(0, cx - half)
    y1, x1 = min(H, cy + half), min(W, cx + half)
    return sl[:, y0:y1, x0:x1].copy()


def _stats(out, inp, THR):
    from dvfopt.core.tri_primitives import tri_areas_flat
    H, W = out.shape[1:]
    a = tri_areas_flat(np.concatenate([out[0].ravel(), out[1].ravel()]), H, W)
    n_neg = int((a <= 0).sum())
    n_bel = int((a < THR - 1e-5).sum())
    l1 = float(np.abs(out - inp).sum())
    return n_neg, n_bel, float(a.min()), l1


def main():
    from dvfopt.core.tri_primitives import tri_areas_flat
    from dvfopt.core.wallbreakers._alm import augmented_lagrangian_2d
    from research.strict_feasibility_2d.algorithms._gn_alm_proto import (
        augmented_lagrangian_2d_gn,
    )

    THR, MARGIN = 0.01, 1e-3
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    for z, half in [(300, 32), (12, 40)]:
        sl = raw[1:3, z].astype(np.float64)
        crop = _fold_crop(sl, half=half)
        a = tri_areas_flat(np.concatenate([crop[0].ravel(), crop[1].ravel()]),
                           *crop.shape[1:])
        n0 = int((a <= 0).sum())
        print(f'\n=== z={z} crop {crop.shape[1:]} '
              f'({2*crop.shape[1]*crop.shape[2]} vars) input n_neg={n0} '
              f'min_T={float(a.min()):+.3f} ===', flush=True)

        # L-BFGS-B baseline.
        t0 = time.time()
        out_b, info_b = augmented_lagrangian_2d(
            crop, threshold=THR, margin=MARGIN, anchor='l2',
            verbose=0, record_history=True,
        )
        wb = time.time() - t0
        out_b = np.asarray(out_b)
        inner_b = sum(r.get('inner_nit', 0)
                      for r in (info_b.get('log_first5', []) + info_b.get('log_last5', [])))
        nn, nb, mt, l1 = _stats(out_b, crop, THR)
        print(f'  L-BFGS-B: wall={wb:6.2f}s  inner~{inner_b:>5}(partial log)  '
              f'n_neg={nn} n<thr={nb} min_T={mt:+.5f} L1={l1:.2f}', flush=True)

        # Gauss-Newton prototype.
        t0 = time.time()
        out_g, info_g = augmented_lagrangian_2d_gn(
            crop, threshold=THR, margin=MARGIN,
        )
        wg = time.time() - t0
        nn, nb, mt, l1 = _stats(out_g, crop, THR)
        print(f'  GN      : wall={wg:6.2f}s  inner={info_g["total_inner"]:>5}       '
              f'  n_neg={nn} n<thr={nb} min_T={mt:+.5f} L1={l1:.2f}', flush=True)
        print(f'  -> GN wall speedup {wb/wg:.2f}x', flush=True)


if __name__ == '__main__':
    main()
