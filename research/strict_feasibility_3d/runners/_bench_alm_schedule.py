"""Benchmark ALM schedule tuning on a small 3D crop.

The profile showed scipy L-BFGS-B (`setulb`) is ~48% of a 3D solve, so the
real 3D lever is FEWER L-BFGS-B iterations. The ALM exposes inner_maxiter,
outer_max, and rho_growth. This sweeps them on a fixed small crop and
reports wall + final feasibility (min_T, n_neg) so we can see whether a
cheaper schedule keeps strict feasibility.

Run ALONE (no other CPU jobs) for clean timing. Guarded for spawn.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def main():
    from dvfopt.core.wallbreakers._alm_3d import augmented_lagrangian_3d
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    OUT = Path(__file__).parent / 'output'
    THR = 0.01
    phi = np.load(OUT / 'b0039_FULL_stage1.npy').astype(np.float64)
    # Small crop that solves in ~10-30s per config.
    crop = phi[:, 44:50, 160:210, 210:260].copy()
    mv = six_tet_min_volume_3d(crop)
    print(f'crop {crop.shape[1:]} n_neg={int((mv<=0).sum())} min_T={float(mv.min()):+.3f}',
          flush=True)

    configs = [
        ('baseline   (inner=200, outer=60, rho_g=5)', dict(inner_maxiter=200, outer_max=60, rho_growth=5.0)),
        ('fewer-inner (inner=50,  outer=60, rho_g=5)', dict(inner_maxiter=50,  outer_max=60, rho_growth=5.0)),
        ('fewer-inner (inner=100, outer=60, rho_g=5)', dict(inner_maxiter=100, outer_max=60, rho_growth=5.0)),
        ('fast-rho    (inner=100, outer=60, rho_g=10)', dict(inner_maxiter=100, outer_max=60, rho_growth=10.0)),
    ]
    for label, kw in configs:
        t0 = time.time()
        out, info = augmented_lagrangian_3d(
            crop, threshold=THR, margin=1e-3, anchor='l2',
            verbose=0, record_history=True, **kw,
        )
        dt = time.time() - t0
        mv = six_tet_min_volume_3d(out)
        n_neg = int((mv <= 0).sum())
        n_bel = int((mv < THR - 1e-5).sum())
        print(f'  {label}: wall={dt:5.1f}s  n_neg={n_neg:4d}  n<thr={n_bel:5d}  '
              f'min_T={float(mv.min()):+.5f}  outer_used={info["outer_used"]}',
              flush=True)


if __name__ == '__main__':
    main()
