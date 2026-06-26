"""Profile where time goes in a representative 3D (tet) and 2D (tri) solve.

Grounds the optimization plan: is the cost in the constraint kernels
(tet_volumes_flat / tet_grad_T_v / tri_areas / tri_grad_T_v — where
float32 would help), the optimizer loop (scipy L-BFGS-B / HiGHS linprog —
where schedule tuning or a better solver helps), or Python overhead?

Guarded for Windows spawn (the 2D auto_slp path may spawn a worker pool).
"""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root (for `research` pkg)


def _profile(label, fn, top=25):
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    fn()
    pr.disable()
    dt = time.time() - t0
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(top)
    print(f'\n===== {label}  (wall {dt:.1f}s) =====', flush=True)
    print(s.getvalue(), flush=True)


def main():
    from dvfopt import correct_dvf_3d
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d
    from research.strict_feasibility_2d.runners._compare import run_method

    OUT = Path(__file__).parent / 'output'

    # ---- 3D: a small dense crop that finishes fast (single-process) ----
    phi = np.load(OUT / 'b0039_FULL_stage1.npy').astype(np.float64)
    # pick a modest sub-block around a folded region; keep it small so the
    # profile completes in ~1 min.
    crop = phi[:, 44:52, 150:230, 200:280].copy()
    n0 = int((six_tet_min_volume_3d(crop) <= 0).sum())
    print(f'3D crop {crop.shape[1:]} n_neg={n0}', flush=True)
    if n0 > 0 and '2donly' not in sys.argv:
        _profile(
            '3D correct_dvf_3d (single-process, n_workers=1)',
            lambda: correct_dvf_3d(crop, threshold=0.01, n_workers=1,
                                   thorough=False, verbose=0),
        )

    # ---- 2D: the cluster-SLP path with n_workers=1 so the per-cluster LP
    # work runs IN-PROCESS and cProfile actually captures it (every B0039
    # slice is 320x456 > 5k px, so auto_slp always takes this path). ----
    from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
        cluster_slp_iter,
    )
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    sl = raw[1:3, 300].astype(np.float64)  # [dy, dx] of z=300
    print(f'\n2D cluster_slp_iter n_workers=1 (z=300) {sl.shape}', flush=True)
    _profile(
        '2D cluster_slp_iter (z=300, n_workers=1, in-process)',
        lambda: cluster_slp_iter(sl, threshold=0.01, max_outer_iters=6,
                                 n_workers=1),
    )


if __name__ == '__main__':
    main()
