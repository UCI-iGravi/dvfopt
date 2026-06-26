"""Experimental speed/accuracy levers for the 2D champion (auto_slp).

(1) SEED-COST SWEEP — the profiled 2D bottleneck is the per-cluster m14
    seed (L-BFGS-B). Sweep inner_seed across the cost spectrum
    (harmonic < m10 < m14_quick < m14_fast[default] < m14) and measure
    wall + feasibility + L1. A cheaper seed that still lets the SLP loop
    reach feasibility at similar L1 is a direct speed win.

(2) L1-OPTIMALITY GAP — compare clustered auto_slp L1 to the global
    (non-clustered) slp_iter L1, i.e. how much L1 the cluster
    decomposition leaves on the table vs solving the whole slice. Small
    gap => no accuracy headroom from decomposition.

Guarded for Windows spawn.
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

    # ---- (1) Seed-cost sweep ----
    print('=== (1) SEED-COST SWEEP (cluster_slp_iter, n_workers=16, continuous) ===', flush=True)
    for z in [300, 450]:
        sl = raw[1:3, z].astype(np.float64)
        print(f'\n slice z={z}  input n_neg={_n_neg(sl)}', flush=True)
        print(f'{"seed":>10} | {"wall(s)":>8} | {"n_neg":>5} | {"L1":>10}', flush=True)
        for seed in ['harmonic', 'm10', 'm14_quick', 'm14_fast']:
            t0 = time.time()
            try:
                out, _ = cluster_slp_iter(
                    sl,
                    threshold=THR,
                    max_outer_iters=6,
                    n_workers=16,
                    scheduler='continuous',
                    inner_seed=seed,
                )
                dt = time.time() - t0
                print(
                    f'{seed:>10} | {dt:>8.1f} | {_n_neg(out):>5} | '
                    f'{float(np.abs(out - sl).sum()):>10.1f}',
                    flush=True,
                )
            except Exception as e:
                print(f'{seed:>10} | FAILED: {type(e).__name__}: {e}', flush=True)

    # ---- (2) L1-optimality gap: clustered vs global SLP ----
    print('\n=== (2) L1-OPTIMALITY GAP (clustered auto_slp vs global slp_iter) ===', flush=True)
    for z in [450]:  # global on one slice (whole-slice LP is slow)
        sl = raw[1:3, z].astype(np.float64)
        t0 = time.time()
        out_c, _ = cluster_slp_iter(
            sl, threshold=THR, max_outer_iters=6, n_workers=16, scheduler='continuous'
        )
        wc, l1c = time.time() - t0, float(np.abs(out_c - sl).sum())
        t0 = time.time()
        out_g, _ = slp_iter(sl, threshold=THR, seed='m14')
        wg, l1g = time.time() - t0, float(np.abs(out_g - sl).sum())
        print(
            f' z={z}: clustered L1={l1c:.1f} ({wc:.1f}s, n_neg={_n_neg(out_c)})  '
            f'global L1={l1g:.1f} ({wg:.1f}s, n_neg={_n_neg(out_g)})  '
            f'gap={(l1c - l1g) / l1g * 100:+.2f}%',
            flush=True,
        )


if __name__ == '__main__':
    main()
