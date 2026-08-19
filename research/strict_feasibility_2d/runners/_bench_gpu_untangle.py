"""Bench the first-order GPU untangler as a faster 2D METHOD.

(1) verify torch 2-tri areas == tri_areas_flat.
(2) whole-slice GPU untangle: wall + feasibility + L1 (standalone).
(3) hand the GPU result to slp_iter for L1 refinement -> final L1 + wall,
    compared against auto_slp from scratch (the current champion).
On representative slices (mild z=300, dense z=13). GPU is idle (marching
is CPU-only), so no contention.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _areas_np(p):
    from dvfopt.core.tri_primitives import tri_areas_flat
    H, W = p.shape[1:]
    return tri_areas_flat(np.concatenate([p[0].ravel(), p[1].ravel()]), H, W)


def _nneg(p):
    return int((_areas_np(p) <= 0).sum())


def main():
    import torch

    from research.strict_feasibility_2d.algorithms._gpu_untangle import (
        _areas_torch,
        gpu_untangle_2d,
    )
    print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}', flush=True)

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']

    # (1) area parity check on a real slice
    sl0 = raw[1:3, 300].astype(np.float64)
    a_np = _areas_np(sl0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dy = torch.tensor(sl0[0], device=dev, dtype=torch.float64)
    dx = torch.tensor(sl0[1], device=dev, dtype=torch.float64)
    t1, t2 = _areas_torch(dy, dx, torch)
    a_t = torch.cat([t1.reshape(-1), t2.reshape(-1)]).cpu().numpy()
    print(f'[parity] max|torch-np| = {np.abs(a_t - a_np).max():.2e}', flush=True)

    from dvfopt.core.slp import cluster_slp_iter, slp_iter

    for z, tag in ((300, 'mild'), (13, 'dense')):
        sl = raw[1:3, z].astype(np.float64)
        print(f'\n=== z={z} ({tag}) {sl.shape[1:]}  raw folds={_nneg(sl)} ===',
              flush=True)

        # (2) whole-slice GPU untangle
        t0 = time.time()
        g = gpu_untangle_2d(sl, threshold=THR, iters=4000, verbose=1)
        wg = time.time() - t0
        print(f'  GPU untangle       : wall={wg:6.1f}s  folds={_nneg(g)}  '
              f'L1={np.abs(g - sl).sum():10.1f}', flush=True)

        # (3a) GPU seed -> SLP refine (only if GPU got close)
        if _nneg(g) < _nneg(sl):
            t0 = time.time()
            gr, _ = slp_iter(g, threshold=THR, seed=None) if _nneg(g) == 0 else \
                cluster_slp_iter(g, threshold=THR, max_outer_iters=6, n_workers=8,
                                 scheduler='continuous')
            wr = time.time() - t0
            print(f'  GPU->SLP refine    : wall={wr:6.1f}s (+{wg:.0f} gpu)  '
                  f'folds={_nneg(gr)}  L1={np.abs(gr - sl).sum():10.1f}', flush=True)

        # (3b) champion baseline from scratch
        t0 = time.time()
        base, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                   n_workers=8, scheduler='continuous')
        wb = time.time() - t0
        print(f'  auto_slp (champion): wall={wb:6.1f}s  folds={_nneg(base)}  '
              f'L1={np.abs(base - sl).sum():10.1f}', flush=True)


if __name__ == '__main__':
    main()
