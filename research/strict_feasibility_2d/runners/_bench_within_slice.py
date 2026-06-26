"""Benchmark auto_slp's WITHIN-SLICE parallelism scaling, with L1.

auto_slp already parallelizes inside a single slice: cluster_slp_iter
solves fold-clusters concurrently across an n_workers process pool. This
sweeps n_workers on a single slice and reports all three metrics —
wall, feasibility (n_neg 2-tri), AND L1 deviation from the input — to
see (a) how within-slice parallelism scales and (b) whether worker count
perturbs solution quality (clustering/batch order can change the result).

Guarded for Windows spawn.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _n_neg_2tri(out):
    from dvfopt.core.tri_primitives import tri_areas_flat

    H, W = out.shape[1:]
    flat = np.concatenate([out[0].ravel(), out[1].ravel()])  # DY_FIRST
    return int((tri_areas_flat(flat, H, W) <= 0).sum())


def main():
    from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
        cluster_slp_iter,
    )

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']

    # (z, worker-counts) — moderate slice gets a full sweep; sparser slice a
    # shorter one. Avoid the pathological z=12 at n_workers=1 (too slow).
    plans = [
        (300, [1, 2, 4, 8, 16, 24]),
        (450, [1, 8, 24]),
    ]
    for z, workers in plans:
        sl = raw[1:3, z].astype(np.float64)
        n0 = _n_neg_2tri(sl)
        print(f'\n=== slice z={z} {sl.shape} input n_neg={n0} ===', flush=True)
        print(
            f'{"n_workers":>9} | {"wall(s)":>8} | {"speedup":>7} | '
            f'{"n_neg":>5} | {"L1":>12} | {"L1 vs nw=1":>10}',
            flush=True,
        )
        base_wall = None
        base_l1 = None
        for nw in workers:
            t0 = time.time()
            out, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6, n_workers=nw)
            dt = time.time() - t0
            nneg = _n_neg_2tri(out)
            l1 = float(np.abs(out - sl).sum())
            if base_wall is None:
                base_wall, base_l1 = dt, l1
            sp = base_wall / dt
            dl1 = (l1 - base_l1) / base_l1 * 100.0
            print(
                f'{nw:>9} | {dt:>8.1f} | {sp:>6.2f}x | {nneg:>5} | {l1:>12.1f} | {dl1:>+9.2f}%',
                flush=True,
            )


if __name__ == '__main__':
    main()
