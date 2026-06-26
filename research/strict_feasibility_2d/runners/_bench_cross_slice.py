"""Benchmark cross-slice parallelism for the 2D auto_slp pipeline.

Profiling showed every B0039 slice takes auto_slp's cluster path, whose
dominant cost is the SERIAL m14 seed (L-BFGS-B). A 16-worker inner pool
only accelerates the (minor) SLP loop, leaving cores idle during the
serial seed. So the better parallelism axis is ACROSS slices: run N slices
concurrently, each with inner n_workers=1 AND numba pinned to 1 thread
(the 1-thread-per-worker lesson — otherwise N slices x prange-threads
oversubscribe).

Compares, on the same slice set:
  A) current: sequential slices, each cluster_slp_iter(n_workers=16)
  B) proposed: ProcessPool over slices, each cluster_slp_iter(n_workers=1)
                with numba threads pinned to 1

Reports wall + per-slice feasibility for both.

GUARDED for Windows spawn.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _pin_one_thread():
    try:
        import numba

        numba.set_num_threads(1)
    except Exception:
        pass


def _n_neg_2tri(out):
    """Count 2-tri folds (area <= 0) in a (2, H, W) [dy, dx] field."""
    from dvfopt.core.tri_primitives import tri_areas_flat

    H, W = out.shape[1:]
    flat = np.concatenate([out[0].ravel(), out[1].ravel()])  # DY_FIRST
    a = tri_areas_flat(flat, H, W)
    return int((a <= 0).sum())


def _solve_slice_serial(args):
    """Outer-pool worker: one slice, inner serial, numba pinned to 1 thread."""
    sl, threshold = args
    _pin_one_thread()
    from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
        cluster_slp_iter,
    )

    out, info = cluster_slp_iter(sl, threshold=threshold, max_outer_iters=6, n_workers=1)
    return _n_neg_2tri(out)


def main():
    from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
        cluster_slp_iter,
    )

    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    # Moderate-density slices (avoid the pathological z=12 to bound runtime).
    zs = [120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 500, 520]
    slices = [raw[1:3, z].astype(np.float64) for z in zs]
    print(
        f'cross-slice bench: {len(zs)} slices {slices[0].shape}, cores={os.cpu_count()}', flush=True
    )

    # A) Sequential, inner 16-worker pool (current auto_slp large-slice path).
    t0 = time.time()
    seq_neg = []
    for sl in slices:
        out, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6, n_workers=16)
        seq_neg.append(_n_neg_2tri(out))
    seq_wall = time.time() - t0
    print(
        f'[A sequential, inner n_workers=16] wall={seq_wall:.1f}s '
        f'feasible={sum(n == 0 for n in seq_neg)}/{len(zs)}',
        flush=True,
    )

    # B) Cross-slice pool, inner serial + numba pinned to 1 thread.
    n_outer = min(len(zs), os.cpu_count() or 1)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_outer, initializer=_pin_one_thread) as ex:
        par_neg = list(ex.map(_solve_slice_serial, [(sl, THR) for sl in slices]))
    par_wall = time.time() - t0
    print(
        f'[B cross-slice, {n_outer}x inner n_workers=1] wall={par_wall:.1f}s '
        f'feasible={sum(n == 0 for n in par_neg)}/{len(zs)}',
        flush=True,
    )

    print(
        f'\nSPEEDUP: {seq_wall / par_wall:.2f}x  '
        f'({seq_wall:.0f}s -> {par_wall:.0f}s on {len(zs)} slices)',
        flush=True,
    )
    print(
        f'feasibility identical: {seq_neg == par_neg} '
        f'(seq {seq_neg}\n                       par {par_neg})',
        flush=True,
    )


if __name__ == '__main__':
    main()
