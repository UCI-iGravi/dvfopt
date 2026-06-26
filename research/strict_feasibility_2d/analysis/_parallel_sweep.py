"""Re-test n_workers parallelism with the fixed shared-pool architecture.

The old code re-created ProcessPoolExecutor per sub-round, paying
Windows spawn cost ~1-2 s per worker on every re-creation, which
erased parallelism gains. The fix in `cluster_lp_2tri.py` shares one
pool across the entire `cluster_slp_iter` call.

This script measures wall + L1 across n_workers in {1, 2, 4, 8} on
3 representative slices.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
    cluster_slp_iter,
)
from research.strict_feasibility_2d.worst_cases._load import load_b0039_slice

SLICES = [12, 100, 300]
N_WORKERS = [1, 2, 4, 8]


def _stats(phi_2hw):
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    Tmin = np.minimum(T1, T2)
    return int((Tmin <= 0).sum()), float(Tmin.min())


def main():
    print(
        f'{"slice":<8s}  {"n_workers":>9s}  {"wall (s)":>9s}  '
        f'{"L1":>11s}  {"n_neg":>5s}  {"min_T":>9s}  {"speedup":>8s}'
    )
    print('-' * 70)
    for z in SLICES:
        case_id, phi_in, meta = load_b0039_slice(z)
        baseline_wall = None
        for nw in N_WORKERS:
            t0 = time.time()
            phi_out, info = cluster_slp_iter(
                phi_in,
                threshold=0.01,
                n_workers=nw,
            )
            wall = time.time() - t0
            n_neg, mt = _stats(phi_out)
            L1 = float(np.abs(phi_out - phi_in).sum())
            if nw == 1:
                baseline_wall = wall
                sp = '1.00x'
            else:
                sp = f'{baseline_wall / wall:.2f}x'
            print(
                f'z={z:<6d}  {nw:>9d}  {wall:>9.1f}  {L1:>11.1f}  '
                f'{n_neg:>5d}  {mt:>+9.4f}  {sp:>8s}'
            )


if __name__ == '__main__':
    main()
