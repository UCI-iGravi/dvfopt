"""Targeted 3D method comparison on a moderate-density B0039 subvolume.

The earlier 24x24x24 test (80% folded, min_T=-3.07) was pathologically
dense — `lp_oneshot` hung in HiGHS for hours. This script uses a
16x16x16 region with ~11% folded tets, which is closer to what
cluster_slp would see per-cluster in a real production pipeline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from research.strict_feasibility_3d.runners._compare_3d import run_method, _stats


def main():
    sub = np.load(_HERE / 'output' / 'b0039_subvol_16_moderate.npy')
    init = _stats(sub)
    print(
        f'B0039 16x16x16 moderate-density subvolume:  shape={sub.shape}  '
        f'init n_neg={init["n_neg"]} ({init["n_neg"]/init.get("total", 1)*100 if False else 100*init["n_neg"]/(6*16**3):.2f}%)  '
        f'min_T={init["min_T"]:+.4f}',
        flush=True,
    )
    for m in ['m10', 'm14', 'lp_oneshot', 'slp_iter']:
        print(f'  starting {m}...', flush=True)
        t0 = time.time()
        rec = run_method(m, sub)
        flag = 'OK ' if rec['feasible'] else 'INF'
        err = f'   err={rec["error"]}' if rec['error'] else ''
        print(
            f'  {m:<14s} {flag}  n_neg={rec["final_n_neg"]:5d}  '
            f'min_T={rec["final_min_T"]:+.4f}  L1={rec["L1_dev"]:>10.2f}  '
            f'({rec["wall_s"]:>6.1f}s){err}',
            flush=True,
        )


if __name__ == '__main__':
    main()
