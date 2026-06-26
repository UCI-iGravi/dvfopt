"""Test 3D methods on a real B0039 subvolume.

Loads the precomputed 24x24x24 subvolume (centered on the densest fold
cluster in B0039 at z=12, y=191, x=189; 80% of tets initially folded)
and runs each method, reporting (feasibility, L1, wall).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from research.strict_feasibility_3d.runners._compare_3d import _stats, run_method


def main():
    sub = np.load(_HERE / 'output' / 'b0039_subvol_24x24x24.npy')
    init = _stats(sub)
    print(
        f'B0039 24x24x24 subvolume:  shape={sub.shape}  '
        f'init n_neg={init["n_neg"]}  min_T={init["min_T"]:+.4f}',
        flush=True,
    )
    for m in ['m10', 'm14', 'lp_oneshot', 'slp_iter']:
        rec = run_method(m, sub)
        flag = 'OK ' if rec['feasible'] else 'INF'
        err = f'   err={rec["error"]}' if rec['error'] else ''
        print(
            f'  {m:<14s} {flag}  n_neg={rec["final_n_neg"]:6d}  '
            f'min_T={rec["final_min_T"]:+.4f}  L1={rec["L1_dev"]:>10.2f}  '
            f'({rec["wall_s"]:>6.1f}s){err}',
            flush=True,
        )


if __name__ == '__main__':
    main()
