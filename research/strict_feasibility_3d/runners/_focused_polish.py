"""Try focused active-set LP polish on the 173-fold residual.

If this works, the same approach scales to the 19-fold final
residual (it should — fewer active constraints, smaller LP).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from research.strict_feasibility_3d.algorithms.focused_lp_6tet import focused_slp

OUTPUT = _HERE / 'output'


def main():
    # Start from the 173-fold checkpoint.
    cache = OUTPUT / 'b0039_FULL_stage3_z000_016.npy'
    phi_in = np.load(cache)
    print(f'Loaded {cache}  shape={phi_in.shape}', flush=True)

    t0 = time.time()
    phi_out, info = focused_slp(
        phi_in,
        threshold=0.01,
        active_buffer=0.0,  # only currently-violated tets in the active set
        trust_radius_0=2.0,  # large trust so the LP can pull deep folds out
        max_iter=30,
        seed=phi_in,
        verbose=1,
    )
    wall = time.time() - t0
    print(
        f'\n=== Focused SLP done ===\n'
        f'  iters:           {info["iters"]}\n'
        f'  final n_neg:     {info["final_n_neg"]}\n'
        f'  final n<0.01:    {info["final_n_below_threshold"]}\n'
        f'  final min_T:     {info["final_min_T"]:+.6f}\n'
        f'  STRICT 100%:     {info["final_n_neg"] == 0 and info["final_n_below_threshold"] == 0}\n'
        f'  total wall:      {wall:.1f}s',
        flush=True,
    )
    if info['final_n_neg'] == 0 and info['final_n_below_threshold'] == 0:
        np.save(OUTPUT / 'b0039_FULL_strict_feas_z000_016_focused.npy', phi_out)
        print('\n*** Saved strict-feasible result ***', flush=True)


if __name__ == '__main__':
    main()
