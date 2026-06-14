"""Head-to-head on z=300: m14_fast vs m10 vs harmonic inner seed.

The cProfile run on z=300 showed _m14_fast_seed is 87% of total wall
(103.5 / 119 s), with l2_refine_2d alone 75 s. Since the outer SLP
polishes L1 anyway, the L2-refine inside m14_fast is redundant.

Hypothesis: cheaper inner seeds (m10, even just harmonic) give the
same final feasibility and L1 in less time.
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
SEEDS = ['m14_fast', 'm14_quick', 'm10', 'harmonic']


def _stats(phi_2hw):
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    Tmin = np.minimum(T1, T2)
    return int((Tmin <= 0).sum()), float(Tmin.min())


def main():
    print(f'{"slice":<8s}  {"seed":<11s}  {"wall (s)":>9s}  {"L1":>11s}  '
          f'{"n_neg":>5s}  {"min_T":>9s}')
    print('-' * 70)
    for z in SLICES:
        case_id, phi_in, meta = load_b0039_slice(z)
        for seed in SEEDS:
            t0 = time.time()
            phi_out, info = cluster_slp_iter(
                phi_in,
                threshold=0.01,
                inner_seed=seed,
            )
            wall = time.time() - t0
            n_neg, mt = _stats(phi_out)
            L1 = float(np.abs(phi_out - phi_in).sum())
            print(f'z={z:<6d}  {seed:<11s}  {wall:>9.1f}  {L1:>11.1f}  '
                  f'{n_neg:>5d}  {mt:>+9.4f}')


if __name__ == '__main__':
    main()
