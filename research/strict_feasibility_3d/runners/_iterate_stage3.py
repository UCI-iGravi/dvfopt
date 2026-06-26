"""Iterate Stage 3 (M10Tet @ threshold=0.015) on the z=0..15 chunk
output until convergence or fold-count plateau.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt import (
    HarmonicALMBarrier3DStrategy,
    L1Objective,
    Solver,
    Tet6Constraint3D,
)
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

OUTPUT = _HERE / 'output'


def _stats(phi):
    V = six_tet_volumes_3d(phi)
    return {
        'n_neg': int((V <= 0).sum()),
        'n_below': int((V < 0.01 - 1e-5).sum()),
        'min_T': float(V.min()),
    }


def main():
    cache = OUTPUT / 'b0039_FULL_stage3_z000_016.npy'
    phi = np.load(cache)
    print(f'Loaded {cache}', flush=True)
    s = _stats(phi)
    print(f'  start: n_neg={s["n_neg"]}  n<0.01={s["n_below"]}  min_T={s["min_T"]:+.6f}', flush=True)

    prev_n_neg = s['n_neg']
    for iteration in range(4):
        if prev_n_neg == 0 and s['n_below'] == 0:
            print('Already strict feasible.', flush=True)
            break
        print(f'\n=== Iter {iteration}: M10Tet @ threshold=0.015 ===', flush=True)
        t0 = time.time()
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=0.015,
        )
        phi = solver.fit(phi).corrected
        wall = time.time() - t0
        s = _stats(phi)
        print(
            f'  result: n_neg={s["n_neg"]}  n<0.01={s["n_below"]}  '
            f'min_T={s["min_T"]:+.6f}  wall={wall:.1f}s',
            flush=True,
        )
        if s['n_neg'] >= prev_n_neg:
            print('  No progress — stopping.', flush=True)
            break
        prev_n_neg = s['n_neg']
        if s['n_neg'] == 0 and s['n_below'] == 0:
            print('  *** STRICT 100% FEASIBLE ***', flush=True)
            np.save(OUTPUT / 'b0039_FULL_stage3_z000_016_strict.npy', phi)
            break


if __name__ == '__main__':
    main()
