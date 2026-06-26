"""Multi-scale v3: start from saved 9-fold MS_V1 state and apply
GENTLE iterative polish strategies that don't break the field.

v2 showed that M14Tet on a 9-fold residual EXPANDS to 497 folds —
the refine-repair step is too aggressive when there's only a few
folds left. v3 sticks with M10Tet (barrier only, no repair) and
varies threshold / iterations.

Stages:
  1. Iterate M10Tet @ 0.015 up to 5 times (warm start each time).
  2. Iterate M10Tet @ 0.012 up to 3 times.
  3. Iterate M10Tet @ 0.020 up to 3 times (tighter target, see if
     it pushes through a different attractor).
  4. Final pass M10Tet @ 0.015 to settle.

Save best result at each stage.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def report(phi, label, phi_input=None):
    V = six_tet_volumes_3d(phi)
    n_neg = int((V <= 0).sum())
    n_below = int((V < THRESHOLD - 1e-5).sum())
    mn = float(V.min())
    L1 = '' if phi_input is None else f'  L1_from_input={float(np.abs(phi - phi_input).sum()):.1f}'
    print(f'{label}: n_neg={n_neg}  n<0.01={n_below}  min_T={mn:+.6f}{L1}', flush=True)
    return n_neg, n_below, mn


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    ms_v1 = np.load(OUTPUT / 'b0039_z0_15_ms_v1.npy').astype(np.float64)
    report(phi_input, 'Original input')
    report(ms_v1, 'MS_V1 start (saved 9-fold)', phi_input)

    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    cur = ms_v1
    best_n_neg = int((six_tet_volumes_3d(cur) <= 0).sum())
    best_state = cur.copy()

    SCHEDULES = [
        ('iter_0.015', 0.015, 5),
        ('iter_0.012', 0.012, 3),
        ('iter_0.020', 0.020, 3),
        ('iter_0.015_final', 0.015, 2),
    ]

    for sched_name, thr, n_iter in SCHEDULES:
        print(f'\n=== {sched_name}: threshold={thr}, iterations={n_iter} ===', flush=True)
        for it in range(n_iter):
            t0 = time.time()
            solver = Solver(
                constraint=Tet6Constraint3D(shape=cur.shape[1:]),
                objective=L1Objective(eps=1e-4),
                strategy=HarmonicALMBarrier3DStrategy(),
                threshold=thr,
            )
            new = solver.fit(cur).corrected
            wall = time.time() - t0
            n_neg, n_below, mn = report(new, f'  {sched_name} iter {it + 1}/{n_iter}', phi_input)
            print(f'    wall={wall:.1f}s', flush=True)
            if n_neg < best_n_neg:
                best_n_neg = n_neg
                best_state = new.copy()
                print(f'    *** new best: n_neg={n_neg} ***', flush=True)
            cur = new
            if n_neg == 0 and n_below == 0:
                print(f'\n*** STRICT 100% feasible at {sched_name} iter {it + 1} ***', flush=True)
                np.save(OUTPUT / 'b0039_z0_15_strict_via_ms_v3.npy', new)
                return

    print('\n=== Final ===', flush=True)
    print(f'  best across all schedules: n_neg={best_n_neg}', flush=True)
    np.save(OUTPUT / 'b0039_z0_15_ms_v3_best.npy', best_state)
    print('  saved best state to b0039_z0_15_ms_v3_best.npy', flush=True)


if __name__ == '__main__':
    main()
