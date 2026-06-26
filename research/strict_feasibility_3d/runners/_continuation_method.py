"""Continuation method: tighten threshold step-by-step.

Start with a threshold that's already satisfied by the input
(min_T = -0.013). Run M10Tet at each threshold in a sequence,
warm-starting from the previous result.

Sequence of thresholds:
   -0.010 -> -0.005 -> 0.000 -> +0.005 -> +0.010 -> +0.015

Hypothesis: each warm-started M10Tet pass perturbs the field
slightly. The cumulative effect of small perturbations may reach
a different local minimum than a direct M10Tet @ 0.015 from the
M10Tet plateau state.
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
TRUE_THRESHOLD = 0.01


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    V0 = six_tet_volumes_3d(phi)
    print(
        f'Start: n_neg={int((V0 <= 0).sum())}  '
        f'n<0.01={int((V0 < TRUE_THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    schedule = [-0.010, -0.005, 0.000, 0.005, 0.010, 0.015, 0.020]
    cur = phi.astype(np.float64).copy()
    for ti, thr in enumerate(schedule):
        V_cur = six_tet_volumes_3d(cur)
        print(
            f'\n--- Step {ti+1}/{len(schedule)}: threshold={thr:+.4f} '
            f'(current n_neg={int((V_cur <= 0).sum())}, '
            f'n<0.01={int((V_cur < TRUE_THRESHOLD - 1e-5).sum())}, '
            f'min_T={float(V_cur.min()):+.6f}) ---',
            flush=True,
        )
        t0 = time.time()
        solver = Solver(
            constraint=Tet6Constraint3D(shape=cur.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=thr,
        )
        cur = solver.fit(cur).corrected
        wall = time.time() - t0
        V = six_tet_volumes_3d(cur)
        n_neg = int((V <= 0).sum())
        n_below = int((V < TRUE_THRESHOLD - 1e-5).sum())
        L1 = float(np.abs(cur - phi).sum())
        print(
            f'  result: n_neg={n_neg}  n<0.01(true)={n_below}  '
            f'min_T={float(V.min()):+.6f}  L1_from_orig={L1:.1f}  wall={wall:.1f}s',
            flush=True,
        )
        if n_neg == 0 and n_below == 0:
            print(f'  *** STRICT 100% feasible at step {ti+1}, threshold={thr} ***', flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_continuation.npy', cur)
            break

    V_final = six_tet_volumes_3d(cur)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < TRUE_THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(cur - phi).sum())
    print(
        f'\n=== Final ===\n'
        f'  n_neg={n_neg}\n'
        f'  n<0.01={n_below}\n'
        f'  min_T={float(V_final.min()):+.6f}\n'
        f'  L1 from orig={L1:.1f}\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )


if __name__ == '__main__':
    main()
