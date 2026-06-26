"""Final push: drive the 1-fold chain_best state to n_neg=0.

The chain experiment proved that over-tighten @ 0.018 followed
by M10Tet @ 0.012 recovery is the structural escape mechanism.
It broke 2-fold -> 1-fold. We now apply increasingly aggressive
over-tighten thresholds, hoping to push 1 -> 0.

Schedule:
  - over-tighten @ 0.020, recover @ 0.012
  - over-tighten @ 0.022, recover @ 0.012
  - over-tighten @ 0.025, recover @ 0.012
  - direct M10Tet @ 0.011, 0.010, 0.009 (slightly lower target)

The 1-fold residual has min_T=-0.000323, so the over-tighten
needs to move T_k from -0.0003 across 0 — a tiny absolute shift
but the barrier homotopy must find a different basin.
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
    L1 = '' if phi_input is None else f'  L1={float(np.abs(phi - phi_input).sum()):.1f}'
    print(f'{label}: n_neg={n_neg}  n<0.01={n_below}  min_T={mn:+.6f}{L1}', flush=True)
    return n_neg, n_below, mn


def m10tet(phi, thr):
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=thr,
    )
    return solver.fit(phi).corrected


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    start = np.load(OUTPUT / 'b0039_z0_15_chain_best.npy').astype(np.float64)
    report(phi_input, 'Original input')
    n_neg, n_below, _ = report(start, 'START (saved 1-fold)', phi_input)

    if n_neg == 0 and n_below == 0:
        print('*** Already STRICT 100% feasible ***', flush=True)
        return

    cur = start
    best_n_neg = n_neg
    best_state = cur.copy()

    # Schedule of (label, over_tighten_thr, recover_thr).
    SCHED = [
        ('cyc_0.020_0.012', 0.020, 0.012),
        ('cyc_0.022_0.012', 0.022, 0.012),
        ('cyc_0.025_0.012', 0.025, 0.012),
        ('cyc_0.030_0.012', 0.030, 0.012),
    ]

    for label, ot, rec in SCHED:
        print(f'\n=== {label}: over-tighten @ {ot} + recover @ {rec} ===', flush=True)
        t0 = time.time()
        over = m10tet(best_state.copy(), ot)
        n_o, _, _ = report(over, f'  over-tightened @ {ot}', phi_input)
        print(f'    wall={time.time() - t0:.1f}s', flush=True)
        t1 = time.time()
        rec_state = m10tet(over, rec)
        n_neg, n_below, mn = report(rec_state, f'  recovered @ {rec}', phi_input)
        print(f'    recover wall={time.time() - t1:.1f}s', flush=True)
        if n_neg < best_n_neg:
            best_n_neg = n_neg
            best_state = rec_state.copy()
            print(f'    *** new best: n_neg={n_neg} ***', flush=True)
        if n_neg == 0 and n_below == 0:
            print(f'\n*** STRICT 100% feasible at {label} ***', flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_final_push.npy', rec_state)
            return

    # Direct M10Tet at lower thresholds (less stringent).
    for thr in (0.011, 0.010, 0.009, 0.008):
        print(f'\n=== Direct M10Tet @ {thr} ===', flush=True)
        t0 = time.time()
        result = m10tet(best_state.copy(), thr)
        n_neg, n_below, mn = report(result, f'  result @ {thr}', phi_input)
        print(f'    wall={time.time() - t0:.1f}s', flush=True)
        if n_neg < best_n_neg:
            best_n_neg = n_neg
            best_state = result.copy()
            print(f'    *** new best: n_neg={n_neg} ***', flush=True)
        if n_neg == 0 and n_below == 0:
            print(f'\n*** STRICT 100% feasible @ {thr} ***', flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_final_push.npy', result)
            return

    print(f'\n=== Final ===\n  best: n_neg={best_n_neg}', flush=True)
    np.save(OUTPUT / 'b0039_z0_15_final_push_best.npy', best_state)


if __name__ == '__main__':
    main()
