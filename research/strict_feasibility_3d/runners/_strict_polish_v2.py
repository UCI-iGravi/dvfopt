"""Strict-feasibility polish: re-run M10Tet on stage-2 output with a
higher target threshold so the barrier-polish lands above 0.01.

stage 2's M10Tet at threshold=0.01 landed at min_T=+0.0049 — it
under-shot the target by ~50%. Re-running on its own output with
threshold=0.02 (2x) should push the barrier path higher.
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
STAGE2_CACHE = OUTPUT / 'b0039_z10_14_stage2.npy'
TRUE_THRESHOLD = 0.01


def main():
    if not STAGE2_CACHE.exists():
        raise SystemExit(f'no stage 2 cache at {STAGE2_CACHE}; run _threestage_pipeline.py first')
    phi_s2 = np.load(STAGE2_CACHE)
    V = six_tet_volumes_3d(phi_s2)
    print(
        f'Stage 2 cache: n_neg={int((V<=0).sum())}  '
        f'n<thresh={int((V<TRUE_THRESHOLD-1e-5).sum())}  min_T={float(V.min()):+.6f}',
        flush=True,
    )

    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    for overshoot_threshold in [0.015, 0.02, 0.03]:
        print(f'\n--- M10Tet polish with threshold={overshoot_threshold} ---', flush=True)
        t0 = time.time()
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi_s2.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=overshoot_threshold,
        )
        result = solver.fit(phi_s2)
        phi_out = result.corrected
        wall = time.time() - t0
        V_out = six_tet_volumes_3d(phi_out)
        n_neg = int((V_out <= 0).sum())
        n_below = int((V_out < TRUE_THRESHOLD - 1e-5).sum())
        L1_from_s2 = float(np.abs(phi_out - phi_s2).sum())
        print(
            f'  result: n_neg={n_neg}  n<thresh(0.01)={n_below}  '
            f'min_T={float(V_out.min()):+.6f}  L1_from_s2={L1_from_s2:.1f}  wall={wall:.1f}s',
            flush=True,
        )
        if n_neg == 0 and n_below == 0:
            print(f'  *** STRICT 100% FEASIBLE *** at threshold={overshoot_threshold}', flush=True)
            np.save(OUTPUT / f'b0039_z10_14_strict_feas_threshold{overshoot_threshold:.3f}.npy', phi_out)
            break


if __name__ == '__main__':
    main()
