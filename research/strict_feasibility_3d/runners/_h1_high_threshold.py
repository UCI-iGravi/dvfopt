"""H1: try M10Tet with progressively higher target thresholds on
the 173-fold residual. Hypothesis: a much higher target threshold
forces the barrier path to push harder and resolve stuck folds."""
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
    phi_in = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    V = six_tet_volumes_3d(phi_in)
    print(
        f'Input: n_neg={int((V<=0).sum())}  n<0.01={int((V<TRUE_THRESHOLD-1e-5).sum())}  '
        f'min_T={float(V.min()):+.6f}',
        flush=True,
    )

    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    for target_threshold in [0.025, 0.05, 0.10]:
        print(f'\n--- M10Tet @ threshold={target_threshold} ---', flush=True)
        t0 = time.time()
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi_in.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=target_threshold,
        )
        result = solver.fit(phi_in)
        phi_out = result.corrected
        wall = time.time() - t0
        V_out = six_tet_volumes_3d(phi_out)
        n_neg = int((V_out <= 0).sum())
        n_below = int((V_out < TRUE_THRESHOLD - 1e-5).sum())
        L1 = float(np.abs(phi_out - phi_in).sum())
        print(
            f'  result: n_neg={n_neg}  n<0.01(true)={n_below}  '
            f'min_T={float(V_out.min()):+.6f}  L1_from_input={L1:.1f}  wall={wall:.1f}s',
            flush=True,
        )
        if n_neg == 0 and n_below == 0:
            print(f'  *** STRICT 100% FEASIBLE *** at target threshold {target_threshold}', flush=True)
            np.save(OUTPUT / f'b0039_z0_15_strict_via_h1_{int(target_threshold * 1000):03d}.npy', phi_out)
            phi_in = phi_out  # continue from improved state
            break
        else:
            phi_in = phi_out  # iterate from improved state


if __name__ == '__main__':
    main()
