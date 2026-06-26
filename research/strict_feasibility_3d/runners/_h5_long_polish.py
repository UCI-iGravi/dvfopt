"""H5: M10Tet with very long polish_max_iter and very tight tolerance.

The default polish_max_iter is 200. The barrier polish stage may
just be giving up before reaching the central path. Try
polish_max_iter=5000 + tighter alm_inner_maxiter and outer_max.
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
    phi_in = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    V = six_tet_volumes_3d(phi_in)
    print(
        f'Input: n_neg={int((V <= 0).sum())}  n<0.01={int((V < TRUE_THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V.min()):+.6f}',
        flush=True,
    )

    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    print('\n--- M10Tet @ threshold=0.015, polish_max_iter=5000, outer_max=200 ---', flush=True)
    t0 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi_in.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(
            polish_max_iter=5000,
            outer_max=200,
            alm_inner_maxiter=1000,
        ),
        threshold=0.015,
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
        print('  *** STRICT 100% FEASIBLE via long polish ***', flush=True)


if __name__ == '__main__':
    main()
