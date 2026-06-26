"""H3: M10Tet with wider harmonic ring_pad to fill larger fold cores.

The harmonic stage in M10Tet finds connected fold cores and
Dirichlet-fills via a 7-point Laplacian. The `ring_pad` parameter
controls how big a "core" to fill (in voxel units). Default is 2.
A wider pad covers larger fold neighborhoods and may resolve folds
that smaller pads leave intact.

Test progression: ring_pad in {2 (default), 5, 10}, with
threshold=0.015 (matches our overshoot target).
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

    for ring_pad in [5, 10]:
        print(f'\n--- M10Tet ring_pad={ring_pad}  threshold=0.015 ---', flush=True)
        t0 = time.time()
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi_in.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(
                ring_pad=ring_pad,
                max_grow_iters=15,
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


if __name__ == '__main__':
    main()
