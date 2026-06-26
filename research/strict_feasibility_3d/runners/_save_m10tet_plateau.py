"""Helper: run M10Tet @ 0.015 on raw input once, save the plateau
result for re-use in subsequent experiments. Skips waiting 76 min
for the pre-pass each time."""
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


def main():
    plateau_path = OUTPUT / 'b0039_z0_15_m10tet_plateau.npy'
    if plateau_path.exists():
        print(f'Plateau already saved at {plateau_path}', flush=True)
        phi = np.load(plateau_path)
        V = six_tet_volumes_3d(phi)
        print(f'  n_neg={int((V<=0).sum())} n<0.01={int((V<0.01-1e-5).sum())} '
              f'min_T={V.min():+.6f}', flush=True)
        return

    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    V0 = six_tet_volumes_3d(phi_input)
    print(f'Input: n_neg={int((V0<=0).sum())} n<0.01={int((V0<0.01-1e-5).sum())}',
          flush=True)

    print('Running M10Tet @ 0.015 ...', flush=True)
    t0 = time.time()
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi_input.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    phi_out = solver.fit(phi_input).corrected
    wall = time.time() - t0
    print(f'M10Tet wall={wall:.1f}s ({wall/60:.1f} min)', flush=True)
    V = six_tet_volumes_3d(phi_out)
    print(f'Result: n_neg={int((V<=0).sum())} n<0.01={int((V<0.01-1e-5).sum())} '
          f'min_T={V.min():+.6f}', flush=True)
    np.save(plateau_path, phi_out)
    print(f'Saved to {plateau_path}', flush=True)


if __name__ == '__main__':
    main()
