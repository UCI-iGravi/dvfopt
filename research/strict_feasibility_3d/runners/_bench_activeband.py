"""Benchmark: active-band M10Tet vs global M10Tet on the full dense band.

Both use the now-parallel tet kernels. Active-band crops to the fold
clusters and solves M10Tet only there; global solves the whole field.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt import HarmonicALMBarrier3DStrategy, L1Objective, Solver, Tet6Constraint3D
from dvfopt.core.wallbreakers._coupled_kring_3d import active_band_alm_recovery_3d
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

OUT = _HERE / 'output'


def main():
    phi = np.load(OUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    mv = six_tet_min_volume_3d(phi)
    n0 = int((mv <= 0).sum())
    print(f'INPUT full band {phi.shape}: n_neg={n0} min_T={mv.min():+.6f}', flush=True)

    print('\n=== GLOBAL M10Tet @ 0.012 (parallel kernels) ===', flush=True)
    t0 = time.time()
    out_g = (
        Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=0.012,
        )
        .fit(phi)
        .corrected
    )
    wg = time.time() - t0
    mvg = six_tet_min_volume_3d(out_g)
    L1g = float(np.abs(out_g - phi).sum())
    print(
        f'  GLOBAL: n_neg={int((mvg <= 0).sum())} min_T={mvg.min():+.6f} '
        f'L1={L1g:.1f} wall={wg:.1f}s',
        flush=True,
    )

    print('\n=== ACTIVE-BAND M10Tet @ 0.012 ===', flush=True)
    t0 = time.time()
    out_a, info = active_band_alm_recovery_3d(phi, threshold=0.012, pad=4, verbose=1)
    wa = time.time() - t0
    mva = six_tet_min_volume_3d(out_a)
    L1a = float(np.abs(out_a - phi).sum())
    print(
        f'  ACTIVE-BAND: n_neg={int((mva <= 0).sum())} min_T={mva.min():+.6f} '
        f'L1={L1a:.1f} wall={wa:.1f}s clusters={info["n_clusters"]}',
        flush=True,
    )

    print(
        f'\n=== SPEEDUP: {wg / max(wa, 0.1):.1f}x  (global {wg:.0f}s -> active-band {wa:.0f}s) ===',
        flush=True,
    )


if __name__ == '__main__':
    main()
