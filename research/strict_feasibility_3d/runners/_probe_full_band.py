"""Tractability probe for a FULL-volume run.

Times the orchestrator on a single representative z-band crop of the full
B0039 stage-1 field, so we can project the cost of the whole 528-slice
volume before committing to it. Picks the fold-heaviest 24-slice band.

NOTE: guarded by ``if __name__ == '__main__'`` — mandatory on Windows.
``correct_dvf_3d(n_workers>1)`` spawns worker processes; on the spawn
start method each worker re-imports this module, so any top-level heavy
work (loading the 1.85 GB field, calling the solver) MUST sit under the
guard or every worker re-runs it (fork bomb).
"""
import sys
import time
from pathlib import Path

import numpy as np


def main():
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from dvfopt import correct_dvf_3d
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    OUT = Path(__file__).parent / 'output'
    phi = np.load(OUT / 'b0039_FULL_stage1.npy').astype(np.float64)
    D = phi.shape[1]
    band = 24

    mv = six_tet_min_volume_3d(phi)
    per_z = (mv <= 0).reshape(mv.shape[0], -1).sum(axis=1)
    csum = np.concatenate([[0], np.cumsum(per_z)])
    best_z0, best_cnt = 0, -1
    for z0 in range(D - band):
        c = int(csum[min(z0 + band, len(csum) - 1)] - csum[z0])
        if c > best_cnt:
            best_cnt, best_z0 = c, z0
    z0, z1 = best_z0, best_z0 + band
    crop = phi[:, z0:z1, :, :].copy()
    cmv = six_tet_min_volume_3d(crop)
    n0 = int((cmv <= 0).sum())
    print(f'worst band z[{z0}:{z1}] crop={crop.shape[1:]} n_neg={n0} '
          f'min_T={float(cmv.min()):+.4f}', flush=True)

    t0 = time.time()
    out, rep = correct_dvf_3d(crop, threshold=0.01, n_workers=24,
                              thorough=False, verbose=1)
    dt = time.time() - t0
    print(f'\nBAND feasible={rep.feasible} {rep.n_neg_in}->{rep.n_neg_out} '
          f'n<0.01={rep.n_below_out} min_T={rep.min_T_out:+.5f} '
          f'floor_out={rep.best_diag_floor_out} wall={dt:.1f}s', flush=True)
    print(f'PROJECTION: ~{D // band} such bands; worst-band wall={dt:.0f}s '
          f'=> rough upper-bound total ~{dt * (D // band) / 3600:.1f} h '
          f'(most bands are far lighter than this worst one)', flush=True)


if __name__ == '__main__':
    main()
