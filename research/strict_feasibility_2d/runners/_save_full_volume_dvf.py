"""Save the NEW (continuous-default auto_slp) corrected DVFs for every
B0039 2D slice, assembled into a full (3, 528, 320, 456) volume.

Matches the b0039_FULL_stage1.npy convention: channel 0 (dz) = 0 (the 2D
per-slice correction produces no z-displacement), channels 1/2 = corrected
[dy, dx]. The result is a drop-in replacement for b0039_FULL_stage1.npy,
produced by the shipped continuous scheduler.

RESUMABLE + crash-safe: the output is an on-disk memmap written one slice
at a time (flushed); a progress JSON records the next slice. Re-running
resumes. Per-slice metrics are also written to a CSV (superset of the
metrics-only benchmark).

GUARDED for Windows spawn.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def main():
    from research.strict_feasibility_2d.runners._compare import run_method

    OUT = Path(__file__).parent / 'output'
    SRC = 'data/dvfs/archive/new_b0039_laplacian_deformation_field.npz'
    VOL = OUT / 'b0039_FULL_stage1_continuous.npy'
    PROG = OUT / 'b0039_FULL_stage1_continuous_progress.json'
    CSV = OUT / 'all_slices_dvf_metrics.csv'

    raw = np.load(SRC)['arr']
    _, D, H, W = raw.shape

    if VOL.exists() and PROG.exists():
        vol = np.lib.format.open_memmap(VOL, mode='r+')
        start = int(json.loads(PROG.read_text())['next_z'])
        print(f'RESUME from z={start}/{D}', flush=True)
    else:
        vol = np.lib.format.open_memmap(VOL, mode='w+', dtype=np.float64,
                                        shape=(3, D, H, W))
        vol[0, :] = 0.0  # dz: 2D correction produces no z-displacement
        vol.flush()
        start = 0
        with open(CSV, 'w', newline='') as f:
            csv.writer(f).writerow(['z', 'init_n_neg', 'final_n_neg',
                                    'feasible', 'L1_dev', 'wall_s'])

    t0 = time.time()
    for z in range(start, D):
        sl = raw[1:3, z].astype(np.float64)
        rec = run_method('auto_slp', sl)
        po = np.asarray(rec['phi_out'])           # (2, H, W) [dy, dx]
        vol[1, z] = po[0]
        vol[2, z] = po[1]
        vol.flush()
        PROG.write_text(json.dumps({'next_z': z + 1, 'D': D}))
        with open(CSV, 'a', newline='') as f:
            csv.writer(f).writerow([
                z, rec.get('init_n_neg_2tri'), rec.get('final_n_neg_2tri'),
                int(bool(rec.get('feasible'))),
                f"{rec.get('L1_dev', float('nan')):.3f}",
                f"{rec.get('wall_s', float('nan')):.2f}",
            ])
        if z % 25 == 0 or z == D - 1:
            print(f'[z={z:3d}] feasible={int(bool(rec.get("feasible")))} '
                  f'final_n_neg={rec.get("final_n_neg_2tri")} '
                  f'({(time.time()-t0)/3600:.2f}h elapsed)', flush=True)

    # Sanity check: the assembled per-slice-2D-feasible volume still has the
    # z-stacking 3D folds (same regime as b0039_FULL_stage1) — confirms the
    # output matches the staging convention.
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d
    arr = np.asarray(vol)
    mv = six_tet_min_volume_3d(arr)
    print(f'\nSAVED {VOL}  shape={arr.shape}', flush=True)
    print(f'3D folds after stacking (expected ~728k, the stage-2/3 target): '
          f'{int((mv <= 0).sum())}  min_T={float(mv.min()):+.3f}', flush=True)


if __name__ == '__main__':
    main()
