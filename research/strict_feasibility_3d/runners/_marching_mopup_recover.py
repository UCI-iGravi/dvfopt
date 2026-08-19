"""Targeted recovery for the marching sweep's crashed global mop-up.

The sweep finished (n_neg=97, all residuals in the compact dense-band box
z0-18, y130-209, x172-244) but the final active_band mop-up called the
recovery on the WHOLE 528-slice field; a merged cluster hit the global-
fallback path and SuperLU OOM'd (jcol 3.1M -> SIGSEGV).

Fix: run active_band_alm_recovery_3d on JUST the compact sub-volume with
merge_dilation=1 so every cluster crop stays tiny (max bbox 15x8x3). Paste
back, re-verify GLOBALLY (catches any boundary breakage), save to a new
file. Idempotent and non-destructive: reads the saved sweep memmap, writes
b0039_FULL_marching25d_mopped.npy.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

OUT = 'research/strict_feasibility_3d/runners/output'
SRC = f'{OUT}/b0039_FULL_marching25d.npy'
DST = f'{OUT}/b0039_FULL_marching25d_mopped.npy'

# compact sub-volume covering all residuals with margin (frozen outside)
Z0, Z1 = 0, 22
Y0, Y1 = 105, 225
X0, X1 = 150, 265


def _stats(V):
    return int((V <= 0).sum()), int((V < 0.01).sum()), float(V.min())


def main():
    from dvfopt.core.wallbreakers._coupled_kring_3d import (
        active_band_alm_recovery_3d,
    )
    from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

    full = np.array(np.load(SRC))          # writable copy
    V0 = six_tet_volumes_3d(full)
    print(f'GLOBAL before: n_neg={_stats(V0)[0]}  n<0.01={_stats(V0)[1]}  '
          f'min_T={_stats(V0)[2]:.5f}', flush=True)

    crop = full[:, Z0:Z1, Y0:Y1, X0:X1].copy()
    Vc = six_tet_volumes_3d(crop)
    print(f'crop {crop.shape[1:]}: n_neg={_stats(Vc)[0]}  '
          f'n<0.01={_stats(Vc)[1]}  min_T={_stats(Vc)[2]:.5f}', flush=True)

    t0 = time.time()
    crop2, info = active_band_alm_recovery_3d(
        crop, threshold=0.012, pad=4, merge_dilation=1, max_widen=2,
        n_workers=1, verbose=1)
    Vc2 = six_tet_volumes_3d(crop2)
    print(f'crop after mop ({time.time() - t0:.1f}s): n_neg={_stats(Vc2)[0]}  '
          f'n<0.01={_stats(Vc2)[1]}  min_T={_stats(Vc2)[2]:.5f}  '
          f'accepted={info.get("accepted")}', flush=True)

    # paste back and re-verify globally (catches any boundary breakage)
    full[:, Z0:Z1, Y0:Y1, X0:X1] = crop2
    V1 = six_tet_volumes_3d(full)
    nn, nsub, mn = _stats(V1)
    addl1 = float(np.abs(crop2 - crop).sum())
    print(f'GLOBAL after : n_neg={nn}  n<0.01={nsub}  min_T={mn:.5f}  '
          f'mop added L1={addl1:.1f}', flush=True)

    np.save(DST, full)
    print(f'saved {DST}', flush=True)


if __name__ == '__main__':
    main()
