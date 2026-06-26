"""End-to-end validation of correct_dvf_3d across real B0039 sections.

Runs the packaged orchestrator on a difficulty spectrum of real sections
and reports, per section: did it reach strict n_neg=0, the residual, the
best-diagonal floor (irreducible-under-fixed-triangulation set), L1, wall.
This answers the standing question: does the packaged pipeline fix the
problematic sections, and where is the genuine floor.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt import correct_dvf_3d
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

OUT = _HERE / 'output'

# Ordered easy -> hard.
SECTIONS = [
    ('subvol_8_easy', 'b0039_subvol_8_easy.npy'),
    ('subvol_16_moderate', 'b0039_subvol_16_moderate.npy'),
    ('dense_band_z0_15', 'b0039_FULL_stage3_z000_016.npy'),
    ('band_z10_14', 'b0039_z10_14_stage1.npy'),
    ('subvol_24_dense', 'b0039_subvol_24x24x24.npy'),
]


def main():
    results = []
    for name, fn in SECTIONS:
        p = OUT / fn
        if not p.exists():
            print(f'[{name}] missing {fn}; skip', flush=True)
            continue
        phi = np.load(p).astype(np.float64)
        mv = six_tet_min_volume_3d(phi)
        n0 = int((mv <= 0).sum())
        print(f'\n=== {name}  shape={phi.shape}  n_neg_in={n0} ===', flush=True)
        t0 = time.time()
        try:
            out, rep = correct_dvf_3d(phi, threshold=0.01, verbose=1)
        except Exception as exc:  # noqa: BLE001
            print(f'  ERROR: {type(exc).__name__}: {exc}', flush=True)
            results.append((name, n0, 'ERROR', None, None, None, time.time() - t0))
            continue
        # independent re-verify
        mvo = six_tet_min_volume_3d(out)
        n_out = int((mvo <= 0).sum())
        n_below = int((mvo < 0.01 - 1e-5).sum())
        print(f'  RESULT feasible={rep.feasible} n_neg {n0}->{n_out} '
              f'n<0.01={n_below} min_T={mvo.min():+.6f} '
              f'floor_in={rep.best_diag_floor_in} floor_out={rep.best_diag_floor_out} '
              f'L1={rep.l1_from_input:.1f} wall={rep.wall_s:.1f}s', flush=True)
        if rep.feasible:
            np.save(OUT / f'orch_strict_{name}.npy', out)
        results.append((name, n0, rep.feasible, n_out, n_below,
                        rep.best_diag_floor_out, rep.wall_s))

    print('\n' + '=' * 84, flush=True)
    print('ORCHESTRATOR VALIDATION SUMMARY', flush=True)
    print('=' * 84, flush=True)
    print(f'{"section":<22}{"n_in":>8}{"feasible":>10}{"n_out":>8}'
          f'{"n<.01":>8}{"floor":>8}{"wall(s)":>10}', flush=True)
    print('-' * 84, flush=True)
    for (name, n0, feas, n_out, n_below, floor, wall) in results:
        print(f'{name:<22}{n0:>8}{str(feas):>10}{str(n_out):>8}'
              f'{str(n_below):>8}{str(floor):>8}{wall:>10.1f}', flush=True)


if __name__ == '__main__':
    main()
