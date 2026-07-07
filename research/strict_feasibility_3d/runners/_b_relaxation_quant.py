"""Part XXI option B+E: quantify the constraint-semantics relaxations.

No new solver — measures, on the real saved artifacts, exactly how much
each relaxation would have bought:

  B-i   per-cell best-diagonal (mixed triangulation) acceptance
  B-ii  margin band: how much "infeasibility" is tightening vs folding
  B-iii tolerance: n_violations at 0 / -1e-5 / -1e-4 / -2e-4
  E     fixable-vs-floor routing split per artifact

Artifacts: stage-3 dense band (173 folds), the 1-fold escape plateau, and
the strict final band.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def main():
    from dvfopt.jacobian.tetrahedron_sign import (
        best_diagonal_min_volume,
        six_tet_min_volume_3d,
    )

    OUT = Path(__file__).parent / 'output'
    artifacts = [
        ('stage3_band (pre-escape)', 'b0039_FULL_stage3_z000_016.npy'),
        ('escape plateau (1 fold)', 'b0039_z0_15_BEST_1fold.npy'),
        ('strict final band', 'orch_v2_strict_denseband.npy'),
    ]
    for label, fname in artifacts:
        p = OUT / fname
        if not p.exists():
            print(f'--- {label}: MISSING ({fname})', flush=True)
            continue
        phi = np.load(p).astype(np.float64)
        mv = six_tet_min_volume_3d(phi)          # fixed Kuhn diagonal
        bd, _ = best_diagonal_min_volume(phi)    # best of 4 main diagonals
        print(f'\n=== {label}  {phi.shape} ===', flush=True)
        print(f'  fixed-diag : n_neg={int((mv <= 0).sum()):>6}  '
              f'n<0.01={int((mv < 0.01 - 1e-5).sum()):>6}  min_T={float(mv.min()):+.6f}',
              flush=True)
        print(f'  best-diag  : n_neg={int((bd <= 0).sum()):>6}  '
              f'n<0.01={int((bd < 0.01 - 1e-5).sum()):>6}  min_T={float(bd.min()):+.6f}',
              flush=True)
        neg = mv <= 0
        fixable = neg & (bd > 0)
        floor = neg & (bd <= 0)
        print(f'  E routing  : folded={int(neg.sum())}  '
              f'diagonal-fixable={int(fixable.sum())}  true-floor={int(floor.sum())}',
              flush=True)
        print('  B-iii tolerance sweep (fixed diag):', flush=True)
        for tol in (0.0, -1e-5, -1e-4, -2e-4, -1e-3):
            print(f'    tol={tol:+.0e}: violations={int((mv < tol).sum())}',
                  flush=True)
        print('  B-iii tolerance sweep (best diag):', flush=True)
        for tol in (0.0, -1e-4, -2e-4):
            print(f'    tol={tol:+.0e}: violations={int((bd < tol).sum())}',
                  flush=True)


if __name__ == '__main__':
    main()
