"""Full-volume B0039 fold elimination via an additive multigrid V-cycle.

The raw stage-1 field is UNIFORMLY, severely folded (728 533 folds, every
z-layer, min_T -4.13), so band-by-band full-resolution solving projected
to ~2.5 days. But the fold count collapses under downsampling:

    fine (528,320,456)  728 533 folds   min_T -4.13
    /2   (264,160,228)   19 500         -0.73
    /4   (132, 80,114)      149         -0.17
    /8   ( 66, 40, 57)        8         -0.03

i.e. the folds are predominantly LOW-frequency. So solve coarsest, then
prolongate the *correction* up the pyramid, ADDING it to the original
fine-detail field at each level (additive / FAS-style multigrid — NOT the
replacement that _multiscale_3d does, so fine detail and low L1 deviation
are preserved), polishing the residual at each level.

Per level we log the seeded fold count (the make-or-break number is the
fine level's count after prolongation) and checkpoint, so the run is
observable and resumable.

GUARDED for Windows spawn.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))


def main():
    from dvfopt import correct_dvf_3d
    from dvfopt.core.wallbreakers._multiscale_3d import (
        _downsample_2x,
        _upsample_2x,
    )
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    OUT = Path(__file__).parent / 'output'
    SRC = OUT / 'b0039_FULL_stage1.npy'
    FINAL = OUT / 'b0039_FULL_corrected_mg.npy'
    THR = 0.01

    def stats(a):
        mv = six_tet_min_volume_3d(a)
        return int((mv <= 0).sum()), int((mv < THR).sum()), float(mv.min())

    t_run = time.time()
    fine = np.load(SRC).astype(np.float64)

    # Build the pyramid: pyr[0]=fine, pyr[1]=/2, pyr[2]=/4, pyr[3]=/8.
    pyr = [fine]
    for _ in range(3):
        pyr.append(_downsample_2x(pyr[-1]))
    for k, p in enumerate(pyr):
        n, nb, mt = stats(p)
        print(f'pyramid L{k} {p.shape[1:]} n_neg={n} n<thr={nb} min_T={mt:+.3f}', flush=True)

    # Solve the coarsest level (L3, /8) to feasibility.
    t0 = time.time()
    sol, rep = correct_dvf_3d(pyr[3], threshold=THR, n_workers=24, thorough=True, verbose=0)
    n, nb, mt = stats(sol)
    print(
        f'[L3 solve] /8 -> n_neg={n} feasible={rep.feasible} min_T={mt:+.4f} '
        f'({time.time() - t0:.0f}s)',
        flush=True,
    )

    # V-cycle upward: prolongate correction, add to original level, polish.
    for lvl in (2, 1, 0):
        target = pyr[lvl]
        coarse = pyr[lvl + 1]
        # Correction the coarser solve made, in coarse-voxel units.
        corr_coarse = sol - coarse
        # Prolongate to this level (trilinear x2 grid, magnitude x2).
        corr_fine = _upsample_2x(corr_coarse, target.shape[1:])
        seeded = target + corr_fine
        n0, nb0, mt0 = stats(seeded)
        tag = {0: 'fine', 1: '/2', 2: '/4'}[lvl]
        print(
            f'[L{lvl} seed] {tag} seeded n_neg={n0} n<thr={nb0} '
            f'min_T={mt0:+.4f}  (was {stats(target)[0]} unseeded)',
            flush=True,
        )
        t0 = time.time()
        sol, rep = correct_dvf_3d(seeded, threshold=THR, n_workers=24, thorough=True, verbose=1)
        n, nb, mt = stats(sol)
        print(
            f'[L{lvl} solve] {tag} -> n_neg={n} feasible={rep.feasible} '
            f'min_T={mt:+.4f} ({(time.time() - t0) / 3600:.2f}h)',
            flush=True,
        )
        np.save(OUT / f'b0039_FULL_mg_L{lvl}.npy', sol)

    n, nb, mt = stats(sol)
    print(
        f'FINAL n_neg={n} n<0.01={nb} min_T={mt:+.6f} '
        f'total_wall={(time.time() - t_run) / 3600:.2f}h',
        flush=True,
    )
    np.save(FINAL, sol)
    print(f'saved {FINAL}', flush=True)


if __name__ == '__main__':
    main()
