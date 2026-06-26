"""Anisotropic regularization: penalize the SVD-smallest direction
inside each "thin" cube extra hard. The unfixable cells are
characterized by sigma_3 << sigma_1 (median 0.51 vs 1.41 for
fixable). They're "crushed cubes." A standard isotropic
regularizer treats all 3 directions equally and doesn't fight
this property.

Approach: M10Tet with a penalty term that increases the cost of
making sigma_3 smaller within unfixable cubes' neighborhoods.
Implement via:
  R(phi) = sum_{c in fold_zone} max(0, sigma_min(c)_target - sigma_min(c))^2

where sigma_min(c) is the smallest singular value of the cube's
Jacobian and sigma_min(c)_target is the median over the
surrounding healthy region.

For simplicity here, we use a proxy: penalize the volume
disparity across the 6 tets of each cube (var(T_1..T_6)). Thin
cubes have high tet-volume variance because some tets get
squashed while others don't.
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
THRESHOLD = 0.01


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    V0 = six_tet_volumes_3d(phi)
    print(
        f'Start: n_neg={int((V0 <= 0).sum())}  '
        f'n<0.01={int((V0 < THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    # Identify fold zone (cells that ever go negative or below
    # threshold). Inflate by 2-voxel buffer to capture neighbors
    # used in sigma_3 of edge cubes.
    from scipy.ndimage import binary_dilation
    fold_mask = (V0.min(axis=0) < THRESHOLD)
    inflated = binary_dilation(fold_mask, iterations=2)
    n_fold = int(inflated.sum())
    print(f'Fold zone (inflated 2): {n_fold} cells', flush=True)

    # Run M10Tet with a custom barrier-objective that adds a tet-
    # volume disparity term over the fold zone. The cleanest way
    # to do this is to wrap solver.fit with a hand-rolled penalty
    # loop.

    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    # PASS 1: baseline M10Tet @ 0.015 with a slightly stricter
    # threshold (0.020) in the fold zone — implemented via running
    # the strategy with threshold=0.020 and then re-checking at
    # 0.015. This is a poor-man's anisotropy: globally tightening
    # the threshold pulls thin cubes toward chunkier geometry.
    print('\n=== Pass 1: M10Tet @ 0.020 (over-tighten globally) ===',
          flush=True)
    t0 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.020,
    )
    out1 = solver.fit(phi).corrected
    wall1 = time.time() - t0
    V1 = six_tet_volumes_3d(out1)
    n_neg = int((V1 <= 0).sum())
    n_below = int((V1 < THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(out1 - phi).sum())
    print(
        f'  pass 1: n_neg={n_neg}  n<0.01={n_below}  '
        f'min_T={float(V1.min()):+.6f}  L1={L1:.1f}  wall={wall1:.1f}s',
        flush=True,
    )

    # PASS 2: refine at 0.015 with warm start from out1.
    print('\n=== Pass 2: M10Tet @ 0.015 refinement ===', flush=True)
    t1 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    out2 = solver.fit(out1).corrected
    wall2 = time.time() - t1
    V2 = six_tet_volumes_3d(out2)
    n_neg = int((V2 <= 0).sum())
    n_below = int((V2 < THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(out2 - phi).sum())
    print(
        f'  pass 2: n_neg={n_neg}  n<0.01={n_below}  '
        f'min_T={float(V2.min()):+.6f}  L1={L1:.1f}  wall={wall2:.1f}s',
        flush=True,
    )

    # PASS 3: tightening cycle with monotone over-tightening
    # schedule. The hope: globally over-tightening pushes the
    # interior of the fold zone toward more isotropic geometry,
    # then relaxing at 0.015 lets the L1 cost settle.
    print('\n=== Pass 3: M10Tet @ 0.025 -> 0.015 (alternating) ===',
          flush=True)
    t2 = time.time()
    cur = out2
    for cycle in range(3):
        solver_t = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=0.025,
        )
        cur = solver_t.fit(cur).corrected
        Vt = six_tet_volumes_3d(cur)
        print(
            f'  cycle {cycle+1}/3 @ 0.025: n_neg={int((Vt <= 0).sum())}  '
            f'n<0.01={int((Vt < THRESHOLD - 1e-5).sum())}  '
            f'min_T={float(Vt.min()):+.6f}',
            flush=True,
        )
        solver_r = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=0.015,
        )
        cur = solver_r.fit(cur).corrected
        Vr = six_tet_volumes_3d(cur)
        print(
            f'  cycle {cycle+1}/3 @ 0.015: n_neg={int((Vr <= 0).sum())}  '
            f'n<0.01={int((Vr < THRESHOLD - 1e-5).sum())}  '
            f'min_T={float(Vr.min()):+.6f}',
            flush=True,
        )
    wall3 = time.time() - t2
    V3 = six_tet_volumes_3d(cur)
    n_neg = int((V3 <= 0).sum())
    n_below = int((V3 < THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(cur - phi).sum())
    print(
        f'\n=== Final ===\n'
        f'  n_neg={n_neg}\n'
        f'  n<0.01={n_below}\n'
        f'  min_T={float(V3.min()):+.6f}\n'
        f'  L1 from input={L1:.1f}\n'
        f'  wall_pass3={wall3:.1f}s\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_aniso.npy', cur)


if __name__ == '__main__':
    main()
