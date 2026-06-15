"""Alternative pipeline: 2D auto_slp per slice + global M10Tet 3D
post-pass on the stacked field.

M10Tet (HarmonicALMBarrier3DStrategy) succeeded on the 8^3 B0039
subvolume (where m14 catastrophically failed). The 2D-corrected
5-slice stack has only 0.18% folded tets (much sparser than the
16^3 subvolume that hung), so a global M10Tet pass may converge.

If this achieves 100% 3D feasibility, we have a working pipeline:

  Stage 1: 2D auto_slp per slice  → 100% 2D feasibility per slice,
                                   0.18% 3D-folded after stacking
  Stage 2: 3D M10Tet on the stack → cleans up the straddling-tet
                                   folds, achieves 100% 3D feasibility
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt import (
    HarmonicALMBarrier3DStrategy,
    L1Objective,
    Solver,
    Tet6Constraint3D,
)
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

from research.strict_feasibility_2d.runners._compare import run_method as run_2d


def main():
    print('Loading B0039 + z=10..14 chunk...', flush=True)
    arr = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr'].astype(np.float64)

    z_range = list(range(10, 15))
    t_start = time.time()

    # Stage 1: 2D auto_slp per slice.
    print(f'\nStage 1: 2D auto_slp on z={z_range[0]}..{z_range[-1]}', flush=True)
    corrected_slices = []
    for z in z_range:
        rec = run_2d('auto_slp', arr[1:3, z])
        corrected_slices.append(rec['phi_out'])
        print(f'  z={z}: feas={rec["feasible"]} wall={rec["wall_s"]:.1f}s', flush=True)
    t_stage1 = time.time() - t_start

    # Stack into 3D.
    n_slices = len(z_range)
    H, W = arr.shape[2:]
    stack = np.zeros((3, n_slices, H, W), dtype=np.float64)
    for i, phi_2d in enumerate(corrected_slices):
        stack[1, i] = phi_2d[0]
        stack[2, i] = phi_2d[1]

    V_stack = six_tet_volumes_3d(stack)
    print(
        f'\nStage 1 stacked: 3D shape={stack.shape}  '
        f'n_neg={int((V_stack <= 0).sum())} ({int((V_stack <= 0).sum())/V_stack.size*100:.4f}%)  '
        f'min_T={float(V_stack.min()):+.4f}  wall={t_stage1:.1f}s',
        flush=True,
    )

    # Stage 2: M10Tet global 3D pass.
    print('\nStage 2: M10Tet (HarmonicALMBarrier3DStrategy) global 3D pass...', flush=True)
    t1 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=stack.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.01,
    )
    result = solver.fit(stack)
    phi_out = result.corrected
    t_stage2 = time.time() - t1

    V_final = six_tet_volumes_3d(phi_out)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < 0.01 - 1e-5).sum())
    L1 = float(np.abs(phi_out[1:3] - arr[1:3, z_range[0]:z_range[-1] + 1]).sum())
    print(f'\nStage 2 wall: {t_stage2:.1f}s', flush=True)
    print(
        f'\n=== Final ===\n'
        f'  shape:                {phi_out.shape}\n'
        f'  n_neg (V<=0):         {n_neg}\n'
        f'  n_below_threshold:    {n_below}\n'
        f'  min_T:                {float(V_final.min()):+.6f}\n'
        f'  total L1 vs input:    {L1:.1f}\n'
        f'  100% feasible:        {n_neg == 0 and n_below == 0}\n'
        f'  total wall:           {time.time() - t_start:.1f}s',
        flush=True,
    )


if __name__ == '__main__':
    main()
