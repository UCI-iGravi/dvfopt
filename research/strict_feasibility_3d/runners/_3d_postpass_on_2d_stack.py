"""Two-stage 100% 3D feasibility pipeline:

  Stage 1: 2D auto_slp per z-slice (achieves per-slice 2-triangle
           feasibility).
  Stage 2: 3D cluster_slp on the stacked field (fixes the straddling-
           tet folds that emerge from mismatched corrections between
           adjacent slices).

The first stage alone leaves ~0.18% of 3D tets folded (vs ~1.6% in
the raw B0039); these folds are localised to z-boundaries between
slices with different corrections, so cluster decomposition can
break them into small per-cluster LP problems.
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
from research.strict_feasibility_2d.runners._compare import run_method as run_2d
from research.strict_feasibility_3d.algorithms.cluster_lp_6tet import (
    cluster_slp_iter_3d,
)


def main():
    print('Loading B0039 + z=10..14 chunk...', flush=True)
    arr = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr'].astype(np.float64)

    z_range = list(range(10, 15))
    t0 = time.time()
    print(f'\nStage 1: 2D auto_slp on z={z_range[0]}..{z_range[-1]}', flush=True)
    corrected_slices = []
    for z in z_range:
        phi_2d = arr[1:3, z]
        rec = run_2d('auto_slp', phi_2d)
        corrected_slices.append(rec['phi_out'])
        print(f'  z={z}: feas={rec["feasible"]} wall={rec["wall_s"]:.1f}s', flush=True)
    print(f'Stage 1 total: {time.time() - t0:.1f}s', flush=True)

    # Stack into 3D field.
    n_slices = len(z_range)
    H, W = arr.shape[2:]
    stack = np.zeros((3, n_slices, H, W), dtype=np.float64)
    for i, phi_2d in enumerate(corrected_slices):
        stack[1, i] = phi_2d[0]
        stack[2, i] = phi_2d[1]

    V_stack = six_tet_volumes_3d(stack)
    n_neg_stack = int((V_stack <= 0).sum())
    print(
        f'\nStage 1 output stacked: 3D shape={stack.shape}  '
        f'n_neg={n_neg_stack} ({n_neg_stack/V_stack.size*100:.4f}%)  '
        f'min_T={float(V_stack.min()):+.4f}',
        flush=True,
    )

    if n_neg_stack == 0:
        print('Already 100% 3D feasible after 2D-only pass — no post-pass needed.', flush=True)
        return

    # Stage 2: 3D cluster_slp post-pass.
    print('\nStage 2: 3D cluster_slp post-pass...', flush=True)
    t1 = time.time()
    phi_out, info = cluster_slp_iter_3d(
        stack, threshold=0.01, inner_seed='m10', verbose=1,
    )
    stage2_wall = time.time() - t1
    V_final = six_tet_volumes_3d(phi_out)
    n_neg_final = int((V_final <= 0).sum())
    below = int((V_final < 0.01 - 1e-5).sum())
    L1_total = float(np.abs(phi_out[1:3] - arr[1:3, z_range[0]:z_range[-1]+1]).sum())
    print(f'\nStage 2 total: {stage2_wall:.1f}s', flush=True)
    print(
        f'\n=== Final ===\n'
        f'  shape: {phi_out.shape}\n'
        f'  n_neg (V<=0):         {n_neg_final}\n'
        f'  n_below_threshold:    {below}\n'
        f'  min_T:                {float(V_final.min()):+.6f}\n'
        f'  total L1 vs input:    {L1_total:.1f}\n'
        f'  total wall:           {time.time() - t0:.1f}s\n'
        f'  cluster solves done:  {info["total_cluster_solves"]}',
        flush=True,
    )


if __name__ == '__main__':
    main()
