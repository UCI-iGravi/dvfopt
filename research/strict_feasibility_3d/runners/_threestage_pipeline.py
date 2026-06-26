"""Three-stage 100% strict-feasibility 3D pipeline:

  Stage 1: 2D auto_slp per z-slice  → 100% 2-tri feasibility per slice
  Stage 2: M10Tet global 3D         → 0 folds across whole stack
  Stage 3: cluster_slp_3d polish    → push below-threshold cells to
                                      >= threshold (strict 100%)

Each stage's output is cached to .npz so reruns of later stages are
cheap.
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

OUTPUT = _HERE / 'output'
STAGE1_CACHE = OUTPUT / 'b0039_z10_14_stage1.npy'
STAGE2_CACHE = OUTPUT / 'b0039_z10_14_stage2.npy'
THRESHOLD = 0.01


def _stats_line(phi, label):
    V = six_tet_volumes_3d(phi)
    n_neg = int((V <= 0).sum())
    n_below = int((V < THRESHOLD - 1e-5).sum())
    min_T = float(V.min())
    return f'{label}  n_neg={n_neg:>6d}  n<thresh={n_below:>6d}  min_T={min_T:+.6f}'


def _stage1(arr, z_range):
    """2D auto_slp per slice → stacked 3D field."""
    if STAGE1_CACHE.exists():
        print(f'[stage 1] loading cache {STAGE1_CACHE}', flush=True)
        return np.load(STAGE1_CACHE)
    n = len(z_range)
    H, W = arr.shape[2:]
    stack = np.zeros((3, n, H, W), dtype=np.float64)
    for i, z in enumerate(z_range):
        t0 = time.time()
        rec = run_2d('auto_slp', arr[1:3, z])
        stack[1, i] = rec['phi_out'][0]
        stack[2, i] = rec['phi_out'][1]
        print(
            f'  z={z}: feas={rec["feasible"]}  wall={rec["wall_s"]:.1f}s',
            flush=True,
        )
    np.save(STAGE1_CACHE, stack)
    print(f'[stage 1] cached to {STAGE1_CACHE}', flush=True)
    return stack


def _stage2(stack):
    """Global M10Tet on the stacked field."""
    if STAGE2_CACHE.exists():
        print(f'[stage 2] loading cache {STAGE2_CACHE}', flush=True)
        return np.load(STAGE2_CACHE)
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    solver = Solver(
        constraint=Tet6Constraint3D(shape=stack.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=THRESHOLD,
    )
    result = solver.fit(stack)
    np.save(STAGE2_CACHE, result.corrected)
    print(f'[stage 2] cached to {STAGE2_CACHE}', flush=True)
    return result.corrected


def _stage3(phi_in):
    """Below-threshold cluster polish via cluster_slp_3d."""
    phi_out, info = cluster_slp_iter_3d(
        phi_in,
        threshold=THRESHOLD,
        inner_seed='m10',
        max_outer_iters=4,
        polish_below_threshold=True,
        verbose=1,
    )
    return phi_out, info


def main():
    print('Loading B0039...', flush=True)
    arr = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr'].astype(
        np.float64
    )
    z_range = list(range(10, 15))
    t_start = time.time()

    print(f'\n=== Stage 1: 2D auto_slp on z={z_range[0]}..{z_range[-1]} ===', flush=True)
    t1 = time.time()
    stack = _stage1(arr, z_range)
    print(_stats_line(stack, '[stage 1 stack]'), flush=True)
    print(f'  wall: {time.time() - t1:.1f}s', flush=True)

    print('\n=== Stage 2: M10Tet global 3D pass ===', flush=True)
    t2 = time.time()
    phi_s2 = _stage2(stack)
    print(_stats_line(phi_s2, '[stage 2 out]'), flush=True)
    print(f'  wall: {time.time() - t2:.1f}s', flush=True)

    print('\n=== Stage 3: cluster_slp_3d polish below threshold ===', flush=True)
    t3 = time.time()
    phi_s3, info = _stage3(phi_s2)
    print(_stats_line(phi_s3, '[stage 3 out]'), flush=True)
    print(f'  wall: {time.time() - t3:.1f}s', flush=True)
    print(f'  cluster solves: {info["total_cluster_solves"]}', flush=True)

    V_final = six_tet_volumes_3d(phi_s3)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(phi_s3[1:3] - arr[1:3, z_range[0] : z_range[-1] + 1]).sum())
    print(
        f'\n=== Final ===\n'
        f'  shape:             {phi_s3.shape}\n'
        f'  n_neg (V<=0):      {n_neg}\n'
        f'  n_below_threshold: {n_below}\n'
        f'  min_T:             {float(V_final.min()):+.6f}\n'
        f'  L1 vs raw input:   {L1:.1f}\n'
        f'  STRICT 100% feas:  {n_neg == 0 and n_below == 0}\n'
        f'  total wall:        {time.time() - t_start:.1f}s',
        flush=True,
    )


if __name__ == '__main__':
    main()
