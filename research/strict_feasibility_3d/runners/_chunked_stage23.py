"""Chunked Stage 2+3 strict-feasibility pass on top of the cached
full-volume Stage 1 output.

Loads `b0039_FULL_stage1.npy` (528-slice 2D-corrected stack), picks
a z-chunk, runs M10Tet at threshold=0.01 (Stage 2) then again at
threshold=0.015 (Stage 3 overshoot). Reports the chunk's residual
3D fold structure.

The point is to demonstrate Stage 2+3 work on a representative
z-band so we can estimate the full-volume runtime budget.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

OUTPUT = _HERE / 'output'
FULL_STAGE1_CACHE = OUTPUT / 'b0039_FULL_stage1.npy'


def _stats(phi, label):
    V = six_tet_volumes_3d(phi)
    return (
        f'{label}  n_neg={int((V<=0).sum()):>7d}  '
        f'n<0.01={int((V<0.01-1e-5).sum()):>7d}  '
        f'min_T={float(V.min()):+.6f}'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--z0', type=int, required=True)
    p.add_argument('--z1', type=int, required=True, help='exclusive')
    args = p.parse_args()

    print(f'Loading full Stage 1 cache: {FULL_STAGE1_CACHE}', flush=True)
    full = np.load(FULL_STAGE1_CACHE)
    print(f'  full shape: {full.shape}', flush=True)

    chunk = np.ascontiguousarray(full[:, args.z0:args.z1])
    print(f'\nChunk z={args.z0}..{args.z1 - 1}  shape={chunk.shape}', flush=True)
    print(_stats(chunk, '  Stage 1 chunk:  '), flush=True)

    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    # Stage 2: M10Tet @ threshold=0.01
    print('\n=== Stage 2: M10Tet @ threshold=0.01 ===', flush=True)
    t0 = time.time()
    solver_s2 = Solver(
        constraint=Tet6Constraint3D(shape=chunk.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.01,
    )
    s2 = solver_s2.fit(chunk).corrected
    t_s2 = time.time() - t0
    print(_stats(s2, '  Stage 2 out:    '), flush=True)
    print(f'  wall: {t_s2:.1f}s', flush=True)

    # Stage 3: M10Tet @ threshold=0.015 (overshoot)
    print('\n=== Stage 3: M10Tet @ threshold=0.015 (overshoot) ===', flush=True)
    t0 = time.time()
    solver_s3 = Solver(
        constraint=Tet6Constraint3D(shape=chunk.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    s3 = solver_s3.fit(s2).corrected
    t_s3 = time.time() - t0
    print(_stats(s3, '  Stage 3 out:    '), flush=True)
    print(f'  wall: {t_s3:.1f}s', flush=True)

    V_final = six_tet_volumes_3d(s3)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < 0.01 - 1e-5).sum())
    print(
        f'\n=== Chunk Final ===\n'
        f'  shape:              {s3.shape}\n'
        f'  n_neg:              {n_neg}\n'
        f'  n_below_threshold:  {n_below}\n'
        f'  min_T:              {float(V_final.min()):+.6f}\n'
        f'  STRICT 100% feas:   {n_neg == 0 and n_below == 0}\n'
        f'  stage 2+3 wall:     {t_s2 + t_s3:.1f}s',
        flush=True,
    )

    # Save chunk result.
    out_path = OUTPUT / f'b0039_FULL_stage3_z{args.z0:03d}_{args.z1:03d}.npy'
    np.save(out_path, s3)
    print(f'\nChunk result saved to {out_path}', flush=True)


if __name__ == '__main__':
    main()
