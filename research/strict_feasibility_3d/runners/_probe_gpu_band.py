"""GPU-barrier speed/feasibility probe on a representative dense band.

Compares the CUDA penalty->log-barrier solver (BarrierTet3DTorchStrategy,
float32, windowed for VRAM) against the CPU M10Tet band cost (band 3
region z[44:76] took 3.75 h to strict 0 on CPU). Measures whether the GPU
path reaches strict 0 and how fast, to decide if the band loop should
switch its inner solver to GPU.

float32 is mandatory: the GPU is a GeForce RTX 3050 (fp64 is ~1/32 fp32).
windowed=True keeps VRAM bounded (full-grid LBFGS history_size=100 over an
~11M-variable band would need ~8.8 GB > 8.6 GB card).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))


def main():
    from dvfopt import (
        Solver, Tet6Constraint3D, L1Objective, BarrierTet3DTorchStrategy,
    )
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    OUT = Path(__file__).parent / 'output'
    THR = 0.01
    phi = np.load(OUT / 'b0039_FULL_stage1.npy').astype(np.float64)
    band = phi[:, 44:76, :, :].copy()  # band-3 region, ~23.8k folds
    mv0 = six_tet_min_volume_3d(band)
    n0, nb0, mt0 = int((mv0 <= 0).sum()), int((mv0 < THR).sum()), float(mv0.min())
    print(f'band z[44:76] {band.shape[1:]} n_neg={n0} n<thr={nb0} '
          f'min_T={mt0:+.4f}', flush=True)

    for mode in ('windowed', 'fullgrid'):
        try:
            strat = BarrierTet3DTorchStrategy(
                dtype='float32', windowed=(mode == 'windowed'), pad=2,
            )
            t0 = time.time()
            res = Solver(
                constraint=Tet6Constraint3D(shape=band.shape[1:]),
                objective=L1Objective(eps=1e-4),
                strategy=strat, threshold=THR,
            ).fit(band)
            out = res.corrected
            dt = time.time() - t0
            mv = six_tet_min_volume_3d(out)
            n, nb, mt = int((mv <= 0).sum()), int((mv < THR).sum()), float(mv.min())
            print(f'[GPU {mode}] n_neg {n0}->{n} n<thr->{nb} min_T={mt:+.5f} '
                  f'feasible={n == 0} ({dt/60:.1f} min)  '
                  f'[CPU M10Tet baseline: 3.75 h to strict 0]', flush=True)
        except Exception as e:  # OOM or other — report and move on
            print(f'[GPU {mode}] FAILED: {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
