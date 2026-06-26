"""Multi-scale v2: extends the v1 pipeline with multiple residual
polish stages, aiming for n_neg=0.

v1 produced 6 residual folds (best of all 14 methods tested).
This script:

  1. Runs the v1 multi-scale pipeline (coarse box-avg + M10Tet,
     trilinear upsample, fine M10Tet polish).
  2. Saves the v1 result unconditionally as
     `b0039_z0_15_ms_v1.npy`.
  3. Applies M14Tet (refine-repair) as additional polish.
  4. If still > 0, applies M14Schwarz3D (cluster-decomposition
     + per-cluster refine-repair) targeting only the residual
     fold cells.
  5. If still > 0, applies M10Tet at threshold=0.012 (slightly
     lower) as a final tightening pass — the cells that survive
     0.015 may resolve at 0.012.

Goal: drive n_neg to 0 from the 6-fold multi-scale baseline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.ndimage import zoom

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d


OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def downsample_2x(phi):
    _, D, H, W = phi.shape
    Dh, Hh, Wh = D // 2, H // 2, W // 2
    phi_t = phi[:, :2*Dh, :2*Hh, :2*Wh]
    coarse = phi_t.reshape(3, Dh, 2, Hh, 2, Wh, 2).mean(axis=(2, 4, 6))
    return coarse * 0.5


def upsample_2x(coarse, target_shape):
    out = np.empty((3, *target_shape))
    for c in range(3):
        out[c] = zoom(coarse[c], 2.0, order=1)
        if out[c].shape != target_shape:
            out_c = out[c]
            out_full = np.zeros(target_shape, dtype=out_c.dtype)
            mz, my, mx = min(out_c.shape[0], target_shape[0]), min(out_c.shape[1], target_shape[1]), min(out_c.shape[2], target_shape[2])
            out_full[:mz, :my, :mx] = out_c[:mz, :my, :mx]
            out[c] = out_full
    return out * 2.0


def report(phi, label, phi_input=None):
    V = six_tet_volumes_3d(phi)
    n_neg = int((V <= 0).sum())
    n_below = int((V < THRESHOLD - 1e-5).sum())
    mn = float(V.min())
    L1 = '' if phi_input is None else f'  L1_from_input={float(np.abs(phi - phi_input).sum()):.1f}'
    print(f'{label}: n_neg={n_neg}  n<0.01={n_below}  min_T={mn:+.6f}{L1}',
          flush=True)
    return n_neg, n_below, mn


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'Input')

    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        HarmonicALMRefineRepair3DStrategy,
        SchwarzHarmonicALMRefineRepair3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    # === STAGE 1: multi-scale (same as v1). ===
    print('\n=== STAGE 1: multi-scale pyramid (coarse + upsample + fine polish) ===',
          flush=True)
    coarse = downsample_2x(phi_input)
    report(coarse, '  coarse')
    t0 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=coarse.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    coarse_polished = solver.fit(coarse).corrected
    print(f'  coarse polish wall={time.time()-t0:.1f}s', flush=True)
    report(coarse_polished, '  coarse polished')

    upsampled = upsample_2x(coarse_polished, phi_input.shape[1:])
    report(upsampled, '  upsampled')

    t1 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi_input.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    ms_v1 = solver.fit(upsampled).corrected
    print(f'  fine polish wall={time.time()-t1:.1f}s', flush=True)
    n_neg, n_below, mn = report(ms_v1, 'MS_V1 result', phi_input)
    np.save(OUTPUT / 'b0039_z0_15_ms_v1.npy', ms_v1)
    if n_neg == 0 and n_below == 0:
        print('\n*** STRICT 100% feasible after stage 1 ***', flush=True)
        np.save(OUTPUT / 'b0039_z0_15_strict_via_ms_v2.npy', ms_v1)
        return

    # === STAGE 2: M14Tet polish (refine + repair). ===
    print('\n=== STAGE 2: M14Tet polish on multi-scale residual ===', flush=True)
    t2 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi_input.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMRefineRepair3DStrategy(),
        threshold=0.015,
    )
    after_m14 = solver.fit(ms_v1).corrected
    print(f'  m14 wall={time.time()-t2:.1f}s', flush=True)
    n_neg, n_below, mn = report(after_m14, 'after M14Tet', phi_input)
    np.save(OUTPUT / 'b0039_z0_15_ms_v2_m14.npy', after_m14)
    if n_neg == 0 and n_below == 0:
        print('\n*** STRICT 100% feasible after stage 2 ***', flush=True)
        np.save(OUTPUT / 'b0039_z0_15_strict_via_ms_v2.npy', after_m14)
        return

    # === STAGE 3: M14Schwarz3D (cluster decomposition + per-cluster refine-repair). ===
    print('\n=== STAGE 3: M14Schwarz3D on residual ===', flush=True)
    t3 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi_input.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=SchwarzHarmonicALMRefineRepair3DStrategy(),
        threshold=0.015,
    )
    after_schwarz = solver.fit(after_m14).corrected
    print(f'  schwarz wall={time.time()-t3:.1f}s', flush=True)
    n_neg, n_below, mn = report(after_schwarz, 'after M14Schwarz3D', phi_input)
    np.save(OUTPUT / 'b0039_z0_15_ms_v2_schwarz.npy', after_schwarz)
    if n_neg == 0 and n_below == 0:
        print('\n*** STRICT 100% feasible after stage 3 ***', flush=True)
        np.save(OUTPUT / 'b0039_z0_15_strict_via_ms_v2.npy', after_schwarz)
        return

    # === STAGE 4: M10Tet at lower threshold 0.012 (final tightening). ===
    print('\n=== STAGE 4: M10Tet @ 0.012 final tightening ===', flush=True)
    t4 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi_input.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.012,
    )
    after_tight = solver.fit(after_schwarz).corrected
    print(f'  tighten wall={time.time()-t4:.1f}s', flush=True)
    n_neg, n_below, mn = report(after_tight, 'after M10Tet @ 0.012', phi_input)
    np.save(OUTPUT / 'b0039_z0_15_ms_v2_tight.npy', after_tight)
    if n_neg == 0 and n_below == 0:
        print('\n*** STRICT 100% feasible after stage 4 ***', flush=True)
        np.save(OUTPUT / 'b0039_z0_15_strict_via_ms_v2.npy', after_tight)
        return

    print(f'\n=== Final ===\n  best residual after 4 stages: n_neg={n_neg}, '
          f'n<0.01={n_below}, min_T={mn:+.6f}', flush=True)
    print('  STRICT 100% feas: False', flush=True)


if __name__ == '__main__':
    main()
