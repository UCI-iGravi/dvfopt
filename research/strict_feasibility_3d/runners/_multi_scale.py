"""Multi-scale pyramid: solve at coarse scale, upsample, polish.

Downsample the field 2x in each dimension (average over 2x2x2 voxel
blocks). The coarse field has DIFFERENT fold structure since
averaging smooths some folds out. Run M10Tet at coarse scale (much
smaller problem). Trilinear-upsample the coarse result as initial
guess for fine scale. Run M10Tet polish at fine scale from this
warm-started state.

Hypothesis: the coarse-scale optimum corresponds to a different
local minimum at fine scale than direct M10Tet @ 0.015 finds.
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
    """Box-filter downsample by 2x along each axis. Returns (3, D/2, H/2, W/2)."""
    _, D, H, W = phi.shape
    Dh, Hh, Wh = D // 2, H // 2, W // 2
    # Truncate to even.
    phi_t = phi[:, : 2 * Dh, : 2 * Hh, : 2 * Wh]
    coarse = phi_t.reshape(3, Dh, 2, Hh, 2, Wh, 2).mean(axis=(2, 4, 6))
    # Scale displacements by 0.5 (since the coarse grid has spacing 2 in original units).
    return coarse * 0.5


def upsample_2x(coarse, target_shape):
    """Trilinear upsample to target_shape. Scale displacements back by 2."""
    out = np.empty((3, *target_shape))
    for c in range(3):
        out[c] = zoom(coarse[c], 2.0, order=1)
        # Crop/pad to exact shape.
        if out[c].shape != target_shape:
            out_c = out[c]
            out_full = np.zeros(target_shape, dtype=out_c.dtype)
            mz, my, mx = (
                min(out_c.shape[0], target_shape[0]),
                min(out_c.shape[1], target_shape[1]),
                min(out_c.shape[2], target_shape[2]),
            )
            out_full[:mz, :my, :mx] = out_c[:mz, :my, :mx]
            out[c] = out_full
    return out * 2.0


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    print(f'Loaded, shape={phi.shape}', flush=True)
    V0 = six_tet_volumes_3d(phi)
    print(
        f'Start: n_neg={int((V0 <= 0).sum())}  '
        f'n<0.01={int((V0 < THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    # Downsample.
    coarse = downsample_2x(phi)
    print(f'Coarse shape: {coarse.shape}', flush=True)
    V_c = six_tet_volumes_3d(coarse)
    print(
        f'  coarse stats: n_neg={int((V_c <= 0).sum())}  min_T={float(V_c.min()):+.6f}',
        flush=True,
    )

    # M10Tet @ 0.015 on coarse field.
    print('\n=== M10Tet on coarse field @ threshold=0.015 ===', flush=True)
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    t0 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=coarse.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    coarse_polished = solver.fit(coarse).corrected
    wall_coarse = time.time() - t0
    V_cp = six_tet_volumes_3d(coarse_polished)
    print(
        f'  coarse polished: n_neg={int((V_cp <= 0).sum())}  '
        f'n<0.01={int((V_cp < THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V_cp.min()):+.6f}  wall={wall_coarse:.1f}s',
        flush=True,
    )

    # Upsample.
    upsampled = upsample_2x(coarse_polished, phi.shape[1:])
    print(f'\nUpsampled shape: {upsampled.shape}', flush=True)
    V_u = six_tet_volumes_3d(upsampled)
    print(
        f'  upsampled stats: n_neg={int((V_u <= 0).sum())}  '
        f'n<0.01={int((V_u < THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V_u.min()):+.6f}',
        flush=True,
    )

    # Final polish at fine scale with warm-start from upsampled.
    print(
        '\n=== Final polish at fine scale (M10Tet @ 0.015 from upsampled warm-start) ===',
        flush=True,
    )
    t1 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    fine_polished = solver.fit(upsampled).corrected
    wall_fine = time.time() - t1
    V_f = six_tet_volumes_3d(fine_polished)
    n_neg = int((V_f <= 0).sum())
    n_below = int((V_f < THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(fine_polished - phi).sum())
    print(
        f'  final: n_neg={n_neg}  n<0.01={n_below}  '
        f'min_T={float(V_f.min()):+.6f}  L1_from_input={L1:.1f}  wall={wall_fine:.1f}s',
        flush=True,
    )
    print(
        f'\n=== Final ===\n  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_multiscale.npy', fine_polished)


if __name__ == '__main__':
    main()
