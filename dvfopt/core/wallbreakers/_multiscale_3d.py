"""Multi-scale (coarse-to-fine) basin-hop seed for 3D fold elimination.

The one stage that drove the thick dense band 173 -> 6 in the research
(REPORT Part VIII) where a single-scale M10Tet plateaus at ~19. Folds
cluster differently at coarse resolution (box-averaging merges them), so
solving coarse then upsampling lands the fine solve in a *different,
better basin* than solving fine directly.

Pipeline:
  1. Downsample 2x (box-average 2x2x2 blocks; displacements x0.5 to match
     the coarse grid spacing).
  2. M10Tet on the small coarse field (cheap).
  3. Trilinear upsample back (displacements x2). This step is destructive
     — it manufactures transient folds — but it seeds a new basin.
  4. M10Tet polish at fine scale, which recovers from the upsample folds
     into the better basin.

Returns the fine-polished field. Use as a bulk-reduction route for thick
dense chunks where active-band / single-scale M10Tet plateau high.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import zoom

from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


def _downsample_2x(phi):
    """Box-average 2x along each axis; scale displacements by 0.5."""
    _, D, H, W = phi.shape
    Dh, Hh, Wh = D // 2, H // 2, W // 2
    if Dh < 1 or Hh < 1 or Wh < 1:
        return None
    phi_t = phi[:, :2 * Dh, :2 * Hh, :2 * Wh]
    coarse = phi_t.reshape(3, Dh, 2, Hh, 2, Wh, 2).mean(axis=(2, 4, 6))
    return coarse * 0.5


def _upsample_2x(coarse, target_shape):
    """Trilinear upsample to ``target_shape``; scale displacements by 2."""
    out = np.empty((3, *target_shape))
    for c in range(3):
        up = zoom(coarse[c], 2.0, order=1)
        if up.shape != tuple(target_shape):
            full = np.zeros(target_shape, dtype=up.dtype)
            mz = min(up.shape[0], target_shape[0])
            my = min(up.shape[1], target_shape[1])
            mx = min(up.shape[2], target_shape[2])
            full[:mz, :my, :mx] = up[:mz, :my, :mx]
            # edge-replicate any uncovered border planes
            if mz < target_shape[0]:
                full[mz:] = full[mz - 1:mz]
            if my < target_shape[1]:
                full[:, my:] = full[:, my - 1:my]
            if mx < target_shape[2]:
                full[:, :, mx:] = full[:, :, mx - 1:mx]
            up = full
        out[c] = up
    return out * 2.0


def multiscale_seed_3d(
    phi,
    *,
    threshold=0.012,
    coarse_threshold=0.015,
    inner_solve=None,
    verbose=0,
):
    """Coarse-to-fine basin-hop reduction. Returns ``(phi_out, info)``.

    Parameters
    ----------
    phi : ndarray (3, D, H, W)
    threshold : float          fine-polish M10Tet target.
    coarse_threshold : float   coarse-solve M10Tet target (slightly looser).
    inner_solve : callable | None
        ``(field, threshold) -> field`` M10Tet solver. None -> default
        HarmonicALMBarrier3D via Solver (lazy import).
    verbose : int

    Notes
    -----
    Falls back to a single fine M10Tet solve if the chunk is too small to
    downsample (any dim < 2 cubes after halving).
    """
    phi = np.asarray(phi, dtype=np.float64)
    if inner_solve is None:
        def inner_solve(field, thr):
            from dvfopt import (
                HarmonicALMBarrier3DStrategy,
                L1Objective,
                Solver,
                Tet6Constraint3D,
            )
            return Solver(
                constraint=Tet6Constraint3D(shape=field.shape[1:]),
                objective=L1Objective(eps=1e-4),
                strategy=HarmonicALMBarrier3DStrategy(),
                threshold=thr,
            ).fit(field).corrected

    t0 = time.time()
    coarse = _downsample_2x(phi)
    if coarse is None or min(coarse.shape[1:]) < 2:
        if verbose:
            print('  [multiscale] too small to downsample — single fine solve',
                  flush=True)
        out = inner_solve(phi, threshold)
        return out, {'used_multiscale': False, 'wall_s': time.time() - t0}

    if verbose:
        cv = six_tet_min_volume_3d(coarse)
        print(f'  [multiscale] coarse {coarse.shape[1:]} '
              f'n_neg={int((cv<=0).sum())}', flush=True)
    coarse_out = inner_solve(coarse, coarse_threshold)
    ups = _upsample_2x(coarse_out, phi.shape[1:])
    if verbose:
        uv = six_tet_min_volume_3d(ups)
        print(f'  [multiscale] upsampled n_neg={int((uv<=0).sum())} '
              f'(transient) -> fine polish', flush=True)
    out = inner_solve(ups, threshold)
    if verbose:
        ov = six_tet_min_volume_3d(out)
        print(f'  [multiscale] fine n_neg={int((ov<=0).sum())} '
              f'({time.time()-t0:.1f}s)', flush=True)
    return out, {'used_multiscale': True, 'wall_s': time.time() - t0}


__all__ = ['multiscale_seed_3d']
