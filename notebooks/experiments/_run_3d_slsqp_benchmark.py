"""3D SLSQP benchmark.

Runs ``iterative_3d`` (windowed SLSQP, Jacobian-determinant constraint) on:

  1. small synthetic random DVF (5x5x5)         -- should converge quickly
  2. medium synthetic random DVF (10x10x10)     -- moderate
  3. real-data downsampled by 1/4 (132x80x114)  -- realistic load
  4. real-data 3D crop, 12 slices around z=12   -- the dense region

Reports init/final neg-Jdet count, init/final min Jdet, wall time, L2 distortion.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO)

import numpy as np
from scipy.ndimage import zoom

from dvfopt import jacobian_det3D, generate_random_dvf_3d, scale_dvf_3d
from dvfopt.core.slsqp.iterative3d import iterative_3d

DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')
THRESHOLD = 0.01
SEED = 42


def jdet_stats(phi):
    """Return (n_neg, min_J, max_J) for a (3,D,H,W) field."""
    J = jacobian_det3D(phi)
    return int((J <= 0).sum()), float(J.min()), float(J.max())


def run_case(label, phi, *, max_iterations=20, max_minimize_iter=200,
             max_window_voxels=1000, verbose=1):
    """Run iterative_3d on phi; return summary dict."""
    n0, m0, mx0 = jdet_stats(phi)
    print(f'\n--- {label} ---', flush=True)
    print(f'  shape={phi.shape}   init: neg={n0}  min_J={m0:+.4f}  '
          f'max_J={mx0:+.4f}', flush=True)
    t0 = time.time()
    try:
        result = iterative_3d(
            phi.copy(),
            threshold=THRESHOLD,
            max_iterations=max_iterations,
            max_minimize_iter=max_minimize_iter,
            max_window_voxels=max_window_voxels,
            verbose=verbose,
        )
    except Exception as exc:
        wall = time.time() - t0
        print(f'  EXCEPTION after {wall:.1f}s: {type(exc).__name__}: {exc}',
              flush=True)
        return dict(label=label, shape=tuple(phi.shape),
                    init_neg=n0, init_min=m0,
                    final_neg=None, final_min=None,
                    wall_s=wall, error=str(exc), feasible=False)
    wall = time.time() - t0
    corrected = result if isinstance(result, np.ndarray) else result[0]
    n1, m1, _ = jdet_stats(corrected)
    l2 = float(np.linalg.norm((corrected - phi).ravel()))
    feas = (n1 == 0 and m1 >= THRESHOLD - 1e-5)
    print(f'  final: neg={n1}  min_J={m1:+.4f}  L2={l2:.4f}  '
          f'wall={wall:.1f}s  {"CONVERGED" if feas else "still folded"}',
          flush=True)
    return dict(label=label, shape=tuple(phi.shape),
                init_neg=n0, init_min=m0,
                final_neg=n1, final_min=m1, L2=l2,
                wall_s=wall, feasible=feas, error=None)


def main():
    rows = []

    # ---- 1. synthetic small (5x5x5) ----
    rng_dvf = generate_random_dvf_3d((3, 3, 3, 3), 4.0, SEED)
    phi = scale_dvf_3d(rng_dvf, (5, 5, 5))
    rows.append(run_case('synthetic 5x5x5', phi,
                         max_iterations=25, max_window_voxels=125))

    # ---- 2. synthetic medium (10x10x10) ----
    rng_dvf = generate_random_dvf_3d((3, 4, 4, 4), 4.0, SEED + 1)
    phi = scale_dvf_3d(rng_dvf, (10, 10, 10))
    rows.append(run_case('synthetic 10x10x10', phi,
                         max_iterations=25, max_window_voxels=500))

    # ---- 3. real downsampled 1/4 ----
    print('\nloading real DVF...', flush=True)
    phi_real = np.load(DATA_PATH)
    print(f'  full shape: {phi_real.shape}', flush=True)
    # zoom each channel with order=1 (linear)
    factor = 1 / 4
    phi_ds = np.stack([zoom(phi_real[c], factor, order=1)
                       for c in range(3)])
    # rescale displacements (zoom resamples values but the spatial scale
    # changed by 4x -> displacements should shrink by 4x to keep the field
    # consistent)
    phi_ds *= factor
    rows.append(run_case('real 1/4 downsampled', phi_ds,
                         max_iterations=20, max_window_voxels=1500))

    # ---- 4. real crop, 12 slices around z=12 (the dense region) ----
    z0, z1 = 6, 18
    phi_crop = phi_real[:, z0:z1].copy()
    rows.append(run_case(f'real crop z={z0}-{z1-1}', phi_crop,
                         max_iterations=15, max_window_voxels=1500))

    # ---- summary ----
    print('\n' + '=' * 78, flush=True)
    print(f'{"case":>26s} | {"shape":>16s} | {"init_neg":>9s} {"final_neg":>10s} | '
          f'{"init_min":>9s} {"final_min":>10s} | {"wall_s":>7s}  result', flush=True)
    for r in rows:
        shape_str = 'x'.join(str(s) for s in r['shape'])
        fneg = r['final_neg'] if r['final_neg'] is not None else '----'
        fmin = (f'{r["final_min"]:+10.4f}' if r['final_min'] is not None
                else '         -')
        result = ('CONVERGED' if r['feasible']
                  else (r['error'][:25] if r.get('error') else 'still folded'))
        print(f'{r["label"]:>26s} | {shape_str:>16s} | '
              f'{r["init_neg"]:9d} {str(fneg):>10s} | '
              f'{r["init_min"]:+9.4f} {fmin:>10s} | '
              f'{r["wall_s"]:7.1f}  {result}', flush=True)


if __name__ == '__main__':
    main()
