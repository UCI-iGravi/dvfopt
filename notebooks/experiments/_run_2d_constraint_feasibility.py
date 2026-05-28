"""Feasibility comparison on the worst-case slices:
2-triangle constraint vs central-difference Jacobian determinant.

Same solver (barrier, full-grid, L2 anchor) -- the only variable is the
constraint function. Reports both metrics on both runs so we can see
cross-feasibility:

    - is a Jdet-feasible field also 2-tri-feasible?  (no)
    - is a 2-tri-feasible field also Jdet-feasible?  (mathematically yes,
      since T1, T2 >= 0 implies J >= 0 in this formulation -- but the
      2-tri solver may not actually reach feasibility on these slices)

Tests on the slices the main runner could not crack (a representative
subset of the densest: z = 12, 13, 17). Each cell of the output table
shows BOTH the constraint each barrier optimised AND the *cross-metric*
(the other constraint evaluated on the same corrected field).
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO)

import numpy as np

from dvfopt.core.iterative2d_barrier import iterative_2d_barrier
from dvfopt.core.iterative2d_tri_barrier import iterative_2d_tri_barrier
from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

THRESHOLD = 0.01
DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')

TARGET_Z = [12, 13, 17]              # representative worst slices

# Truncated schedules so the benchmark finishes in reasonable time;
# the trends are visible by lam=1e6.
LAM = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
MU = (1e-1, 1e-2, 1e-3)
MAX_ITER = 200


def _stats(phi2):
    """phi2: (2, H, W) [dy, dx]. Returns dict with 2-tri and Jdet stats."""
    T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
    J = np.squeeze(jacobian_det2D(phi2))
    return dict(
        tri_neg=int((T1 <= 0).sum() + (T2 <= 0).sum()),
        tri_min=float(min(T1.min(), T2.min())),
        jdet_neg=int((J <= 0).sum()),
        jdet_min=float(J.min()),
    )


def _coerce_2hw(arr):
    """Anything returned by the solvers -> (2, H, W) [dy, dx]."""
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 3:        # (3, 1, H, W)
        return np.stack([arr[1, 0], arr[2, 0]])
    if arr.ndim == 3 and arr.shape[0] == 3:        # (3, H, W)
        return np.stack([arr[1], arr[2]])
    if arr.ndim == 3 and arr.shape[0] == 2:        # (2, H, W)
        return arr
    raise ValueError(f'cannot coerce shape {arr.shape}')


def run_jdet_barrier(phi_full_volume, z):
    """Existing Jdet barrier (full-grid). Returns (corrected 2hw, wall)."""
    deformation = phi_full_volume[:, z:z + 1].copy()
    t0 = time.time()
    out = iterative_2d_barrier(
        deformation, threshold=THRESHOLD, margin=1e-3,
        lam_schedule=LAM, mu_schedule=MU,
        max_minimize_iter=MAX_ITER, windowed=False, verbose=0)
    return _coerce_2hw(out), time.time() - t0


def run_tri_barrier(phi_full_volume, z):
    """New 2-tri barrier (full-grid). Returns (corrected 2hw, wall)."""
    phi = np.stack([phi_full_volume[1, z].copy(), phi_full_volume[2, z].copy()])
    t0 = time.time()
    out = iterative_2d_tri_barrier(
        phi, threshold=THRESHOLD, margin=1e-3,
        lam_schedule=LAM, mu_schedule=MU,
        max_minimize_iter=MAX_ITER, anchor='l2', verbose=0)
    return _coerce_2hw(out), time.time() - t0


def main():
    phi_full = np.load(DATA_PATH)
    rows = []
    for z in TARGET_Z:
        phi0 = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
        s_init = _stats(phi0)
        print(f'\n========== z={z} ==========', flush=True)
        print(f'  init: 2tri n_neg={s_init["tri_neg"]} min={s_init["tri_min"]:+.4f}  '
              f'jdet n_neg={s_init["jdet_neg"]} min={s_init["jdet_min"]:+.4f}',
              flush=True)

        print(f'  [Jdet barrier (central diff)]', flush=True)
        phi_j, wall_j = run_jdet_barrier(phi_full, z)
        s_j = _stats(phi_j)
        print(f'    -> 2tri n_neg={s_j["tri_neg"]:5d} min={s_j["tri_min"]:+.5f} |  '
              f'jdet n_neg={s_j["jdet_neg"]:5d} min={s_j["jdet_min"]:+.5f}  '
              f'({wall_j:.0f}s)', flush=True)

        print(f'  [2-tri barrier]', flush=True)
        phi_t, wall_t = run_tri_barrier(phi_full, z)
        s_t = _stats(phi_t)
        print(f'    -> 2tri n_neg={s_t["tri_neg"]:5d} min={s_t["tri_min"]:+.5f} |  '
              f'jdet n_neg={s_t["jdet_neg"]:5d} min={s_t["jdet_min"]:+.5f}  '
              f'({wall_t:.0f}s)', flush=True)

        rows.append(dict(z=z, init=s_init, jdet=s_j, tri=s_t,
                         wall_j=wall_j, wall_t=wall_t))

    # --- summary table ---
    print('\n' + '=' * 92, flush=True)
    print('FEASIBILITY COMPARISON  (threshold = 0.01 for both metrics)', flush=True)
    print(f'{"z":>4s} | {"solver":>14s} | {"2-tri":>20s} | {"Jdet (central diff)":>22s} | {"wall":>6s}',
          flush=True)
    print(f'{"":>4s} | {"":>14s} | {"n_neg":>5s} {"min":>14s} | {"n_neg":>5s} {"min":>14s} |',
          flush=True)
    for r in rows:
        print(f'{r["z"]:4d} | {"(init)":>14s} | '
              f'{r["init"]["tri_neg"]:5d} {r["init"]["tri_min"]:+14.5f} | '
              f'{r["init"]["jdet_neg"]:5d} {r["init"]["jdet_min"]:+14.5f} |', flush=True)
        print(f'{"":>4s} | {"Jdet barrier":>14s} | '
              f'{r["jdet"]["tri_neg"]:5d} {r["jdet"]["tri_min"]:+14.5f} | '
              f'{r["jdet"]["jdet_neg"]:5d} {r["jdet"]["jdet_min"]:+14.5f} | '
              f'{r["wall_j"]:6.0f}', flush=True)
        print(f'{"":>4s} | {"2-tri barrier":>14s} | '
              f'{r["tri"]["tri_neg"]:5d} {r["tri"]["tri_min"]:+14.5f} | '
              f'{r["tri"]["jdet_neg"]:5d} {r["tri"]["jdet_min"]:+14.5f} | '
              f'{r["wall_t"]:6.0f}', flush=True)


if __name__ == '__main__':
    main()
