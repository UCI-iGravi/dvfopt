"""Benchmark correction methods on the four reference 64x91 B0039 slices.

Methods are split by their *target* criterion — Jdet methods and
triangle methods are NOT the same problem. Mixing the two constraints
in a single solver run is redundant: the strict 4-triangle constraint
already implies Jdet >= threshold at the cell level.

Jdet-target methods:
  * iterative_serial      (SLSQP windowed, Jdet)
  * iterative_parallel    (SLSQP windowed, Jdet, multiprocess)
  * iterative_2d_barrier  (penalty -> log-barrier, Jdet)
  * DVFopt(constraint='jdet')

Triangle-target methods:
  * iterative_serial(enforce_triangles=True)  (SLSQP windowed, strict 4-tri)
  * iterative_2d_tri_barrier                  (penalty -> log-barrier, 2-tri)

The four slices are the 'old ones' from data/test_cases:
  02a_64x91_slice90, 02b_64x91_slice200, 02c_64x91_slice350, 02d_64x91_slice500
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import contextlib
import io

import numpy as np

# Run from repo root so dvfopt imports work without install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dvfopt import (
    iterative_serial, iterative_parallel,
    jacobian_det2D,
)
from dvfopt.core.iterative2d_barrier import iterative_2d_barrier
from dvfopt.core.iterative2d_tri_barrier import iterative_2d_tri_barrier
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.unified import DVFopt, DVFoptConfig


SLICES = [
    ("slice090", "data/test_cases/02a_64x91_slice90.npy"),
    ("slice200", "data/test_cases/02b_64x91_slice200.npy"),
    ("slice350", "data/test_cases/02c_64x91_slice350.npy"),
    ("slice500", "data/test_cases/02d_64x91_slice500.npy"),
]


def _stats_2d(phi2: np.ndarray) -> dict:
    """phi2 shape (2, H, W) with channels [dy, dx]."""
    J = jacobian_det2D(phi2)
    T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
    return dict(
        jdet_neg=int((J <= 0).sum()),
        jdet_min=float(J.min()),
        tri_neg=int((T1 <= 0).sum() + (T2 <= 0).sum()),
        tri_min=float(min(T1.min(), T2.min())),
    )


def _quiet_call(fn, *args, **kwargs):
    """Run fn capturing stdout to keep the table output clean."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn(*args, **kwargs)
    return result


def run_slsqp_serial_jdet(deformation):
    return _quiet_call(iterative_serial, deformation, verbose=0,
                        threshold=0.01, max_iterations=200,
                        max_minimize_iter=200)


def run_slsqp_serial_tri(deformation):
    # Triangle-only constraint (Jdet constraint is now skipped internally
    # since the 4-triangle constraint already implies Jdet feasibility at
    # the cell level).
    return _quiet_call(iterative_serial, deformation, verbose=0,
                        threshold=0.01, max_iterations=80,
                        max_minimize_iter=100, enforce_triangles=True)


def run_slsqp_parallel_jdet(deformation):
    return _quiet_call(iterative_parallel, deformation, verbose=0,
                        threshold=0.01, max_iterations=200,
                        max_minimize_iter=200, max_workers=4)


def run_barrier_jdet(deformation):
    return _quiet_call(iterative_2d_barrier, deformation, verbose=0,
                        threshold=0.01, max_iterations=20,
                        max_minimize_iter=200, windowed=True, pad=2)


def run_barrier_tri(deformation):
    phi2 = np.stack([deformation[1, 0], deformation[2, 0]])
    return _quiet_call(iterative_2d_tri_barrier, phi2, threshold=0.01,
                        max_minimize_iter=300, verbose=0)


def run_dvfopt_auto_jdet(deformation):
    cfg = DVFoptConfig(solver='auto', constraint='jdet',
                       threshold=0.01, verbose=0)
    res = _quiet_call(DVFopt(cfg).fit, deformation)
    return res.corrected  # (3, H, W) for (3, 1, H, W) input


# Group by target criterion. "jdet" methods enforce only Jdet >= threshold;
# "tri" methods enforce all 4 triangles per cell >= threshold (strict
# PL-bijectivity, full vertex coverage). These are *different* problems,
# so per-group feasibility is the right comparison axis.
METHODS_JDET = [
    ("slsqp_serial_jdet",    run_slsqp_serial_jdet),
    ("slsqp_parallel_jdet",  run_slsqp_parallel_jdet),
    ("barrier_jdet",         run_barrier_jdet),
    ("dvfopt_auto_jdet",     run_dvfopt_auto_jdet),
]
METHODS_TRI = [
    ("slsqp_serial_tri",     run_slsqp_serial_tri),
    ("barrier_tri",          run_barrier_tri),
]
METHODS = METHODS_JDET + METHODS_TRI


def _coerce_to_phi2(corrected) -> np.ndarray:
    """Normalize a method's return value to (2, H, W) = (dy, dx)."""
    arr = np.asarray(corrected)
    if arr.ndim == 4:                            # (3, 1, H, W)
        return np.stack([arr[1, 0], arr[2, 0]])
    if arr.ndim == 3 and arr.shape[0] == 2:      # (2, H, W) — already
        return arr
    if arr.ndim == 3 and arr.shape[0] == 3:      # (3, H, W)
        return np.stack([arr[1], arr[2]])
    raise ValueError(f"unexpected shape {arr.shape}")


PER_METHOD_TIMEOUT_S = 300.0  # any method exceeding this aborts with status='timeout'


def run_one(slice_path: str, method_name: str, method_fn):
    deformation = np.load(slice_path)
    phi_init = np.stack([deformation[1, 0], deformation[2, 0]])
    init = _stats_2d(phi_init)
    t0 = time.perf_counter()
    timed_out = False
    try:
        out = method_fn(deformation.copy())
    except Exception as exc:
        wall = time.perf_counter() - t0
        return dict(
            method=method_name, wall_s=wall,
            init_jdet_neg=init['jdet_neg'], init_jdet_min=init['jdet_min'],
            init_tri_neg=init['tri_neg'],   init_tri_min=init['tri_min'],
            final_jdet_neg=-1, final_jdet_min=float('nan'),
            final_tri_neg=-1, final_tri_min=float('nan'),
            l2=float('nan'),
            jdet_feasible=False, tri_feasible=False,
            error=f'{type(exc).__name__}: {exc}',
        )
    wall = time.perf_counter() - t0
    phi2 = _coerce_to_phi2(out)
    final = _stats_2d(phi2)
    l2 = float(np.linalg.norm(phi2 - phi_init))
    return dict(
        method=method_name,
        wall_s=wall,
        init_jdet_neg=init['jdet_neg'], init_jdet_min=init['jdet_min'],
        init_tri_neg=init['tri_neg'],   init_tri_min=init['tri_min'],
        final_jdet_neg=final['jdet_neg'], final_jdet_min=final['jdet_min'],
        final_tri_neg=final['tri_neg'],   final_tri_min=final['tri_min'],
        l2=l2,
        jdet_feasible=(final['jdet_neg'] == 0 and final['jdet_min'] >= 0.01 - 1e-5),
        tri_feasible=(final['tri_neg'] == 0),
        error='',
    )


def main():
    rows = []
    print(f"\n{'slice':<10} {'method':<22} {'wall_s':>7} "
          f"{'jdet_neg':>10} {'jdet_min':>10} "
          f"{'tri_neg':>9} {'tri_min':>10} "
          f"{'L2':>8} {'feas_J':>7} {'feas_T':>7}")
    print('-' * 110)
    init_printed = set()
    for slice_name, slice_path in SLICES:
        for method_name, method_fn in METHODS:
            row = run_one(slice_path, method_name, method_fn)
            row['slice'] = slice_name
            rows.append(row)
            if slice_name not in init_printed:
                print(f"{slice_name:<10} {'(initial)':<22} {'':>7} "
                      f"{row['init_jdet_neg']:>10d} {row['init_jdet_min']:>+10.4f} "
                      f"{row['init_tri_neg']:>9d} {row['init_tri_min']:>+10.4f} "
                      f"{'':>8} {'':>7} {'':>7}")
                init_printed.add(slice_name)
            print(f"{slice_name:<10} {method_name:<22} {row['wall_s']:>7.2f} "
                  f"{row['final_jdet_neg']:>10d} {row['final_jdet_min']:>+10.4f} "
                  f"{row['final_tri_neg']:>9d} {row['final_tri_min']:>+10.4f} "
                  f"{row['l2']:>8.2f} {str(row['jdet_feasible']):>7} "
                  f"{str(row['tri_feasible']):>7}")
        print()

    # Save CSV
    out_dir = os.path.join(_REPO_ROOT, 'benchmarks', 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'b0039_slice_comparison.csv')
    cols = ['slice', 'method', 'wall_s',
            'init_jdet_neg', 'init_jdet_min', 'init_tri_neg', 'init_tri_min',
            'final_jdet_neg', 'final_jdet_min', 'final_tri_neg', 'final_tri_min',
            'l2', 'jdet_feasible', 'tri_feasible']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f"\nCSV written to {csv_path}")

    # Per-method aggregate across all four slices.
    # For Jdet methods, the "own-criterion" feasibility is jdet_feasible.
    # For triangle methods, the "own-criterion" feasibility is tri_feasible.
    print(f"\n=== Jdet-target methods (criterion: Jdet >= 0.01 everywhere) ===")
    print(f"{'method':<22} {'avg_wall_s':>11} {'feas_rate':>10} {'avg_L2':>9}")
    print('-' * 60)
    for m, _ in METHODS_JDET:
        m_rows = [r for r in rows if r['method'] == m]
        avg_wall = np.mean([r['wall_s'] for r in m_rows])
        avg_l2 = np.mean([r['l2'] for r in m_rows])
        rate = sum(1 for r in m_rows if r['jdet_feasible']) / len(m_rows)
        print(f"{m:<22} {avg_wall:>11.2f} {rate*100:>9.0f}% {avg_l2:>9.2f}")

    print(f"\n=== Triangle-target methods (criterion: all 2 triangle areas >= 0 per cell) ===")
    print(f"{'method':<22} {'avg_wall_s':>11} {'feas_rate':>10} {'avg_L2':>9}")
    print('-' * 60)
    for m, _ in METHODS_TRI:
        m_rows = [r for r in rows if r['method'] == m]
        avg_wall = np.mean([r['wall_s'] for r in m_rows])
        avg_l2 = np.mean([r['l2'] for r in m_rows])
        rate = sum(1 for r in m_rows if r['tri_feasible']) / len(m_rows)
        print(f"{m:<22} {avg_wall:>11.2f} {rate*100:>9.0f}% {avg_l2:>9.2f}")


if __name__ == '__main__':
    main()
