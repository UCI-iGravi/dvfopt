"""Slice-by-slice benchmark of m14 vs barrier on the full B0039 z-stack.

528 slices total. Both methods use ``objective=L2Objective()`` and the package
default ``threshold=0.01``. The CSV is written incrementally so partial
progress is visible while the benchmark runs (~hours).

Per slice, the script:

1. Computes initial fold stats (skip if already feasible).
2. Runs ``iterative_2d_tri_barrier(objective=L2Objective())`` and records L2/L1/wall.
3. Runs ``iterative_2d_tri_refine_repair(objective=L2Objective())`` and records.
4. Appends two CSV rows (one per method) and flushes to disk.

Generous per-slice time budget; failures are captured in the
``error`` column rather than aborting the run.

CSV schema
----------
z, shape, init_n_neg, init_min_T, method, wall_s,
final_n_neg, final_min_T, L1, L2, feasible, error
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dvfopt.core.barrier.tri2d import iterative_2d_tri_barrier
from dvfopt.core.wallbreakers import iterative_2d_tri_harmonic_polished  # noqa
from dvfopt.core.wallbreakers import iterative_2d_tri_refine_repair as iterative_m14
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.objectives import make_objective

THRESHOLD = 0.01
ANCHOR = 'l2'
OBJECTIVE = make_objective(ANCHOR, eps_l1=1e-4)
TIME_BUDGET_PER_SLICE = 600.0


def _silent(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*a, **k)


def _stats(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    n_neg = int((np.minimum(T1, T2) <= 0).sum())
    min_T = float(min(T1.min(), T2.min()))
    return n_neg, min_T


def _run_method(name, fn, phi_in):
    t0 = time.perf_counter()
    err = ''
    phi_out = None
    try:
        phi_out = fn(phi_in.copy())
    except Exception as exc:
        err = f'{type(exc).__name__}: {exc}'
    wall = time.perf_counter() - t0
    if phi_out is None:
        return dict(
            method=name,
            wall_s=wall,
            error=err,
            final_n_neg=-1,
            final_min_T=float('nan'),
            L1=float('nan'),
            L2=float('nan'),
            feasible=False,
        )
    n_neg, min_T = _stats(phi_out)
    diff = (phi_out - phi_in).ravel()
    return dict(
        method=name,
        wall_s=wall,
        error='',
        final_n_neg=n_neg,
        final_min_T=min_T,
        L1=float(np.abs(diff).sum()),
        L2=float(np.sqrt(np.dot(diff, diff))),
        feasible=(n_neg == 0 and min_T >= THRESHOLD - 1e-5),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z-start', type=int, default=0, help='first slice index (inclusive)')
    parser.add_argument(
        '--z-end', type=int, default=None, help='last slice index (exclusive); default = D'
    )
    parser.add_argument(
        '--out-csv',
        type=str,
        default=None,
        help='output csv path (default: benchmarks/results/b0039_full_volume_m14_vs_barrier.csv)',
    )
    parser.add_argument(
        '--skip-feasible',
        action='store_true',
        default=True,
        help='skip slices with no initial folds',
    )
    args = parser.parse_args()

    arr_path = os.path.join(
        _REPO_ROOT, 'data', 'dvfs', 'b0039', 'b0039_laplacian_deformation_field.npy'
    )
    print(f'Loading {arr_path}...', flush=True)
    arr = np.load(arr_path)
    D = arr.shape[1]
    z_start = max(0, args.z_start)
    z_end = D if args.z_end is None else min(D, args.z_end)
    print(f'Volume: {arr.shape}  iterating z={z_start}..{z_end - 1}', flush=True)

    if args.out_csv is None:
        out_csv = os.path.join(
            _REPO_ROOT, 'benchmarks', 'results', 'b0039_full_volume_m14_vs_barrier.csv'
        )
    else:
        out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not os.path.exists(out_csv)

    cols = [
        'z',
        'shape',
        'init_n_neg',
        'init_min_T',
        'method',
        'wall_s',
        'final_n_neg',
        'final_min_T',
        'L1',
        'L2',
        'feasible',
        'error',
    ]

    methods = [
        (
            'barrier_l2',
            lambda p: _silent(
                iterative_2d_tri_barrier,
                p,
                threshold=THRESHOLD,
                margin=1e-3,
                max_minimize_iter=500,
                objective=OBJECTIVE,
                verbose=0,
            ),
        ),
        (
            'm14_l2',
            lambda p: _silent(
                iterative_m14,
                p,
                threshold=THRESHOLD,
                margin=1e-3,
                objective=OBJECTIVE,
                time_budget_s=TIME_BUDGET_PER_SLICE,
                verbose=0,
            ),
        ),
    ]

    overall_t0 = time.perf_counter()
    n_slices_run = 0
    with open(out_csv, 'a', buffering=1) as f:  # line-buffered
        if write_header:
            f.write(','.join(cols) + '\n')

        for z in range(z_start, z_end):
            dy = arr[1, z].astype(np.float64).copy()
            dx = arr[2, z].astype(np.float64).copy()
            phi = np.stack([dy, dx])
            H, W = phi.shape[1], phi.shape[2]
            init_n_neg, init_min_T = _stats(phi)

            if args.skip_feasible and init_n_neg == 0:
                continue

            n_slices_run += 1
            print(
                f'\n[z={z:>3}/{z_end - 1}] ({H}x{W})  '
                f'init n_neg={init_n_neg:>5}  init min_T={init_min_T:+.3f}  '
                f'(slices run so far: {n_slices_run}, '
                f'elapsed {time.perf_counter() - overall_t0:.1f}s)',
                flush=True,
            )

            for method_label, fn in methods:
                r = _run_method(method_label, fn, phi)
                row_dict = dict(
                    z=z,
                    shape=f'{H}x{W}',
                    init_n_neg=init_n_neg,
                    init_min_T=init_min_T,
                    **r,
                )
                f.write(','.join(str(row_dict.get(c, '')) for c in cols) + '\n')
                tag = 'OK' if r['feasible'] else ('ERR' if r['error'] else 'FAIL')
                print(
                    f'  [{tag:>3}] {method_label:<12} '
                    f'wall={r["wall_s"]:>7.2f}s  '
                    f'n_neg={r["final_n_neg"]:>5}  '
                    f'min_T={r["final_min_T"]:+.4f}  '
                    f'L1={r["L1"]:>10.1f}  L2={r["L2"]:>8.2f}'
                    + (f'  err={r["error"]}' if r["error"] else ''),
                    flush=True,
                )

    print(
        f'\nDone. {n_slices_run} slices processed in {time.perf_counter() - overall_t0:.1f}s.',
        flush=True,
    )
    print(f'CSV: {out_csv}', flush=True)


if __name__ == '__main__':
    main()
