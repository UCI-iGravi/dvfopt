"""Full B0039 z=12 slice: which methods break the wall?

The 320×456 z=12 slice has 8,978 initial folds (~3% of all cells
inverted) and min_T = -135.985. This is the canonical "really dense"
test case from the manuscript. Earlier benchmarks claimed m10 reaches
100% feasibility on B0039; this script audits that claim with the
current code.

Methods run:

* ``barrier_l1``             — iterative_2d_tri_barrier(anchor='l1').
* ``m10_default``            — iterative_2d_tri_harmonic_polished
                               (max_grow_iters=8, manuscript default).
* ``m10_grow20``             — same with max_grow_iters=20.
* ``m14_default_l2``         — iterative_2d_tri_refine_repair
                               (anchor='l2', max_grow_iters=8).
* ``m14_l1_grow20``          — refine_repair anchor='l1' (the
                               manuscript's lowest-L1 variant),
                               max_grow_iters=20.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dvfopt.core.iterative2d_tri_barrier import iterative_2d_tri_barrier
from dvfopt.core.wallbreakers import (
    iterative_2d_tri_harmonic_polished,
    iterative_2d_tri_refine_repair,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

THRESHOLD = 0.01


def _silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def _stats(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return int((T1 <= 0).sum() + (T2 <= 0).sum()), float(min(T1.min(), T2.min()))


def run_one(label, fn, phi_in):
    print(f'=== {label} ===', flush=True)
    t0 = time.time()
    phi_out = fn(phi_in.copy())
    wall = time.time() - t0
    n_neg, min_T = _stats(phi_out)
    diff = (phi_out - phi_in).ravel()
    l1 = float(np.abs(diff).sum())
    l2 = float(np.sqrt(np.dot(diff, diff)))
    feasible = n_neg == 0 and min_T >= THRESHOLD - 1e-5
    print(
        f'  wall={wall:.2f}s  n_neg={n_neg}  min_T={min_T:+.5f}  '
        f'L1={l1:.1f}  L2={l2:.2f}  feasible={feasible}',
        flush=True,
    )
    return dict(
        method=label, wall_s=wall, n_neg=n_neg, min_T=min_T, L1=l1, L2=l2, feasible=feasible
    )


def main():
    arr = np.load(
        os.path.join(_REPO_ROOT, 'data', 'dvfs', 'b0039', 'b0039_laplacian_deformation_field.npy')
    )
    dy = arr[1, 12].astype(np.float64).copy()
    dx = arr[2, 12].astype(np.float64).copy()
    phi = np.stack([dy, dx])
    H, W = phi.shape[1], phi.shape[2]
    init_n_neg, init_min = _stats(phi)
    print(f'B0039 z=12 full slice: {phi.shape}', flush=True)
    print(
        f'  {H * W:,} corners, {2 * H * W:,} variables, {2 * (H - 1) * (W - 1):,} constraints',
        flush=True,
    )
    print(f'  init n_neg={init_n_neg:,}  init min_T={init_min:+.3f}', flush=True)
    print(flush=True)

    methods = [
        (
            'barrier_l1',
            lambda p: _silent(
                iterative_2d_tri_barrier,
                p,
                threshold=THRESHOLD,
                margin=1e-3,
                max_minimize_iter=500,
                anchor='l1',
                eps_l1=1e-4,
                verbose=0,
            ),
        ),
        (
            'm10_default',
            lambda p: _silent(
                iterative_2d_tri_harmonic_polished,
                p,
                threshold=THRESHOLD,
                margin=1e-3,
                anchor='l1',
                eps_l1=1e-4,
                max_grow_iters=8,
                time_budget_s=900.0,
                verbose=0,
            ),
        ),
        (
            'm10_grow20',
            lambda p: _silent(
                iterative_2d_tri_harmonic_polished,
                p,
                threshold=THRESHOLD,
                margin=1e-3,
                anchor='l1',
                eps_l1=1e-4,
                max_grow_iters=20,
                time_budget_s=900.0,
                verbose=0,
            ),
        ),
        (
            'm14_default_l2',
            lambda p: _silent(
                iterative_2d_tri_refine_repair,
                p,
                threshold=THRESHOLD,
                margin=1e-3,
                anchor='l2',
                eps_l1=1e-4,
                max_grow_iters=8,
                time_budget_s=900.0,
                verbose=0,
            ),
        ),
        (
            'm14_l1_grow20',
            lambda p: _silent(
                iterative_2d_tri_refine_repair,
                p,
                threshold=THRESHOLD,
                margin=1e-3,
                anchor='l1',
                eps_l1=1e-4,
                max_grow_iters=20,
                time_budget_s=900.0,
                verbose=0,
            ),
        ),
    ]
    rows = []
    for label, fn in methods:
        rows.append(run_one(label, fn, phi))
        print(flush=True)

    print('=== Summary (sorted by L1 among feasible, then wall) ===', flush=True)
    feasible = [r for r in rows if r['feasible']]
    infeasible = [r for r in rows if not r['feasible']]
    feasible.sort(key=lambda r: r['L1'])
    infeasible.sort(key=lambda r: r['n_neg'])
    print(
        f'{"method":<20}  {"wall_s":>8}  {"n_neg":>6}  {"min_T":>8}  '
        f'{"L1":>10}  {"L2":>8}  {"feas":>5}',
        flush=True,
    )
    for r in feasible + infeasible:
        print(
            f'{r["method"]:<20}  {r["wall_s"]:>8.2f}  '
            f'{r["n_neg"]:>6}  {r["min_T"]:>+8.4f}  '
            f'{r["L1"]:>10.1f}  {r["L2"]:>8.2f}  '
            f'{r["feasible"]!s:>5}',
            flush=True,
        )

    # CSV.
    out_csv = os.path.join(_REPO_ROOT, 'benchmarks', 'results', 'b0039_z12_full_slice.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w') as f:
        f.write('method,wall_s,n_neg,min_T,L1,L2,feasible\n')
        for r in rows:
            f.write(
                f'{r["method"]},{r["wall_s"]:.4f},{r["n_neg"]},'
                f'{r["min_T"]:.6f},{r["L1"]:.4f},{r["L2"]:.4f},'
                f'{r["feasible"]}\n'
            )
    print(f'\nCSV: {out_csv}', flush=True)


if __name__ == '__main__':
    main()
