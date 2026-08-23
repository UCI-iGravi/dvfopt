"""Validate 2-tri + full-coverage on the cases in notebook 14.

Notebook ``14_l1-warmstart-2d-cases.ipynb`` claims full feasibility
(``min_TR = +0.010``) on six synthetic cases:

    01a_10x10_crossing, 01b_10x10_opposite, 03a_10x10_opposite,
    03b_10x10_crossing, 03c_20x20_opposite, 03d_20x20_crossing

Solvers compared — **all imported from the installed dvfopt package**.
No inline scipy.optimize wrappers, no notebook-only experimental code.

* **barrier_l2 / barrier_l1** — ``dvfopt.iterative_2d_tri_barrier`` with
  ``full_coverage=True`` (penalty -> log-barrier L-BFGS-B homotopy).
* **slsqp_l2 / slsqp_l1** — ``dvfopt.iterative_2d_tri_slsqp`` with
  ``full_coverage=True`` (full-grid SLSQP + reactive warm-restart, the
  notebook-14 approach promoted into the package).
* **slsqp_windowed_4tri** — ``dvfopt.iterative_serial(enforce_triangles=True)``
  (windowed SLSQP, L2 anchor, strict 4-triangle-per-cell constraint;
  already gives ≥3-coverage at every grid vertex).
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
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dvfopt.core.barrier.tri2d import iterative_2d_tri_barrier
from dvfopt.core.slsqp_fullgrid.tri2d import iterative_2d_tri_slsqp
from dvfopt.core.slsqp_windowed.iterative import iterative_serial
from dvfopt.jacobian.triangle_sign import (
    _corner_patch_areas_2d,
    _triangle_areas_2d,
)
from dvfopt.objectives import make_objective
from dvfopt.testdata import canonical_2tri_2d


def _l1(a, b):
    return float(np.abs(a - b).sum())


def _l2(a, b):
    return float(np.linalg.norm(a - b))


def _stats(phi):
    """Full-coverage 2-tri stats: standard cells + 2 corner patches."""
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    patches = _corner_patch_areas_2d(phi[0], phi[1])
    cell_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    patches_neg = int((patches <= 0).sum())
    return dict(
        cell_neg=cell_neg,
        cell_min=float(min(T1.min(), T2.min())),
        patches_neg=patches_neg,
        patches_min=float(patches.min()),
        full_neg=cell_neg + patches_neg,
        full_min=float(min(T1.min(), T2.min(), patches.min())),
    )


def _silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# ============================================================
# Solver wrappers — all return phi (2, H, W) and accept phi_init (2, H, W).
# ============================================================

THRESHOLD = 0.01
EPS_L1 = 1e-4


def solve_barrier(phi_init, anchor):
    return _silent(
        iterative_2d_tri_barrier,
        phi_init.copy(),
        threshold=THRESHOLD,
        margin=1e-3,
        max_minimize_iter=500,
        objective=make_objective(anchor, eps_l1=EPS_L1),
        verbose=0,
        full_coverage=True,
    )


def solve_slsqp_windowed(phi_init):
    """Package iterative_serial(enforce_triangles=True). L2 anchor only.
    The 4-triangle-per-cell constraint already gives full vertex coverage."""
    H, W = phi_init.shape[1], phi_init.shape[2]
    deformation = np.zeros((3, 1, H, W), dtype=np.float64)
    deformation[1, 0] = phi_init[0]  # dy
    deformation[2, 0] = phi_init[1]  # dx
    out = _silent(
        iterative_serial,
        deformation,
        verbose=0,
        threshold=THRESHOLD,
        max_iterations=80,
        max_minimize_iter=120,
        enforce_triangles=True,
    )
    if out.ndim == 4:
        return np.stack([out[1, 0], out[2, 0]])
    if out.ndim == 3 and out.shape[0] == 2:
        return out
    if out.ndim == 3 and out.shape[0] == 3:
        return np.stack([out[1], out[2]])
    raise ValueError(out.shape)


def solve_slsqp_fullgrid(phi_init, anchor):
    """Package iterative_2d_tri_slsqp with full_coverage=True."""
    return _silent(
        iterative_2d_tri_slsqp,
        phi_init.copy(),
        threshold=THRESHOLD,
        max_iter=80,
        warm_max_iter=1200,
        objective=make_objective(anchor, eps_l1=EPS_L1),
        full_coverage=True,
        verbose=0,
    )


# ============================================================

# (method_label, fn(phi_init) -> phi_out). All imported from dvfopt.
METHODS = [
    ('barrier_l2', lambda p: solve_barrier(p, 'l2')),
    ('barrier_l1', lambda p: solve_barrier(p, 'l1')),
    ('slsqp_l2', lambda p: solve_slsqp_fullgrid(p, 'l2')),
    ('slsqp_l1', lambda p: solve_slsqp_fullgrid(p, 'l1')),
    ('slsqp_windowed_4tri', solve_slsqp_windowed),
]


def run_one(case_key, phi_init, method_label, fn):
    init = _stats(phi_init)

    t0 = time.perf_counter()
    try:
        phi_out = fn(phi_init.copy())
    except Exception as exc:
        return dict(
            case=case_key,
            method=method_label,
            shape=phi_init.shape[1:],
            wall_s=time.perf_counter() - t0,
            init_cell_neg=init['cell_neg'],
            init_cell_min=init['cell_min'],
            init_patches_min=init['patches_min'],
            final_cell_neg=-1,
            final_cell_min=float('nan'),
            final_patches_min=float('nan'),
            final_full_neg=-1,
            final_full_min=float('nan'),
            l1=float('nan'),
            l2=float('nan'),
            feasible=False,
            error=f'{type(exc).__name__}: {exc}',
        )
    wall = time.perf_counter() - t0
    final = _stats(phi_out)
    return dict(
        case=case_key,
        method=method_label,
        shape=phi_init.shape[1:],
        wall_s=wall,
        init_cell_neg=init['cell_neg'],
        init_cell_min=init['cell_min'],
        init_patches_min=init['patches_min'],
        final_cell_neg=final['cell_neg'],
        final_cell_min=final['cell_min'],
        final_patches_min=final['patches_min'],
        final_full_neg=final['full_neg'],
        final_full_min=final['full_min'],
        l1=_l1(phi_out, phi_init),
        l2=_l2(phi_out, phi_init),
        feasible=(final['full_neg'] == 0 and final['full_min'] >= 0.01 - 1e-5),
        error='',
    )


def main():
    rows = []
    hfmt = (
        "{case:<22} {shape:<7} {method:<22} {wall_s:>6} "
        "{cell_neg_f:>8} {cell_min_f:>10} {p_min_f:>10} "
        "{l1:>8} {l2:>8} {feas:>6}"
    )
    print(
        hfmt.format(
            case='case',
            shape='shape',
            method='method',
            wall_s='wall',
            cell_neg_f='cell_n_f',
            cell_min_f='cell_min_f',
            p_min_f='patch_f',
            l1='L1',
            l2='L2',
            feas='feas',
        )
    )
    print('-' * 130)
    # Load the canonical suite once — the synthetic correspondences pass
    # through the same Laplacian interpolation that produced the legacy
    # data/test_cases/*.npy snapshots.
    cases = canonical_2tri_2d()
    for case, phi_init, _meta in cases:
        for method_label, fn in METHODS:
            r = run_one(case, phi_init, method_label, fn)
            rows.append(r)
            if r.get('error'):
                print(
                    f"{case:<22} {r['shape']!s:<7} {method_label:<22} "
                    f"{r['wall_s']:.2f}  ERROR: {r['error']}"
                )
                continue
            print(
                hfmt.format(
                    case=case,
                    shape=f"{r['shape'][0]}x{r['shape'][1]}",
                    method=method_label,
                    wall_s=f"{r['wall_s']:.2f}",
                    cell_neg_f=r['final_cell_neg'],
                    cell_min_f=f"{r['final_cell_min']:+.4f}",
                    p_min_f=f"{r['final_patches_min']:+.4f}",
                    l1=f"{r['l1']:.2f}",
                    l2=f"{r['l2']:.2f}",
                    feas=str(r['feasible']),
                )
            )
        print()

    # Save CSV
    out_dir = os.path.join(_REPO_ROOT, 'benchmarks', 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'l1_2tri_warmstart_cases.csv')
    cols = [
        'case',
        'method',
        'wall_s',
        'init_cell_neg',
        'init_cell_min',
        'init_patches_min',
        'final_cell_neg',
        'final_cell_min',
        'final_patches_min',
        'final_full_neg',
        'final_full_min',
        'l1',
        'l2',
        'feasible',
    ]
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            if r.get('error'):
                continue
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f"CSV: {csv_path}\n")

    # Per-method aggregate
    method_labels = [m for m, _ in METHODS]
    print(f"{'method':<22} {'avg_wall':>10} {'feas_rate':>11} {'avg_L1':>9} {'avg_L2':>9}")
    print('-' * 70)
    for m in method_labels:
        s_rows = [r for r in rows if r['method'] == m and not r.get('error')]
        if not s_rows:
            continue
        avg_wall = np.mean([r['wall_s'] for r in s_rows])
        avg_l1 = np.mean([r['l1'] for r in s_rows])
        avg_l2 = np.mean([r['l2'] for r in s_rows])
        f_rate = sum(1 for r in s_rows if r['feasible']) / len(s_rows)
        print(f"{m:<22} {avg_wall:>10.2f} {f_rate * 100:>10.0f}% {avg_l1:>9.2f} {avg_l2:>9.2f}")


if __name__ == '__main__':
    main()
