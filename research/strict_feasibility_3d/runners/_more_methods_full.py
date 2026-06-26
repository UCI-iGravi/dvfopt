"""Run methods B, C, D, E from _more_methods.py independently
(no early-exit on success) so we can see which methods also reach
n_neg=0 starting from BEST_1fold.

  B. Coupled k=3 SLSQP + M10Tet recovery.
  C. Multi-restart k=2 (8 random seeds, sigma=0.01).
  D. Lagrangian relaxation halo at k=2 (FEASIBILITY_THR=1e-3) + M10Tet.
  E. Coupled k=4 trust-constr + M10Tet recovery.

Each method runs to completion; results all logged.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

import research.strict_feasibility_3d.runners._coupled_kring as ck
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._more_methods import (
    m10tet_recover,
    report,
    run_coupled,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
FOLD_CZ, FOLD_CY, FOLD_CX = 1, 215, 220


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    cur = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)
    n0, b0, _ = report(cur, 'START (BEST_1fold)', phi_input)

    # Storage for cross-method comparison.
    results = []

    # ============================================================
    # METHOD B: Coupled k=3 SLSQP -> M10Tet @ 0.012.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD B: coupled k=3 SLSQP + M10Tet @ 0.012 recovery', flush=True)
    print('='*70, flush=True)
    phi_b, res_b, wall_b, cubes_b, free_b = run_coupled(
        cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=3, max_iter=300)
    if res_b is not None:
        print(f'  k=3 SLSQP: success={res_b.success}  fun={res_b.fun:.4f}  '
              f'iter={res_b.nit}  wall={wall_b:.1f}s  '
              f'cubes={len(cubes_b)}  DOF={3*len(free_b)}', flush=True)
    n, b, _ = report(phi_b, 'B.1 k=3 SLSQP only', phi_input)
    results.append(('B.1 k=3 SLSQP only', n, b, wall_b))

    print('  M10Tet @ 0.012 recovery...', flush=True)
    t0 = time.time()
    phi_b_r = m10tet_recover(phi_b, 0.012)
    wall_b_r = time.time() - t0
    n, b, _ = report(phi_b_r, 'B.2 k=3 SLSQP + M10Tet@0.012', phi_input)
    results.append(('B.2 k=3 + M10Tet@0.012', n, b, wall_b + wall_b_r))
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_B.npy', phi_b_r)
        print('  *** STRICT FEASIBLE via B ***', flush=True)

    # ============================================================
    # METHOD C: Multi-restart k=2 SLSQP -> M10Tet @ 0.012.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD C: Multi-restart k=2 SLSQP (8 seeds) + M10Tet @ 0.012',
          flush=True)
    print('='*70, flush=True)
    best_c_n_neg = 999
    best_c_phi = None
    best_c_seed = None
    for seed in range(8):
        phi_c, res_c, wall_c, _, _ = run_coupled(
            cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=2,
            init_perturbation=(seed, 0.01))
        n_c, b_c, mn = report(phi_c, f'  C seed={seed} SLSQP only', phi_input)
        results.append((f'C seed={seed} SLSQP only', n_c, b_c, wall_c))
        if n_c < best_c_n_neg:
            best_c_n_neg = n_c
            best_c_phi = phi_c.copy()
            best_c_seed = seed
    print(f'\n  Best multi-restart SLSQP: seed={best_c_seed} n_neg={best_c_n_neg}',
          flush=True)
    print('  Now M10Tet @ 0.012 recovery on best...', flush=True)
    t0 = time.time()
    phi_c_r = m10tet_recover(best_c_phi, 0.012)
    wall_c_r = time.time() - t0
    n, b, _ = report(phi_c_r, f'C.best (seed={best_c_seed}) + M10Tet@0.012', phi_input)
    results.append((f'C best seed={best_c_seed} + M10Tet@0.012', n, b, wall_c_r))
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_C.npy', phi_c_r)
        print('  *** STRICT FEASIBLE via C ***', flush=True)

    # ============================================================
    # METHOD D: k=2 SLSQP with relaxed FEASIBILITY_THR=1e-3 -> M10Tet @ 0.012.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD D: k=2 SLSQP with FEASIBILITY_THR=1e-3 (Lagrangian) + M10Tet',
          flush=True)
    print('='*70, flush=True)
    original_thr = ck.FEASIBILITY_THR
    ck.FEASIBILITY_THR = 1e-3
    try:
        phi_d, res_d, wall_d, _, _ = run_coupled(
            cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=2)
        if res_d is not None:
            print(f'  D k=2 (thr=1e-3): success={res_d.success}  '
                  f'fun={res_d.fun:.4f}  wall={wall_d:.1f}s', flush=True)
        n, b, _ = report(phi_d, 'D.1 k=2 thr=1e-3 SLSQP only', phi_input)
        results.append(('D.1 k=2 thr=1e-3 SLSQP only', n, b, wall_d))
        print('  M10Tet @ 0.012 recovery...', flush=True)
        t0 = time.time()
        phi_d_r = m10tet_recover(phi_d, 0.012)
        wall_d_r = time.time() - t0
        n, b, _ = report(phi_d_r, 'D.2 k=2 thr=1e-3 + M10Tet@0.012', phi_input)
        results.append(('D.2 k=2 thr=1e-3 + M10Tet@0.012', n, b, wall_d + wall_d_r))
        if n == 0 and b == 0:
            np.save(OUTPUT / 'b0039_z0_15_strict_via_D.npy', phi_d_r)
            print('  *** STRICT FEASIBLE via D ***', flush=True)
    finally:
        ck.FEASIBILITY_THR = original_thr

    # ============================================================
    # METHOD E: coupled k=4 trust-constr -> M10Tet @ 0.012.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD E: coupled k=4 trust-constr + M10Tet @ 0.012 recovery',
          flush=True)
    print('='*70, flush=True)
    phi_e, res_e, wall_e, cubes_e, free_e = run_coupled(
        cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=4,
        method='trust-constr', max_iter=200)
    if res_e is not None:
        print(f'  k=4 trust-constr: status={res_e.status} fun={res_e.fun:.4f}  '
              f'iter={res_e.nit}  wall={wall_e:.1f}s  '
              f'cubes={len(cubes_e)}  DOF={3*len(free_e)}', flush=True)
    n, b, _ = report(phi_e, 'E.1 k=4 trust-constr only', phi_input)
    results.append(('E.1 k=4 trust-constr only', n, b, wall_e))
    print('  M10Tet @ 0.012 recovery...', flush=True)
    t0 = time.time()
    phi_e_r = m10tet_recover(phi_e, 0.012)
    wall_e_r = time.time() - t0
    n, b, _ = report(phi_e_r, 'E.2 k=4 + M10Tet@0.012', phi_input)
    results.append(('E.2 k=4 + M10Tet@0.012', n, b, wall_e + wall_e_r))
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_E.npy', phi_e_r)
        print('  *** STRICT FEASIBLE via E ***', flush=True)

    # Final comparison.
    print('\n' + '='*70, flush=True)
    print('SUMMARY OF METHODS B-E', flush=True)
    print('='*70, flush=True)
    print(f'{"Method":<40} {"n_neg":>6} {"n<0.01":>8} {"wall (s)":>10}',
          flush=True)
    print('-' * 68, flush=True)
    for (label, n, b, w) in results:
        marker = '  *** STRICT ***' if n == 0 and b == 0 else ''
        print(f'{label:<40} {n:>6} {b:>8} {w:>10.1f}{marker}',
              flush=True)


if __name__ == '__main__':
    main()
