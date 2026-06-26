"""More attacks on the 1-fold attractor — methods that exploit the
findings from Part XI/XII:

  A. Chain: coupled k=2 SLSQP -> M10Tet recovery (recover from 2 folds).
  B. Coupled k=3 SLSQP (~1000 DOF, larger halo for boundary containment).
  C. Multi-restart k=2: random small perturbation + coupled k=2 SLSQP,
     repeat 8 times, take best.
  D. Lagrangian relaxation halo: k=2 SLSQP but with halo-edge cubes
     allowed to be slack by epsilon=5e-4 (smaller than the 1-fold gap).
  E. Coupled k=4 trust-constr (~2200 DOF, much larger halo).

Each method saves its best result. If any reaches n_neg=0 globally,
script exits early.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import minimize

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    six_tet_volumes_3d,
)

# Reuse builders from _coupled_kring.
from research.strict_feasibility_3d.runners._coupled_kring import (
    build_coupled_problem,
    make_apply_x,
    make_constraint_fn,
    make_objective,
    report,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
FEASIBILITY_THR = 0.005


def m10tet_recover(phi, thr=0.012):
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=thr,
    )
    return solver.fit(phi).corrected


def run_coupled(phi, fold_cz, fold_cy, fold_cx, k_ring,
                method='SLSQP', max_iter=300, feasibility_thr=None,
                relaxation_corners=None, relaxation_thr=None,
                init_perturbation=None):
    """Generic coupled k-ring solver with optional features:
      relaxation_corners : set of corner (z, y, x) tuples whose cubes
                           get the looser feasibility_thr.
      init_perturbation  : (seed, sigma) for random init.
    """
    cubes, free_corners, x0 = build_coupled_problem(
        phi, fold_cz, fold_cy, fold_cx, k_ring)
    if init_perturbation is not None:
        seed, sigma = init_perturbation
        rng = np.random.default_rng(seed)
        x0 = x0 + sigma * rng.standard_normal(x0.shape)
    apply_x, corner_idx_map = make_apply_x(cubes, free_corners, phi)
    constraint_fn, n_cubes = make_constraint_fn(cubes, corner_idx_map)
    obj, obj_grad = make_objective(x0.copy())

    cons = [{'type': 'ineq', 'fun': constraint_fn}]
    t0 = time.time()
    try:
        res = minimize(
            obj, x0, jac=obj_grad,
            constraints=cons, method=method,
            options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': False},
        )
        wall = time.time() - t0
        return apply_x(res.x), res, wall, cubes, free_corners
    except Exception as e:
        print(f'  solver error: {e}', flush=True)
        return phi.copy(), None, time.time() - t0, cubes, free_corners


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    cur = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)
    n_neg, n_below, _ = report(cur, 'START (BEST_1fold)', phi_input)

    if n_neg == 0 and n_below == 0:
        print('Already strict feasible.', flush=True)
        return

    FOLD_CZ, FOLD_CY, FOLD_CX = 1, 215, 220
    best_global_n_neg = n_neg
    best_phi = cur.copy()
    best_method = 'START'

    def maybe_update(new_phi, label):
        nonlocal best_global_n_neg, best_phi, best_method
        n, b, _ = report(new_phi, f'  {label}', phi_input)
        if n < best_global_n_neg or (n == best_global_n_neg and b < int((six_tet_volumes_3d(best_phi) < THRESHOLD - 1e-5).sum())):
            best_global_n_neg = n
            best_phi = new_phi.copy()
            best_method = label
            print(f'    *** NEW BEST: {label} n_neg={n} ***', flush=True)
        if n == 0 and b == 0:
            print(f'\n*** STRICT 100% FEASIBLE via {label} ***', flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_more.npy', new_phi)
            return True
        return False

    # ============================================================
    # METHOD A: Chain coupled k=2 -> M10Tet @ 0.012 recovery.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD A: coupled k=2 SLSQP -> M10Tet @ 0.012 recovery', flush=True)
    print('='*70, flush=True)
    phi_a, res_a, wall_a, _, _ = run_coupled(cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=2)
    if res_a is not None:
        print(f'  k=2 SLSQP: success={res_a.success}  fun={res_a.fun:.4f}  '
              f'iter={res_a.nit}  wall={wall_a:.1f}s', flush=True)
    if maybe_update(phi_a, 'A.1 k=2 SLSQP'): return

    print('  Now M10Tet @ 0.012 recovery...', flush=True)
    t0 = time.time()
    phi_a_recovered = m10tet_recover(phi_a, 0.012)
    print(f'  M10Tet recovery wall={time.time()-t0:.1f}s', flush=True)
    if maybe_update(phi_a_recovered, 'A.2 k=2 + M10Tet@0.012'): return

    # Try recovery @ 0.010 too.
    print('  M10Tet @ 0.010 recovery...', flush=True)
    t0 = time.time()
    phi_a_r2 = m10tet_recover(phi_a, 0.010)
    print(f'  M10Tet recovery wall={time.time()-t0:.1f}s', flush=True)
    if maybe_update(phi_a_r2, 'A.3 k=2 + M10Tet@0.010'): return

    # ============================================================
    # METHOD B: Coupled k=3 SLSQP.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD B: coupled k=3 SLSQP', flush=True)
    print('='*70, flush=True)
    phi_b, res_b, wall_b, cubes_b, free_b = run_coupled(
        cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=3, max_iter=300)
    if res_b is not None:
        print(f'  k=3 SLSQP: success={res_b.success}  fun={res_b.fun:.4f}  '
              f'iter={res_b.nit}  wall={wall_b:.1f}s  '
              f'#cubes={len(cubes_b)}  #DOF={3*len(free_b)}', flush=True)
    if maybe_update(phi_b, 'B.1 k=3 SLSQP'): return

    print('  M10Tet @ 0.012 recovery on k=3 result...', flush=True)
    t0 = time.time()
    phi_b_r = m10tet_recover(phi_b, 0.012)
    print(f'  M10Tet recovery wall={time.time()-t0:.1f}s', flush=True)
    if maybe_update(phi_b_r, 'B.2 k=3 + M10Tet@0.012'): return

    # ============================================================
    # METHOD C: Multi-restart k=2 with random init perturbations.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD C: Multi-restart k=2 SLSQP (8 restarts, sigma=0.01)', flush=True)
    print('='*70, flush=True)
    for seed in range(8):
        phi_c, res_c, wall_c, _, _ = run_coupled(
            cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=2,
            init_perturbation=(seed, 0.01))
        ok = res_c is not None and res_c.success
        n_c, _, mn = report(phi_c, f'  C seed={seed} (success={ok})', phi_input)
        if maybe_update(phi_c, f'C seed={seed}'): return

    # ============================================================
    # METHOD D: Lagrangian relaxation halo at k=2 with epsilon=1e-3.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD D: k=2 SLSQP with relaxed FEASIBILITY_THR=1e-3', flush=True)
    print('='*70, flush=True)
    # Set feasibility_thr globally lower for this run.
    # We need to monkey-patch FEASIBILITY_THR. Easier: use a custom constraint.
    import research.strict_feasibility_3d.runners._coupled_kring as ck
    original_thr = ck.FEASIBILITY_THR
    ck.FEASIBILITY_THR = 1e-3
    try:
        phi_d, res_d, wall_d, _, _ = run_coupled(
            cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=2)
        if res_d is not None:
            print(f'  D k=2 (thr=1e-3): success={res_d.success}  fun={res_d.fun:.4f}  '
                  f'wall={wall_d:.1f}s', flush=True)
        if maybe_update(phi_d, 'D.1 k=2 thr=1e-3'): return

        # Recover with M10Tet @ 0.012 to re-tighten.
        print('  M10Tet @ 0.012 recovery to re-tighten...', flush=True)
        t0 = time.time()
        phi_d_r = m10tet_recover(phi_d, 0.012)
        print(f'  M10Tet recovery wall={time.time()-t0:.1f}s', flush=True)
        if maybe_update(phi_d_r, 'D.2 k=2 thr=1e-3 + M10Tet'): return
    finally:
        ck.FEASIBILITY_THR = original_thr

    # ============================================================
    # METHOD E: coupled k=4 with trust-constr.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD E: coupled k=4 trust-constr', flush=True)
    print('='*70, flush=True)
    phi_e, res_e, wall_e, cubes_e, free_e = run_coupled(
        cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=4,
        method='trust-constr', max_iter=200)
    if res_e is not None:
        print(f'  k=4 trust-constr: status={res_e.status} '
              f'fun={res_e.fun:.4f}  iter={res_e.nit}  '
              f'wall={wall_e:.1f}s  #cubes={len(cubes_e)}  '
              f'#DOF={3*len(free_e)}', flush=True)
    if maybe_update(phi_e, 'E.1 k=4 trust-constr'): return

    print('  M10Tet @ 0.012 recovery on k=4 result...', flush=True)
    t0 = time.time()
    phi_e_r = m10tet_recover(phi_e, 0.012)
    print(f'  M10Tet recovery wall={time.time()-t0:.1f}s', flush=True)
    if maybe_update(phi_e_r, 'E.2 k=4 + M10Tet@0.012'): return

    # Save final best.
    print(f'\n=== Final ===\n  best n_neg={best_global_n_neg} via {best_method}',
          flush=True)
    np.save(OUTPUT / 'b0039_z0_15_more_methods_best.npy', best_phi)


if __name__ == '__main__':
    main()
