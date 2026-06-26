"""Simplified pipelines: skip multi-scale + break-recover; just go
M14Tet or M10Tet -> iterate coupled k=2 SLSQP per fold cube ->
M10Tet recovery.

Variant A: M14Tet (refine-repair) on raw input -> per-fold k=2 SLSQP -> M10Tet polish.
Variant B: M10Tet (barrier only) on raw input -> per-fold k=2 SLSQP -> M10Tet polish.

Each iteration of the SLSQP step picks the worst fold cube
(lowest min tet volume) and runs coupled k=2 SLSQP @
FEASIBILITY_THR=1e-3 (Method D, fastest) centered on it. We
iterate until n_neg=0 globally or no further progress.

If we can reach n_neg=0 in say 5-30 SLSQP iterations (each
~5s) plus the M14/M10Tet pre-pass (~30 min) plus a recovery
(~30 min), the total pipeline is ~1 hour — much simpler than
the 12-hour 5-stage pipeline.
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

import research.strict_feasibility_3d.runners._coupled_kring as ck
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._coupled_kring import (
    build_coupled_problem,
    make_apply_x,
    make_constraint_fn,
    make_objective,
    report,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
MAX_ITER_SLSQP_LOOP = 30  # max number of SLSQP iterations per variant
KRING = 2
SLSQP_THR = 1e-3  # Method D Lagrangian relaxation


def run_slsqp_around(phi, cz, cy, cx, k_ring=2):
    """Single coupled k=2 SLSQP @ thr=SLSQP_THR centered at (cz, cy, cx)."""
    cubes, free_corners, x0 = build_coupled_problem(
        phi, cz, cy, cx, k_ring)
    apply_x, corner_idx_map = make_apply_x(cubes, free_corners, phi)
    ck.FEASIBILITY_THR = SLSQP_THR
    constraint_fn, _ = make_constraint_fn(cubes, corner_idx_map)
    obj, obj_grad = make_objective(x0.copy())
    cons = [{'type': 'ineq', 'fun': constraint_fn}]
    t0 = time.time()
    res = minimize(
        obj, x0, jac=obj_grad,
        constraints=cons, method='SLSQP',
        options={'maxiter': 200, 'ftol': 1e-9, 'disp': False},
    )
    wall = time.time() - t0
    return apply_x(res.x), res, wall


def find_worst_fold_cube(phi):
    """Return (cz, cy, cx) of cube with lowest min tet volume; None if none."""
    V = six_tet_volumes_3d(phi)
    min_per_cube = V.min(axis=0)
    # Find cubes with min_T <= 0 (folded).
    folded = (min_per_cube <= 0)
    if not folded.any():
        return None
    # Return location of overall minimum.
    flat = min_per_cube.flatten()
    idx = int(flat.argmin())
    cz, cy, cx = np.unravel_index(idx, min_per_cube.shape)
    return (int(cz), int(cy), int(cx)), float(min_per_cube[cz, cy, cx])


def m10tet(phi, thr=0.015):
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


def m14tet(phi, thr=0.015):
    from dvfopt import (
        HarmonicALMRefineRepair3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMRefineRepair3DStrategy(),
        threshold=thr,
    )
    return solver.fit(phi).corrected


def iterate_slsqp_per_fold(phi, label='', phi_input=None):
    """Run coupled k=2 SLSQP @ thr=SLSQP_THR centered on the worst
    fold cube; repeat until n_neg=0 or stall."""
    cur = phi.copy()
    iter_count = 0
    total_wall = 0.0
    for iter_i in range(MAX_ITER_SLSQP_LOOP):
        result = find_worst_fold_cube(cur)
        if result is None:
            print(f'  [{label}] iter {iter_i}: no folds remain (n_neg=0)',
                  flush=True)
            break
        loc, min_T = result
        cz, cy, cx = loc
        # Ensure cube is interior-enough for k=2 halo.
        D, H, W = cur.shape[1:]
        if not (KRING <= cz < D - 1 - KRING
                and KRING <= cy < H - 1 - KRING
                and KRING <= cx < W - 1 - KRING):
            print(f'  [{label}] iter {iter_i}: worst fold at ({cz},{cy},{cx}) '
                  f'(min={min_T:+.4f}) is too close to chunk boundary; stopping',
                  flush=True)
            break

        print(f'  [{label}] iter {iter_i}: worst fold at ({cz},{cy},{cx}) '
              f'min={min_T:+.6f}', flush=True)
        new, res, wall = run_slsqp_around(cur, cz, cy, cx, KRING)
        total_wall += wall
        n_after, b_after, mn_after = report(
            new, f'    after k={KRING} SLSQP wall={wall:.1f}s', phi_input)
        if res is None or not res.success:
            print(f'  [{label}] iter {iter_i}: SLSQP did not converge; stopping',
                  flush=True)
            break
        # Accept the move only if global n_neg didn't INCREASE substantially.
        V_before = six_tet_volumes_3d(cur)
        n_before = int((V_before <= 0).sum())
        if n_after > n_before + 3:
            print(f'  [{label}] iter {iter_i}: SLSQP increased global n_neg '
                  f'({n_before} -> {n_after}); reverting',
                  flush=True)
            break
        cur = new
        if n_after == 0:
            print(f'  [{label}] iter {iter_i}: STRICT global feasibility achieved',
                  flush=True)
            break
    return cur, iter_count, total_wall


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    n0, b0, _ = report(phi_input, 'INPUT (raw)', None)

    # ============================================================
    # Variant A: M14Tet -> per-fold k=2 SLSQP -> M10Tet recovery.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('VARIANT A: M14Tet -> per-fold k=2 SLSQP @ thr=1e-3 -> M10Tet recover',
          flush=True)
    print('='*70, flush=True)
    t0 = time.time()
    phi_a1 = m14tet(phi_input, 0.015)
    wall_a1 = time.time() - t0
    print(f'  M14Tet wall={wall_a1:.1f}s', flush=True)
    report(phi_a1, '  after M14Tet @ 0.015', phi_input)

    phi_a2, iter_a, wall_a2 = iterate_slsqp_per_fold(
        phi_a1, 'VARIANT-A', phi_input)
    print(f'  per-fold SLSQP total wall={wall_a2:.1f}s', flush=True)
    n_a, b_a, _ = report(phi_a2, '  after SLSQP loop', phi_input)
    if n_a > 0:
        print('  M10Tet @ 0.012 recovery on residual...', flush=True)
        t_rec = time.time()
        phi_a3 = m10tet(phi_a2, 0.012)
        wall_a_rec = time.time() - t_rec
        print(f'  recovery wall={wall_a_rec:.1f}s', flush=True)
        n_a, b_a, _ = report(phi_a3, '  after M10Tet recovery', phi_input)
    else:
        # Even if n_neg=0, run recovery to push n<0.01=0.
        if b_a > 0:
            print('  M10Tet @ 0.012 recovery to tighten...', flush=True)
            t_rec = time.time()
            phi_a3 = m10tet(phi_a2, 0.012)
            wall_a_rec = time.time() - t_rec
            print(f'  recovery wall={wall_a_rec:.1f}s', flush=True)
            n_a, b_a, _ = report(phi_a3, '  after M10Tet recovery', phi_input)
        else:
            phi_a3 = phi_a2
            wall_a_rec = 0.0
    total_a = wall_a1 + wall_a2 + wall_a_rec
    print(f'\n  VARIANT A FINAL: n_neg={n_a}  n<0.01={b_a}  total wall={total_a:.1f}s',
          flush=True)
    if n_a == 0 and b_a == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_simple_A.npy', phi_a3)
        print('  *** STRICT FEASIBLE via VARIANT A ***', flush=True)

    # ============================================================
    # Variant B: M10Tet -> per-fold k=2 SLSQP -> M10Tet recovery.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('VARIANT B: M10Tet -> per-fold k=2 SLSQP @ thr=1e-3 -> M10Tet recover',
          flush=True)
    print('='*70, flush=True)
    t0 = time.time()
    phi_b1 = m10tet(phi_input, 0.015)
    wall_b1 = time.time() - t0
    print(f'  M10Tet wall={wall_b1:.1f}s', flush=True)
    report(phi_b1, '  after M10Tet @ 0.015', phi_input)

    phi_b2, iter_b, wall_b2 = iterate_slsqp_per_fold(
        phi_b1, 'VARIANT-B', phi_input)
    print(f'  per-fold SLSQP total wall={wall_b2:.1f}s', flush=True)
    n_b, b_b, _ = report(phi_b2, '  after SLSQP loop', phi_input)
    if n_b > 0:
        print('  M10Tet @ 0.012 recovery on residual...', flush=True)
        t_rec = time.time()
        phi_b3 = m10tet(phi_b2, 0.012)
        wall_b_rec = time.time() - t_rec
        print(f'  recovery wall={wall_b_rec:.1f}s', flush=True)
        n_b, b_b, _ = report(phi_b3, '  after M10Tet recovery', phi_input)
    else:
        if b_b > 0:
            print('  M10Tet @ 0.012 recovery to tighten...', flush=True)
            t_rec = time.time()
            phi_b3 = m10tet(phi_b2, 0.012)
            wall_b_rec = time.time() - t_rec
            print(f'  recovery wall={wall_b_rec:.1f}s', flush=True)
            n_b, b_b, _ = report(phi_b3, '  after M10Tet recovery', phi_input)
        else:
            phi_b3 = phi_b2
            wall_b_rec = 0.0
    total_b = wall_b1 + wall_b2 + wall_b_rec
    print(f'\n  VARIANT B FINAL: n_neg={n_b}  n<0.01={b_b}  total wall={total_b:.1f}s',
          flush=True)
    if n_b == 0 and b_b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_simple_B.npy', phi_b3)
        print('  *** STRICT FEASIBLE via VARIANT B ***', flush=True)

    # ============================================================
    # Comparison summary.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('SIMPLE PIPELINES COMPARISON', flush=True)
    print('='*70, flush=True)
    print(f'{"Variant":<10} {"n_neg":>6} {"n<0.01":>8} {"wall (s)":>12}',
          flush=True)
    print('-'*38, flush=True)
    print(f'{"A (M14)":<10} {n_a:>6} {b_a:>8} {total_a:>12.1f}', flush=True)
    print(f'{"B (M10)":<10} {n_b:>6} {b_b:>8} {total_b:>12.1f}', flush=True)


if __name__ == '__main__':
    main()
