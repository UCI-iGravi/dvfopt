"""Validation: apply Method D (k=2 SLSQP @ thr=1e-3) + M10Tet recovery
to existing partial checkpoints to confirm they each reach n_neg=0.

This validates the central claim of Part XIV: that from any 1-3 fold
state, Method D + recovery reaches strict feasibility.

Checkpoints tested (those that exist):
  - chain_best.npy (1 fold)
  - ms_v2_tight.npy (2 folds)
  - iter result (will exist once iter pipeline saves it)
  - any others
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._coupled_kring import (
    build_coupled_problem, make_apply_x, make_constraint_fn,
    make_objective, report,
)
import research.strict_feasibility_3d.runners._coupled_kring as ck
from scipy.optimize import minimize


OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def m10tet(phi, thr):
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


def run_method_d(phi, fold_cz, fold_cy, fold_cx, k_ring=2):
    """Method D: k=2 SLSQP @ thr=1e-3 (FEASIBILITY_THR=1e-3)."""
    ck.FEASIBILITY_THR = 1e-3
    cubes, free_corners, x0 = build_coupled_problem(
        phi, fold_cz, fold_cy, fold_cx, k_ring)
    apply_x, corner_idx_map = make_apply_x(cubes, free_corners, phi)
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


def find_worst_fold(phi):
    V = six_tet_volumes_3d(phi)
    min_per_cube = V.min(axis=0)
    if (min_per_cube > 0).all():
        return None
    idx = int(min_per_cube.argmin())
    cz, cy, cx = np.unravel_index(idx, min_per_cube.shape)
    return (int(cz), int(cy), int(cx))


def validate(npy_path, label, phi_input):
    if not npy_path.exists():
        print(f'\n[{label}] file {npy_path} not found, skipping', flush=True)
        return None
    print(f'\n=== Validating: {label} ===', flush=True)
    phi = np.load(npy_path).astype(np.float64)
    n0, b0, mn0 = report(phi, '  START', phi_input)
    if n0 == 0 and b0 == 0:
        print('  Already strict feasible.', flush=True)
        return phi

    # Find worst fold.
    fold = find_worst_fold(phi)
    if fold is None:
        return phi
    cz, cy, cx = fold
    print(f'  Worst fold at ({cz},{cy},{cx})', flush=True)
    # Method D.
    new, res, wall = run_method_d(phi, cz, cy, cx, k_ring=2)
    print(f'  Method D SLSQP: success={res.success} wall={wall:.1f}s', flush=True)
    n, b, mn = report(new, '  after Method D', phi_input)
    # Recovery.
    print('  M10Tet @ 0.012 recovery...', flush=True)
    t0 = time.time()
    final = m10tet(new, 0.012)
    print(f'  recovery wall={time.time()-t0:.1f}s', flush=True)
    n, b, mn = report(final, '  FINAL', phi_input)
    if n == 0 and b == 0:
        print(f'  *** STRICT FEASIBLE via Method D on {label} ***', flush=True)
        return final
    return final


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'RAW INPUT', None)

    # Test on existing 1-3 fold checkpoints.
    candidates = [
        ('CHAIN_BEST (1 fold)', OUTPUT / 'b0039_z0_15_chain_best.npy'),
        ('MS_V2_TIGHT (2 folds)', OUTPUT / 'b0039_z0_15_ms_v2_tight.npy'),
    ]
    for label, path in candidates:
        validate(path, label, phi_input)


if __name__ == '__main__':
    main()
