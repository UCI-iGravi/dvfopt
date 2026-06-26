"""Method E corrected: coupled k=4 trust-constr (proper options) + M10Tet.

Previous run failed because options dict contained 'ftol' which is
not valid for trust-constr. Use 'xtol' / 'gtol' instead.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

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
FOLD_CZ, FOLD_CY, FOLD_CX = 1, 215, 220


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


def run_trust_constr(phi, k_ring=4, max_iter=200):
    print(f'\n=== coupled k={k_ring} trust-constr ===', flush=True)
    cubes, free_corners, x0 = build_coupled_problem(
        phi, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring)
    apply_x, corner_idx_map = make_apply_x(cubes, free_corners, phi)
    constraint_fn, _ = make_constraint_fn(cubes, corner_idx_map)
    obj, obj_grad = make_objective(x0.copy())
    print(f'  cubes={len(cubes)}  DOF={3*len(free_corners)}  '
          f'constraints={6*len(cubes)}', flush=True)

    # trust-constr uses NonlinearConstraint object.
    nlc = NonlinearConstraint(constraint_fn, 0, np.inf)
    t0 = time.time()
    res = minimize(
        obj, x0, jac=obj_grad,
        constraints=nlc, method='trust-constr',
        options={'maxiter': max_iter, 'xtol': 1e-8, 'gtol': 1e-6,
                 'disp': True, 'verbose': 2},
    )
    wall = time.time() - t0
    print(f'  trust-constr: status={res.status}  fun={res.fun:.4f}  '
          f'iter={res.nit}  cv={res.constr_violation:.4e}  wall={wall:.1f}s',
          flush=True)
    print(f'  message: {res.message}', flush=True)
    return apply_x(res.x), res, wall, cubes, free_corners


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    cur = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)
    report(cur, 'START (BEST_1fold)', phi_input)

    # k=4 with proper trust-constr options.
    print('\n' + '='*70, flush=True)
    print('METHOD E (FIXED): coupled k=4 trust-constr + M10Tet @ 0.012',
          flush=True)
    print('='*70, flush=True)
    phi_e, res_e, wall_e, cubes_e, free_e = run_trust_constr(cur, k_ring=4, max_iter=300)
    n, b, _ = report(phi_e, 'E.1 k=4 trust-constr only', phi_input)

    print('\n  M10Tet @ 0.012 recovery...', flush=True)
    t0 = time.time()
    phi_e_r = m10tet_recover(phi_e, 0.012)
    print(f'  recovery wall={time.time()-t0:.1f}s', flush=True)
    n, b, _ = report(phi_e_r, 'E.2 k=4 trust-constr + M10Tet@0.012', phi_input)
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_E.npy', phi_e_r)
        print('  *** STRICT FEASIBLE via E (fixed) ***', flush=True)


if __name__ == '__main__':
    main()
