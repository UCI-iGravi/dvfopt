"""End-to-end smoke test: run several 3D methods on a small synthetic
fold case, report per-method (feasibility, L1, wall) — the 3D analog
of the 2D ``run_method`` dispatcher.
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
from research.strict_feasibility_3d.algorithms.cluster_lp_6tet import (
    cluster_slp_iter_3d,
)
from research.strict_feasibility_3d.algorithms.lp_direct_6tet import (
    lp_oneshot,
    slp_iter,
)
from research.strict_feasibility_3d.worst_cases._synthetic_3d import (
    bowtie_3d_cube,
    dense_random_3d,
)

THRESHOLD = 0.01
SAFETY_TOL = 1e-5


def _stats(phi_3dhw):
    V = six_tet_volumes_3d(phi_3dhw)
    return {
        'n_neg': int((V <= 0).sum()),
        'min_T': float(V.min()),
    }


def _solve_via_3d_strategy(strategy_cls, phi_in_3dhw, **strategy_kwargs):
    """Helper to run a 3D strategy via the v0.2 Solver API."""
    from dvfopt import (
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    D, H, W = phi_in_3dhw.shape[1:]
    solver = Solver(
        constraint=Tet6Constraint3D(shape=(D, H, W)),
        objective=L1Objective(eps=1e-4),
        strategy=strategy_cls(**strategy_kwargs),
        threshold=THRESHOLD,
    )
    return solver.fit(phi_in_3dhw).corrected


def run_method(name: str, phi_in_3dhw: np.ndarray) -> dict:
    init = _stats(phi_in_3dhw)
    t0 = time.time()
    try:
        if name == 'm10':
            from dvfopt import HarmonicALMBarrier3DStrategy
            phi_out = _solve_via_3d_strategy(HarmonicALMBarrier3DStrategy, phi_in_3dhw)
            extra = {}
        elif name == 'm14':
            from dvfopt import HarmonicALMRefineRepair3DStrategy
            phi_out = _solve_via_3d_strategy(HarmonicALMRefineRepair3DStrategy, phi_in_3dhw)
            extra = {}
        elif name == 'lp_oneshot':
            phi_out, extra = lp_oneshot(phi_in_3dhw, threshold=THRESHOLD)
        elif name == 'slp_iter':
            # m10 seed: more robust than m14 in 3D — m14's L2-refine can
            # overshoot back into infeasibility on extreme-fold cases.
            phi_out, extra = slp_iter(phi_in_3dhw, threshold=THRESHOLD, seed='m10')
        elif name == 'cluster_slp':
            phi_out, extra = cluster_slp_iter_3d(phi_in_3dhw, threshold=THRESHOLD)
        else:
            raise ValueError(f'unknown method: {name!r}')
        error = None
    except Exception as exc:
        import traceback
        traceback.print_exc()
        phi_out = phi_in_3dhw.copy()
        extra = {}
        error = f'{type(exc).__name__}: {exc}'
    wall = time.time() - t0
    final = _stats(phi_out)
    diff = phi_out.astype(np.float64) - phi_in_3dhw.astype(np.float64)
    return {
        'method': name,
        'phi_out': phi_out,
        'init_n_neg': init['n_neg'],
        'init_min_T': init['min_T'],
        'final_n_neg': final['n_neg'],
        'final_min_T': final['min_T'],
        'feasible': final['n_neg'] == 0 and final['min_T'] >= THRESHOLD - SAFETY_TOL,
        'L1_dev': float(np.abs(diff).sum()),
        'wall_s': wall,
        'error': error,
        'extra': extra,
    }


def _print_row(rec):
    flag = 'OK ' if rec['feasible'] else 'INF'
    err = f'   err={rec["error"]}' if rec['error'] else ''
    print(
        f'  {rec["method"]:<14s} {flag}  n_neg={rec["final_n_neg"]:4d}  '
        f'min_T={rec["final_min_T"]:+.4f}  L1={rec["L1_dev"]:>10.2f}  '
        f'({rec["wall_s"]:.2f}s){err}',
        flush=True,
    )


def main():
    cases = [
        ('bowtie_3d_cube_8',  bowtie_3d_cube(size=8)),
        ('bowtie_3d_cube_12', bowtie_3d_cube(size=12)),
        ('dense_random_3d_10', dense_random_3d(size=10)),
    ]
    methods = ['m10', 'm14', 'lp_oneshot', 'slp_iter']

    for case_id, phi_in in cases:
        init = _stats(phi_in)
        print(
            f'\n=== {case_id}  shape={phi_in.shape}  '
            f'init n_neg={init["n_neg"]}  min_T={init["min_T"]:+.4f} ===',
            flush=True,
        )
        for m in methods:
            rec = run_method(m, phi_in)
            _print_row(rec)


if __name__ == '__main__':
    main()
