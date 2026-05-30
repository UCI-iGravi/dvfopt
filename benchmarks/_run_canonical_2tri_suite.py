"""Canonical 2-tri benchmark — declarative variant using BenchmarkSuite.

This is the same comparison as ``_run_l1_2tri_warmstart_cases.py`` (the
six notebook-14 cases × the production 2-tri solvers) but expressed
declaratively through :class:`BenchmarkSuite`. Demonstrates the
intended migration path: each ``_run_*.py`` becomes a 30-line script
that builds ``cases``, ``solvers``, and calls ``suite.run()``.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from benchmark_suite import BenchmarkSuite

from dvfopt import Solver
from test_cases import canonical_2tri_2d


def main():
    # Cases — canonical 2-tri suite from notebook 14.
    cases = {name: phi for (name, phi, _meta) in canonical_2tri_2d()}

    # Per-case shape sometimes differs (10×10 vs 20×20); the Solver's
    # constraint is shape-bound, so we need one Solver per (shape, method).
    # The suite handles this gracefully — it just passes phi to solver.fit,
    # which uses Constraint.coerce/flatten to validate the shape match.
    # For a one-shape demo, build a 10×10 suite; users running the full
    # battery on heterogeneous shapes should group by shape.
    # We pick the most representative 10×10 case to demo:
    sample_name, sample_phi, _ = canonical_2tri_2d()[0]
    shape = sample_phi.shape[1:]
    print(
        f'Running suite on cases of shape {shape} only (others omitted '
        f'for the BenchmarkSuite single-shape demo)'
    )
    cases_for_shape = {n: p for n, p in cases.items() if p.shape[1:] == shape}

    solvers = {
        'barrier_l1': Solver.from_spec(
            constraint='2tri',
            objective='l1',
            strategy='barrier',
            shape=shape,
        ),
        'barrier_l2': Solver.from_spec(
            constraint='2tri',
            objective='l2',
            strategy='barrier',
            shape=shape,
        ),
        'slsqp_l1': Solver.from_spec(
            constraint='2tri',
            objective='l1',
            strategy='slsqp',
            shape=shape,
            strategy_kwargs=dict(max_iter=80, warm_max_iter=1200),
        ),
        'slsqp_l2': Solver.from_spec(
            constraint='2tri',
            objective='l2',
            strategy='slsqp',
            shape=shape,
            strategy_kwargs=dict(max_iter=80, warm_max_iter=1200),
        ),
        'm10_l1': Solver.from_spec(
            constraint='2tri',
            objective='l1',
            strategy='m10',
            shape=shape,
        ),
        'm14_l1': Solver.from_spec(
            constraint='2tri',
            objective='l1',
            strategy='m14',
            shape=shape,
        ),
    }

    suite = BenchmarkSuite(
        cases=cases_for_shape,
        solvers=solvers,
        out_csv=os.path.join(_REPO_ROOT, 'benchmarks', 'results', 'canonical_2tri_suite.csv'),
        verbose=True,
    )
    df = suite.run()
    print(suite.summary(df))


if __name__ == '__main__':
    main()
