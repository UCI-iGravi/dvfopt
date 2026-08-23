"""record_history=True surfaces per-major-iteration SLSQP traces in SolveInfo."""

import numpy as np

from dvfopt import L1Objective, SLSQPFullGridStrategy, Solver, TriConstraint2D


def _folded_field(h=8, w=8):
    rng = np.random.default_rng(3)
    phi = rng.normal(0, 0.8, (2, h, w))  # [dy, dx], strong enough to fold
    return phi


def test_fullgrid_trace_in_solveinfo():
    phi = _folded_field()
    solver = Solver(
        constraint=TriConstraint2D(shape=phi.shape[1:]),
        objective=L1Objective(),
        strategy=SLSQPFullGridStrategy(),
    )
    res = solver.fit(phi, record_history=True)
    traces = res.info.extras.get("slsqp_trace")
    assert traces, "expected slsqp_trace in SolveInfo.extras"
    assert traces[0]["iters"], "trace must contain major-iteration records"
    assert {"obj", "max_viol", "opt", "alpha"} <= set(traces[0]["iters"][0])
