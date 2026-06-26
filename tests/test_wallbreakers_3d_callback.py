"""3D wallbreaker step_callback hook: live staging + stop."""
from __future__ import annotations

import numpy as np
import pytest

from dvfopt.core.wallbreakers._refine_repair_3d import iterative_3d_tet_refine_repair


def _folded_volume_3d(D=4, H=10, W=10):
    _, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    phi = np.zeros((3, D, H, W))
    phi[2, :, 4:6, 4:6] = 1.5  # local dx bump -> a few folded cells
    return phi


def test_m14tet_core_fires_step_callback_per_phase():
    phi = _folded_volume_3d()
    seen = []

    def cb(state):
        assert state['phi'].shape == phi.shape
        seen.append(state['stage'])

    iterative_3d_tet_refine_repair(phi, time_budget_s=30.0, verbose=0, step_callback=cb)
    # seed + pull always fire; repair fires only if residual; polish if strict.
    assert 'seed' in seen and 'pull' in seen
    assert seen == sorted(seen, key=['seed', 'pull', 'repair', 'polish'].index)


def test_m14tet_core_stop_via_callback_raises():
    phi = _folded_volume_3d()

    def cb(state):
        raise KeyboardInterrupt('stop')

    with pytest.raises(KeyboardInterrupt):
        iterative_3d_tet_refine_repair(phi, time_budget_s=30.0, verbose=0, step_callback=cb)


def test_m14tet_core_default_callback_none_unchanged():
    phi = _folded_volume_3d()
    out = iterative_3d_tet_refine_repair(phi, time_budget_s=30.0, verbose=0)
    assert out.shape == phi.shape


# ---------------------------------------------------------------------------
# Task 4: strategy-level step_callback wiring
# ---------------------------------------------------------------------------

def test_m10tet_strategy_fires_harmonic_and_alm():
    from dvfopt import L2Objective, M10TetStrategy, Solver, Tet6Constraint3D

    phi = _folded_volume_3d()
    seen = []
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L2Objective(),
        strategy=M10TetStrategy(),
    )
    solver.fit(phi, step_callback=lambda s: seen.append(s['stage']))
    assert 'harmonic' in seen and 'alm' in seen


def test_m14tet_strategy_forwards_callback():
    from dvfopt import L2Objective, M14TetStrategy, Solver, Tet6Constraint3D

    phi = _folded_volume_3d()
    seen = []
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L2Objective(),
        strategy=M14TetStrategy(time_budget_s=30.0),
    )
    solver.fit(phi, step_callback=lambda s: seen.append(s['stage']))
    assert 'seed' in seen


def test_m14schwarz3d_stop_via_callback_raises():
    from dvfopt import L2Objective, M14Schwarz3DStrategy, Solver, Tet6Constraint3D

    phi = _folded_volume_3d()
    with pytest.raises(KeyboardInterrupt):
        Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L2Objective(),
            strategy=M14Schwarz3DStrategy(time_budget_s=30.0),
        ).fit(phi, step_callback=lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))
