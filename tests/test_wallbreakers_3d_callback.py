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
