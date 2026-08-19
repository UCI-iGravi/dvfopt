"""Menu <-> strategy-registry parity.

Every method the GUI menus offer must construct through dvfopt's public
strategy registry (one ``_MID_TO_LABEL`` table), so adding/renaming a Strategy
cannot silently drift between the library and the GUI.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip('PyQt5', reason='dvfopt_gui requires the [gui] extra (PyQt5)')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dvfopt import Strategy
from dvfopt.strategies.base import _STRATEGY_REGISTRY
from dvfopt_gui._shared import _METHOD_SPECS_BY_CONSTRAINT, _compose_method_id
from dvfopt_gui.worker import _MID_TO_LABEL, SolverWorker
from tests.conftest import planted_fold_3d as _folded_volume

# Method-ids that deliberately do NOT construct via the registry table:
# auto_* routes by fold stats (auto_strategy + make_strategy fallback),
# the 2D windowed path drives iterative_serial directly (live windows),
# and pipeline3d runs correct_dvf_3d, not a Strategy.
SPECIAL = {
    'auto_2tri',
    'auto_jdet',
    'auto_tet3d',
    'auto_jdet3d',
    'slsqp_windowed_2tri',
    'slsqp_windowed_jdet',
    'pipeline3d_tet3d',
}


def _folded_2d(H=8, W=8):
    """(3, 1, H, W) [dz, dy, dx] with a central fold."""
    rng = np.random.default_rng(0)
    d = rng.normal(0, 0.05, (3, 1, H, W))
    d[0] = 0.0
    d[1, 0, 3:5, 3:5] -= 1.3
    return d


def _all_menu_mids():
    return {
        _compose_method_id(algo, constraint)
        for constraint, specs in _METHOD_SPECS_BY_CONSTRAINT.items()
        for algo, _display in specs
    }


def test_menu_ids_are_registry_table_plus_known_specials():
    assert _all_menu_mids() == set(_MID_TO_LABEL) | SPECIAL


def test_every_mapped_label_is_registered():
    assert not {lbl for lbl in _MID_TO_LABEL.values() if lbl not in _STRATEGY_REGISTRY}


@pytest.mark.parametrize('mid', sorted(_MID_TO_LABEL))
def test_build_strategy_constructs_via_registry(mid):
    vol = _folded_volume() if mid.endswith(('_tet3d', '_jdet3d')) else _folded_2d()
    w = SolverWorker(deformation_i=vol, method_id=mid)
    strategy = w._build_strategy()
    assert isinstance(strategy, Strategy)
    assert isinstance(strategy, _STRATEGY_REGISTRY[_MID_TO_LABEL[mid]])
