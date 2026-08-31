"""The GUI's Auto objective: same dispatch as correct_dvf, via the shared helper."""

from types import SimpleNamespace

import numpy as np
import pytest

from dvfopt import ISQPWindowedStrategy
from dvfopt.constraints import SimplexConstraint2DBilinear
from dvfopt.solver import resolve_auto_objective
from dvfopt.testdata import make_random_dvf

pytest.importorskip("dvfopt_gui.worker")


def _stub_with_field(scale=1.0):
    from dvfopt_gui.worker import SolverWorker

    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    phi = np.zeros((2, 64, 64))
    phi[:, 24 : 24 + patch.shape[1], 26 : 26 + patch.shape[2]] = patch * scale
    stub = SimpleNamespace(phi=phi, _params={"objective_id": "auto"}, _resolved_auto_objective=None)
    stub._phi_for_auto = lambda: stub.phi
    return SolverWorker, stub


def test_resolver_matches_the_library_dispatch():
    assert resolve_auto_objective(10, -1.0) == ("none", True)
    assert resolve_auto_objective(3000, -1.0) == ("l2", False)
    assert resolve_auto_objective(10, -50.0) == ("l2", False)


def test_gui_worker_auto_mild_injects_polish():
    cls, stub = _stub_with_field()
    c = SimplexConstraint2DBilinear(shape=(64, 64))
    out = cls._resolve_auto_and_polish(stub, c, ISQPWindowedStrategy())
    assert stub._resolved_auto_objective == "none" and out.polish == "l2"


def test_gui_worker_auto_deep_keeps_l2_no_polish():
    cls, stub = _stub_with_field(scale=25.0)
    c = SimplexConstraint2DBilinear(shape=(64, 64))
    out = cls._resolve_auto_and_polish(stub, c, ISQPWindowedStrategy())
    assert stub._resolved_auto_objective == "l2" and out.polish is None


def test_gui_worker_non_auto_is_untouched():
    cls, stub = _stub_with_field()
    stub._params = {"objective_id": "l1"}
    strat = ISQPWindowedStrategy()
    assert (
        cls._resolve_auto_and_polish(stub, SimplexConstraint2DBilinear(shape=(64, 64)), strat)
        is strat
    )
