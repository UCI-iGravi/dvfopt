"""``objective='auto'``: 'l2' on trap-heavy fields, 'none' + per-window polish elsewhere."""

import numpy as np
import pytest

import dvfopt.strategies.windowed as win_mod
from dvfopt import correct_dvf
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.objectives import L2Objective, NoneObjective
from dvfopt.testdata import make_random_dvf

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")


def _mild_field(H=64, W=64, at=(24, 26)):
    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    phi = np.zeros((2, H, W))
    phi[:, at[0] : at[0] + patch.shape[1], at[1] : at[1] + patch.shape[2]] = patch
    return phi


def _spy(monkeypatch):
    """Record the engine kwargs and the objective the strategy passes through."""
    seen = {}
    real = win_mod.windowed_correct

    def wrapper(phi, inner, **kw):
        seen.update(kw)
        return real(phi, inner, **kw)

    monkeypatch.setattr(win_mod, "windowed_correct", wrapper)
    return seen


@needs_osqp
def test_auto_mild_field_takes_the_fast_branch_with_polish(monkeypatch):
    seen = _spy(monkeypatch)
    phi = _mild_field()
    res = correct_dvf(
        phi, constraint="bilinear", objective="auto", strategy="isqp_windowed", verbose=0
    )
    assert res.final_n_neg == 0
    assert seen.get("polish") == "l2"
    assert isinstance(seen.get("objective"), NoneObjective)


@needs_osqp
def test_auto_deep_field_takes_the_l2_branch(monkeypatch):
    seen = _spy(monkeypatch)
    phi = _mild_field() * 25.0  # min bilinear value far below -10
    res = correct_dvf(
        phi, constraint="bilinear", objective="auto", strategy="isqp_windowed", verbose=0
    )
    assert res.final_n_neg == 0
    assert seen.get("polish") in (None,)
    assert isinstance(seen.get("objective"), L2Objective)


@needs_osqp
def test_auto_respects_an_explicit_polish(monkeypatch):
    seen = _spy(monkeypatch)
    phi = _mild_field()
    res = correct_dvf(
        phi,
        constraint="bilinear",
        objective="auto",
        strategy="isqp_windowed",
        polish=None,
        verbose=0,
    )
    assert res.final_n_neg == 0
    assert seen.get("polish") is None
    assert isinstance(seen.get("objective"), NoneObjective)


@needs_osqp
def test_auto_with_auto_strategy_resolves_and_solves(monkeypatch):
    """objective='auto' + strategy='auto' on a mild bilinear field routes to the
    windowed engine with the polish injected into the resolved strategy."""
    seen = _spy(monkeypatch)
    phi = _mild_field()
    res = correct_dvf(phi, constraint="bilinear", objective="auto", strategy="auto", verbose=0)
    assert res.final_n_neg == 0
    assert seen.get("polish") == "l2"
