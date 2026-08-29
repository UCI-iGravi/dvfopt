"""Terminal harmonic re-seed stage of the windowed engine.

The residual the round loop and the mop plateau on sits on the rotated orientation
branch of its ring (see ``windowed_correct``'s docstring); the stage replaces each
residual cluster's neighbourhood by the harmonic interpolation of its ring and
polishes. These tests pin: the harmonic fill itself, the stage's mechanics on a
folded field (it fires, reaches 0 folds, books its pixels as touched), and that
it never runs on a field the mop cleared (so those runs are byte-identical).
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import pixel_fold_mask, windowed_correct
from dvfopt.objectives import NoneObjective
from dvfopt.testdata import make_random_dvf

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")


def _localized_fold(H=64, W=64, at=(24, 26)):
    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    phi = np.zeros((2, H, W))
    y, x = at
    phi[:, y : y + patch.shape[1], x : x + patch.shape[2]] = patch
    return phi


def test_harmonic_fill_is_discrete_harmonic_and_keeps_the_boundary():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(2, 20, 20))
    mask = np.zeros((20, 20), bool)
    mask[6:14, 5:15] = True
    ref = phi.copy()
    engine._harmonic_fill(phi, mask)
    assert np.array_equal(phi[:, ~mask], ref[:, ~mask])  # boundary untouched
    inner = mask.copy()
    inner[[6, 13], :] = False
    inner[:, [5, 14]] = False
    lap = (
        4 * phi[:, 1:-1, 1:-1]
        - phi[:, :-2, 1:-1]
        - phi[:, 2:, 1:-1]
        - phi[:, 1:-1, :-2]
        - phi[:, 1:-1, 2:]
    )
    assert np.abs(lap[:, inner[1:-1, 1:-1]]).max() < 1e-9  # zero Laplacian inside


def test_harmonic_fill_empty_mask_is_a_no_op():
    phi = np.ones((2, 5, 5))
    engine._harmonic_fill(phi, np.zeros((5, 5), bool))
    assert np.array_equal(phi, np.ones((2, 5, 5)))


@needs_osqp
def test_stage_reseeds_a_folded_field_to_zero_and_books_touched():
    phi = _localized_fold()
    c = SimplexConstraint2D(shape=phi.shape[1:])
    assert pixel_fold_mask(c, phi, 0.01).any()
    touched = np.zeros(phi.shape[1:], bool)
    rep = engine.SliceReport()
    opts = engine._InnerOpts()
    sub_kw = dict(
        margin=3,
        maxiter=400,
        max_rounds=8,
        margin_delta=1e-3,
        max_window_area=3000,
        mop_margin=25,
        verbose=0,
    )
    sub_kw.update(engine.asdict(opts))
    work = phi.copy()
    engine._reseed_stage(
        work,
        c,
        0.01,
        NoneObjective(),
        "isqp",
        opts,
        rep,
        touched,
        c.ring if hasattr(c, "ring") else 1,
        3,
        2,
        sub_kw,
        expired=lambda: False,
    )
    assert rep.reseed_rounds_run >= 1 and rep.reseed_px > 0
    assert rep.reseed_folds_before > 0 and rep.reseed_folds_after == 0
    assert not pixel_fold_mask(c, work, 0.01).any()
    assert touched.any() and np.all(
        touched[np.any(work != phi, axis=0)]
    )  # every moved pixel is booked


@needs_osqp
def test_stage_never_runs_on_a_field_the_mop_cleared():
    phi = _localized_fold()
    c = SimplexConstraint2D(shape=phi.shape[1:])
    kw = dict(constraint=c, objective=NoneObjective(), threshold=0.01, verbose=0)
    out_on, rep_on = windowed_correct(phi, "isqp", **kw)
    out_off, rep_off = windowed_correct(phi, "isqp", reseed_rounds=0, **kw)
    assert (
        rep_on.folds_after == 0
        and rep_on.reseed_rounds_run == 0
        and rep_on.reseed_folds_before == -1
    )
    assert np.array_equal(out_on, out_off)


@needs_osqp  # the strategy checks for osqp before calling the (patched) engine
def test_strategy_forwards_the_reseed_knobs(monkeypatch):
    from dvfopt.strategies.windowed import ISQPWindowedStrategy

    seen = {}

    def fake(phi, inner, **kw):
        seen.update(kw)
        return np.asarray(phi, dtype=float).copy(), engine.SliceReport()

    monkeypatch.setattr("dvfopt.strategies.windowed.windowed_correct", fake)
    phi = np.zeros((2, 16, 16))
    s = ISQPWindowedStrategy(reseed_rounds=5, reseed_radius=4)
    s.solve(
        phi,
        constraint=SimplexConstraint2D(shape=(16, 16)),
        objective=NoneObjective(),
        threshold=0.01,
    )
    assert seen["reseed_rounds"] == 5 and seen["reseed_radius"] == 4
    assert seen["reseed_before_mop"] is True and ISQPWindowedStrategy().reseed_before_mop is True
    assert ISQPWindowedStrategy().reseed_rounds == 3 and ISQPWindowedStrategy().reseed_radius == 2
