"""Post-feasibility re-anchor stage of the windowed engine.

``reanchor='l2'`` / ``'l1'`` re-solves tiles over the MOVED region against the
distance-to-INPUT objective once the field is fold-free, keeping a tile only if
every enforced row stays at or above ``threshold``. These tests pin: the fidelity
gain (with 0 folds and damage 0 preserved), the per-tile revert, the untouched
default path, and the knob plumbing through the strategy dataclass.
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import pixel_fold_mask, windowed_correct
from dvfopt.objectives import NoneObjective
from dvfopt.strategies.windowed import ISQPWindowedStrategy
from dvfopt.testdata import make_random_dvf

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")


def _localized_fold(H=64, W=64, at=(24, 26)):
    """A zero field with one folded 10x10 patch pasted in (as in the c2f tests)."""
    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    phi = np.zeros((2, H, W))
    y, x = at
    phi[:, y : y + patch.shape[1], x : x + patch.shape[2]] = patch
    return phi


def _solve(phi, **kw):
    return windowed_correct(
        phi,
        "isqp",
        constraint=SimplexConstraint2D(shape=phi.shape[1:]),
        objective=NoneObjective(),  # the robust recipe: pure feasibility, no anchor
        threshold=0.01,
        **kw,
    )


# ---------------------------------------------------------------------------
# (a) / (b) the stage buys fidelity and keeps the certificate
# ---------------------------------------------------------------------------


@needs_osqp
@pytest.mark.parametrize("kind", ["l2", "l1"])
def test_reanchor_reduces_the_move_and_keeps_zero_folds(kind):
    phi = _localized_fold()
    base, rep0 = _solve(phi)
    out, rep = _solve(phi, reanchor=kind)
    assert rep0.folds_after == rep.folds_after == 0
    assert rep.damage == 0

    # each kind is judged in its own norm
    def measure(d):
        return float(np.linalg.norm(d)) if kind == "l2" else float(np.abs(d).sum())

    assert measure(out - phi) < measure(base - phi)
    assert rep.reanchor_sweeps_run >= 1
    assert rep.reanchor_accepted > 0
    # the recorded before/after bracket the stage and agree with the L2 of the field
    assert rep.reanchor_l2_before == pytest.approx(float(np.linalg.norm(base - phi)), rel=1e-6)
    assert rep.reanchor_l2_after == pytest.approx(float(np.linalg.norm(out - phi)), rel=1e-6)


@needs_osqp
def test_reanchor_only_moves_pixels_the_main_solve_moved():
    """No-damage accounting is unaffected: the stage frees a SUBSET of the moved set."""
    phi = _localized_fold()
    base, _ = _solve(phi)
    out, _ = _solve(phi, reanchor="l2")
    moved_base = np.any(np.abs(base - phi) > 1e-9, axis=0)
    moved_out = np.any(np.abs(out - phi) > 1e-9, axis=0)
    assert not (moved_out & ~moved_base).any()


# ---------------------------------------------------------------------------
# (c) per-tile verify-and-revert
# ---------------------------------------------------------------------------


def _feasible_pair(H=32, W=32):
    """``(phi_ref, phi)``: a smooth fold-free field and a moved, still-feasible copy."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    ref = np.stack([0.1 * np.sin(xx / 5.0), 0.1 * np.cos(yy / 5.0)])
    phi = ref + 0.05
    c = SimplexConstraint2D(shape=(H, W))
    assert not pixel_fold_mask(c, phi, 0.01).any()
    return ref, phi


def test_infeasible_tile_is_reverted(monkeypatch):
    ref, phi = _feasible_pair()
    before = phi.copy()
    rep = engine.SliceReport()
    # inner returns a wildly infeasible iterate -> every tile must be rejected
    monkeypatch.setattr(
        engine, "solve_window_inner", lambda sub, *a, **k: (sub.flat0 - 50.0, 0, True)
    )
    engine._reanchor_pass(
        phi,
        ref,
        SimplexConstraint2D(shape=phi.shape[1:]),
        0.01,
        engine._ReanchorOpts("l2", 10, 1, 16),
        1e-3,
        rep,
        "isqp",
        engine._InnerOpts(),
    )
    assert rep.reanchor_tiles > 0
    assert rep.reanchor_accepted == 0
    assert np.array_equal(phi, before)


def test_tile_that_buys_nothing_is_reverted(monkeypatch):
    """Feasible but no fidelity gain -> still reverted (the stage is monotone)."""
    ref, phi = _feasible_pair()
    before = phi.copy()
    rep = engine.SliceReport()
    monkeypatch.setattr(engine, "solve_window_inner", lambda sub, *a, **k: (sub.flat0, 0, True))
    engine._reanchor_pass(
        phi,
        ref,
        SimplexConstraint2D(shape=phi.shape[1:]),
        0.01,
        engine._ReanchorOpts("l2", 10, 1, 16),
        1e-3,
        rep,
        "isqp",
        engine._InnerOpts(),
    )
    assert rep.reanchor_tiles > 0 and rep.reanchor_accepted == 0
    assert np.array_equal(phi, before)


# ---------------------------------------------------------------------------
# (d) the default path is untouched
# ---------------------------------------------------------------------------


@needs_osqp
def test_reanchor_none_is_the_default_path_byte_for_byte():
    phi = _localized_fold()
    a, rep_a = _solve(phi)
    b, rep_b = _solve(phi, reanchor="none")
    assert np.array_equal(a, b)
    assert rep_a.reanchor_sweeps_run == rep_b.reanchor_sweeps_run == 0
    assert rep_a.reanchor_tiles == rep_b.reanchor_tiles == 0
    assert rep_a.reanchor_l2_before == rep_a.reanchor_l2_after == 0.0


def test_build_subproblem_free_extra_defaults_to_the_whole_box():
    phi = np.zeros((2, 20, 20))
    c = SimplexConstraint2D(shape=(20, 20))
    box = (5, 11, 5, 11)
    full = engine.build_subproblem(c, phi, box, 0.01)
    everything = engine.build_subproblem(c, phi, box, 0.01, free_extra=np.ones((20, 20), bool))
    assert np.array_equal(full.free_mask, everything.free_mask)
    half = np.zeros((20, 20), bool)
    half[5:8, 5:11] = True
    sub = engine.build_subproblem(c, phi, box, 0.01, free_extra=half)
    assert sub.free_mask.sum() == 18 < full.free_mask.sum()


def test_unknown_reanchor_kind_raises():
    with pytest.raises(ValueError, match="unknown reanchor"):
        _solve(np.zeros((2, 16, 16)), reanchor="l3")


# ---------------------------------------------------------------------------
# (e) strategy plumbing
# ---------------------------------------------------------------------------


def test_strategy_forwards_the_reanchor_knobs(monkeypatch):
    seen = {}

    def fake(phi, inner, **kw):
        seen.update(kw)
        return np.asarray(phi), engine.SliceReport()

    monkeypatch.setattr("dvfopt.strategies.windowed.windowed_correct", fake)
    strat = ISQPWindowedStrategy(
        reanchor="l1", reanchor_maxiter=7, reanchor_sweeps=1, reanchor_tile=16
    )
    strat.solve(
        np.zeros((2, 8, 8)),
        constraint=SimplexConstraint2D(shape=(8, 8)),
        objective=NoneObjective(),
        threshold=0.01,
    )
    assert seen["reanchor"] == "l1"
    assert seen["reanchor_maxiter"] == 7
    assert seen["reanchor_sweeps"] == 1
    assert seen["reanchor_tile"] == 16


def test_strategy_defaults_to_no_reanchor():
    s = ISQPWindowedStrategy()
    assert s.reanchor == "none"
    assert (s.reanchor_maxiter, s.reanchor_sweeps, s.reanchor_tile) == (60, 3, 48)
