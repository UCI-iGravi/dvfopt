"""Coarse-to-fine warm start + the exposed trust-region knobs.

The engine can prepend a coarse-grid solve and seed the fine solve with the
prolongated correction (``coarse_to_fine``). The stage is only allowed to move
pixels the fine engine would have freed anyway, so the no-damage invariant is
unchanged — that masking is what these tests pin, alongside the restriction /
prolongation unit math, the skip path, and the ``tr_delta`` / ``tr_max``
plumbing down to :func:`~dvfopt.core.primitives.isqp.isqp_solve`.
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import find_windows, pixel_fold_mask, windowed_correct
from dvfopt.objectives import NoneObjective
from dvfopt.testdata import make_random_dvf

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")

MARGIN, RING = 3, 1  # SimplexConstraint2D ring; margin is the engine default


def _localized_fold(H=64, W=64, at=(24, 26)):
    """A zero field with one folded 10x10 patch pasted in from ``dvfopt.testdata``.

    Everything outside ``at + (10, 10)`` starts perfectly healthy, so any change
    out there is a no-damage violation, not solver work.
    """
    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]  # (2, 10, 10)
    phi = np.zeros((2, H, W))
    y, x = at
    phi[:, y : y + patch.shape[1], x : x + patch.shape[2]] = patch
    return phi


def _solve(phi, **kw):
    return windowed_correct(
        phi,
        "isqp",
        constraint=SimplexConstraint2D(shape=phi.shape[1:]),
        objective=NoneObjective(),
        threshold=0.01,
        **kw,
    )


# ---------------------------------------------------------------------------
# (a) restriction / prolongation unit math
# ---------------------------------------------------------------------------


def test_restrict_rescales_to_coarse_pixel_units():
    phi = np.zeros((2, 8, 8))
    phi[0], phi[1] = 4.0, -2.0
    coarse = engine._restrict(phi, 2)
    assert coarse.shape == (2, 4, 4)
    # displacements are in COARSE pixel units -> halved by a 2x restriction
    assert np.allclose(coarse[0], 2.0)
    assert np.allclose(coarse[1], -1.0)


def test_restrict_drops_the_trailing_partial_block():
    assert engine._restrict(np.ones((2, 7, 9)), 2).shape == (2, 3, 4)


def test_prolongate_rescales_back_and_zero_pads_odd_dims():
    delta = engine._prolongate(np.ones((2, 3, 4)), (7, 9), 2)
    assert delta.shape == (2, 7, 9)
    assert np.allclose(delta[:, :6, :8], 2.0)  # units scale back up by the factor
    # the row/col the integer factor cannot cover stays zero (the fine solve owns it)
    assert np.all(delta[:, 6, :] == 0.0)
    assert np.all(delta[:, :, 8] == 0.0)


def test_restrict_prolongate_round_trip_is_identity_on_a_constant_field():
    phi = np.full((2, 8, 8), 3.0)
    assert np.allclose(engine._prolongate(engine._restrict(phi, 2), (8, 8), 2), phi)


# ---------------------------------------------------------------------------
# (b) the no-damage invariant: nothing outside the fold neighbourhood moves
# ---------------------------------------------------------------------------


@needs_osqp
def test_warm_start_and_output_leave_healthy_area_byte_identical(monkeypatch):
    """The prolongated delta is dense everywhere the coarse solve moved; masking it
    to the fine window free boxes is what keeps untouched area untouched."""
    phi = _localized_fold()
    fold = np.argwhere(pixel_fold_mask(SimplexConstraint2D(shape=phi.shape[1:]), phi, 0.01))
    (y0, x0), (y1, x1) = fold.min(0), fold.max(0)
    pad = MARGIN + RING + 8 + 1  # free box = fold bbox + margin; +8 for two grows
    box = (max(0, y0 - pad), y1 + pad + 1, max(0, x0 - pad), x1 + pad + 1)

    warm = {}
    real = engine._coarse_warm_start

    def spy(*a, **kw):
        res = real(*a, **kw)
        warm["delta"] = res[0]
        return res

    monkeypatch.setattr(engine, "_coarse_warm_start", spy)
    out, rep = _solve(phi, giant_tile=8)  # 4*8 <= 64 -> the stage runs

    assert rep.warm_folds >= 0, "the coarse-to-fine stage did not run"
    assert rep.folds_before > 0 and rep.folds_after == 0
    assert rep.damage == 0  # accounted against the ORIGINAL input
    assert rep.rounds == 1 and rep.mop_windows == 0  # no extra reach beyond the boxes

    outside = np.ones(phi.shape[1:], bool)
    outside[box[0] : box[1], box[2] : box[3]] = False
    assert np.all(warm["delta"][:, outside] == 0.0)  # warm field == input out there
    assert np.array_equal(out[:, outside], phi[:, outside])  # and so does the result


@needs_osqp
def test_warm_delta_is_confined_to_the_engines_own_window_boxes():
    """Sharper than the bbox check: the mask is exactly the free boxes
    :func:`find_windows` opens on the ORIGINAL fold mask."""
    phi = _localized_fold()
    c = SimplexConstraint2D(shape=phi.shape[1:])
    allow = np.zeros(phi.shape[1:], bool)
    for fy0, fy1, fx0, fx1 in find_windows(pixel_fold_mask(c, phi, 0.01), MARGIN, RING):
        allow[fy0:fy1, fx0:fx1] = True
    delta, _rep_c, _boxes = engine._coarse_warm_start(
        phi, c, NoneObjective(), 0.01, 2, MARGIN, RING, "isqp", dict(giant_tile=8)
    )
    assert np.any(delta[:, allow] != 0.0)  # it did something
    assert np.all(delta[:, ~allow] == 0.0)  # and only there


# ---------------------------------------------------------------------------
# (c) coarse_to_fine=False and the small-field skip path
# ---------------------------------------------------------------------------


@needs_osqp
def test_small_field_skips_the_stage_and_is_byte_identical():
    """min(H, W) < 4*giant_tile -> no coarse solve, and the default path is then
    bit-for-bit the pre-change engine (what every other windowed gate exercises)."""
    phi = _localized_fold()  # 64x64 vs the default giant_tile=64 -> skipped
    out_default, rep_default = _solve(phi)
    out_off, rep_off = _solve(phi, coarse_to_fine=False)

    assert rep_default.warm_folds == -1  # -1 == stage did not run
    assert rep_default.coarse_solve_s == 0.0
    assert (rep_default.coarse_folds_before, rep_default.coarse_iters) == (-1, 0)
    assert np.array_equal(out_default, out_off)


@needs_osqp
def test_fold_free_field_skips_the_stage():
    _out, rep = _solve(np.zeros((2, 64, 64)), giant_tile=8)
    assert rep.folds_before == 0 and rep.warm_folds == -1


@needs_osqp
def test_coarse_solve_is_never_recursive(monkeypatch):
    seen = []
    real = windowed_correct

    def spy(phi_in, inner, **kw):
        seen.append(kw.get("coarse_to_fine", True))
        return real(phi_in, inner, **kw)

    monkeypatch.setattr(engine, "windowed_correct", spy)
    engine._coarse_warm_start(
        _localized_fold(),
        SimplexConstraint2D(shape=(64, 64)),
        NoneObjective(),
        0.01,
        2,
        MARGIN,
        RING,
        "isqp",
        dict(giant_tile=8),
    )
    assert seen == [False]


# ---------------------------------------------------------------------------
# (d) tr_delta / tr_max reach isqp_solve, and their defaults change nothing
# ---------------------------------------------------------------------------


def test_engine_forwards_trust_region_knobs(monkeypatch):
    seen = []

    def fake_inner(sub, label, maxiter, **kw):
        seen.append((kw.get("tr_delta"), kw.get("tr_max")))
        x = np.array(sub.flat0, dtype=float)
        x[sub.free_idx] = 0.0  # clears the window; frozen vars stay put
        return x, 1, True

    monkeypatch.setattr(engine, "solve_window_inner", fake_inner)
    _solve(_localized_fold(), tr_delta=0.5, tr_max=4.0)
    assert seen and set(seen) == {(0.5, 4.0)}


@needs_osqp
def test_tr_delta_default_output_is_unchanged():
    """Passing the documented defaults explicitly must be bit-for-bit the
    hard-coded behaviour it replaced."""
    rng = np.random.default_rng(0)
    n = 12
    flat0 = rng.normal(0, 0.3, n)
    target = flat0 + 5.0  # far enough that the trust region bounds the step
    kw = dict(
        cons=lambda f: f + 50.0,
        cons_jac=lambda f: np.eye(n),
        obj=lambda f: float((f - target) @ (f - target)),
        obj_grad=lambda f: 2.0 * (f - target),
        maxiter=5,
    )
    a, ia, fa = isqp_mod.isqp_solve(flat0, **kw)
    b, ib, fb = isqp_mod.isqp_solve(flat0, **kw, tr_delta=2.0, tr_max=16.0)
    assert np.array_equal(a, b) and (ia, fa) == (ib, fb)
    tight, _it, _ok = isqp_mod.isqp_solve(flat0, **kw, tr_delta=1e-3)
    assert not np.array_equal(a, tight)  # the knob actually bites


# ---------------------------------------------------------------------------
# (e) the strategy exposes and forwards the new knobs
# ---------------------------------------------------------------------------


@needs_osqp
def test_strategy_forwards_new_knobs(monkeypatch):
    from dvfopt import ISQPWindowedStrategy
    from dvfopt.core.windowed._common import SliceReport
    from dvfopt.strategies import windowed as strat_mod

    seen = {}
    monkeypatch.setattr(
        strat_mod,
        "windowed_correct",
        lambda phi_in, inner, **kw: (seen.update(kw), (phi_in, SliceReport()))[1],
    )
    ISQPWindowedStrategy(tr_delta=0.5, tr_max=4.0, coarse_to_fine=False, coarse_factor=3).solve(
        np.zeros((2, 8, 8)),
        constraint=SimplexConstraint2D(shape=(8, 8)),
        objective=NoneObjective(),
        threshold=0.01,
    )
    assert (seen["tr_delta"], seen["tr_max"]) == (0.5, 4.0)
    assert seen["coarse_to_fine"] is False and seen["coarse_factor"] == 3

    d = ISQPWindowedStrategy()
    assert (d.tr_delta, d.tr_max) == (2.0, 16.0)
    assert (d.coarse_to_fine, d.coarse_factor) == (True, 4)


# (d) a run cut by ``time_budget_s`` before the fine loop reaches a warm-started box


@needs_osqp
def test_budget_cut_after_warm_start_books_no_damage(monkeypatch):
    """The warm start is a move over the engine's own window boxes, so a fold it
    creates there on a run the budget cuts before the fine loop reaches it is a
    RESIDUAL inside a fold neighbourhood — never damage to untouched area.
    (Measured before the fix: raw B0039 z16 under a 40 s budget -> damage 3.)"""
    phi = _localized_fold()
    c = SimplexConstraint2D(shape=phi.shape[1:])
    boxes = find_windows(pixel_fold_mask(c, phi, 0.01), MARGIN, RING)

    def folding_warm_start(
        phi, constraint, objective, threshold, factor, margin, ring, inner, sub_kw
    ):
        # a deliberately bad "correction" confined to the engine's boxes: it folds them
        delta = np.zeros_like(phi)
        for fy0, fy1, fx0, fx1 in boxes:
            delta[0, fy0:fy1, fx0:fx1] = 3.0 * np.arange(fx1 - fx0)[None, :]
        return delta, engine.SliceReport(), boxes

    monkeypatch.setattr(engine, "_coarse_warm_start", folding_warm_start)
    out, rep = _solve(phi, giant_tile=8, time_budget_s=0)  # expires before any fine window
    assert rep.coarse_solve_s >= 0 and rep.rounds == 0  # the stage ran, the fine loop did not
    assert rep.folds_after > 0  # the bad warm start left folds behind ...
    assert rep.damage == 0  # ... booked as residual, not damage
    assert rep.residual_in_window == rep.folds_after
    outside = np.ones(phi.shape[1:], bool)
    for fy0, fy1, fx0, fx1 in boxes:
        outside[fy0:fy1, fx0:fx1] = False
    assert np.array_equal(out[:, outside], phi[:, outside])


def test_giant_tiler_stops_at_the_deadline(monkeypatch):
    """``_solve_giant_schwarz`` checks ``expired`` between tiles — a giant region is
    many window solves, not one (measured: a 40 s budget ran 189 s before this)."""
    calls = []
    monkeypatch.setattr(engine, "_solve_window", lambda *a, **k: calls.append(a[2]))
    ticks = iter(range(100))
    expired = lambda: next(ticks) >= 1  # False for the first tile, True after  # noqa: E731
    phi = np.zeros((2, 64, 64))
    c = SimplexConstraint2D(shape=(64, 64))
    rep = engine.SliceReport()
    r = engine._solve_giant_schwarz(
        phi,
        c,
        (0, 64, 0, 64),
        0.01,
        NoneObjective(),
        50,
        RING,
        rep,
        1e-3,
        inner="isqp",
        opts=engine._InnerOpts(giant_tile=16),
        expired=expired,
    )
    assert len(calls) == 1  # one tile solved, then the deadline stopped the sweep
    assert r == -1  # no completed sweep to report a residual count for
