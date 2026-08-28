"""Exact merit line search in the elastic-QP inner (``step_rule='exact_ls'``).

The 2D constraint rows are BILINEAR in ``(dy, dx)``, so along the QP step a row
is exactly quadratic and the merit is an exact piecewise quadratic whose global
minimiser is closed form. These tests pin what that rests on: the line model
itself, the closed-form minimiser against a brute-force scan of the TRUE merit,
the guard that falls back to the stock trust-region acceptance when the fitted
objective misleads, and ``step_rule='tr'`` still being the stock path byte for
byte.
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2D, SimplexConstraint2DBilinear
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import build_subproblem, find_windows, pixel_fold_mask, windowed_correct
from dvfopt.objectives import L1Objective, NoneObjective
from dvfopt.testdata import make_random_dvf

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")

THR = 0.01


def _localized_fold(H=64, W=64, at=(24, 26)):
    """A zero field with one folded 10x10 patch pasted in (the coarse-to-fine
    suite's fixture: healthy everywhere outside ``at + (10, 10)``)."""
    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    phi = np.zeros((2, H, W))
    y, x = at
    phi[:, y : y + patch.shape[1], x : x + patch.shape[2]] = patch
    return phi


def _solve(phi, constraint=None, objective=None, **kw):
    c = SimplexConstraint2D(shape=phi.shape[1:]) if constraint is None else constraint
    return windowed_correct(
        phi,
        "isqp",
        constraint=c,
        objective=NoneObjective() if objective is None else objective,
        threshold=THR,
        **kw,
    )


def _first_sub(phi, cls=SimplexConstraint2DBilinear):
    """The engine's OWN first window sub-problem for *phi* — a real window, with
    the real frozen ring and the real enforced-row set."""
    c = cls(shape=phi.shape[1:])
    box = find_windows(pixel_fold_mask(c, phi, THR), 3, 1)[0]
    return build_subproblem(c, phi, box, THR, NoneObjective())


def _rand_dir(sub, rng, scale):
    d = np.zeros(sub.flat0.size)
    d[sub.free_idx] = rng.normal(0.0, scale, sub.free_idx.size)
    return d


# ---------------------------------------------------------------------------
# (a) 'tr' is the stock path, byte for byte
# ---------------------------------------------------------------------------


def test_primitive_default_is_tr_engine_default_is_exact_ls():
    import inspect

    assert inspect.signature(isqp_mod.isqp_solve).parameters["step_rule"].default == "tr"
    assert inspect.signature(windowed_correct).parameters["step_rule"].default == "exact_ls"
    assert engine._InnerOpts().step_rule == "exact_ls"

    from dvfopt import ISQPWindowedStrategy

    assert ISQPWindowedStrategy().step_rule == "exact_ls"


@needs_osqp
def test_tr_never_touches_the_exact_machinery(monkeypatch):
    """Sabotaging the line minimiser leaves a ``step_rule='tr'`` run bit-for-bit
    unchanged — the new code is unreachable from the stock path."""
    phi = _localized_fold()
    ref, rep = _solve(phi.copy(), step_rule="tr")
    assert rep.folds_after == 0 and rep.damage == 0

    def boom(*a, **k):
        raise AssertionError("step_rule='tr' must not run the exact line search")

    monkeypatch.setattr(isqp_mod, "_exact_line_min", boom)
    out, _ = _solve(phi.copy(), step_rule="tr")
    assert np.array_equal(out, ref)


def test_unknown_step_rule_raises():
    pytest.importorskip(
        'osqp'
    )  # isqp_solve checks the optional osqp dep before validating arguments
    with pytest.raises(ValueError, match="step_rule"):
        isqp_mod.isqp_solve(np.zeros(2), None, None, None, 1, step_rule="nope")
    with pytest.raises(ValueError, match="step_rule"):
        _solve(np.zeros((2, 8, 8)), step_rule="nope")


def test_exact_ls_is_rejected_on_a_3d_field():
    """A 6-tet volume row is trilinear, hence CUBIC along a line — the quadratic
    model does not transfer, so the engine refuses at its entry."""
    with pytest.raises(ValueError, match="2D"):
        windowed_correct(
            np.zeros((3, 4, 8, 8)),
            "isqp",
            constraint=SimplexConstraint2D(shape=(8, 8)),
            objective=NoneObjective(),
            threshold=THR,
        )


# ---------------------------------------------------------------------------
# (b) the line model is exact on a real window
# ---------------------------------------------------------------------------


@needs_osqp
def test_quadratic_line_model_matches_cons_exactly():
    """``c(x + a d) == c + a (J d) + a^2 q`` with ``q = c(x+d) - c - J d``, and
    that same ``q`` equals a CENTRAL DIFFERENCE of ``cons`` along ``d`` — which it
    only can if the row really is quadratic along the line."""
    sub = _first_sub(_localized_fold())
    rng = np.random.default_rng(0)
    x = sub.flat0
    c = np.asarray(sub.cons(x))
    for scale in (0.05, 0.5, 2.0):
        d = _rand_dir(sub, rng, scale)
        g = np.asarray(sub.cons_jac(x) @ d)
        q = np.asarray(sub.cons(x + d)) - c - g
        q_fd = 0.5 * (np.asarray(sub.cons(x + d)) + np.asarray(sub.cons(x - d)) - 2.0 * c)
        assert np.abs(q - q_fd).max() / max(np.abs(q).max(), 1e-12) < 1e-10
        for a in rng.uniform(-1.0, 1.0, 4):
            ref = np.asarray(sub.cons(x + a * d))
            got = c + a * g + a * a * q
            assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12) < 1e-10


# ---------------------------------------------------------------------------
# (c) the closed-form minimiser beats a dense scan of the TRUE merit
# ---------------------------------------------------------------------------


@needs_osqp
@pytest.mark.parametrize("scale", [0.2, 1.0])
def test_exact_minimiser_matches_a_dense_scan_of_the_true_merit(scale):
    sub = _first_sub(_localized_fold())
    rng = np.random.default_rng(1)
    x = sub.flat0
    c = np.asarray(sub.cons(x))
    w = np.full(c.size, 1e3)
    d = _rand_dir(sub, rng, scale)
    g = np.asarray(sub.cons_jac(x) @ d)
    q = np.asarray(sub.cons(x + d)) - c - g
    f0, fh, f1 = (float(sub.obj(x + a * d)) for a in (0.0, 0.5, 1.0))
    fco = (f0, 4.0 * fh - f1 - 3.0 * f0, 2.0 * f1 + 2.0 * f0 - 4.0 * fh)
    a_star, m_star, m_zero = isqp_mod._exact_line_min(c, g, q, w, fco, 1.0)

    def true_merit(a):
        y = x + a * d
        return sub.obj(y) + float(w @ np.clip(-np.asarray(sub.cons(y)), 0, None))

    grid = np.linspace(0.0, 1.0, 201)
    vals = np.array([true_merit(a) for a in grid])
    span = float(vals.max() - vals.min()) + 1e-12
    assert abs(m_zero - vals[0]) <= 1e-9 * span  # m(0) is the merit at the iterate
    assert abs(m_star - true_merit(a_star)) <= 1e-8 * span  # model merit IS the true one
    assert true_merit(a_star) <= vals.min() + 1e-9 * span  # nothing on the grid beats it


# ---------------------------------------------------------------------------
# (d) the guard: a misleading model degenerates to the stock acceptance
# ---------------------------------------------------------------------------


@needs_osqp
def test_guard_falls_back_to_tr_acceptance_when_the_model_misleads(monkeypatch):
    """Only the OBJECTIVE part of the line model is fitted (exact for L2/none,
    approximate for the eps-smoothed L1), so the true merit at ``a*`` is checked
    before stepping. A minimiser that never finds a real decrease must leave the
    run bit-for-bit identical to ``step_rule='tr'``."""
    phi = _localized_fold()
    obj = L1Objective(eps=1e-2)
    ref, _ = _solve(phi.copy(), objective=obj, step_rule="tr")
    monkeypatch.setattr(isqp_mod, "_exact_line_min", lambda *a, **k: (0.0, 0.0, 0.0))
    out, rep = _solve(phi.copy(), objective=obj, step_rule="exact_ls")
    assert np.array_equal(out, ref)
    assert rep.folds_after == 0 and rep.damage == 0


@needs_osqp
def test_l1_run_never_increases_the_merit():
    """The guard's invariant, on a real L1 window: every accepted step strictly
    decreases the merit, so the traced merit sequence is non-increasing."""
    sub = _first_sub(_localized_fold(), cls=SimplexConstraint2D)
    trace = {}
    isqp_mod.isqp_solve(
        sub.flat0,
        sub.cons,
        sub.cons_jac,
        sub.obj_grad,
        60,
        obj=sub.obj,
        hess_diag=sub.hess_diag,
        free_idx=sub.free_idx,
        trace=trace,
        step_rule="exact_ls",
    )
    merits = [r["merit"] for r in trace["iters"]]
    assert len(merits) > 1
    assert all(b <= a for a, b in zip(merits, merits[1:])), merits
    stepped = [r for r in trace["iters"] if r.get("stepped")]
    assert stepped and all("alpha" in r and r["rule"] in ("exact_ls", "tr") for r in stepped)


# ---------------------------------------------------------------------------
# (e) the knob reaches the driver, (f) both rules keep the engine invariants
# ---------------------------------------------------------------------------


@needs_osqp
def test_strategy_knob_reaches_the_driver(monkeypatch):
    """Strategy dataclass -> windowed_correct -> _InnerOpts -> isqp_solve."""
    from dvfopt import ISQPWindowedStrategy
    from dvfopt.core.windowed import _inners

    seen = []
    real = _inners.isqp_solve
    monkeypatch.setattr(
        _inners, "isqp_solve", lambda *a, **kw: (seen.append(kw), real(*a, **kw))[1]
    )
    phi = _localized_fold()
    kw = dict(
        constraint=SimplexConstraint2D(shape=phi.shape[1:]),
        objective=NoneObjective(),
        threshold=THR,
    )
    ISQPWindowedStrategy(step_rule="tr").solve(phi, **kw)
    assert seen and all(k["step_rule"] == "tr" for k in seen)
    seen.clear()
    ISQPWindowedStrategy().solve(phi, **kw)
    assert seen and all(k["step_rule"] == "exact_ls" for k in seen)


@needs_osqp
@pytest.mark.parametrize("rule", ["tr", "exact_ls"])
@pytest.mark.parametrize("objective", [NoneObjective(), L1Objective(eps=1e-2)])
def test_both_rules_clear_the_field_without_damage(rule, objective):
    phi = _localized_fold()
    _out, rep = _solve(phi.copy(), objective=objective, step_rule=rule)
    # `damage` is the engine's own no-damage accounting against the ORIGINAL
    # input: any pixel outside every window's enforced footprint that moved.
    assert rep.folds_before > 0 and rep.folds_after == 0 and rep.damage == 0


# ---------------------------------------------------------------------------
# (g) the a*-collapse bail (``exact_ls_fallback_steps``)
# ---------------------------------------------------------------------------


@needs_osqp
@pytest.mark.parametrize("k", [1, 3])
def test_a_star_collapse_hands_the_window_to_the_escalation_ladder(monkeypatch, k):
    """A scripted a*-collapse: the minimiser always returns a tiny step that still
    decreases the merit, so neither the futility test nor step-tol would ever give
    up on the window. After ``k`` consecutive collapses the call must stop, with
    the step taken and the reason attributable."""
    sub = _first_sub(_localized_fold(), cls=SimplexConstraint2D)
    monkeypatch.setattr(isqp_mod, "_exact_line_min", lambda *a: (0.01, 0.0, 0.0))
    trace = {}
    isqp_mod.isqp_solve(
        sub.flat0,
        sub.cons,
        sub.cons_jac,
        sub.obj_grad,
        40,
        obj=sub.obj,
        hess_diag=sub.hess_diag,
        free_idx=sub.free_idx,
        trace=trace,
        step_rule="exact_ls",
        exact_ls_fallback_steps=k,
    )
    assert trace["exit"] == "a-collapse"
    assert trace["nit"] == k  # stopped on the k-th collapse, not one later
    assert all(r["stepped"] and r["rule"] == "exact_ls" for r in trace["iters"])


@needs_osqp
def test_a_star_collapse_bail_is_off_at_zero(monkeypatch):
    """``exact_ls_fallback_steps=0`` disables the bail: the same scripted collapse
    keeps taking exact steps for the whole budget."""
    sub = _first_sub(_localized_fold(), cls=SimplexConstraint2D)
    monkeypatch.setattr(isqp_mod, "_exact_line_min", lambda *a: (0.01, 0.0, 0.0))
    trace = {}
    isqp_mod.isqp_solve(
        sub.flat0,
        sub.cons,
        sub.cons_jac,
        sub.obj_grad,
        12,
        obj=sub.obj,
        hess_diag=sub.hess_diag,
        free_idx=sub.free_idx,
        trace=trace,
        step_rule="exact_ls",
        exact_ls_fallback_steps=0,
    )
    assert trace["exit"] != "a-collapse" and trace["nit"] > 3  # never stops on the collapse
    assert all("alpha" in r for r in trace["iters"])


@needs_osqp
def test_collapse_bail_knob_reaches_the_driver(monkeypatch):
    """Strategy dataclass -> windowed_correct -> _InnerOpts -> isqp_solve, and the
    strategy default is the engine default the gate was measured at."""
    from dvfopt import ISQPWindowedStrategy
    from dvfopt.core.windowed import _inners

    default = engine._InnerOpts().exact_ls_fallback_steps
    assert ISQPWindowedStrategy().exact_ls_fallback_steps == default
    seen = []
    real = _inners.isqp_solve
    monkeypatch.setattr(
        _inners, "isqp_solve", lambda *a, **kw: (seen.append(kw), real(*a, **kw))[1]
    )
    phi = _localized_fold()
    ISQPWindowedStrategy(exact_ls_fallback_steps=7).solve(
        phi,
        constraint=SimplexConstraint2D(shape=phi.shape[1:]),
        objective=NoneObjective(),
        threshold=THR,
    )
    assert seen and all(k["exact_ls_fallback_steps"] == 7 for k in seen)
