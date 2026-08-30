"""Fast + robust windowed-isqp knobs: the no-trust-region fallback, the
two-tier OSQP iteration caps, and the giant-region tiler's tile / sweep caps.

The engine retries a failed window ONCE with the trust region off (the TR
ratio test freezes on sliver-scale violations the legacy line search clears)
before paying for a grow, and caps the OSQP ADMM iterations per subproblem
(2000 normal / 500 fallback). These tests drive the plumbing with fake inner
solvers, so only the strategy-level test needs osqp installed.
"""

import math
import types

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import windowed_correct
from dvfopt.objectives import NoneObjective

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")


def _sparse_folds(H=60, W=60, seed=3):
    """Zero field with a few separated sharp fold blobs (folds under 2-tri)."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((2, H, W))
    for cy, cx in [(15, 15), (15, 45), (45, 20)]:
        phi[0, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (5, 5))
        phi[1, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (5, 5))
    return phi


def _fake_inner(calls):
    """Inner that FAILS every trust-region solve and clears the window without
    it (zeroing the free pixels restores the identity field there)."""

    def inner(sub, label, maxiter, trace=None, trust_region=True, osqp_max_iter=None, **_qp):
        calls.append(dict(trust_region=trust_region, maxiter=maxiter, osqp_max_iter=osqp_max_iter))
        x = np.array(sub.flat0, dtype=float)
        if trust_region:
            return x, 1, False
        x[sub.free_idx] = 0.0  # frozen vars stay at flat0 — no-damage still holds
        return x, 1, True

    return inner


def _run(phi, monkeypatch, calls, **kw):
    monkeypatch.setattr(engine, "solve_window_inner", _fake_inner(calls))
    return windowed_correct(
        phi,
        "isqp",
        constraint=SimplexConstraint2D(shape=phi.shape[1:]),
        objective=NoneObjective(),
        threshold=0.01,
        **kw,
    )


# ---------------------------------------------------------------------------
# (a) the no-TR fallback fires on a failed window, before grow-on-failure
# ---------------------------------------------------------------------------


def test_fallback_clears_window_without_growing(monkeypatch):
    calls = []
    phi = _sparse_folds()
    _out, rep = _run(phi, monkeypatch, calls)

    assert rep.folds_before > 0 and rep.folds_after == 0
    assert rep.damage == 0
    assert any(w.fallback for w in rep.windows)
    assert all(w.grows == 0 for w in rep.windows)  # cleared without a single grow
    # first attempt: trust region ON at the normal cap; retry: OFF, short budget
    assert calls[0] == dict(trust_region=True, maxiter=400, osqp_max_iter=1000)
    assert calls[1] == dict(trust_region=False, maxiter=200, osqp_max_iter=500)


def test_no_tr_fallback_off_falls_through_to_grow(monkeypatch):
    calls = []
    phi = _sparse_folds()
    # reseed off: the terminal re-seed stage would clear this deliberate plateau
    _out, rep = _run(phi, monkeypatch, calls, no_tr_fallback=False, reseed_rounds=0)

    assert rep.folds_after > 0  # the TR-only fake never clears anything
    assert not any(c["trust_region"] is False for c in calls)
    assert max(w.grows for w in rep.windows) > 0  # grow-on-failure instead
    assert not any(w.fallback for w in rep.windows)


def test_engine_knobs_reach_the_inner(monkeypatch):
    calls = []
    _out, _rep = _run(
        _sparse_folds(),
        monkeypatch,
        calls,
        maxiter=77,
        fallback_maxiter=11,
        qp_max_iter=13,
        qp_max_iter_fallback=17,
    )
    assert calls[0] == dict(trust_region=True, maxiter=77, osqp_max_iter=13)
    assert calls[1] == dict(trust_region=False, maxiter=11, osqp_max_iter=17)


def test_giant_tile_knobs_reach_the_tiler(monkeypatch):
    """``giant_tile`` / ``giant_max_sweeps`` ride ``_InnerOpts`` into the
    giant-region Schwarz tiler (default 64/8; overridable per call)."""
    seen = []
    monkeypatch.setattr(
        engine,
        "_solve_giant_schwarz",
        lambda *a, opts=None, **kw: seen.append((opts.giant_tile, opts.giant_max_sweeps)),
    )
    # max_window_area=1 sends every cluster down the giant path
    _run(_sparse_folds(), monkeypatch, [], max_window_area=1)
    assert seen and set(seen) == {(64, 8)}  # the new default tile

    seen.clear()
    _run(_sparse_folds(), monkeypatch, [], max_window_area=1, giant_tile=48, giant_max_sweeps=2)
    assert seen and set(seen) == {(48, 2)}


# ---------------------------------------------------------------------------
# (a2) giant_tile_fit: tile fitted to the region's geometry
# ---------------------------------------------------------------------------


def test_fit_tile():
    """``_fit_tile`` divides the longest side into an integer number of
    near-equal tiles, clamped to [0.75, 1.5] x target."""
    assert engine._fit_tile(125, 152, 64) == 51  # measured B0039 z16 giant
    assert engine._fit_tile(152, 125, 64) == 51  # longest side, either axis
    assert engine._fit_tile(128, 128, 64) == 64  # exactly k * target -> target
    assert engine._fit_tile(64, 64, 64) == 64
    # smaller than the target: clamped up, never below ceil(0.75 * target)
    assert engine._fit_tile(20, 20, 64) == 48
    assert engine._fit_tile(1, 1, 64) == 48
    assert engine._fit_tile(0, 0, 64) == 48  # degenerate box: no crash, no ZeroDivision
    # huge regions and odd targets stay in band
    for h, w, t in [(100000, 100000, 64), (3, 300000, 7), (999, 1, 10), (5, 5, 1)]:
        assert math.ceil(0.75 * t) <= engine._fit_tile(h, w, t) <= math.ceil(1.5 * t)


def test_giant_tile_fit_sizes_the_tiles(monkeypatch):
    """fit=True tiles the region at the fitted size, fit=False at the raw
    target — checked on the tile boxes the tiler actually builds."""
    boxes = []
    monkeypatch.setattr(engine, "_solve_window", lambda _phi, _c, tb, *a, **kw: boxes.append(tb))
    phi = np.zeros((2, 200, 200))  # fold-free -> the tiler exits after one sweep
    con = SimplexConstraint2D(shape=phi.shape[1:])

    def max_tile(**opts_kw):
        boxes.clear()
        engine._solve_giant_schwarz(
            phi,
            con,
            (10, 135, 10, 162),  # 125 x 152, the measured B0039 z16 giant
            0.01,
            NoneObjective(),
            400,
            1,
            engine.SliceReport(),
            1e-3,
            opts=engine._InnerOpts(giant_tile=64, **opts_kw),
        )
        return max(max(b[1] - b[0], b[3] - b[2]) for b in boxes)

    assert max_tile(giant_tile_fit=True) == 51
    assert max_tile(giant_tile_fit=False) == 64


# ---------------------------------------------------------------------------
# (b) osqp_max_iter reaches OSQP.setup
# ---------------------------------------------------------------------------


def test_osqp_max_iter_passes_through_to_setup(monkeypatch):
    captured = []

    class _FakeOSQP:
        def setup(self, p, q, a, lo, up, **kw):
            captured.append(kw)
            self._n = p.shape[0]

        def solve(self):
            return types.SimpleNamespace(
                x=np.zeros(self._n), info=types.SimpleNamespace(status="solved")
            )

    monkeypatch.setattr(isqp_mod, "HAS_OSQP", True)
    monkeypatch.setattr(isqp_mod, "osqp", types.SimpleNamespace(OSQP=_FakeOSQP), raising=False)

    kw = dict(
        cons=lambda f: np.array([1.0]),
        cons_jac=lambda f: np.ones((1, 2)),
        obj_grad=lambda f: np.zeros(2),
        maxiter=1,
    )
    isqp_mod.isqp_solve(np.zeros(2), **kw, osqp_max_iter=123)
    assert captured[-1]["max_iter"] == 123
    isqp_mod.isqp_solve(np.zeros(2), **kw)  # default: OSQP's own 8000
    assert captured[-1]["max_iter"] == 8000


# ---------------------------------------------------------------------------
# (c) the strategy exposes the knobs and forwards them to the engine
# ---------------------------------------------------------------------------


@needs_osqp
def test_strategy_forwards_knobs(monkeypatch):
    from dvfopt import ISQPWindowedStrategy
    from dvfopt.core.windowed._common import SliceReport
    from dvfopt.strategies import windowed as strat_mod

    seen = {}

    def fake_windowed_correct(phi_in, inner, **kwargs):
        seen.update(kwargs)
        return phi_in, SliceReport()

    monkeypatch.setattr(strat_mod, "windowed_correct", fake_windowed_correct)
    strategy = ISQPWindowedStrategy(
        no_tr_fallback=False,
        fallback_maxiter=5,
        qp_max_iter=7,
        qp_max_iter_fallback=9,
        giant_tile=48,
        giant_max_sweeps=2,
        giant_tile_fit=False,
    )
    phi = np.zeros((2, 8, 8))
    strategy.solve(
        phi,
        constraint=SimplexConstraint2D(shape=(8, 8)),
        objective=NoneObjective(),
        threshold=0.01,
    )
    assert seen["no_tr_fallback"] is False
    assert seen["fallback_maxiter"] == 5
    assert (seen["qp_max_iter"], seen["qp_max_iter_fallback"]) == (7, 9)
    assert (seen["giant_tile"], seen["giant_max_sweeps"]) == (48, 2)
    assert seen["giant_tile_fit"] is False

    defaults = ISQPWindowedStrategy()
    assert (defaults.no_tr_fallback, defaults.fallback_maxiter) == (True, 200)
    assert (defaults.qp_max_iter, defaults.qp_max_iter_fallback) == (1000, 500)
    assert (defaults.giant_tile, defaults.giant_max_sweeps) == (64, 8)
    assert defaults.giant_tile_fit is True
