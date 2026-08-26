"""Hybrid QP backend for the isqp driver (``dvfopt.core.primitives.isqp``).

Covers the IP-vs-ADMM dispatch policy (against fakes, so no solver noise), the
``'osqp'`` byte-identity pin, the missing-clarabel degradation, and the knob
plumbing strategy -> engine -> driver.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from dvfopt.constraints import JdetConstraint2D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.primitives.isqp import HAS_OSQP, _HybridQP, _make_qp
from dvfopt.core.windowed import windowed_correct
from dvfopt.objectives import L2Objective
from dvfopt.strategies.windowed import ISQPWindowedStrategy

if not HAS_OSQP:
    pytest.skip("osqp not installed", allow_module_level=True)


# --------------------------------------------------------------------------
# Fakes: an OSQP whose ADMM iteration counts are scripted, and a clarabel
# whose solve is recorded. Both are installed over the module-level names the
# backend resolves (``isqp.osqp`` / ``isqp.clarabel``).
# --------------------------------------------------------------------------
class _FakeOSQP:
    def __init__(self, admm_iters):
        self.script = list(admm_iters)
        self.admm_calls = []
        self.warm_starts = []

    def setup(self, *a, **k):
        pass

    def update(self, **k):
        pass

    def warm_start(self, x=None, y=None):
        self.warm_starts.append(np.asarray(x))

    def solve(self):
        n = self.script.pop(0)
        self.admm_calls.append(n)
        return SimpleNamespace(x=np.zeros(3), info=SimpleNamespace(iter=n, status="solved"))


def _fake_clarabel(log, status="Solved"):
    def solver(p, q, a, b, cones, settings):
        log.append("ip")
        return SimpleNamespace(
            solve=lambda: SimpleNamespace(x=np.zeros(q.size), status=status, iterations=7)
        )

    return SimpleNamespace(
        DefaultSettings=SimpleNamespace,
        NonnegativeConeT=lambda n: n,
        DefaultSolver=solver,
    )


def _fitted(monkeypatch, admm_iters, ip_log, status="Solved", **kw):
    """A ``_HybridQP`` set up over the fakes, plus the fake OSQP behind it."""
    fake = _FakeOSQP(admm_iters)
    monkeypatch.setattr(isqp_mod, "osqp", SimpleNamespace(OSQP=lambda: fake))
    monkeypatch.setattr(isqp_mod, "clarabel", _fake_clarabel(ip_log, status))
    qp = _HybridQP(**kw)
    eye = sparse.eye(3, format="csc")
    qp.setup(eye, np.zeros(3), eye, np.zeros(3), np.full(3, np.inf))
    return qp, fake


def test_policy_cold_then_admm_then_tail(monkeypatch):
    """IP on the cold solve, then ADMM, then IP again only after a long ADMM run."""
    ip = []
    qp, fake = _fitted(monkeypatch, [100, 900, 50], ip, ip_after_admm_iters=800)
    qp.solve()  # 1: cold -> IP
    assert ip == ["ip"] and fake.admm_calls == []
    assert len(fake.warm_starts) == 1  # the IP point seeds OSQP's warm start
    qp.solve()  # 2: last=0 -> ADMM (100)
    qp.solve()  # 3: last=100 < 800 -> ADMM (900)
    assert ip == ["ip"] and fake.admm_calls == [100, 900]
    qp.solve()  # 4: last=900 >= 800 -> IP
    assert ip == ["ip", "ip"]
    qp.solve()  # 5: last=0 -> ADMM (50)
    assert fake.admm_calls == [100, 900, 50]


def test_policy_no_cold(monkeypatch):
    """``ip_cold=False`` drops the cold leg; the tail signal still fires."""
    ip = []
    qp, fake = _fitted(monkeypatch, [900, 10], ip, ip_cold=False, ip_after_admm_iters=800)
    qp.solve()  # ADMM (900)
    assert ip == [] and fake.admm_calls == [900]
    qp.solve()  # IP (tail)
    assert ip == ["ip"]


def test_ip_failure_falls_through_to_admm(monkeypatch):
    """A non-'Solved' IP status is a fall-through, not an error."""
    ip = []
    qp, fake = _fitted(monkeypatch, [10, 20], ip, status="PrimalInfeasible")
    res = qp.solve()
    assert ip == ["ip"] and fake.admm_calls == [10]  # tried IP, used ADMM
    assert res.info.iter == 10
    assert not fake.warm_starts  # a failed IP never seeds the warm start


def test_ip_exception_falls_through_to_admm(monkeypatch):
    """So is an exception anywhere inside the IP leg."""
    ip = []
    qp, fake = _fitted(monkeypatch, [10], ip)

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(isqp_mod.clarabel, "DefaultSolver", boom)
    qp.solve()
    assert fake.admm_calls == [10]


def test_make_qp_backends(monkeypatch):
    import osqp

    assert isinstance(_make_qp("osqp", True, 800), osqp.OSQP)
    monkeypatch.setattr(isqp_mod, "HAS_CLARABEL", True)
    assert isinstance(_make_qp("hybrid", True, 800), _HybridQP)
    monkeypatch.setattr(isqp_mod, "HAS_CLARABEL", False)
    assert isinstance(_make_qp("hybrid", True, 800), osqp.OSQP)  # degrades, never raises
    with pytest.raises(ValueError, match="unknown qp_backend"):
        _make_qp("gurobi", True, 800)


# --------------------------------------------------------------------------
# End-to-end through the engine.
# --------------------------------------------------------------------------
def _folded(H=60, W=60, seed=3):
    rng = np.random.default_rng(seed)
    phi = np.zeros((2, H, W))
    for cy, cx in [(18, 18), (40, 42)]:
        phi[0, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (5, 5))
        phi[1, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (5, 5))
    return phi


def _run(phi, **kw):
    return windowed_correct(
        phi,
        constraint=JdetConstraint2D(shape=phi.shape[1:]),
        objective=L2Objective(),
        threshold=0.01,
        **kw,
    )


def test_osqp_backend_is_byte_identical(monkeypatch):
    """``qp_backend='osqp'`` never touches the hybrid machinery — sabotaging
    every hybrid ingredient leaves the output bit-for-bit unchanged."""
    phi = _folded()
    ref, rep = _run(phi, qp_backend="osqp")
    assert rep.folds_after == 0 and rep.damage == 0

    def boom(*a, **k):
        raise AssertionError("osqp backend must not build the hybrid path")

    monkeypatch.setattr(isqp_mod, "_HybridQP", boom)
    monkeypatch.setattr(isqp_mod, "clarabel", SimpleNamespace(DefaultSolver=boom))
    out, _ = _run(phi, qp_backend="osqp")
    assert np.array_equal(out, ref)


def test_hybrid_without_clarabel_is_the_osqp_path(monkeypatch):
    """No clarabel installed -> 'hybrid' silently IS 'osqp' (same bits, no error)."""
    phi = _folded()
    ref, _ = _run(phi, qp_backend="osqp")
    monkeypatch.setattr(isqp_mod, "HAS_CLARABEL", False)
    out, rep = _run(phi, qp_backend="hybrid")
    assert np.array_equal(out, ref)
    assert rep.folds_after == 0 and rep.damage == 0


def test_hybrid_default_clears_folds_without_damage():
    phi = _folded()
    _, rep = _run(phi)  # engine default is 'hybrid'
    assert rep.folds_after == 0 and rep.damage == 0


def test_defaults_engine_hybrid_primitive_osqp():
    """The engine defaults to hybrid; the primitive keeps the pre-hybrid default."""
    import inspect

    from dvfopt.core.windowed._common import _InnerOpts

    assert _InnerOpts().qp_backend == "hybrid"
    assert inspect.signature(windowed_correct).parameters["qp_backend"].default == "hybrid"
    assert inspect.signature(isqp_mod.isqp_solve).parameters["qp_backend"].default == "osqp"
    assert ISQPWindowedStrategy().qp_backend == "hybrid"
    assert ISQPWindowedStrategy().ip_after_admm_iters == 800


def test_strategy_knobs_reach_the_driver(monkeypatch):
    """Strategy dataclass -> windowed_correct -> _InnerOpts -> isqp_solve."""
    from dvfopt.core.windowed import _inners

    seen = []
    real = _inners.isqp_solve

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(_inners, "isqp_solve", spy)
    phi = _folded()
    ISQPWindowedStrategy(qp_backend="osqp", ip_cold=False, ip_after_admm_iters=123).solve(
        phi,
        constraint=JdetConstraint2D(shape=phi.shape[1:]),
        objective=L2Objective(),
        threshold=0.01,
    )
    assert seen, "the strategy never reached the isqp driver"
    assert all(k["qp_backend"] == "osqp" for k in seen)
    assert all(k["ip_cold"] is False for k in seen)
    assert all(k["ip_after_admm_iters"] == 123 for k in seen)


# --------------------------------------------------------------------------
# The backend rung of the escalation ladder: a window whose interior-point
# trajectory ends GENUINELY folded is retried on plain OSQP, from the ORIGINAL
# start state, before grow-on-failure.
# --------------------------------------------------------------------------
def _backend_fake(calls):
    """Inner that fails every 'hybrid' attempt (leaving the window folded) and
    clears the window on 'osqp' (zeroing the free pixels restores identity)."""

    def inner(
        sub,
        label,
        maxiter,
        trace=None,
        trust_region=True,
        osqp_max_iter=None,
        qp_backend="osqp",
        **_,
    ):
        calls.append(
            dict(
                qp_backend=qp_backend, trust_region=trust_region, start=np.asarray(sub.flat0).copy()
            )
        )
        x = np.asarray(sub.flat0, dtype=float).copy()
        if qp_backend != "osqp":
            x[sub.free_idx] *= 1.5  # amplify the fold: still infeasible, and != flat0
            return x, 1, False
        x[sub.free_idx] = 0.0  # frozen vars stay at flat0 — no-damage still holds
        return x, 1, True

    return inner


def test_backend_fallback_clears_window_without_growing(monkeypatch):
    from dvfopt.core.windowed import _common as engine

    calls = []
    monkeypatch.setattr(engine, "solve_window_inner", _backend_fake(calls))
    phi = _folded()
    _out, rep = _run(phi, qp_backend="hybrid")

    assert rep.folds_before > 0 and rep.folds_after == 0
    assert rep.damage == 0
    assert rep.backend_fallbacks > 0
    assert any(w.backend_fallback for w in rep.windows)
    assert all(w.grows == 0 for w in rep.windows)  # cleared without a single grow

    # attempt 1 hybrid (TR on) -> no-TR retry from the FAILED iterate -> attempt 2
    # on osqp from the ORIGINAL start state (the IP trajectory is what led astray).
    assert [c["qp_backend"] for c in calls[:3]] == ["hybrid", "hybrid", "osqp"]
    assert [c["trust_region"] for c in calls[:3]] == [True, False, True]
    assert not np.array_equal(calls[1]["start"], calls[0]["start"])
    assert np.array_equal(calls[2]["start"], calls[0]["start"])


def test_osqp_backend_never_takes_the_backend_rung(monkeypatch):
    """Nothing to fall back TO when the run is already on osqp."""
    from dvfopt.core.windowed import _common as engine

    calls = []
    monkeypatch.setattr(engine, "solve_window_inner", _backend_fake(calls))
    _out, rep = _run(_folded(), qp_backend="osqp")
    assert rep.backend_fallbacks == 0
    assert {c["qp_backend"] for c in calls} == {"osqp"}


def test_giant_tiles_skip_the_backend_rung(monkeypatch):
    """A tile (``allow_grow=False``) is re-swept by the Schwarz loop and then the
    mop, so a second full attempt per tile defeats tiling exactly like growing
    one does — measured at 505 s vs 264 s on raw B0039 z16 for no fold gain."""
    from dvfopt.core.windowed import _common as engine

    calls = []
    monkeypatch.setattr(engine, "solve_window_inner", _backend_fake(calls))
    phi = _folded()
    rep = engine.SliceReport()
    engine._solve_window(
        phi,
        JdetConstraint2D(shape=phi.shape[1:]),
        (14, 24, 14, 24),
        0.01,
        L2Objective(),
        400,
        1,
        rep,
        allow_grow=False,
        opts=engine._InnerOpts(qp_backend="hybrid"),
    )
    assert rep.backend_fallbacks == 0
    assert {c["qp_backend"] for c in calls} == {"hybrid"}
