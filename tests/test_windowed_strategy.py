"""Tests for the windowed wrapper strategies (``dvfopt.strategies.windowed``)
and the promoted ``FiniteJdetConstraint2D``.

The solve-path tests need the optional ``osqp`` dependency (skipped per-test
via ``HAS_OSQP``); construction, rejection, the osqp ImportError gate, and the
finite-Jdet constraint checks run everywhere.
"""

import numpy as np
import pytest

from dvfopt import (
    ISQPWindowedStrategy,
    L2Objective,
    Solver,
    WindowedWrapperStrategy,
)
from dvfopt.constraints import (
    FiniteJdetConstraint2D,
    JdetConstraint2D,
    Tet6Constraint3D,
    TriConstraint2D,
    TriConstraint2DBilinear,
    TriConstraint2DFullCoverage,
    make_constraint,
)
from dvfopt.core.primitives.coloring import dense_jacobian
from dvfopt.core.primitives.isqp import HAS_OSQP
from dvfopt.exceptions import IncompatibleConstraintError
from dvfopt.metrics import constraint_fold_stats
from dvfopt.strategies import make_strategy

needs_osqp = pytest.mark.skipif(not HAS_OSQP, reason="osqp not installed")


def _sparse_folds(H=100, W=100, seed=3):
    """A mostly fold-free field with a few separated sharp fold blobs — folds
    under all three registered 2D families (same fixture as test_windowed_isqp)."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((2, H, W))
    for cy, cx in [(20, 20), (20, 75), (70, 25), (72, 72)]:
        phi[0, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (5, 5))
        phi[1, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (5, 5))
    return phi


# ---------------------------------------------------------------------------
# (a) Solver composition reaches zero folds (needs osqp)
# ---------------------------------------------------------------------------


@needs_osqp
@pytest.mark.parametrize(
    "ctype", [TriConstraint2D, JdetConstraint2D, FiniteJdetConstraint2D, TriConstraint2DBilinear]
)
def test_solver_composition_reaches_zero_folds(ctype):
    phi = _sparse_folds()
    res = Solver(
        constraint=ctype(shape=phi.shape[1:]),
        objective=L2Objective(),
        strategy=ISQPWindowedStrategy(),
    ).fit(phi)
    assert res.init_n_neg > 0  # the fixture folds under every family
    assert res.final_n_neg == 0
    assert res.feasible
    assert res.corrected.shape == phi.shape


# ---------------------------------------------------------------------------
# (b) registry labels
# ---------------------------------------------------------------------------


def test_make_strategy_zero_arg():
    s = make_strategy("isqp_windowed")
    assert isinstance(s, ISQPWindowedStrategy)
    assert s.inner == "isqp"


def test_from_spec_resolves_registry_label():
    solver = Solver.from_spec(strategy="isqp_windowed", constraint="2tri_standard", shape=(20, 20))
    assert isinstance(solver.strategy, ISQPWindowedStrategy)
    assert isinstance(solver.constraint, TriConstraint2D)


# ---------------------------------------------------------------------------
# (c) rejections (no osqp needed)
# ---------------------------------------------------------------------------


def test_rejects_fullcoverage_constraint_at_construction():
    with pytest.raises(IncompatibleConstraintError):
        Solver(
            constraint=TriConstraint2DFullCoverage(shape=(12, 12)),
            objective=L2Objective(),
            strategy=ISQPWindowedStrategy(),
        )


def test_rejects_6tet_constraint_at_construction():
    with pytest.raises(IncompatibleConstraintError):
        Solver(
            constraint=Tet6Constraint3D(shape=(4, 6, 6)),
            objective=L2Objective(),
            strategy=ISQPWindowedStrategy(),
        )


def test_wrapper_requires_inner():
    with pytest.raises(ValueError, match="inner"):
        WindowedWrapperStrategy()


def test_wrapper_rejects_unknown_inner_label():
    with pytest.raises(ValueError, match="unknown inner"):
        WindowedWrapperStrategy(inner="nope")


# ---------------------------------------------------------------------------
# (d) the osqp gate (works with or without osqp installed)
# ---------------------------------------------------------------------------


def test_isqp_strategy_raises_friendly_importerror_without_osqp(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "osqp":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    phi = np.zeros((2, 10, 10))
    with pytest.raises(ImportError, match=r"dvfopt\[solvers\]"):
        ISQPWindowedStrategy().solve(
            phi,
            constraint=JdetConstraint2D(shape=(10, 10)),
            objective=L2Objective(),
            threshold=0.01,
        )


# ---------------------------------------------------------------------------
# (e) FiniteJdetConstraint2D (no osqp needed)
# ---------------------------------------------------------------------------


def test_finite_analytic_jacobian_matches_dense_adjoint_exactly():
    """The analytic sparse Jacobian must equal the dense adjoint build exactly:
    each row has 6 distinct-column nonzeros, so ``adjoint(flat, e_i)`` copies
    row ``i`` with no summation — bitwise equality, not just tolerance."""
    c = FiniteJdetConstraint2D(shape=(9, 7))
    rng = np.random.default_rng(4)
    flat = rng.normal(0, 0.5, c.n_variables)
    dense = dense_jacobian(c, flat)  # rows via adjoint probes
    analytic = np.asarray(c.jacobian(flat).todense())
    assert np.array_equal(dense, analytic)


def test_make_constraint_finite_round_trip():
    c = make_constraint("finite", (8, 8))
    assert isinstance(c, FiniteJdetConstraint2D)
    assert c.n_constraints == 7 * 7 and c.n_variables == 2 * 8 * 8
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.3, (2, 8, 8))
    assert np.array_equal(c.unflatten(c.flatten(phi)), phi)


def test_constraint_fold_stats_finite_smoke():
    phi = _sparse_folds(seed=3)
    name, stats = constraint_fold_stats(phi, constraint="finite")
    assert name == "finite"
    assert stats.n_neg > 0 and stats.n_below >= stats.n_neg
    assert stats.min_val < 0 and not stats.feasible
    # a clean identity field is feasible under 'finite' (cell dets == 1)
    name2, clean = constraint_fold_stats(np.zeros((2, 12, 12)), constraint="finite")
    assert name2 == "finite" and clean.n_neg == 0 and clean.feasible
