"""``polish``: per-window anchored polish after a successful solve (default off)."""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2DBilinear
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import windowed_correct
from dvfopt.objectives import L2Objective, NoneObjective
from dvfopt.testdata import make_random_dvf

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")


def _fold_field(H=64, W=64, at=(24, 26)):
    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    phi = np.zeros((2, H, W))
    phi[:, at[0] : at[0] + patch.shape[1], at[1] : at[1] + patch.shape[2]] = patch
    return phi


@needs_osqp
def test_polish_off_is_byte_identical():
    phi = _fold_field()
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    kw = dict(constraint=c, objective=NoneObjective(), threshold=0.01, verbose=0)
    out_a, rep_a = windowed_correct(phi.copy(), "isqp", **kw)
    out_b, rep_b = windowed_correct(phi.copy(), "isqp", polish=None, **kw)
    assert np.array_equal(out_a, out_b)
    assert rep_a.polish_windows == rep_b.polish_windows == 0


@needs_osqp
def test_polish_recovers_fidelity_at_zero_folds():
    """objective 'none' + polish 'l2' must keep 0 folds and damage 0 while moving
    the field LESS than plain 'none' (that is the whole point), and at least one
    window's polish must have been accepted."""
    phi = _fold_field()
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    kw = dict(constraint=c, objective=NoneObjective(), threshold=0.01, verbose=0)
    out_none, rep_none = windowed_correct(phi.copy(), "isqp", **kw)
    out_pol, rep_pol = windowed_correct(phi.copy(), "isqp", polish="l2", **kw)
    assert rep_pol.folds_after == 0 and rep_pol.damage == 0
    assert rep_pol.polish_windows > 0 and rep_pol.polish_accepted > 0
    m_none = float(np.linalg.norm((out_none - phi).ravel()))
    m_pol = float(np.linalg.norm((out_pol - phi).ravel()))
    assert m_pol < m_none


@needs_osqp
def test_polish_never_worsens_the_l2_objective_path():
    """polish under an in-solve L2 objective is allowed and must stay 0 folds /
    damage 0 (the verify-and-revert guards it); it may or may not accept."""
    phi = _fold_field()
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    _out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=L2Objective(),
        threshold=0.01,
        verbose=0,
        polish="l2",
    )
    assert rep.folds_after == 0 and rep.damage == 0


@needs_osqp
def test_strategy_forwards_polish():
    assert engine._InnerOpts().polish is None
    from dvfopt import ISQPWindowedStrategy

    s = ISQPWindowedStrategy(polish="l2", polish_maxiter=7)
    assert s.polish == "l2" and s.polish_maxiter == 7
