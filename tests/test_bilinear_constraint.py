"""``TriConstraint2DBilinear`` — the bilinear cell-min-Jdet certificate as four
smooth triangle rows per cell (both diagonal splits).

The contract under test: ``min over the 4 rows == 0.5 * cell_min_jdet_2d``, so
feasibility certifies the bilinear interpolant injective on every cell. Also
pins ``cell_min_jdet_2d`` itself against brute-force sampling (it had no
coverage before).
"""

import numpy as np
import pytest

from dvfopt import (
    BarrierStrategy,
    ISQPWindowedStrategy,
    L1Objective,
    L2Objective,
    SLPStrategy,
    Solver,
    auto_strategy,
)
from dvfopt.constraints import TriConstraint2D, TriConstraint2DBilinear, make_constraint
from dvfopt.core.primitives.isqp import HAS_OSQP
from dvfopt.exceptions import IncompatibleConstraintError
from dvfopt.jacobian.injectivity_radius import cell_min_jdet_2d
from dvfopt.metrics import constraint_fold_stats
from dvfopt.strategies import make_strategy
from tests.conftest import planted_fold


def _rows(c, phi):
    H, W = c.shape
    return c.values(c.flatten(phi)).reshape(4, H - 1, W - 1)


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


def test_identity_is_half_everywhere():
    c = TriConstraint2DBilinear((5, 7))
    assert c.n_constraints == 4 * 4 * 6
    np.testing.assert_allclose(_rows(c, np.zeros((2, 5, 7))), 0.5)


def test_cell_min_matches_cell_min_jdet_2d():
    phi = planted_fold(9, 11, seed=4, scale=0.6)
    c = TriConstraint2DBilinear(phi.shape[1:])
    np.testing.assert_allclose(_rows(c, phi).min(0), 0.5 * cell_min_jdet_2d(phi), atol=1e-12)


def test_first_two_blocks_are_the_2tri_rows():
    phi = planted_fold(8, 6, seed=1)
    c = TriConstraint2DBilinear(phi.shape[1:])
    t = TriConstraint2D(phi.shape[1:])
    np.testing.assert_array_equal(
        c.values(c.flatten(phi))[: t.n_constraints], t.values(t.flatten(phi))
    )


def test_opposite_diagonal_rows_differ_from_2tri_on_a_skewed_cell():
    """A reflex vertex ON the TR-BL diagonal folds the quad (bilinear map) without
    folding either TR-BL triangle — invisible to 2-tri, caught by both diagonals."""
    phi = np.zeros((2, 3, 3))
    # push pixel (0, 1) — cell (0, 0)'s TR corner — inward past that cell's TL-BR diagonal
    phi[0, 0, 1], phi[1, 0, 1] = 0.7, -0.7
    t = TriConstraint2D((3, 3))
    c = TriConstraint2DBilinear((3, 3))
    assert t.values(t.flatten(phi)).min() > 0  # 2-tri: TR-BL split sees no fold
    assert c.values(c.flatten(phi)).min() < 0  # both diagonals: the fold is caught
    assert cell_min_jdet_2d(phi).min() < 0


def test_adjoint_matches_finite_differences():
    phi = planted_fold(6, 5, seed=2)
    c = TriConstraint2DBilinear(phi.shape[1:])
    flat = c.flatten(phi)
    rng = np.random.default_rng(0)
    v = rng.normal(size=c.n_constraints)
    eps = 1e-6
    num = np.empty(c.n_variables)
    for i in range(c.n_variables):
        p, m = flat.copy(), flat.copy()
        p[i] += eps
        m[i] -= eps
        num[i] = np.dot((c.values(p) - c.values(m)) / (2 * eps), v)
    np.testing.assert_allclose(c.adjoint(flat, v), num, atol=1e-6)


def test_cell_min_jdet_2d_is_the_bilinear_minimum():
    """Brute force: the bilinear map's Jdet sampled over each cell is never below
    the closed-form corner minimum, and attains it (biaffine => corner extremum)."""
    rng = np.random.default_rng(0)
    H, W = 4, 5
    phi = rng.normal(0, 0.5, (2, H, W))
    ref = np.mgrid[:H, :W].astype(float)
    Y, X = ref[0] + phi[0], ref[1] + phi[1]
    a = np.linspace(0, 1, 41)
    A, B = np.meshgrid(a, a, indexing='ij')  # alpha down rows, beta across cols
    cm = cell_min_jdet_2d(phi)
    for i in range(H - 1):
        for j in range(W - 1):
            # bilinear map P(a, b) = (1-a)(1-b) TL + (1-a) b TR + a (1-b) BL + a b BR
            def d_beta(F):
                return (1 - A) * (F[i, j + 1] - F[i, j]) + A * (F[i + 1, j + 1] - F[i + 1, j])

            def d_alpha(F):
                return (1 - B) * (F[i + 1, j] - F[i, j]) + B * (F[i + 1, j + 1] - F[i, j + 1])

            jdet = d_beta(X) * d_alpha(Y) - d_alpha(X) * d_beta(Y)
            assert jdet.min() >= cm[i, j] - 1e-12
            assert abs(jdet.min() - cm[i, j]) < 1e-12


# ---------------------------------------------------------------------------
# registry / metrics / auto
# ---------------------------------------------------------------------------


def test_registry_and_fold_stats():
    assert isinstance(make_constraint('bilinear', (6, 6)), TriConstraint2DBilinear)
    name, st = constraint_fold_stats(planted_fold(10, 10, seed=0), constraint='bilinear')
    assert name == 'bilinear' and st.n_neg > 0
    _, clean = constraint_fold_stats(np.zeros((2, 8, 8)), constraint='bilinear')
    assert clean.feasible and clean.n_neg == 0


@pytest.mark.parametrize('n_neg,min_val', [(5, -0.1), (10_000, -20.0)])
def test_auto_strategy_picks_something_that_accepts_it(n_neg, min_val):
    c = TriConstraint2DBilinear((12, 12))
    label = auto_strategy(c, n_neg, min_val, 'l1')
    Solver(constraint=c, objective=L1Objective(), strategy=make_strategy(label))


def test_two_triangle_specialised_strategy_rejects_it():
    with pytest.raises(IncompatibleConstraintError):
        Solver(
            constraint=TriConstraint2DBilinear((12, 12)),
            objective=L1Objective(),
            strategy=SLPStrategy(),
        )


# ---------------------------------------------------------------------------
# solves: feasible under the constraint => bilinear-injective on every cell
# ---------------------------------------------------------------------------


THR, ERR_TOL = 0.01, 1e-5


def _certified(phi_out):
    """Feasible under the constraint <=> every cell's bilinear min-Jdet >= 2*threshold."""
    return cell_min_jdet_2d(phi_out).min() >= 2 * (THR - ERR_TOL)


def test_barrier_reaches_feasibility_and_certifies_the_cells():
    phi = planted_fold(14, 14, seed=3, scale=0.3)
    c = TriConstraint2DBilinear(phi.shape[1:])
    res = Solver(
        constraint=c, objective=L2Objective(), strategy=BarrierStrategy(), threshold=THR
    ).fit(phi)
    assert res.init_n_neg > 0
    assert res.final_n_neg == 0 and res.feasible
    assert _certified(res.corrected)


@pytest.mark.skipif(not HAS_OSQP, reason='osqp not installed')
def test_isqp_windowed_reaches_feasibility_and_certifies_the_cells():
    rng = np.random.default_rng(3)
    phi = np.zeros((2, 60, 60))
    for cy, cx in [(15, 15), (15, 44), (44, 18)]:
        phi[:, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (2, 5, 5))
    c = TriConstraint2DBilinear(phi.shape[1:])
    res = Solver(
        constraint=c, objective=L2Objective(), strategy=ISQPWindowedStrategy(), threshold=THR
    ).fit(phi)
    assert res.init_n_neg > 0
    assert res.final_n_neg == 0 and res.feasible
    assert _certified(res.corrected)
