"""``TriConstraint2DBilinear`` — the bilinear cell-min-Jdet certificate as four
smooth triangle rows per cell (both diagonal splits).

Contract under test: ``min over the 4 rows == 0.5 * cell_min_jdet_2d``, so
feasibility certifies the bilinear interpolant injective on every cell. Also
pins ``cell_min_jdet_2d`` itself against brute-force sampling (it had no
coverage before). The hypothesis adjoint-vs-FD check lives in
``test_constraint_properties.py``; the windowed no-damage / zero-fold checks
in ``test_windowed_isqp.py`` / ``test_windowed_strategy.py``.
"""

import importlib.util

import matplotlib
import numpy as np
import pytest

from dvfopt import (
    BarrierStrategy,
    DVFopt,
    DVFoptConfig,
    L1Objective,
    L2Objective,
    SLPStrategy,
    SLSQPFullGridStrategy,
    SLSQPWindowedStrategy,
    Solver,
    auto_strategy,
)
from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.constraints import (
    _CONSTRAINT_REGISTRY,
    TriConstraint2D,
    TriConstraint2DBilinear,
    make_constraint,
)
from dvfopt.core.primitives.coloring import dense_jacobian
from dvfopt.core.windowed._locality import _cell_pattern
from dvfopt.exceptions import IncompatibleConstraintError
from dvfopt.jacobian.injectivity_radius import cell_min_jdet_2d
from dvfopt.jacobian.shoelace import _all_triangle_areas_2d
from dvfopt.metrics import constraint_fold_stats
from dvfopt.objectives import make_objective
from dvfopt.strategies import make_strategy
from tests.conftest import planted_fold

THR, ERR_TOL = DEFAULT_PARAMS['threshold'], DEFAULT_PARAMS['err_tol']


def _rows(c, phi):
    H, W = c.shape
    return c.values(c.flatten(phi)).reshape(4, H - 1, W - 1)


def _certified(phi_out):
    """Feasible under the constraint <=> every cell's bilinear min-Jdet >= 2*threshold."""
    return cell_min_jdet_2d(phi_out).min() >= 2 * (THR - ERR_TOL)


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


def test_rows_are_the_2tri_pair_then_shoelaces_other_diagonal():
    phi = planted_fold(8, 6, seed=1)
    c = TriConstraint2DBilinear(phi.shape[1:])
    t = TriConstraint2D(phi.shape[1:])
    vals = c.values(c.flatten(phi))
    np.testing.assert_array_equal(vals[: t.n_constraints], t.values(t.flatten(phi)))
    # the same four triangles shoelace.py defines (its T1..T4 = our rows reversed)
    np.testing.assert_allclose(
        _rows(c, phi), _all_triangle_areas_2d(phi[0], phi[1])[::-1], atol=1e-12
    )


def test_reflex_vertex_on_the_2tri_diagonal_is_invisible_to_2tri():
    """A reflex vertex ON the TR-BL diagonal folds the quad (bilinear map) without
    folding either TR-BL triangle — invisible to 2-tri, caught by both diagonals."""
    phi = np.zeros((2, 3, 3))
    # push pixel (0, 1) — cell (0, 0)'s TR corner — inward past that cell's TL-BR diagonal
    phi[0, 0, 1], phi[1, 0, 1] = 0.7, -0.7
    t = TriConstraint2D((3, 3))
    c = TriConstraint2DBilinear((3, 3))
    assert t.values(t.flatten(phi)).min() > 0
    assert c.values(c.flatten(phi)).min() < 0
    assert cell_min_jdet_2d(phi).min() < 0


def test_cell_min_jdet_2d_is_the_bilinear_minimum():
    """Brute force: the bilinear map's Jdet sampled over each cell is never below
    the closed-form corner minimum, and attains it (biaffine => corner extremum)."""
    rng = np.random.default_rng(0)
    H, W = 4, 5
    phi = rng.normal(0, 0.5, (2, H, W))
    Y, X = np.mgrid[:H, :W] + phi
    a = np.linspace(0, 1, 41)
    A, B = np.meshgrid(a, a, indexing='ij')  # alpha down rows, beta across cols

    def d_beta(F):  # (H-1, W-1, 41, 41): x-direction edge, blended top->bottom
        top, bot = F[:-1, 1:] - F[:-1, :-1], F[1:, 1:] - F[1:, :-1]
        return (1 - A) * top[..., None, None] + A * bot[..., None, None]

    def d_alpha(F):  # y-direction edge, blended left->right
        left, right = F[1:, :-1] - F[:-1, :-1], F[1:, 1:] - F[:-1, 1:]
        return (1 - B) * left[..., None, None] + B * right[..., None, None]

    jdet = d_beta(X) * d_alpha(Y) - d_alpha(X) * d_beta(Y)
    cm = cell_min_jdet_2d(phi)
    assert (jdet.min((2, 3)) >= cm - 1e-12).all()
    np.testing.assert_allclose(jdet.min((2, 3)), cm, atol=1e-12)


# ---------------------------------------------------------------------------
# registry / metrics / auto / windowed pattern
# ---------------------------------------------------------------------------


def test_registry_and_fold_stats():
    assert isinstance(make_constraint('bilinear', (6, 6)), TriConstraint2DBilinear)
    name, st = constraint_fold_stats(planted_fold(10, 10, seed=0), constraint='bilinear')
    assert name == 'bilinear' and st.n_neg > 0
    _, clean = constraint_fold_stats(np.zeros((2, 8, 8)), constraint='bilinear')
    assert clean.feasible and clean.n_neg == 0


_LABELS_2D = sorted(k for k, v in _CONSTRAINT_REGISTRY.items() if v.dim == 2)


@pytest.mark.parametrize('label', _LABELS_2D)
@pytest.mark.parametrize('osqp', [True, False])
@pytest.mark.parametrize('n_neg,min_val', [(5, -0.1), (10_000, -20.0)])
@pytest.mark.parametrize('objective', ['l1', 'l2'])
def test_auto_strategy_always_picks_an_accepting_strategy(
    monkeypatch, label, osqp, n_neg, min_val, objective
):
    """The invariant behind ``auto_strategy``: whatever it returns must compose
    with the constraint it was asked about — for every registered 2D label, with
    and without ``osqp`` installed."""
    if not osqp:
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, 'find_spec', lambda n, *a: None if n == 'osqp' else real(n, *a)
        )
    c = make_constraint(label, (12, 12))
    strategy = make_strategy(auto_strategy(c, n_neg, min_val, objective))
    Solver(constraint=c, objective=make_objective(objective), strategy=strategy)


@pytest.mark.parametrize('strategy', [SLPStrategy, SLSQPFullGridStrategy])
def test_two_tri_specialised_strategies_reject_it_at_construction(strategy):
    with pytest.raises(IncompatibleConstraintError):
        Solver(
            constraint=TriConstraint2DBilinear((12, 12)),
            objective=L1Objective(),
            strategy=strategy(),
        )


@pytest.mark.parametrize('cls,k', [(TriConstraint2D, 2), (TriConstraint2DBilinear, 4)])
def test_structural_sparsity_pattern_matches_the_probed_one(cls, k):
    """The windowed engine's index-arithmetic pattern is exactly what dense
    probing found (same columns, same order) — for both triangle families."""
    c = cls((6, 7))
    J = dense_jacobian(c, np.random.default_rng(0).normal(0, 0.3, c.n_variables))
    probed = [np.nonzero(J[r])[0] for r in range(c.n_constraints)]
    struct = _cell_pattern(6, 7, k)
    assert len(struct) == len(probed)
    assert all(np.array_equal(a, b) for a, b in zip(probed, struct))


# ---------------------------------------------------------------------------
# solves: feasible under the constraint => bilinear-injective on every cell
# ---------------------------------------------------------------------------


def test_barrier_reaches_feasibility_and_certifies_the_cells():
    phi = planted_fold(14, 14, seed=3, scale=0.3)
    c = TriConstraint2DBilinear(phi.shape[1:])
    res = Solver(constraint=c, objective=L2Objective(), strategy=BarrierStrategy()).fit(phi)
    assert res.init_n_neg > 0
    assert res.final_n_neg == 0 and res.feasible
    assert _certified(res.corrected)


def test_slsqp_windowed_triangle_mode_solves_it():
    """The legacy windowed SLSQP's triangle mode enforces all four triangles —
    exactly this constraint — so it accepts and solves it."""
    phi = planted_fold(10, 10, seed=0)
    c = TriConstraint2DBilinear(phi.shape[1:])
    res = Solver(constraint=c, objective=L2Objective(), strategy=SLSQPWindowedStrategy()).fit(phi)
    assert res.init_n_neg > 0
    assert res.final_n_neg == 0 and res.feasible
    assert _certified(res.corrected)


@pytest.mark.parametrize('label', ['2tri', 'bilinear', 'jdet', 'finite'])
def test_facade_resolves_every_2d_label_and_plots(label):
    """The DVFopt facade goes through the constraint registry, and
    ``plot_feasibility`` maps every row layout (2/4 triangle rows, the default
    '2tri' snapshot's corner-patch rows, per-pixel Jdet, per-cell finite)."""
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from dvfopt._plots import plot_feasibility

    phi = planted_fold(8, 8, seed=3)
    cfg = DVFoptConfig(constraint=label, solver='barrier', objective='l2', record_snapshots=True)
    res = DVFopt(cfg).fit(phi)
    plot_feasibility(res, z=0)  # snapshot path
    plot_feasibility(DVFopt(DVFoptConfig(constraint=label, solver='barrier')).fit(phi), z=0)
    plt.close('all')
