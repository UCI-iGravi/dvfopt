"""Linear orientation rows (``orientation_delta``) of the windowed engine.

Deformed grid edges must keep a positive projection of at least ``delta`` on their
own direction (plus the anti-diagonal convexity rows). A cell on the rotated
orientation branch violates them, so the QP never heads there. These tests pin the
row values on the identity, that a rotated cell violates them while a compressed
but properly oriented cell does not, the Jacobian, the pack guard, and the plumbing.
"""

import numpy as np
import pytest

from dvfopt.constraints import JdetConstraint2D, SimplexConstraint2D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import build_subproblem
from dvfopt.objectives import NoneObjective

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")


def _sub(phi, delta):
    c = SimplexConstraint2D(shape=phi.shape[1:])
    return build_subproblem(c, phi, (3, 9, 3, 9), 0.01, None, 1e-3, orientation_delta=delta), c


def test_rows_on_the_identity_equal_one_minus_delta():
    phi = np.zeros((2, 12, 12))
    sub0, _ = _sub(phi, None)
    sub1, _ = _sub(phi, 0.05)
    v = sub1.cons(sub1.flat0)
    assert sub1.n_enforced > sub0.n_enforced
    assert np.allclose(v[sub0.n_enforced :], 0.95)  # 1 - delta on every appended row
    assert np.array_equal(v[: sub0.n_enforced], sub0.cons(sub0.flat0))  # bilinear rows untouched


def test_rotated_cell_violates_and_compressed_cell_does_not():
    H = W = 12
    yy, xx = np.mgrid[0:H, 0:W]
    # a properly oriented, strongly compressed field: image = 0.2 * grid -> edges 0.2 >= delta
    phi_c = np.stack([0.2 * yy - yy, 0.2 * xx - xx]).astype(float)
    sub_c, _ = _sub(phi_c, 0.05)
    assert sub_c.cons(sub_c.flat0).min() >= 0.0  # all rows satisfied (areas 0.04 >= 0.01 too)
    # a 180-degree rotated patch inside: image = 2*center - grid -> edge projections negative
    phi_r = np.zeros((2, H, W))
    cy, cx = 6.0, 6.0
    phi_r[0, 4:9, 4:9] = (2 * cy - yy[4:9, 4:9]) - yy[4:9, 4:9]
    phi_r[1, 4:9, 4:9] = (2 * cx - xx[4:9, 4:9]) - xx[4:9, 4:9]
    sub_r, _ = _sub(phi_r, 0.05)
    n_bil = _sub(phi_r, None)[0].n_enforced
    v = sub_r.cons(sub_r.flat0)
    assert v[n_bil:].min() < -1.0  # the rotated cells' edges point backwards


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    phi = 0.1 * rng.normal(size=(2, 12, 12))
    sub, _ = _sub(phi, 0.01)
    x0 = np.asarray(sub.flat0, dtype=float)
    J = sub.cons_jac(x0).toarray()
    e = 1e-6
    for k in rng.choice(x0.size, 6, replace=False):
        d = np.zeros_like(x0)
        d[k] = e
        fd = (sub.cons(x0 + d) - sub.cons(x0 - d)) / (2 * e)
        assert np.allclose(fd, J[:, k], atol=1e-6)


def test_rows_reject_a_non_dy_first_pack():
    phi = np.zeros((2, 12, 12))
    with pytest.raises(ValueError, match="DY_FIRST"):
        build_subproblem(
            JdetConstraint2D(shape=(12, 12)),
            phi,
            (3, 9, 3, 9),
            0.01,
            None,
            1e-3,
            orientation_delta=0.01,
        )


def test_default_off_is_byte_identical_in_the_subproblem():
    rng = np.random.default_rng(2)
    phi = rng.normal(size=(2, 12, 12))
    a, _ = _sub(phi, None)
    b = build_subproblem(SimplexConstraint2D(shape=(12, 12)), phi, (3, 9, 3, 9), 0.01, None, 1e-3)
    assert a.n_enforced == b.n_enforced and np.array_equal(a.cons(a.flat0), b.cons(b.flat0))


@needs_osqp
def test_strategy_forwards_orientation_delta(monkeypatch):
    from dvfopt.strategies.windowed import ISQPWindowedStrategy

    seen = {}

    def fake(phi, inner, **kw):
        seen.update(kw)
        return np.asarray(phi, dtype=float).copy(), engine.SliceReport()

    monkeypatch.setattr("dvfopt.strategies.windowed.windowed_correct", fake)
    s = ISQPWindowedStrategy(orientation_delta=0.02)
    s.solve(
        np.zeros((2, 16, 16)),
        constraint=SimplexConstraint2D(shape=(16, 16)),
        objective=NoneObjective(),
        threshold=0.01,
    )
    assert seen["orientation_delta"] == 0.02
    assert ISQPWindowedStrategy().orientation_delta is None


def test_edges_kind_drops_the_anti_diagonal_rows():
    phi = np.zeros((2, 12, 12))
    c = SimplexConstraint2D(shape=(12, 12))
    base = build_subproblem(c, phi, (3, 9, 3, 9), 0.01, None, 1e-3).n_enforced
    full = build_subproblem(
        c, phi, (3, 9, 3, 9), 0.01, None, 1e-3, orientation_delta=0.01
    ).n_enforced
    edges = build_subproblem(
        c, phi, (3, 9, 3, 9), 0.01, None, 1e-3, orientation_delta=0.01, orientation_rows="edges"
    ).n_enforced
    assert base < edges < full  # edge rows only: fewer rows, still some
    assert (full - edges) % 2 == 0 and (full - edges) > 0  # two anti-diagonal rows per cell dropped
