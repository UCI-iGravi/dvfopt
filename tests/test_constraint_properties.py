"""Property-based tests for the :class:`Constraint` hierarchy.

The constraint adjoint (``J^T @ v``) is the single most safety-critical
piece of math in the package — every barrier/SLSQP/wallbreaker solver
calls it on every iteration. If the analytical adjoint disagrees with
the numerical finite-difference Jacobian, every solver above it silently
produces wrong results.

These tests use Hypothesis to randomize shape, RNG seed, and amplitude
across many calls per test, catching boundary cases (small grids, large
amplitudes, near-degenerate triangles) that fixed-seed tests can miss.
"""

from __future__ import annotations

import numpy as np
import pytest

# hypothesis lives in the ``dev`` extra — skip cleanly on lean installs
# (e.g. the ci.yml ``[fast]`` env), matching the PySide6/SimpleITK/torch
# optional-dependency convention used across the suite.
pytest.importorskip('hypothesis', reason='property tests need the dev extra (hypothesis)')

from hypothesis import given, settings
from hypothesis import strategies as st

from dvfopt.constraints import (
    JdetConstraint2D,
    JdetConstraint3D,
    TriConstraint2D,
    TriConstraint2DFullCoverage,
)


def _fd_jacobian_T_v(constraint, phi_flat, v, eps=1e-6):
    """Numerical adjoint: J^T @ v via central-difference column sweep."""
    n_v = constraint.n_variables
    out = np.zeros(n_v)
    for i in range(n_v):
        p = phi_flat.copy()
        p[i] += eps
        m = phi_flat.copy()
        m[i] -= eps
        dT = (constraint.values(p) - constraint.values(m)) / (2 * eps)
        out[i] = float(np.dot(dT, v))
    return out


# Hypothesis strategies for problem dimensions + amplitudes.
shapes_2d = st.tuples(st.integers(3, 7), st.integers(3, 7))
# All axes >= 3 to satisfy the input validator's minimum spatial size.
shapes_3d = st.tuples(st.integers(3, 4), st.integers(3, 5), st.integers(3, 5))
seeds = st.integers(0, 10000)
amps = st.floats(min_value=0.01, max_value=0.3, allow_nan=False)


@given(shape=shapes_2d, seed=seeds, amp=amps)
@settings(max_examples=30, deadline=None)
def test_tri_constraint_2d_adjoint_matches_fd(shape, seed, amp):
    H, W = shape
    rng = np.random.default_rng(seed)
    phi = np.stack([rng.normal(0, amp, (H, W)), rng.normal(0, amp, (H, W))])
    c = TriConstraint2D((H, W))
    flat = c.flatten(phi)
    v = rng.normal(size=c.n_constraints)
    ana = c.adjoint(flat, v)
    num = _fd_jacobian_T_v(c, flat, v)
    err = float(np.abs(ana - num).max())
    # FD truncation error is ~eps; analytical should match to <1e-6.
    assert err < 1e-6, f'shape={shape} seed={seed} amp={amp}: err={err:.2e}'


@given(shape=shapes_2d, seed=seeds, amp=amps)
@settings(max_examples=20, deadline=None)
def test_tri_constraint_2d_full_coverage_adjoint(shape, seed, amp):
    H, W = shape
    rng = np.random.default_rng(seed)
    phi = np.stack([rng.normal(0, amp, (H, W)), rng.normal(0, amp, (H, W))])
    c = TriConstraint2DFullCoverage((H, W))
    flat = c.flatten(phi)
    v = rng.normal(size=c.n_constraints)
    err = float(np.abs(c.adjoint(flat, v) - _fd_jacobian_T_v(c, flat, v)).max())
    assert err < 1e-6, f'shape={shape}: err={err:.2e}'


@given(shape=shapes_2d, seed=seeds, amp=amps)
@settings(max_examples=20, deadline=None)
def test_jdet_constraint_2d_adjoint(shape, seed, amp):
    H, W = shape
    rng = np.random.default_rng(seed)
    phi = np.stack([rng.normal(0, amp, (H, W)), rng.normal(0, amp, (H, W))])
    c = JdetConstraint2D((H, W))
    flat = c.flatten(phi)
    v = rng.normal(size=c.n_constraints)
    err = float(np.abs(c.adjoint(flat, v) - _fd_jacobian_T_v(c, flat, v)).max())
    assert err < 1e-6, f'shape={shape}: err={err:.2e}'


@given(shape=shapes_3d, seed=seeds, amp=st.floats(min_value=0.01, max_value=0.15))
@settings(max_examples=10, deadline=None)
def test_jdet_constraint_3d_adjoint(shape, seed, amp):
    D, H, W = shape
    rng = np.random.default_rng(seed)
    phi = np.stack(
        [
            rng.normal(0, amp, (D, H, W)),
            rng.normal(0, amp, (D, H, W)),
            rng.normal(0, amp, (D, H, W)),
        ]
    )
    c = JdetConstraint3D((D, H, W))
    flat = c.flatten(phi)
    v = rng.normal(size=c.n_constraints)
    err = float(np.abs(c.adjoint(flat, v) - _fd_jacobian_T_v(c, flat, v)).max())
    assert err < 1e-6, f'shape={shape}: err={err:.2e}'


@given(shape=shapes_2d, seed=seeds, amp=amps)
@settings(max_examples=20, deadline=None)
def test_tri_constraint_2d_sparse_jacobian_matches_dense(shape, seed, amp):
    """The sparse forward Jacobian (used by SLSQP) should be the dense
    transpose of the adjoint."""
    H, W = shape
    rng = np.random.default_rng(seed)
    phi = np.stack([rng.normal(0, amp, (H, W)), rng.normal(0, amp, (H, W))])
    c = TriConstraint2D((H, W))
    flat = c.flatten(phi)
    J_sparse = c.jacobian(flat).toarray()
    # Build dense Jacobian via FD column sweep — should match within FD tol.
    eps = 1e-6
    J_num = np.zeros((c.n_constraints, c.n_variables))
    for i in range(c.n_variables):
        p = flat.copy()
        p[i] += eps
        m = flat.copy()
        m[i] -= eps
        J_num[:, i] = (c.values(p) - c.values(m)) / (2 * eps)
    err = float(np.abs(J_sparse - J_num).max())
    assert err < 1e-6, f'shape={shape}: err={err:.2e}'


@pytest.mark.parametrize(
    'cls,shape',
    [
        (TriConstraint2D, (5, 6)),
        (TriConstraint2DFullCoverage, (5, 6)),
        (JdetConstraint2D, (5, 6)),
        (JdetConstraint3D, (3, 4, 5)),
    ],
)
def test_round_trip_flatten_unflatten(cls, shape):
    """``unflatten(flatten(phi))`` should be a no-op on the canonical form."""
    rng = np.random.default_rng(0)
    if cls is JdetConstraint3D:
        D, H, W = shape
        phi = np.stack(
            [
                rng.normal(scale=0.1, size=(D, H, W)),
                rng.normal(scale=0.1, size=(D, H, W)),
                rng.normal(scale=0.1, size=(D, H, W)),
            ]
        )
    else:
        H, W = shape
        phi = np.stack([rng.normal(scale=0.1, size=(H, W)), rng.normal(scale=0.1, size=(H, W))])
    c = cls(shape)
    np.testing.assert_array_equal(c.unflatten(c.flatten(phi)), phi)


def test_tri_constraint_accepts_3channel_input():
    """TriConstraint2D.coerce should accept (3, 1, H, W) and (3, H, W)."""
    rng = np.random.default_rng(0)
    H, W = 6, 7
    phi_2 = np.stack([rng.normal(scale=0.1, size=(H, W)), rng.normal(scale=0.1, size=(H, W))])
    phi_3 = np.stack([np.zeros((H, W)), phi_2[0], phi_2[1]])
    phi_31hw = phi_3[:, None, :, :]
    c = TriConstraint2D((H, W))
    f2 = c.flatten(phi_2)
    f3 = c.flatten(phi_3)
    f31hw = c.flatten(phi_31hw)
    np.testing.assert_array_equal(f2, f3)
    np.testing.assert_array_equal(f2, f31hw)


def test_jdet_constraint_2d_accepts_3channel_input():
    rng = np.random.default_rng(0)
    H, W = 6, 7
    phi_2 = np.stack([rng.normal(scale=0.1, size=(H, W)), rng.normal(scale=0.1, size=(H, W))])
    phi_3 = np.stack([np.zeros((H, W)), phi_2[0], phi_2[1]])
    c = JdetConstraint2D((H, W))
    np.testing.assert_array_equal(c.flatten(phi_2), c.flatten(phi_3))
