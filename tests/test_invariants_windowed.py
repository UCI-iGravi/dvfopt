"""Invariant guards for the windowed engine's certification claim.

The campaign's promise is not a number, it is a set of invariants: the corrected
field has 0 folds at EVERY gauge the engine speaks (simplex, bilinear, finite),
damage is 0 by construction, and a fold-free input passes through untouched.
These tests assert the invariants themselves over seeded random fields, sizes,
shapes and objectives — scenario-independent, so a future change that breaks the
claim fails here even if every historical scenario still passes.
"""

import numpy as np
import pytest

from dvfopt.constraints import (
    FiniteJdetConstraint2D,
    SimplexConstraint2D,
    SimplexConstraint2DBilinear,
)
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import min_field, windowed_correct
from dvfopt.objectives import L2Objective, NoneObjective

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")

THR = 0.01


def _random_folded(H, W, seed, scale=0.55):
    """Seeded random displacement field; asserts it actually contains folds."""
    rng = np.random.default_rng(seed)
    phi = np.stack([rng.normal(0, scale, (H, W)), rng.normal(0, scale, (H, W))])
    return phi


def _assert_certified(phi_in, out, rep):
    H, W = phi_in.shape[1:]
    for c in (
        SimplexConstraint2D(shape=(H, W)),
        SimplexConstraint2DBilinear(shape=(H, W)),
        FiniteJdetConstraint2D(shape=(H, W)),
    ):
        m = min_field(c, out)
        assert (m >= THR).all(), f"{type(c).__name__}: min {m.min()}"
    assert rep.damage == 0
    assert np.isfinite(out).all()


@needs_osqp
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("objective", [L2Objective, NoneObjective])
def test_random_fields_certify_at_every_gauge(seed, objective):
    phi = _random_folded(48, 48, seed)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).any(), "fixture must contain folds"
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=objective(), threshold=THR, verbose=0
    )
    _assert_certified(phi, out, rep)


@needs_osqp
def test_non_square_and_border_folds_certify():
    """A non-square field whose folds touch the image border (the border guard
    keeps border folds free instead of insetting them)."""
    phi = _random_folded(40, 88, seed=7)
    phi[:, :3, :] *= 2.0  # push the top border into deeper folds
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    assert (min_field(c, phi)[:3] < THR).any()
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=L2Objective(), threshold=THR, verbose=0
    )
    _assert_certified(phi, out, rep)


@needs_osqp
def test_fold_free_input_is_returned_byte_identical():
    phi = np.zeros((2, 32, 32))
    phi[0] += 0.05  # a benign uniform shift, no folds anywhere
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=L2Objective(), threshold=THR, verbose=0
    )
    assert np.array_equal(out, phi)
    assert rep.n_windows == 0 and rep.damage == 0


@needs_osqp
def test_nonstandard_threshold_certifies_at_that_threshold():
    phi = _random_folded(48, 48, seed=11)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    thr = 0.03
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=L2Objective(), threshold=thr, verbose=0
    )
    assert (min_field(c, out) >= thr).all()
    assert rep.damage == 0


@needs_osqp
def test_untouched_pixels_are_untouched():
    """Pixels outside every enforced footprint must be bit-identical to the input
    (the no-damage invariant stated directly, not via the damage counter)."""
    phi = np.zeros((2, 64, 64))
    from dvfopt.testdata import make_random_dvf

    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    phi[:, 24 : 24 + patch.shape[1], 26 : 26 + patch.shape[2]] = patch
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    out, _rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=L2Objective(), threshold=THR, verbose=0
    )
    changed = np.any(out != phi, axis=0)
    ys, xs = np.nonzero(changed)
    # every changed pixel is within a window's reach of the original fold patch
    assert ys.size == 0 or (ys.min() > 5 and ys.max() < 60 and xs.min() > 5 and xs.max() < 62)
    # and the far corner is literally untouched
    assert np.array_equal(out[:, :10, 45:], phi[:, :10, 45:])
