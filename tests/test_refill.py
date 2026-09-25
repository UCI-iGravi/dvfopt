"""dvfopt.dvf.refill.harmonic_refill against direct Laplace solves."""

import sys

import numpy as np
import pytest

from dvfopt.dvf.pins import detect_pins
from dvfopt.dvf.refill import harmonic_refill
from dvfopt.laplacian.solver import solveLaplacianFromCorrespondences

SHAPE = (6, 20, 20)


def _pins(n=30, seed=0):
    rng = np.random.default_rng(seed)
    tgt = np.column_stack([rng.integers(1, s - 1, n) for s in SHAPE]).astype(float)
    tgt = np.unique(tgt, axis=0)
    src = tgt.copy()
    src[:, 1:] += rng.normal(0, 0.5, (len(tgt), 2))
    return src, tgt


def _solve(src, tgt):
    return solveLaplacianFromCorrespondences(SHAPE, src, tgt, rtol=1e-10, maxiter=20000)


@pytest.mark.parametrize('amg', [True, False])
def test_refill_with_every_pin_kept_reproduces_the_field(monkeypatch, amg):
    if not amg:
        monkeypatch.setitem(sys.modules, 'pyamg', None)  # force the Jacobi fallback
    src, tgt = _pins()
    phi = _solve(src, tgt)
    pins = detect_pins(phi)
    out = harmonic_refill(phi, pins, rtol=1e-10)
    np.testing.assert_allclose(out, phi, atol=1e-6)


def test_refill_without_a_contradictory_pin_matches_a_direct_solve():
    src, tgt = _pins(seed=1)
    src[0, 1:] += 25.0  # a pin that contradicts the smooth rest
    phi = _solve(src, tgt)
    dirichlet = detect_pins(phi)
    bad = tuple(tgt[0].astype(int))
    assert dirichlet[bad]
    dirichlet[bad] = False
    out = harmonic_refill(phi, dirichlet, rtol=1e-10)
    np.testing.assert_allclose(out, _solve(src[1:], tgt[1:]), atol=1e-6)
    assert np.abs(out[1:, bad[0], bad[1], bad[2]]).max() < 5.0  # the 25 px spike is gone


def test_refill_2d_and_bad_input():
    rng = np.random.default_rng(2)
    phi2 = rng.normal(0, 1, (2, 12, 12))
    keep = np.zeros((12, 12), bool)
    keep[0], keep[-1], keep[:, 0], keep[:, -1] = True, True, True, True
    out = harmonic_refill(phi2, keep, rtol=1e-10)
    np.testing.assert_array_equal(out[:, keep], phi2[:, keep])
    inner = out[:, 1:-1, 1:-1]  # harmonic: each interior value = mean of its 4 neighbours
    nb = (out[:, :-2, 1:-1] + out[:, 2:, 1:-1] + out[:, 1:-1, :-2] + out[:, 1:-1, 2:]) / 4
    np.testing.assert_allclose(inner, nb, atol=1e-6)
    with pytest.raises(ValueError, match='empty'):
        harmonic_refill(phi2, np.zeros((12, 12), bool))
