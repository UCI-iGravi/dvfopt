"""dvfopt.dvf.pins: pin read-back, pairwise test, cover, tau rule."""

import numpy as np

from dvfopt.dvf.pins import (
    auto_tau,
    detect_pins,
    greedy_cover,
    inconsistent_pins,
    violating_pairs,
    violating_pairs_pruned,
)
from dvfopt.laplacian.solver import solveLaplacianFromCorrespondences


def test_planted_pins_read_back_and_contradiction_dropped():
    rng = np.random.default_rng(0)
    shape = (8, 24, 24)
    tgt = np.column_stack([rng.integers(1, s - 1, 40) for s in shape]).astype(float)
    src = tgt.copy()
    src[:, 1:] += rng.normal(0, 0.3, (40, 2))
    src[0, 1:] = src[1, 1:] + 30.0  # one pin that contradicts its neighbour
    tgt[1] = tgt[0] + [0, 1, 0]
    phi = solveLaplacianFromCorrespondences(shape, src, tgt, rtol=1e-8, maxiter=5000)
    truth = np.zeros(shape, bool)
    truth[tuple(np.round(tgt).astype(int).T)] = True
    found = detect_pins(phi)
    assert (found == truth).all(), (found.sum(), truth.sum(), (found & ~truth).sum())
    coords, drop, _bad = inconsistent_pins(phi, found)
    dropped = {tuple(c) for c in coords[drop]}
    assert tuple(np.round(tgt[0]).astype(int)) in dropped and len(dropped) <= 2, dropped


def test_pruned_pairs_equal_full_search():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 40, (400, 3))
    d = rng.normal(0, 3, (400, 2))
    full = violating_pairs(pts, d, 1.0, 15.0)
    pruned = violating_pairs_pruned(pts, d, 1.0, 15.0)
    assert len(full) > 0
    as_set = lambda a: {tuple(r) for r in a.tolist()}  # noqa: E731
    assert as_set(full) == as_set(pruned)


def test_greedy_cover_is_a_vertex_cover():
    rng = np.random.default_rng(2)
    pairs = rng.integers(0, 50, (200, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    drop = greedy_cover(50, pairs)
    assert (drop[pairs[:, 0]] | drop[pairs[:, 1]]).all()
    assert drop.sum() < 50
    assert not greedy_cover(5, np.zeros((0, 2), np.int64)).any()


def test_auto_tau_floor_and_scaling():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, (3, 6, 16, 16))
    assert auto_tau(1e-3 * noise) == 0.7
    assert auto_tau(10.0 * noise) > 0.7


def test_pruned_pairs_rejects_nonpositive_c():
    import pytest

    from dvfopt.dvf.pins import violating_pairs_pruned

    pts = np.zeros((3, 3))
    pts[:, 2] = [0.0, 1.0, 2.0]
    d = np.zeros((3, 2))
    with pytest.raises(ValueError):
        violating_pairs_pruned(pts, d, 0.0, 60.0)
