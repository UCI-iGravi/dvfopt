"""``giant_workers``: RAS (Jacobi) giant-tile sweeps on the shared spawn pool."""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2DBilinear
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import windowed_correct
from dvfopt.objectives import NoneObjective
from dvfopt.testdata import make_random_dvf

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")


def test_ras_cores_partition_the_inset_region():
    """Tiler-generated tiles: the step-grid cores are pairwise disjoint, cover the
    inset region exactly, and each core lies inside its own tile."""
    it0, it1, ix0, ix1, tile, step = 0, 115, 0, 115, 64, 51
    tiles = [
        (ty, min(ty + tile, it1), tx, min(tx + tile, ix1))
        for ty in range(it0, it1, step)
        for tx in range(ix0, ix1, step)
    ]
    cores = engine._ras_cores(tiles, step, (it0, it1, ix0, ix1))
    seen = np.zeros((it1, ix1), dtype=bool)
    for t, c in zip(tiles, cores):
        cy0, cy1, cx0, cx1 = c
        assert t[0] <= cy0 and cy1 <= t[1] and t[2] <= cx0 and cx1 <= t[3]
        assert not seen[cy0:cy1, cx0:cx1].any()
        seen[cy0:cy1, cx0:cx1] = True
    assert seen.all()


def _giant_field(H=120, W=120):
    """One CONNECTED fold region bigger than max_window_area (3000 px) so the giant
    tiler runs: the 10x10 random fold patch tiled contiguously over a 60x60 block
    (free bbox ~66x66 = 4356 px after margin)."""
    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    phi = np.zeros((2, H, W))
    for by in range(6):
        for bx in range(6):
            y, x = 25 + by * 10, 25 + bx * 10
            phi[:, y : y + patch.shape[1], x : x + patch.shape[2]] = patch
    return phi


@needs_osqp
def test_giant_workers_off_is_byte_identical():
    phi = _giant_field()
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    kw = dict(
        constraint=c, objective=NoneObjective(), threshold=0.01, verbose=0, coarse_to_fine=False
    )
    out_a, rep_a = windowed_correct(phi.copy(), "isqp", **kw)
    out_b, rep_b = windowed_correct(phi.copy(), "isqp", giant_workers=0, **kw)
    assert np.array_equal(out_a, out_b)
    assert rep_a.giant_regions >= 1  # the fixture really exercises the tiler


@needs_osqp
def test_giant_workers_ras_reaches_zero_folds_damage_zero():
    """The RAS sweep must clear the giant region with 0 folds and damage 0; the
    trajectory may differ from the serial multiplicative sweep."""
    phi = _giant_field()
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=0.01,
        verbose=0,
        coarse_to_fine=False,
        giant_workers=2,
    )
    assert rep.giant_regions >= 1
    assert rep.folds_after == 0 and rep.damage == 0


@needs_osqp
def test_strategy_forwards_giant_workers():
    from dvfopt import ISQPWindowedStrategy

    assert engine._InnerOpts().giant_workers == 0
    assert ISQPWindowedStrategy(giant_workers=3).giant_workers == 3
