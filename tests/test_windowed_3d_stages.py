"""Phase 2 of the 3D port: the engine's stages (coarse warm start, giant tiler, mop,
harmonic re-seed, re-anchor, polish) on (3, D, H, W) fields. Additive: the 2D stages
are covered by their own suites and by benchmarks/windowed_2d_identity.py.
"""

import itertools

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import min_field, pixel_fold_mask, windowed_correct
from dvfopt.objectives import L2Objective, NoneObjective

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")
THR = 0.01


def _blob(shape, centre, size, amp, seed=0):
    """Identity field with one random blob (a (3, *size) normal patch) at ``centre``."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((3, *shape))
    sl = tuple(slice(c - s // 2, c - s // 2 + s) for c, s in zip(centre, size))
    phi[(slice(None), *sl)] = rng.normal(0, amp, (3, *size))
    return phi


def test_restrict_3d_rescales_to_coarse_units_and_drops_partial_blocks():
    phi = np.zeros((3, 8, 9, 10))
    phi[0], phi[1], phi[2] = 4.0, -2.0, 1.0
    coarse = engine._restrict(phi, 2)
    assert coarse.shape == (3, 4, 4, 5)
    assert (
        np.allclose(coarse[0], 2.0) and np.allclose(coarse[1], -1.0) and np.allclose(coarse[2], 0.5)
    )


def test_restrict_2d_is_unchanged():
    phi = np.zeros((2, 8, 8))
    phi[0], phi[1] = 4.0, -2.0
    coarse = engine._restrict(phi, 2)
    assert (
        coarse.shape == (2, 4, 4) and np.allclose(coarse[0], 2.0) and np.allclose(coarse[1], -1.0)
    )
    assert engine._restrict(np.ones((2, 7, 9)), 2).shape == (2, 3, 4)


def test_prolongate_3d_rescales_back_and_zero_pads_odd_dims():
    delta = engine._prolongate(np.ones((3, 3, 4, 2)), (7, 9, 5), 2)
    assert delta.shape == (3, 7, 9, 5)
    assert np.allclose(delta[:, :6, :8, :4], 2.0)
    assert (
        np.all(delta[:, 6] == 0.0)
        and np.all(delta[:, :, 8] == 0.0)
        and np.all(delta[:, :, :, 4] == 0.0)
    )


def test_restrict_prolongate_round_trip_3d_is_identity_on_a_constant_field():
    phi = np.full((3, 8, 8, 8), 3.0)
    assert np.allclose(engine._prolongate(engine._restrict(phi, 2), (8, 8, 8), 2), phi)


def test_inner_opts_have_no_3d_twin():
    from dvfopt.core.windowed._common import _InnerOpts

    assert _InnerOpts().giant_tile == 64 and not hasattr(_InnerOpts(), "giant_tile_3d")


def _x_compression_64():
    """64^3 identity field with an x-compression steeper than 1 on a central block."""
    phi = np.zeros((3, 64, 64, 64))
    zz, yy, xx = np.ogrid[0:64, 0:64, 0:64]
    block = (zz >= 28) & (zz < 36) & (yy >= 28) & (yy < 36) & (xx >= 28) & (xx < 36)
    phi[2] = np.where(block, -1.3 * (xx - 32), 0.0)
    return phi


@needs_osqp
def test_coarse_warm_start_is_off_by_default_on_3d(monkeypatch):
    """Spike 2: on 3D the warm start INTRODUCES negative axial edge projections (12 at
    factor 4, 568 at factor 2) and took the 2.5D residual's densest box 493 -> 23,277
    folds, so the 3D column of ``DEFAULTS_BY_DIM`` is False. An explicit
    ``coarse_to_fine=True`` reads as the 2D default and resolves to False;
    ``dim_defaults=False`` is the escape.
    """
    calls = []

    def spy(phi, *a, **k):
        calls.append(True)
        return np.zeros_like(phi), engine.SliceReport(), []

    monkeypatch.setattr(engine, "_coarse_warm_start", spy)
    phi = _x_compression_64()
    kw = dict(
        constraint=SimplexConstraint3D(shape=phi.shape[1:]),
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        time_budget_s=1e-6,  # the coarse block runs before the budget bites; the rounds do not
        giant_tile=16,
    )
    windowed_correct(phi.copy(), "isqp", **kw)
    assert calls == []  # the 3D column is False
    windowed_correct(phi.copy(), "isqp", coarse_to_fine=True, **kw)
    assert calls == []  # an explicit True still reads as the 2D default
    windowed_correct(phi.copy(), "isqp", dim_defaults=False, coarse_to_fine=True, **kw)
    assert calls == [True]  # the escape takes the literal


@needs_osqp
def test_coarse_warm_start_runs_on_3d_and_keeps_healthy_area_byte_identical():
    # min(shape) >= 4 * giant_tile (= 64) -> the stage fires; coarse_factor 4 -> a 16^3 coarse
    # problem. A smooth x-compression steeper than 1 (dx = -1.3 * (x - 32) on a block) inverts the
    # map at EVERY scale, so the coarse field folds too and the prolongated delta / `allow` mask are
    # really exercised — a random blob averages away at factor 4 (coarse fold-free -> zero delta).
    # Block [26, 38) also works (coarse_folds_before 9, 0 folds, damage 0) but takes 465 s; [28, 36)
    # is the same behaviour in 47 s.
    # The stage is OFF in the 3D column of DEFAULTS_BY_DIM (spike 2), so this test takes the
    # `dim_defaults=False` escape and restates the 3D column's other knobs literally.
    phi = _x_compression_64()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).any()
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        dim_defaults=False,
        coarse_to_fine=True,
        giant_tile=16,
        mop_margin=6,
        max_window_area=8000,
    )
    assert rep.coarse_folds_before > 0 and rep.coarse_iters > 0  # the coarse solve did real work
    assert rep.folds_after == 0 and rep.damage == 0
    assert np.array_equal(out[:, :12], phi[:, :12]) and np.array_equal(
        out[:, :, :, 52:], phi[:, :, :, 52:]
    )


@needs_osqp
def test_strategy_forwards_the_3d_knobs(monkeypatch):
    from dvfopt import ISQPWindowedStrategy
    from dvfopt.strategies import windowed as strat

    seen = {}

    def fake(phi, inner, **kw):
        seen.update(kw)
        return np.asarray(phi, float), engine.SliceReport()

    monkeypatch.setattr(strat, "windowed_correct", fake)
    s = ISQPWindowedStrategy(giant_tile=12, mop_margin=4)
    s.solve(
        np.zeros((3, 6, 8, 8)),
        constraint=SimplexConstraint3D(shape=(6, 8, 8)),
        objective=NoneObjective(),
        threshold=THR,
    )
    assert seen["giant_tile"] == 12 and seen["mop_margin"] == 4


def test_fit_tile_nd_matches_the_2d_fit_and_clamps():
    assert engine._fit_tile_nd((125, 152), 64) == engine._fit_tile(125, 152, 64) == 51
    assert engine._fit_tile_nd((24, 24, 24), 16) == 12  # ceil(24/2), clamped to [12, 24]
    assert (
        engine._fit_tile_nd((10, 10, 10), 16) == 12
    )  # a region smaller than the target keeps a usable tile


def test_ras_cores_partition_a_3d_inset_region():
    inset = (0, 10, 0, 10, 0, 10)
    tiles = [
        (z, min(z + 6, 10), y, min(y + 6, 10), x, min(x + 6, 10))
        for z in range(0, 10, 4)
        for y in range(0, 10, 4)
        for x in range(0, 10, 4)
    ]
    cores = engine._ras_cores(tiles, 4, inset)
    cover = np.zeros((10, 10, 10), int)
    for core in cores:
        cover[engine._box_slices(core)] += 1
    assert (cover == 1).all()  # a partition: every voxel in exactly one core


@needs_osqp
def test_3d_giant_region_is_tiled_and_cleared():
    # a 10^3 blob -> one connected free box of ~16^3 = 4096 > max_window_area=800 -> the tiler;
    # _fit_tile_nd((16, 16, 16), 10) == 8, step 4 -> 4 tiles per axis of <= 8^3 voxels each.
    # (step = max(tile - overlap, tile // 2, 1) = max(8 - 4, 4, 1) = 4: the tile // 2 floor
    # ties here and only binds for tiles nearer the overlap, see the step-floor test below.)
    # NOTE: the smaller shared fixture (16^3 field, max_window_area=200, giant_tile=6)
    # was tried here first per the fix-wave's conditional — it clears (giant_regions >= 1,
    # folds_after == 0, damage == 0) but in 188.67s (not "well under 60s") and its far
    # planes are no longer untouched (the blob is too close to the field boundary at that
    # size), so this test keeps the original 22^3 fixture. It also made the RAS test below
    # impractically slow for an unrelated reason (see its own note), so no test here uses
    # the smaller fixture.
    phi = _blob((22, 22, 22), (11, 11, 11), (10, 10, 10), amp=0.4, seed=3)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).sum() > 100
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        max_window_area=800,
        giant_tile=10,
        mop_margin=0,
        reseed_rounds=0,
    )
    assert rep.giant_regions >= 1 and len(rep.giant_boxes[0]) == 6
    assert rep.n_windows > 1 and all(
        engine._box_size(w.patch_box) < engine._box_size(rep.giant_boxes[0]) for w in rep.windows
    )  # every solved window is a tile strictly inside the giant box
    assert rep.damage == 0 and rep.folds_after == 0
    # damage == 0 carries the inset invariant; the far planes are a sanity check
    assert np.array_equal(out[:, :2], phi[:, :2])


@needs_osqp
def test_3d_giant_workers_ras_reaches_zero_folds_damage_zero():
    # NOTE: the smaller shared fixture (16^3, max_window_area=200, giant_tile=6) tried
    # here per the fix-wave's conditional does not fit the way the brief assumed: fitted
    # tile is 5 (not 6), step 1 (not 2), so `giant_tile_fit` produces 2744 near-duplicate
    # overlapping tiles per sweep (vs 64 below) -- a combinatorial blow-up that made this
    # RAS test impractically slow. Keeping the original 22^3 fixture here too.
    phi = _blob((22, 22, 22), (11, 11, 11), (10, 10, 10), amp=0.4, seed=3)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        max_window_area=800,
        giant_tile=10,
        giant_workers=2,
        mop_margin=0,
        reseed_rounds=0,
    )
    assert rep.giant_regions >= 1 and rep.damage == 0 and rep.folds_after == 0


def test_harmonic_fill_3d_is_discrete_harmonic_and_keeps_the_boundary():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(3, 12, 12, 12))
    mask = np.zeros((12, 12, 12), bool)
    mask[3:9, 3:9, 3:9] = True
    ref = phi.copy()
    engine._harmonic_fill(phi, mask)
    assert np.array_equal(phi[:, ~mask], ref[:, ~mask])
    inner = mask.copy()
    inner[[3, 8], :, :] = inner[:, [3, 8], :] = inner[:, :, [3, 8]] = False
    lap = 6 * phi[:, 1:-1, 1:-1, 1:-1]
    for ax in (1, 2, 3):
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[ax - 1], hi[ax - 1] = slice(0, -2), slice(2, None)
        lap = lap - phi[(slice(None), *lo)] - phi[(slice(None), *hi)]
    assert np.abs(lap[:, inner[1:-1, 1:-1, 1:-1]]).max() < 1e-9


def test_harmonic_fill_2d_neighbour_order_is_unchanged():
    # the 2D neighbour order (x+1, x-1, y+1, y-1) decides the float accumulation of the
    # boundary sum; pin it against a hand-built matrix so the n-D loop cannot drift
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(2, 6, 6))
    mask = np.zeros((6, 6), bool)
    mask[2:4, 2:4] = True
    a = phi.copy()
    engine._harmonic_fill(a, mask)
    b = phi.copy()
    ys, xs = np.nonzero(mask)
    rhs = np.zeros((2, ys.size))
    for k, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if not mask[y + dy, x + dx]:
                rhs[:, k] += phi[:, y + dy, x + dx]
    # 4 interior pixels, each with 2 masked neighbours: solve 4 I - A directly
    A = np.zeros((4, 4))
    for k, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if mask[y + dy, x + dx]:
                j = int(np.flatnonzero((ys == y + dy) & (xs == x + dx))[0])
                A[k, j] = -1.0
        A[k, k] = 4.0
    for ch in range(2):
        b[ch, ys, xs] = np.linalg.solve(A, rhs[ch])
    assert np.allclose(a, b)


# The round loop's default 'hybrid' QP backend converges this blob's window fully
# regardless of how small `maxiter`/`fallback_maxiter` are (a single well-converged
# QP step already reaches feasibility), so a residual never survives to the mop/
# re-seed stages under the brief's original knobs. Capping the QP backend's own
# ADMM iterations (`qp_max_iter`/`qp_max_iter_fallback=2`) on plain `'osqp'` (no
# interior-point rung) forces genuinely inaccurate per-iteration steps so a real
# residual survives the round loop and exercises the ported n-D mop / re-seed code.
_FORCE_RESIDUAL = dict(qp_max_iter=2, qp_max_iter_fallback=2, qp_backend='osqp')


@needs_osqp
def test_3d_mop_fires_on_a_residual_and_never_damages():
    # a hard blob under a short round-loop budget leaves a residual the mop re-windows;
    # blob y 21..27 -> free box y 17..31 -> two grows reach y 8 -> mop boxes stay at
    # y >= 9 (measured): nothing below y = 6 is ever freed
    phi = _blob((14, 40, 30), (7, 24, 15), (4, 6, 6), amp=1.4, seed=0)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        maxiter=10,
        fallback_maxiter=10,
        max_rounds=1,
        reseed_rounds=0,
        **_FORCE_RESIDUAL,
    )
    assert rep.damage == 0 and np.isfinite(out).all()
    assert rep.mop_windows >= 1
    assert rep.mop_cleared >= 0  # the mop only helps
    assert np.array_equal(out[:, :, :6], phi[:, :, :6])


@needs_osqp
def test_3d_reseed_stage_runs_on_a_residual_and_books_touched():
    phi = _blob((14, 40, 30), (7, 24, 15), (4, 6, 6), amp=1.4, seed=0)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        maxiter=10,
        fallback_maxiter=10,
        max_rounds=1,
        mop_margin=0,
        **_FORCE_RESIDUAL,
    )
    assert rep.damage == 0
    assert rep.reseed_folds_before > 0  # the round loop really left the stage a residual
    assert rep.reseed_rounds_run >= 1 and rep.reseed_px > 0
    assert rep.reseed_folds_after <= rep.reseed_folds_before
    # the re-seed mask is the residual's corners dilated by 2, inside the touched region
    assert np.array_equal(out[:, :, :6], phi[:, :, :6])


@needs_osqp
@pytest.mark.parametrize("kind", ["l2", "l1"])
def test_3d_reanchor_reduces_the_move_and_keeps_zero_folds(kind):
    phi = _blob((8, 40, 20), (4, 20, 10), (4, 6, 6), amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    base, rep0 = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR, verbose=0
    )
    assert rep0.folds_after == 0
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        reanchor=kind,
        giant_tile=16,
        reanchor_maxiter=8,  # unit-test budget; tile 16 steps by 8 (the overlap is 8), so the moved region is ~8 tiles
        reanchor_sweeps=1,
    )
    assert rep.folds_after == 0 and rep.damage == 0
    assert rep.reanchor_sweeps_run >= 1 and rep.reanchor_tiles >= 1
    assert rep.reanchor_accepted > 0
    assert rep.reanchor_l2_after < rep.reanchor_l2_before
    moved0 = np.any(base != phi, axis=0)
    assert not np.any(out != phi, axis=0)[~moved0].any()  # only voxels the main solve moved


@needs_osqp
def test_3d_polish_runs_and_keeps_zero_folds():
    phi = _blob((8, 40, 20), (4, 20, 10), (4, 6, 6), amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        polish="l2",
    )
    assert rep.folds_after == 0 and rep.damage == 0
    assert rep.polish_windows >= 1 and rep.polish_accepted > 0


@needs_osqp
def test_3d_mop_margin_zero_still_disables_the_mop():
    phi = _blob((14, 40, 30), (7, 24, 15), (4, 6, 6), amp=1.4, seed=0)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    _out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        maxiter=10,
        fallback_maxiter=10,
        max_rounds=1,
        reseed_rounds=0,
        mop_margin=0,
        **_FORCE_RESIDUAL,
    )
    assert rep.damage == 0 and rep.mop_windows == 0


def test_tiler_and_reanchor_steps_never_collapse_for_small_3d_tiles():
    # tile 5 with overlap 4 used to step by 1 (2744 tiles on a 14^3 region); the floor is tile // 2
    assert (
        max(5 - 4, 5 // 2, 1) == 2
        and max(16 - 8, 16 // 2, 1) == 8
        and max(48 - 8, 48 // 2, 1) == 40
    )
    inset = (0, 14, 0, 14, 0, 14)
    tile, step = 5, max(5 - 4, 5 // 2, 1)
    n_tiles = len(
        list(itertools.product(*(range(inset[2 * a], inset[2 * a + 1], step) for a in range(3))))
    )
    assert n_tiles == 7**3  # not 14**3
    assert tile > step  # tiles still overlap


def test_giant_tiles_drop_a_border_strip_too_thin_to_window():
    # the 17^3 B0039 sub-volume: inset 17 per axis, fitted tile 12, step 8 -> starts 0, 8, 16;
    # the 16..17 strip's ring-padded patch clips to width 2 at the border (validate_dvf min 3)
    tiles = engine._giant_tiles((0, 17, 0, 17, 0, 17), 12, 8, (17, 17, 17), 1)
    assert len(tiles) == 2**3 and {t[0] for t in tiles} == {0, 8}
    cores = engine._ras_cores(tiles, 8, (0, 17, 0, 17, 0, 17))
    assert {c[:2] for c in cores} == {(0, 8), (8, 17)}  # the previous core absorbs the strip
    # the same inset one voxel wider keeps the strip: its padded patch is 3 wide
    tiles = engine._giant_tiles((0, 18, 0, 18, 0, 18), 12, 8, (18, 18, 18), 1)
    assert {t[0] for t in tiles} == {0, 8, 16}
    # interior region: a 1-voxel strip pads to 3 on both sides and is kept (2D shape, old behaviour)
    tiles = engine._giant_tiles((10, 137, 20, 100), 48, 42, (200, 200), 1)
    assert {t[0] for t in tiles} == {10, 52, 94, 136}
    cores = engine._ras_cores(tiles, 42, (10, 137, 20, 100))
    assert {c[:2] for c in cores} == {(10, 52), (52, 94), (94, 136), (136, 137)}
