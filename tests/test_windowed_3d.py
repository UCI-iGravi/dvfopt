"""3D (SimplexConstraint3D) path of the windowed engine — the 3D port (phases 1-2).

Every test here is additive: the 2D families are covered by the existing
windowed suites and by benchmarks/windowed_2d_identity.py (byte-identity).
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import (
    LOCALITY,
    build_subproblem,
    find_windows,
    min_field,
    windowed_correct,
)
from dvfopt.objectives import L2Objective, NoneObjective

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")

THR = 0.01


def test_box_helpers_are_dimension_agnostic():
    from dvfopt.core.windowed._common import _box_size, _box_slices, _pad_box

    assert _box_slices((1, 4, 2, 6)) == (slice(1, 4), slice(2, 6))
    assert _box_slices((0, 2, 1, 4, 2, 6)) == (slice(0, 2), slice(1, 4), slice(2, 6))
    assert _box_size((1, 4, 2, 6)) == 12
    assert _box_size((0, 2, 1, 4, 2, 6)) == 24
    assert _pad_box((1, 4, 2, 6), (5, 8), 2) == (0, 5, 0, 8)
    assert _pad_box((2, 4, 3, 5, 4, 6), (10, 10, 10), 1) == (1, 5, 2, 6, 3, 7)


def test_find_windows_3d_boxes_and_border_rule():
    mask = np.zeros((10, 12, 14), bool)
    mask[4, 5, 6] = True  # interior fold
    mask[0, 1, 1] = True  # fold on the z=0 face (and near the y/x=0 faces)
    boxes = sorted(find_windows(mask, margin=2, ring=1))
    # dilation by margin+ring=3 is an L1 ball; the bbox is inset by ring except on a
    # side that reached the volume border, where the fold must stay free
    assert boxes == [(0, 3, 0, 4, 0, 4), (2, 7, 3, 8, 4, 9)]


def test_find_windows_2d_is_unchanged():
    mask = np.zeros((20, 20), bool)
    mask[10, 10] = True
    assert find_windows(mask, margin=3, ring=1) == [(7, 14, 7, 14)]


def test_locality_registers_tet3d_with_ring_one():
    assert LOCALITY[SimplexConstraint3D].ring == 1


def test_min_field_tet3d_matches_values_and_pads_inf():
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.3, (3, 5, 6, 7))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    m = min_field(c, phi)
    assert m.shape == (5, 6, 7)
    ref = np.asarray(c.values(c.flatten(phi))).reshape(6, 4, 5, 6).min(0)
    np.testing.assert_allclose(m[:4, :5, :6], ref, atol=1e-12)
    assert np.isinf(m[4]).all() and np.isinf(m[:, 5]).all() and np.isinf(m[:, :, 6]).all()


def test_influenced_tet3d_is_the_eight_corner_rule():
    c = SimplexConstraint3D(shape=(5, 5, 5))
    fm = np.zeros((5, 5, 5), bool)
    fm[2, 2, 2] = True
    idx, _jac_of = LOCALITY[SimplexConstraint3D].influenced(c, fm, 5, 5, 5, (False,) * 6)
    m = 4 * 4 * 4
    expect = sorted((z * 4 + y) * 4 + x for z in (1, 2) for y in (1, 2) for x in (1, 2))
    assert sorted(set(int(i) % m for i in idx)) == expect  # the 8 cubes cornered by the voxel
    assert sorted(idx.tolist()) == sorted(b * m + cf for b in range(6) for cf in expect)  # x 6 tets


def test_tet3d_patch_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    phi = rng.normal(0, 0.2, (3, 4, 4, 5))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    f0 = np.asarray(c.flatten(phi), dtype=np.float64)
    fm = np.ones(phi.shape[1:], bool)
    _idx, jac_of = LOCALITY[SimplexConstraint3D].influenced(c, fm, 4, 4, 5, (True,) * 6)
    J = jac_of(f0)
    assert J.shape == (c.n_constraints, c.n_variables)
    Jd = J.toarray()
    eps = 1e-6
    for k in rng.choice(f0.size, 12, replace=False):
        e = np.zeros_like(f0)
        e[k] = eps
        fd = (np.asarray(c.values(f0 + e)) - np.asarray(c.values(f0 - e))) / (2 * eps)
        np.testing.assert_allclose(Jd[:, k], fd, atol=1e-6)


def test_build_subproblem_3d_geometry_and_patch_identity():
    from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

    rng = np.random.default_rng(2)
    phi = rng.normal(0, 0.25, (3, 9, 10, 11))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    box = (2, 5, 3, 7, 4, 8)
    sub = build_subproblem(c, phi, box, THR, NoneObjective(), 1e-3)
    assert sub.patch_box == (1, 6, 2, 8, 3, 9)  # ring 1 on every side (interior box)
    assert sub.free_mask.shape == (5, 6, 6)
    assert sub.free_mask.sum() == 3 * 4 * 4 and sub.free_mask[1:4, 1:5, 1:5].all()
    n = sub.free_mask.size
    assert sub.free_idx.size == 3 * 48  # every free voxel in each of the dx / dy / dz blocks
    assert set((sub.free_idx // n).tolist()) == {0, 1, 2}
    # every cube of this patch has a free corner -> all 6 * cells rows are enforced, and
    # their values at flat0 are the GLOBAL tet volumes (exact cells: no interior-cut mismatch)
    m = 4 * 5 * 5
    assert sub.n_enforced == 6 * m
    vals = (sub.cons(sub.flat0) + (THR + 1e-3)).reshape(6, 4, 5, 5)
    np.testing.assert_allclose(vals, six_tet_volumes_3d(phi)[:, 1:5, 2:7, 3:8], atol=1e-12)
    J = sub.cons_jac(sub.flat0)
    assert J.shape == (sub.n_enforced, sub.flat0.size)


def test_build_subproblem_3d_free_extra_and_volume_border():
    phi = np.zeros((3, 9, 10, 11))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    fe = np.zeros(phi.shape[1:], bool)
    fe[3, 5, 6] = True
    sub = build_subproblem(c, phi, (2, 5, 3, 7, 4, 8), THR, None, 1e-3, free_extra=fe)
    assert sub.free_idx.size == 3 and sub.n_enforced == 6 * 8  # one voxel: its 8 cubes
    sub = build_subproblem(c, phi, (0, 3, 3, 7, 4, 8), THR, None, 1e-3)
    assert sub.patch_box == (0, 4, 2, 8, 3, 9)  # no ring past the z=0 face
    assert sub.free_mask[0:3, 1:5, 1:5].all() and sub.free_mask.sum() == 3 * 4 * 4


def test_rows_3d_identity_equals_one_minus_delta_and_counts_every_edge():
    from dvfopt.core.windowed._common import _orientation_rows_3d

    c = SimplexConstraint3D(shape=(4, 4, 4))
    a, b = _orientation_rows_3d(c, np.ones((4, 4, 4), bool), 0.01)
    assert a.shape == (3 * 48, 3 * 64)  # 48 edges per axis on a 4^3 grid
    np.testing.assert_allclose(a @ np.zeros(3 * 64) + b, 0.99)


def test_rows_3d_keep_free_to_frozen_edges_and_drop_frozen_frozen():
    from dvfopt.core.slsqp_windowed.constraints3d import _injectivity_linear_constraint_3d
    from dvfopt.core.windowed._common import _orientation_rows_3d

    c = SimplexConstraint3D(shape=(3, 3, 3))
    fm = np.zeros((3, 3, 3), bool)
    fm[1, 1, 1] = True
    a, _b = _orientation_rows_3d(c, fm, 0.01)
    assert a.shape[0] == 6  # the centre voxel's six axial edges, each free-to-frozen
    n, centre = 27, 13
    for r in range(6):
        assert {centre, n + centre, 2 * n + centre} & set(a[r].indices.tolist())
    # the 3D injectivity helper's own filter (BOTH endpoints free) would keep none of them
    assert _injectivity_linear_constraint_3d((3, 3, 3), 0.01, freeze_mask=~fm) is None


def test_rows_3d_rotated_cube_violates_and_compressed_cube_does_not():
    from dvfopt.core.windowed._common import _orientation_rows_3d

    c = SimplexConstraint3D(shape=(3, 3, 3))  # validate_dvf's minimum spatial size is 3
    a, b = _orientation_rows_3d(c, np.ones((3, 3, 3), bool), 0.01)
    phi = np.zeros((3, 3, 3, 3))
    phi[2, :, :, 0], phi[2, :, :, 1] = 1.0, -1.0  # the x-edge 0 -> 1 flips: deformed x goes 1 -> 0
    assert (a @ np.asarray(c.flatten(phi)) + b).min() < 0
    phi = np.zeros((3, 3, 3, 3))
    for j in range(3):  # 50 % compression on every axis: every edge projection is +0.5
        phi[2, :, :, j] = phi[1, :, j, :] = phi[0, j] = -0.5 * j
    assert (a @ np.asarray(c.flatten(phi)) + b).min() > 0


def test_subproblem_3d_with_rows_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    phi = rng.normal(0, 0.2, (3, 5, 5, 6))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    sub = build_subproblem(
        c,
        phi,
        (1, 4, 1, 4, 1, 5),
        THR,
        NoneObjective(),
        1e-3,
        orientation_delta=0.01,
        orientation_rows='edges',
    )
    assert sub.n_enforced > 6 * 4 * 4 * 5  # tet rows plus the edge rows
    J = sub.cons_jac(sub.flat0).toarray()
    assert J.shape == (sub.n_enforced, sub.flat0.size)
    for k in rng.choice(sub.free_idx, 10, replace=False):
        e = np.zeros_like(sub.flat0)
        e[k] = 1e-6
        fd = (sub.cons(sub.flat0 + e) - sub.cons(sub.flat0 - e)) / 2e-6
        np.testing.assert_allclose(J[:, k], fd, atol=1e-6)


def test_subproblem_3d_rejects_the_full_rows_kind():
    c = SimplexConstraint3D(shape=(4, 4, 4))
    with pytest.raises(ValueError, match='edges'):
        build_subproblem(
            c,
            np.zeros((3, 4, 4, 4)),
            (1, 3, 1, 3, 1, 3),
            THR,
            None,
            1e-3,
            orientation_delta=0.01,
            orientation_rows='full',
        )


def _planted_3d(shape=(8, 40, 20), seed=0, amp=0.5, size=(4, 6, 6)):
    """Identity field with one random blob of folds in the interior (default: a mild
    blob whose free box, y 13..26 after margin 3 + ring 1, stays above y = 5 even after
    two grow-on-failure steps of 4 — so y < 5 is outside every window)."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((3, *shape))
    z, y, x = (s // 2 for s in shape)
    dz, dy, dx = size
    phi[:, z - dz // 2 : z + dz // 2, y - dy // 2 : y + dy // 2, x - dx // 2 : x + dx // 2] = (
        rng.normal(0, amp, (3, *size))
    )
    return phi


@needs_osqp
@pytest.mark.parametrize("objective", [NoneObjective, L2Objective])
def test_3d_planted_folds_no_damage_and_untouched_voxels_bit_identical(objective):
    phi = _planted_3d()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).any(), "fixture must contain folds"
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=objective(), threshold=THR, verbose=0
    )
    assert rep.damage == 0 and np.isfinite(out).all()
    assert rep.folds_after < rep.folds_before
    assert rep.n_windows >= 1 and len(rep.windows[0].patch_box) == 6
    lo = min(w.patch_box[2] for w in rep.windows)  # lowest y any window's patch reached
    assert lo >= 1
    assert np.array_equal(out[:, :, :lo], phi[:, :, :lo])
    # 3D certificate fields are filled; the 2D ones stay at their defaults
    assert rep.best_diag_floor_after >= 0 and rep.best_diag_floor_after_zero >= 0
    assert rep.folds_after_zero >= 0 and rep.best_diag_floor_after <= rep.folds_after
    assert rep.coarse_folds_before == -1  # too small for the coarse stage (min(shape) < 64)


@needs_osqp
def test_3d_hard_blob_no_damage_under_a_short_budget():
    phi = _planted_3d(amp=1.4)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).any(), "fixture must contain folds"
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        maxiter=40,
        fallback_maxiter=40,
    )
    assert rep.damage == 0 and np.isfinite(out).all()
    lo = min(w.patch_box[2] for w in rep.windows)  # lowest y any window's patch reached
    assert lo >= 1
    assert np.array_equal(out[:, :, :lo], phi[:, :, :lo])


@needs_osqp
def test_3d_mild_blob_reaches_zero_folds_on_tr():
    phi = _planted_3d(amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).any(), "fixture must contain folds"
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR, verbose=0
    )
    assert rep.folds_after == 0 and rep.damage == 0
    assert (min_field(c, out) >= THR).all()
    assert rep.folds_after_zero == 0 and rep.best_diag_floor_after == 0


@needs_osqp
def test_3d_fold_free_input_is_returned_byte_identical():
    phi = np.zeros((3, 6, 8, 8))
    phi[2] = 0.1
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(phi.copy(), "isqp", constraint=c, threshold=THR, verbose=0)
    assert np.array_equal(out, phi) and rep.n_windows == 0 and rep.damage == 0
    assert rep.folds_after_zero == 0 and rep.best_diag_floor_after == 0


@needs_osqp
def test_3d_exact_ls_default_degrades_to_tr(monkeypatch):
    import dvfopt.core.windowed._common as cm

    seen = []
    orig = cm.solve_window_inner

    def spy(sub, inner, maxiter, **kw):
        seen.append(kw.get("step_rule"))
        return orig(sub, inner, maxiter, **kw)

    monkeypatch.setattr(cm, "solve_window_inner", spy)
    phi = _planted_3d((6, 10, 10), amp=1.0)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    windowed_correct(phi, "isqp", constraint=c, threshold=THR, verbose=0, maxiter=3)  # 'exact_ls'
    assert seen and set(seen) == {"tr"}


def test_3d_refuses_the_full_rows_kind():
    phi = np.zeros((3, 6, 8, 8))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    with pytest.raises(ValueError, match="edges"):
        windowed_correct(phi, "isqp", constraint=c, threshold=THR, orientation_rows="full")


@needs_osqp
def test_3d_over_cap_region_is_tiled_and_counted():
    phi = _planted_3d(amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        max_window_area=100,
    )
    assert rep.giant_regions >= 1 and rep.damage == 0
    assert rep.folds_after == 0  # the over-cap region went through the tiler (phase 2) and cleared


def test_2d_report_keeps_the_3d_fields_at_minus_one():
    from dvfopt.core.windowed import SliceReport

    rep = SliceReport()
    assert (rep.folds_after_zero, rep.best_diag_floor_after, rep.best_diag_floor_after_zero) == (
        -1,
        -1,
        -1,
    )


@needs_osqp
def test_3d_solver_composition_and_string_recipe():
    from dvfopt import ISQPWindowedStrategy, Solver, correct_dvf

    phi = _planted_3d((8, 14, 14), amp=0.5)
    res = Solver(
        constraint=SimplexConstraint3D(shape=phi.shape[1:]),
        objective=NoneObjective(),
        strategy=ISQPWindowedStrategy(),
    ).fit(phi)
    assert res.corrected.shape == phi.shape and np.isfinite(res.corrected).all()
    res2 = correct_dvf(phi, constraint='simplex_3d', strategy='isqp_windowed', objective='none')
    assert np.array_equal(res2.corrected, res.corrected)  # same deterministic solve


@needs_osqp
def test_3d_auto_objective_recipe_does_not_inject_polish():
    """``objective='auto'`` asks for the per-window ``polish='l2'`` on mild fields; that
    is a 2D-measured recipe the 3D engine refuses, so the resolver gates it on ``dim``."""
    from dvfopt import correct_dvf
    from dvfopt.solver import resolve_auto_objective

    # the single source of truth: same objective either way, the polish only in 2D
    assert resolve_auto_objective(10, -1.0, dim=3)[1] is False
    assert resolve_auto_objective(10, -1.0) == ("none", True)  # 2D counterpart, unchanged

    phi = _planted_3d((8, 14, 14), amp=0.5)
    res = correct_dvf(phi, constraint="simplex_3d", strategy="isqp_windowed", objective="auto")
    assert res.corrected.shape == phi.shape and np.isfinite(res.corrected).all()


def test_defaults_table_resolves_only_knobs_left_at_their_2d_default():
    from dvfopt.core.windowed._common import DEFAULTS_BY_DIM, resolve_dim_defaults

    two = {k: v[2] for k, v in DEFAULTS_BY_DIM.items()}
    assert resolve_dim_defaults(2, **two) == two  # 2D: the table is the identity
    three = resolve_dim_defaults(3, **two)
    assert three["giant_tile"] == 16 and three["mop_margin"] == 6
    assert three["max_window_area"] == DEFAULTS_BY_DIM["max_window_area"][3]
    # ip_cold stays True in both columns (False was measured and REJECTED: it wins the
    # 17^3 whole window but breaks the tiled crops); the QP cap also stays at its 2D
    # value on 3D (2000/1000 was measured and REJECTED — see DEFAULTS_BY_DIM).
    assert three["ip_cold"] is True and three["qp_max_iter"] == 1000
    assert three["max_window_area"] == 8000
    # the coarse warm start is OFF on 3D (spike 2: the measured source of the introduced
    # negative axial edge projections) and unchanged on 2D
    assert three["coarse_to_fine"] is False
    assert resolve_dim_defaults(2, **two)["coarse_to_fine"] is True
    assert resolve_dim_defaults(3, coarse_to_fine=True)["coarse_to_fine"] is False
    assert (
        resolve_dim_defaults(3, **three) == three
    )  # the recursive solves re-resolve the resolved values
    # an explicit non-default value is honoured in any dimension; 0 still disables the mop
    assert resolve_dim_defaults(3, **{**two, "giant_tile": 12})["giant_tile"] == 12
    assert resolve_dim_defaults(3, **{**two, "mop_margin": 0})["mop_margin"] == 0
    assert resolve_dim_defaults(2, **{**two, "giant_tile": 12})["giant_tile"] == 12
    assert resolve_dim_defaults(3, **{**two, "qp_max_iter": 1500})["qp_max_iter"] == 1500


def test_twin_knobs_are_gone():
    import inspect

    from dvfopt.core.windowed import windowed_correct as wc
    from dvfopt.strategies.windowed import ISQPWindowedStrategy

    params = inspect.signature(wc).parameters
    assert "giant_tile_3d" not in params and "mop_margin_3d" not in params
    assert not hasattr(ISQPWindowedStrategy(), "giant_tile_3d")


def test_dim_defaults_false_takes_every_knob_literally(monkeypatch):
    from dvfopt.core.windowed import _common as engine

    seen = {}
    real = engine._InnerOpts

    def spy(*a, **k):
        o = real(*a, **k)
        seen["giant_tile"] = o.giant_tile
        seen["qp_max_iter"] = o.qp_max_iter
        return o

    monkeypatch.setattr(engine, "_InnerOpts", spy)
    phi = np.zeros((3, 6, 8, 8))
    c = SimplexConstraint3D(shape=(6, 8, 8))
    engine.windowed_correct(
        phi,
        "isqp",
        constraint=c,
        objective=NoneObjective(),
        threshold=THR,
        verbose=0,
        dim_defaults=False,
    )
    assert seen["giant_tile"] == 64 and seen["qp_max_iter"] == 1000  # the 2D defaults
    engine.windowed_correct(
        phi, "isqp", constraint=c, objective=NoneObjective(), threshold=THR, verbose=0
    )
    # qp_max_iter now stays at the 2D value (1000) on the table path too, so it can't
    # distinguish "resolved" from "literal" here; giant_tile still can.
    assert seen["giant_tile"] == 16  # the table
    # the flag must ride into both recursive solves (coarse warm start, re-seed polish),
    # or a literal run's 2D-default knobs are re-resolved there. Both stages need a large
    # folded field to fire, so this is a source check rather than a run.
    import inspect

    assert inspect.getsource(engine.windowed_correct).count("dim_defaults=dim_defaults") == 2
