"""Phase 4, task 1: box-scoped tiler fold counts + patch-only RAS tasks.

Both changes are efficiency-only — the pins here are that they are byte-identical
to what they replace: ``_folds_in_box`` reproduces the whole-field count restricted
to the box, and a RAS worker handed only its tile's ring+1-padded patch reproduces
the solve it used to run on a private copy of the whole snapshot.
"""

import numpy as np
import pytest

from dvfopt.constraints import (
    FiniteJdetConstraint2D,
    JdetConstraint2D,
    SimplexConstraint2D,
    SimplexConstraint2DBilinear,
    SimplexConstraint3D,
)
from dvfopt.core.windowed import _common as cm
from dvfopt.core.windowed import min_field


def _rand_field(rng, shape, amp=0.6):
    return rng.normal(0.0, amp, size=(len(shape), *shape))


@pytest.mark.parametrize(
    'cls, shape',
    [
        (JdetConstraint2D, (23, 29)),
        (FiniteJdetConstraint2D, (23, 29)),
        (SimplexConstraint2D, (23, 29)),
        (SimplexConstraint2DBilinear, (23, 29)),
        (SimplexConstraint3D, (11, 13, 12)),
    ],
)
def test_folds_in_box_matches_whole_field_slice(cls, shape):
    rng = np.random.default_rng(7)
    phi = _rand_field(rng, shape)
    c = cls(shape=shape)
    whole = min_field(c, phi) < 0.01
    boxes = [
        tuple(v for n in shape for v in (0, n)),  # the whole grid
        tuple(v for n in shape for v in (0, max(3, n // 2))),  # touching the low border
        tuple(v for n in shape for v in (n // 3, n)),  # touching the high border
        tuple(v for n in shape for v in (2, n - 2)),  # strictly interior
    ]
    assert whole.any()  # the fixture really has folds to count
    for box in boxes:
        want = int(whole[cm._box_slices(box)].sum())
        assert cm._folds_in_box(c, phi, box, 0.01) == want


@pytest.mark.parametrize(
    'cls, scale',  # jdet (central differences) needs a rougher field to fold at all
    [(SimplexConstraint2DBilinear, 0.4), (JdetConstraint2D, 1.2)],
)
def test_ras_tile_task_on_patch_equals_full_snapshot_solve(cls, scale):
    """The patch-only worker reproduces, byte for byte, a tile solved on a private copy
    of the WHOLE snapshot (the pre-phase-4 semantics), records translated to global.

    ``JdetConstraint2D`` is the case that pins the ``ring + 1`` pad: it is the one family
    whose enforced rows depend on the patch's image-border flags (``_eval_valid_jdet``),
    so a patch padded by only ``ring`` would read its own edge as the image border and
    enforce rows the whole-field solve leaves out.
    """
    pytest.importorskip('osqp')
    from dataclasses import replace

    from dvfopt.objectives import L2Objective
    from tests.conftest import planted_fold

    phi = planted_fold(40, 44, scale=scale).astype(np.float64)  # (2, 40, 44) [dy, dx]
    c = cls(shape=phi.shape[1:])
    opts = cm._resolve_opts_for_test(2)
    if cls is JdetConstraint2D:  # what windowed_correct does for a non-DY_FIRST pack
        opts = replace(opts, orientation_delta=None)
    _assert_patch_task_matches_whole_field(phi, c, (12, 30, 10, 32), (14, 26, 12, 28), opts)


def test_ras_tile_task_3d_translates_every_axis():
    """The 3D worker: same byte-for-byte pin on a ``SimplexConstraint3D`` volume.

    The only dimension-dependent arithmetic in the worker is the translation
    (``off[i // 2]`` on the boxes, ``off[nd - 2]`` / ``off[nd - 1]`` on ``fy0`` / ``fx0``),
    so the tile box is chosen to give three DISTINCT axis offsets — a swapped or
    2D-hardcoded index lands on the wrong axis and the record comparison fails.
    """
    pytest.importorskip('osqp')
    from tests.conftest import planted_fold_3d

    # (3, 12, 13, 14) [dz, dy, dx]; the fixture punches its fold at y, x = 2:4, too close
    # to the low border for three distinct offsets — rolled deeper in (the base is iid
    # noise, so the wrap introduces no seam).
    phi = np.roll(planted_fold_3d(12, 13, 14), (3, 4), axis=(2, 3)).astype(np.float64)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    tb = (6, 11, 4, 9, 5, 11)  # lows 6 / 4 / 5 -> offsets 4 / 2 / 3 with ring + 1 == 2
    core = (7, 10, 5, 8, 6, 10)
    off = tuple(cm._pad_box(tb, phi.shape[1:], 2)[2 * a] for a in range(3))
    assert len(set(off)) == 3, off  # the point of this case
    _assert_patch_task_matches_whole_field(phi, c, tb, core, cm._resolve_opts_for_test(3))


def _assert_patch_task_matches_whole_field(phi, c, tb, core, opts):
    """``_ras_tile_task`` on the tile's ring+1 patch == ``_solve_window`` on a private copy
    of the whole field, values byte for byte and every record translated back to global."""
    from dvfopt.objectives import L2Objective

    ring = cm._locality_of(c).ring
    shape = phi.shape[1:]
    assert (min_field(c, phi)[cm._box_slices(tb)] < 0.01).any()  # the tile has folds to fix
    # reference: old semantics — the whole field handed to the worker
    ref = np.array(phi, dtype=np.float64, copy=True)
    rep_ref = cm.SliceReport()
    cm._solve_window(
        ref,
        c,
        tb,
        0.01,
        L2Objective(),
        400,
        ring,
        rep_ref,
        margin_delta=1e-3,
        allow_grow=False,
        inner='isqp',
        opts=opts,
    )
    want = ref[(slice(None), *cm._box_slices(core))].copy()
    # new: patch-only
    pb = cm._pad_box(tb, shape, ring + 1)
    off = tuple(pb[2 * a] for a in range(len(shape)))
    patch = phi[(slice(None), *cm._box_slices(pb))].copy()
    tb_local = tuple(tb[i] - off[i // 2] for i in range(len(tb)))
    core_local = tuple(core[i] - off[i // 2] for i in range(len(core)))
    got_core, got, rep = cm._ras_tile_task(
        (patch, off, c, tb_local, core_local, 0.01, L2Objective(), 400, ring, 1e-3, 'isqp', opts)
    )
    assert got_core == core
    assert rep_ref.windows and rep_ref.windows[0].inner_iters > 0  # the solve really ran
    assert not np.array_equal(want, phi[(slice(None), *cm._box_slices(core))])  # and moved pixels
    np.testing.assert_array_equal(got, want)
    assert [w.patch_box for w in rep.windows] == [w.patch_box for w in rep_ref.windows]
    assert [(w.fy0, w.fx0) for w in rep.windows] == [(w.fy0, w.fx0) for w in rep_ref.windows]
    assert [(w.ph, w.pw) for w in rep.windows] == [(w.ph, w.pw) for w in rep_ref.windows]
    assert [w.inner_iters for w in rep.windows] == [w.inner_iters for w in rep_ref.windows]


def test_resolve_opts_for_test_matches_the_engine_defaults():
    """The hook must hand back exactly what ``windowed_correct`` builds (2D signature
    defaults; the 3D column of ``DEFAULTS_BY_DIM`` plus the ``step_rule`` degrade)."""
    o2 = cm._resolve_opts_for_test(2)
    assert (o2.giant_tile, o2.step_rule, o2.orientation_rows) == (64, 'exact_ls', 'edges')
    assert o2.orientation_delta == 0.01 and o2.qp_max_iter == 1000
    o3 = cm._resolve_opts_for_test(3)
    assert (o3.giant_tile, o3.step_rule) == (16, 'tr')
    assert o3.qp_max_iter == 1000 and o3.ip_cold is True
