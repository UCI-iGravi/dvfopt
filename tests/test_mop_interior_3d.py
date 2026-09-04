"""Tests for the frozen-rim 3D-interior elastic-SLP mop.

The mop preserves the 2.5D precondition ``dz == 0``: it only ever writes
``phi[1:3]`` (``[dy, dx]``). We therefore plant an INTER-LAYER simplex (3D) fold
with ``dz`` left at zero, by making two adjacent slices' ``dy`` differ
strongly (unlike ``tests/test_coupled_kring_3d.py``, which plants folds in
channel 0 -- that would violate ``dz == 0`` here).
"""

import numpy as np
import pytest

from dvfopt.core.marching._mop_interior_3d import mop_interior_3d
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


def _planted_inter_layer_fold():
    """(3,6,20,20) field, dz==0, with a planted inter-layer simplex (3D) fold."""
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, 6, 20, 20)).astype(np.float64)
    phi[0] = 0.0  # dz == 0 (2.5D precondition)
    phi[1, 2, 8:11, 8:11] = +1.5  # dy on slice z=2
    phi[1, 3, 8:11, 8:11] = -1.5  # dy on slice z=3 -> inter-layer fold
    return phi


def _smooth_field():
    """(3,6,20,20) fold-free field: dz==0 and tiny noise."""
    rng = np.random.default_rng(1)
    phi = rng.normal(0, 0.001, (3, 6, 20, 20)).astype(np.float64)
    phi[0] = 0.0
    return phi


def test_mop_reduces_folds():
    phi = _planted_inter_layer_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    phi_out, info = mop_interior_3d(phi)
    assert info["n_neg_after"] < info["n_neg_before"]


def test_mop_preserves_dz_zero():
    phi = _planted_inter_layer_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    phi_out, _ = mop_interior_3d(phi)
    assert np.all(phi_out[0] == 0.0)


def test_mop_does_not_mutate_input():
    phi = _planted_inter_layer_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    before = phi.copy()
    mop_interior_3d(phi)
    assert np.array_equal(phi, before)


def test_mop_noop_when_no_folds():
    phi = _smooth_field()
    assert int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0
    phi_out, info = mop_interior_3d(phi)
    assert info["n_neg_before"] == 0
    assert info["passes"] == []
    assert np.array_equal(phi_out, phi)


def test_mop_rejects_nonzero_dz():
    phi = _planted_inter_layer_fold()
    phi[0, 2, 5, 5] = 0.3  # violate the dz==0 precondition
    with pytest.raises(ValueError, match="dz"):
        mop_interior_3d(phi)


def test_mop_rejects_nan():
    phi = _planted_inter_layer_fold()
    phi[1, 2, 5, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mop_interior_3d(phi)


def test_mop_dil0_no_dilation():
    # dil=0 must mean "no dilation" (scipy's iterations=0 would instead
    # dilate until convergence, ballooning a single cluster to the whole
    # grid). The mop must still work and reduce folds.
    phi = _planted_inter_layer_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    phi_out, info = mop_interior_3d(phi, dil=0)
    assert info["n_neg_after"] < info["n_neg_before"]
    # With dil=0 the crop stays local: the far corner region is untouched.
    assert np.array_equal(phi_out[:, :, :4, :4], phi[:, :, :4, :4])


def test_mop_negative_dil_raises():
    phi = _planted_inter_layer_fold()
    with pytest.raises(ValueError, match="dil"):
        mop_interior_3d(phi, dil=-1)


def test_mop_repairs_subthreshold_cube():
    # The mop must repair cubes whose min volume is positive but below the
    # threshold, not just negatives. Bisect a fold amplitude so the worst
    # cube's min volume lands strictly inside (0, threshold=0.01).
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.002, (3, 6, 20, 20)).astype(np.float64)
    base[0] = 0.0

    def field(amp):
        phi = base.copy()
        phi[1, 2, 8:11, 8:11] += +amp
        phi[1, 3, 8:11, 8:11] += -amp
        return phi

    def min_T(amp):
        return float(six_tet_min_volume_3d(field(amp)).min())

    lo, hi = 0.0, 1.0
    if not (min_T(lo) > 0.005 and min_T(hi) < 0.005):
        pytest.skip("bisection bracket not achievable for this fixture")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if min_T(mid) > 0.005:
            lo = mid
        else:
            hi = mid
    amp = 0.5 * (lo + hi)
    mn = min_T(amp)
    if not (0.0 < mn < 0.01):
        pytest.skip(f"could not land min_T in (0, 0.01); got {mn}")

    phi = field(amp)
    assert int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0, (
        "fixture must have NO negatives (sub-threshold only)"
    )
    phi_out, info = mop_interior_3d(phi)
    assert info["n_below_before"] > 0
    assert info["n_below_after"] < info["n_below_before"]


def test_mop_copy_false_operates_in_place():
    phi = _planted_inter_layer_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    phi_out, info = mop_interior_3d(phi, copy=False)
    assert phi_out is phi  # caller relinquished the array
    assert info["n_neg_after"] < info["n_neg_before"]


def test_mop_rim_frozen():
    phi = _planted_inter_layer_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    phi_out, _ = mop_interior_3d(phi)
    # Each repaired crop freezes its whole rim, so the global boundary
    # planes of the volume must be untouched.
    assert np.array_equal(phi_out[:, 0, :, :], phi[:, 0, :, :])
    assert np.array_equal(phi_out[:, -1, :, :], phi[:, -1, :, :])
    assert np.array_equal(phi_out[:, :, 0, :], phi[:, :, 0, :])
    assert np.array_equal(phi_out[:, :, -1, :], phi[:, :, -1, :])
    assert np.array_equal(phi_out[:, :, :, 0], phi[:, :, :, 0])
    assert np.array_equal(phi_out[:, :, :, -1], phi[:, :, :, -1])


def _several_separated_clusters():
    """(3,8,30,30) dz==0 field with four inter-layer fold clusters: three
    pairwise-disjoint under the mop's padding (one batch) and a fourth whose
    padded box overlaps the first (forces a second batch)."""
    rng = np.random.default_rng(2)
    phi = rng.normal(0, 0.02, (3, 8, 30, 30)).astype(np.float64)
    phi[0] = 0.0
    for z, y, x in ((1, 5, 5), (3, 20, 20), (5, 5, 20), (2, 12, 5)):
        phi[1, z, y : y + 3, x : x + 3] = +1.5
        phi[1, z + 1, y : y + 3, x : x + 3] = -1.5
    return phi


def test_mop_parallel_is_byte_identical_to_serial():
    phi = _several_separated_clusters()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    ref, info_ref = mop_interior_3d(phi)
    assert info_ref["n_fixed"] >= 2 and not np.array_equal(ref, phi)
    # In-process pool_map seam: exercises the batching / paste-back path alone.
    out, info = mop_interior_3d(phi, n_workers=2, pool_map=lambda w, a, n: [w(x) for x in a])
    assert np.array_equal(out, ref) and info["n_fixed"] == info_ref["n_fixed"]
    # The real shared spawn pool.
    out2, info2 = mop_interior_3d(phi, n_workers=2)
    assert np.array_equal(out2, ref) and info2["n_fixed"] == info_ref["n_fixed"]


def _wide_fold_band():
    """(3,6,12,60) dz==0 field with one inter-layer fold band 52 columns wide —
    a single connected residual cluster far wider than a small max_box."""
    rng = np.random.default_rng(3)
    phi = rng.normal(0, 0.02, (3, 6, 12, 60)).astype(np.float64)
    phi[0] = 0.0
    phi[1, 2, 4:8, 4:56] = +1.5
    phi[1, 3, 4:8, 4:56] = -1.5
    return phi


def test_mop_giant_box_is_tiled_and_parallel_identical():
    phi = _wide_fold_band()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    # Tiled (max_box=12 on a ~60-wide padded box -> several tiles) still repairs
    # and reports MULTIPLE fixed boxes — the tiles, proving the cap engaged.
    tiled, info = mop_interior_3d(phi, max_box=12)
    assert info["n_fixed"] >= 2
    assert info["n_below_after"] < info["n_below_before"]
    # Tiles of one giant box are pairwise disjoint -> parallel == serial exactly.
    par, info_p = mop_interior_3d(
        phi, max_box=12, n_workers=2, pool_map=lambda w, a, n: [w(x) for x in a]
    )
    assert np.array_equal(par, tiled) and info_p["n_fixed"] == info["n_fixed"]
    # The cap is inert on boxes that fit: uncapped == a max_box that swallows the box.
    a, _ = mop_interior_3d(phi, max_box=None)
    b, _ = mop_interior_3d(phi, max_box=1000)
    assert np.array_equal(a, b)


def test_elastic_engine_futility_stop_ends_micro_step_grind():
    """A box whose every candidate is accepted by a hair (1e-6) doubles its
    trust back each time and never reaches the trust floor: without the stop
    it burns all max_iters solves; with stall_iters=4 / 1 % it ends at <= 5."""
    from scipy import sparse

    from dvfopt.core.marching._elastic_engine import elastic_trust_solve

    def run(stall_iters):
        n_lp = [0]
        v = [1.0]
        x0 = np.zeros(2)

        def blocks(_state):
            n_lp[0] += 1
            return [(sparse.csr_matrix(np.array([[1.0, 1.0]])), np.array([0.5]), 1.0)]

        def viol(_state):
            v[0] *= 1.0 - 1e-6  # every candidate a hair better -> accepted
            return v[0]

        elastic_trust_solve(
            x0,
            x0,
            blocks,
            viol,
            lambda s, _x: s,
            state=object(),
            mu=1.0,
            max_iters=20,
            stall_iters=stall_iters,
        )
        return n_lp[0]

    assert run(0) == 20
    assert run(4) <= 5


def test_mop_levels_run_spread_boxes_together():
    """A(0) < B overlaps A -> 1; C disjoint -> 0; D overlaps C -> 1: two levels
    where the old consecutive-prefix rule made three batches ([A],[B,C],[D])."""
    from dvfopt.core.marching._mop_interior_3d import _levels

    A = (0, 5, 0, 10, 0, 10)
    B = (0, 5, 8, 18, 0, 10)  # overlaps A on y
    C = (0, 5, 40, 50, 0, 10)  # far away
    D = (0, 5, 48, 58, 0, 10)  # overlaps C on y
    assert _levels([A, B, C, D]) == [0, 1, 0, 1]
    # transitive: E overlaps B (level 1) -> 2, even though E is disjoint from A
    E = (0, 5, 16, 26, 0, 10)
    assert _levels([A, B, C, D, E]) == [0, 1, 0, 1, 2]
