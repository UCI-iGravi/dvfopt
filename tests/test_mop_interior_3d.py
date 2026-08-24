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
