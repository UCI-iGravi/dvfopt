"""Tests for the frozen-rim 3D-interior elastic-SLP mop.

The mop preserves the 2.5D precondition ``dz == 0``: it only ever writes
``phi[1:3]`` (``[dy, dx]``). We therefore plant an INTER-LAYER 6-tet fold
with ``dz`` left at zero, by making two adjacent slices' ``dy`` differ
strongly (unlike ``tests/test_coupled_kring_3d.py``, which plants folds in
channel 0 -- that would violate ``dz == 0`` here).
"""

import numpy as np
import pytest

from dvfopt.core.marching._mop_interior_3d import mop_interior_3d
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


def _planted_inter_layer_fold():
    """(3,6,20,20) field, dz==0, with a planted inter-layer 6-tet fold."""
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
