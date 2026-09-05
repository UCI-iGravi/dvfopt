"""Axial edge-monotonicity rows in the marching engines (2.5D prevention port)."""

import numpy as np
import pytest

from dvfopt import correct_dvf_25d
from dvfopt.core.marching._mono_rows import axial_mono_rows, mono_block
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


def _planted_25d_fold():
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, 6, 20, 20)).astype(np.float64)
    phi[0] = 0.0
    phi[1, 2, 8:11, 8:11] = +1.5
    phi[1, 3, 8:11, 8:11] = -1.5
    return phi


def test_axial_mono_rows_values():
    """Row values are 1 + (gap diff); a collapsed x-edge dips below delta."""
    D, H, W = 1, 3, 4
    a = axial_mono_rows(D, H, W)
    assert a.shape == (H * (W - 1) + (H - 1) * W, 2 * D * H * W)
    dx = np.zeros((H, W))
    dy = np.zeros((H, W))
    dx[1, 2] = -0.5  # edge (1,1)-(1,2): 1 + dx[1,2] - dx[1,1] = 0.5
    t = 1.0 + a @ np.concatenate([dx.ravel(), dy.ravel()])
    assert np.isclose(t.min(), 0.5)
    # identity map: every row exactly 1
    assert np.allclose(1.0 + a @ np.zeros(2 * D * H * W), 1.0)


def test_mono_block_filters_and_slices():
    a = axial_mono_rows(1, 3, 4)
    dxdy = np.zeros(2 * 12)
    dxdy[6] = -0.9  # dx[1,2] -> one violated gap at delta=0.2 window
    free = np.arange(2 * 12)
    blk = mono_block(a, dxdy, free, delta=0.01, active_window=0.5)
    assert blk is not None
    j, t, thr = blk
    assert thr == 0.01 and (t < 0.52).all()
    # nothing active on the identity map with a tiny window
    assert mono_block(a, np.zeros(2 * 12), free, 0.01, 0.05) is None
    # no free columns -> None
    assert mono_block(a, dxdy, np.array([], dtype=int), 0.01, 0.5) is None


def test_25d_pipeline_with_rows_still_repairs():
    """orientation_delta on the 2.5D pipeline: same-or-better fold count, and the
    knob is byte-identical when None."""
    phi = _planted_25d_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip("no fold planted")
    out_a, rep_a = correct_dvf_25d(phi.copy(), n_workers=1)
    out_b, rep_b = correct_dvf_25d(phi.copy(), n_workers=1, orientation_delta=None)
    assert np.array_equal(out_a, out_b)
    out_c, rep_c = correct_dvf_25d(phi.copy(), n_workers=1, orientation_delta=0.01)
    assert rep_c.n_neg_out <= rep_a.n_neg_out
    assert rep_c.n_neg_best_diag_out >= 0 and rep_a.n_neg_best_diag_out >= 0
