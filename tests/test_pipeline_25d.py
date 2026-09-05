"""Tests for the public 2.5D marching pipeline: ``correct_dvf_25d``.

The pipeline sweeps each in-plane-only (dz==0) z-slice against its
already-repaired neighbour, then runs a frozen-rim 3D-interior mop. Unlike
the 3D k-ring idiom (which plants folds in dz), the 2.5D precondition is
``dz == 0``, so folds here are planted in dy across an adjacent slice pair.
"""

import numpy as np
import pytest

from dvfopt import Correct25DReport, correct_dvf_25d
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


def _planted_25d_fold():
    """Small dz==0 field with an inter-layer simplex (3D) fold between z=2 and z=3."""
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, 6, 20, 20)).astype(np.float64)
    phi[0] = 0.0  # dz == 0 (2.5D precondition)
    phi[1, 2, 8:11, 8:11] = +1.5  # dy on slice z=2
    phi[1, 3, 8:11, 8:11] = -1.5  # dy on slice z=3 -> inter-layer fold
    return phi


def test_correct_dvf_25d_reduces_folds():
    phi = _planted_25d_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip('no fold planted')
    phi_out, report = correct_dvf_25d(phi, n_workers=1)
    assert report.n_neg_out < report.n_neg_in
    # Verified empirically: the bidirectional sweep alone clears this fixture
    # (the mop is not even needed here), so strict feasibility is reachable.
    assert report.n_neg_out == 0
    assert report.feasible is True
    # Output really is fold-free.
    assert int((six_tet_min_volume_3d(phi_out) <= 0).sum()) == 0


def test_preserves_dz_zero():
    phi = _planted_25d_fold()
    phi_out, _ = correct_dvf_25d(phi, n_workers=1)
    assert np.all(phi_out[0] == 0.0)


def test_does_not_mutate_input():
    phi = _planted_25d_fold()
    phi_ref = phi.copy()
    correct_dvf_25d(phi, n_workers=1)
    assert np.array_equal(phi, phi_ref)


def test_rejects_nonzero_dz():
    phi = _planted_25d_fold()
    phi[0, 2, 5, 5] = 0.3  # violate the dz==0 precondition
    with pytest.raises(ValueError, match='dz'):
        correct_dvf_25d(phi, n_workers=1)


def test_rejects_nan_in_dy():
    phi = _planted_25d_fold()
    phi[1, 2, 5, 5] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        correct_dvf_25d(phi, n_workers=1)


def test_rejects_nan_in_dz():
    # NaN in dz would be invisible to the |dz| > tol comparison itself,
    # so the finite check must catch it first.
    phi = _planted_25d_fold()
    phi[0, 2, 5, 5] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        correct_dvf_25d(phi, n_workers=1)


def test_down_sweep_repairs_origin_slice():
    # Fold planted between slices 2 and 3. With origin=2 the up sweep repairs
    # z=3 against the frozen (still-folded) z=2 and cannot fully clear the
    # inter-layer; the down sweep must then repair the ORIGIN slice itself
    # against its already-repaired upper neighbour. The pre-fix code skipped
    # the origin in the down sweep and left residual folds here (verified: 8
    # residual negatives with the origin slice untouched).
    phi = _planted_25d_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip('no fold planted')
    phi_out, report = correct_dvf_25d(phi, origin=2, mop=False, n_workers=1)
    assert not np.array_equal(phi_out[1:3, 2], phi[1:3, 2]), (
        'origin slice was not repaired by the down sweep'
    )
    assert report.n_neg_out == 0


def test_noop_when_already_feasible():
    rng = np.random.default_rng(1)
    phi = rng.normal(0, 0.001, (3, 6, 20, 20)).astype(np.float64)
    phi[0] = 0.0  # dz == 0
    assert int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0, 'fixture should be feasible'
    phi_out, report = correct_dvf_25d(phi, n_workers=1)
    assert report.n_neg_in == 0
    assert report.feasible is True
    assert np.array_equal(phi_out, phi)


def test_origin_auto_picks_mildest_layer():
    # Fold only between slices 2 and 3 -> inter-layer index 2 is the folded
    # one; auto must seed from a mild (fold-free) layer, not index 2.
    phi = _planted_25d_fold()
    if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
        pytest.skip('no fold planted')
    _, report = correct_dvf_25d(phi, origin='auto', n_workers=1)
    assert report.origin != 2


def test_explicit_origin_accepted():
    phi = _planted_25d_fold()
    phi_out, report = correct_dvf_25d(phi, origin=0, n_workers=1)
    assert report.origin == 0
    assert phi_out.shape == phi.shape


def test_returns_report_dataclass():
    phi = _planted_25d_fold()
    _, report = correct_dvf_25d(phi, n_workers=1)
    assert isinstance(report, Correct25DReport)
    assert isinstance(report.stages, list)


def _planted_25d_folds_12():
    """D=12 variant of ``_planted_25d_fold``: the z=2|3 fold plus a milder one
    across the z=7|8 seam of a 3-segment split (both clear single-origin)."""
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, 12, 20, 20)).astype(np.float64)
    phi[0] = 0.0
    for z, amp in ((2, 1.5), (7, 1.0)):
        phi[1, z, 8:11, 8:11] = +amp
        phi[1, z + 1, 8:11, 8:11] = -amp
    return phi


def test_segments_default_and_one_are_the_single_origin_sweep():
    vol = _planted_25d_folds_12()
    ref, rep = correct_dvf_25d(vol)
    out, _ = correct_dvf_25d(vol, n_segments=1)
    assert np.array_equal(out, ref)
    assert rep.stages[0]['stage'] == 'sweep'
    with pytest.raises(ValueError, match='n_segments'):
        correct_dvf_25d(vol, n_segments=7)


def test_segment_parallel_sweep_clears_and_is_deterministic(tmp_path):
    import json

    vol = _planted_25d_folds_12()
    if int((six_tet_min_volume_3d(vol) <= 0).sum()) == 0:
        pytest.skip('no fold planted')
    _, rep_single = correct_dvf_25d(vol)
    out, rep = correct_dvf_25d(vol, n_segments=3)
    assert [s['stage'] for s in rep.stages][:2] == ['segments', 'seams']
    assert rep.n_neg_out == 0 == rep_single.n_neg_out
    assert np.all(out[0] == 0.0)
    # Segments are deterministic given their slab: the pooled run is byte-identical.
    out_par, _ = correct_dvf_25d(vol, n_segments=3, n_workers=3)
    assert np.array_equal(out_par, out)

    # Checkpoint units: stop after the first segment lands, resume, reload.
    ck = tmp_path / 'ck'

    class _Stop(Exception):
        pass

    def stop_first(e):
        raise _Stop

    with pytest.raises(_Stop):
        correct_dvf_25d(vol, n_segments=3, checkpoint_dir=ck, progress_callback=stop_first)
    assert json.loads((ck / 'state.json').read_text())['done'] == [0, 1, 2, 3]
    events = []
    out_ck, _ = correct_dvf_25d(
        vol, n_segments=3, checkpoint_dir=ck, progress_callback=lambda e: events.append(e['index'])
    )
    assert events[0] == 8, 'resume must skip the finished segment'
    assert np.array_equal(out_ck, out)
    done = json.loads((ck / 'state.json').read_text())['done']
    assert done[4:] == [4, 5, 6, 7, 8, 9, 10, 11, 'seam:3', 'seam:7']
    out_re, rep_re = correct_dvf_25d(vol, n_segments=3, checkpoint_dir=ck)
    assert rep_re.stages[-1]['stage'] == 'resumed' and np.array_equal(out_re, out)
