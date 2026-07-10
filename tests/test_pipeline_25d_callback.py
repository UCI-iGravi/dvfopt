"""progress_callback hook on the 2.5D marching pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt.pipeline_25d import correct_dvf_25d


def _interlayer_folded_volume(D=4, H=8, W=8):
    """dz==0, per-slice 2D-feasible, but adjacent slices' dx alternate sign
    strongly at one column -> inter-layer 6-tet folds.

    Magnitude 1.5 (not the smaller values one might reach for first):
    empirically the 6-tet fold test only trips once the alternating swing
    crosses a full grid unit (0.7 -> 0 folds, 1.0 -> the knife-edge, 1.5 ->
    a comfortable margin) -- 1.5 also matches the working fixture's
    magnitude in ``tests/test_pipeline_25d.py::_planted_25d_fold``.
    """
    vol = np.zeros((3, D, H, W), dtype=np.float64)
    for k in range(D):
        vol[2, k, :, 4] = 1.5 if k % 2 == 0 else -1.5
    # Self-check the construction: it must actually have 3D folds.
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    assert (six_tet_min_volume_3d(vol) <= 0).sum() > 0, 'fixture has no 3D folds'
    return vol


def test_progress_callback_fires_with_contract_keys():
    vol = _interlayer_folded_volume()
    events = []
    correct_dvf_25d(vol, verbose=0, progress_callback=lambda e: events.append(dict(e)))
    assert events, 'no progress events fired'
    sweep = [e for e in events if e['phase'] == 'sweep']
    assert sweep, 'no sweep events'
    for e in events:
        assert set(e) == {'phase', 'index', 'total', 'n_neg', 'phi'}
        assert e['phase'] in ('sweep', 'mop')
        assert e['phi'].shape == vol.shape
    # sweep indices increase, total is D
    assert [e['index'] for e in sweep] == sorted(e['index'] for e in sweep)
    assert all(e['total'] == vol.shape[1] for e in sweep)


def test_progress_callback_keyboardinterrupt_propagates():
    vol = _interlayer_folded_volume()

    def cb(e):
        raise KeyboardInterrupt('stop')

    with pytest.raises(KeyboardInterrupt):
        correct_dvf_25d(vol, verbose=0, progress_callback=cb)


def test_default_none_unchanged():
    vol = _interlayer_folded_volume()
    out, report = correct_dvf_25d(vol, verbose=0)
    assert out.shape == vol.shape
    assert report.n_neg_out <= report.n_neg_in
