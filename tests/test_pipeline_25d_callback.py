"""progress_callback hook on the 2.5D marching pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt.pipeline_25d import correct_dvf_25d


def _interlayer_folded_volume(D=4, H=8, W=8):
    """dz==0, per-slice 2D-feasible, but adjacent slices' dx alternate sign
    strongly at one column -> inter-layer simplex (3D) folds.

    Magnitude 1.5 (not the smaller values one might reach for first):
    empirically the simplex (3D) fold test only trips once the alternating swing
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


def test_progress_callback_events_are_independent_snapshots():
    """Default callback_copies=True: each event's 'phi' is an independent
    snapshot of the output buffer at emit time — retained events must not
    alias each other or the live buffer."""
    vol = _interlayer_folded_volume()
    events = []
    out, _ = correct_dvf_25d(vol, verbose=0, progress_callback=lambda e: events.append(dict(e)))
    assert len(events) >= 2, 'need >= 2 events to check snapshot independence'
    # No event carries the live output buffer, and no two events share storage.
    for e in events:
        assert e['phi'] is not out
    for a, b in zip(events, events[1:]):
        assert a['phi'] is not b['phi']
        assert not np.shares_memory(a['phi'], b['phi'])
    # The snapshots capture *different* pipeline states: at least one earlier
    # event differs from a later one (repairs kept mutating the buffer).
    assert any(not np.array_equal(events[0]['phi'], e['phi']) for e in events[1:]), (
        'per-event snapshots are identical — aliasing regression?'
    )
    # Mutation independence: writing into one snapshot leaves the rest intact.
    ref = events[-1]['phi'].copy()
    events[0]['phi'][:] = 12345.0
    assert np.array_equal(events[-1]['phi'], ref)


def test_progress_callback_zero_copy_mode_aliases_live_buffer():
    """callback_copies=False restores the zero-copy contract: every event's
    'phi' IS the live mutable output buffer (the same object the pipeline
    returns), so consumers must copy if they retain it."""
    vol = _interlayer_folded_volume()
    events = []
    out, _ = correct_dvf_25d(
        vol,
        verbose=0,
        progress_callback=lambda e: events.append(dict(e)),
        callback_copies=False,
    )
    assert events, 'no progress events fired'
    for e in events:
        assert e['phi'] is out, 'zero-copy mode must pass the live buffer'


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
