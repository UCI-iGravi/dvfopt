"""``checkpoint_dir`` on the 2.5D pipeline: a run interrupted mid-sweep resumes
from the last finished slice and reproduces the uninterrupted result byte for
byte; a finished checkpoint reloads; a mismatched one refuses."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dvfopt.pipeline_25d import correct_dvf_25d
from tests.test_pipeline_25d_callback import _interlayer_folded_volume


class _Stop(Exception):
    pass


def _stop_after(n):
    seen = []

    def cb(e):
        if e['phase'] == 'sweep':
            seen.append(e['index'])
            if len(seen) == n:
                raise _Stop

    cb.seen = seen
    return cb


def test_interrupted_sweep_resumes_and_matches_cold_run(tmp_path):
    vol = _interlayer_folded_volume(D=6)
    ref, ref_rep = correct_dvf_25d(vol, verbose=0)
    ck = tmp_path / 'ck'

    cb = _stop_after(2)
    with pytest.raises(_Stop):
        correct_dvf_25d(vol, verbose=0, checkpoint_dir=ck, progress_callback=cb, callback_copies=False)
    state = json.loads((ck / 'state.json').read_text())
    assert state['n_done'] == 2 and state['stage'] == 'sweep'

    events = []
    out, rep = correct_dvf_25d(
        vol, verbose=0, checkpoint_dir=ck, progress_callback=lambda e: events.append(e['index'])
    )
    sweep_idx = [i for i in events[:-1]] if rep.stages[-1]['stage'] == 'mop' else events
    assert sweep_idx[0] == 3, 'resume must start at the slice after the last finished one'
    assert np.array_equal(out, ref)
    assert (rep.n_neg_out, rep.n_below_out) == (ref_rep.n_neg_out, ref_rep.n_below_out)
    assert json.loads((ck / 'state.json').read_text())['stage'] == 'done'

    # A finished checkpoint just reloads — no sweep events, same field.
    events.clear()
    out2, rep2 = correct_dvf_25d(vol, verbose=0, checkpoint_dir=ck, progress_callback=lambda e: events.append(e))
    assert not events and np.array_equal(out2, ref)
    assert rep2.stages[-1]['stage'] == 'resumed' and rep2.n_neg_out == ref_rep.n_neg_out


def test_mismatched_checkpoint_refuses(tmp_path):
    vol = _interlayer_folded_volume(D=4)
    ck = tmp_path / 'ck'
    correct_dvf_25d(vol, verbose=0, checkpoint_dir=ck)
    with pytest.raises(ValueError, match='does not match'):
        correct_dvf_25d(vol, verbose=0, checkpoint_dir=ck, threshold=0.02)
    other = vol.copy()
    other[2, 0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match='input_sha256'):
        correct_dvf_25d(other, verbose=0, checkpoint_dir=ck)


def test_no_checkpoint_is_byte_identical_to_checkpointed(tmp_path):
    vol = _interlayer_folded_volume(D=5)
    a, _ = correct_dvf_25d(vol, verbose=0)
    b, _ = correct_dvf_25d(vol, verbose=0, checkpoint_dir=tmp_path / 'ck')
    assert np.array_equal(a, b)
