"""``checkpoint_dir`` on the per-slice 2D sweep (``DVFopt.fit``, serial and
pooled), ``correct_dvf_3d`` and the CLI's ``--checkpoint``: a run interrupted
after its first checkpointed unit resumes, reproduces the cold run byte for
byte with a complete report, refuses a mismatched checkpoint, and is
byte-identical to the un-checkpointed run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dvfopt import DVFopt, DVFoptConfig, correct_dvf_3d
from dvfopt.checkpoint import RunCheckpoint
from dvfopt.cli import main


class _Stop(Exception):
    pass


@pytest.fixture
def stop_after_marks(monkeypatch):
    """Interrupt the run right after its ``n``-th ``RunCheckpoint.mark``."""

    def _arm(n):
        calls = []
        orig = RunCheckpoint.mark

        def mark(self, *a, **kw):
            orig(self, *a, **kw)
            calls.append(1)
            if len(calls) == n:
                raise _Stop

        monkeypatch.setattr(RunCheckpoint, 'mark', mark)
        return calls

    return _arm


def _volume(seed=2, D=3):
    return np.random.default_rng(seed).normal(0, 0.3, (3, D, 10, 10))


_CFG = dict(solver='barrier', constraint='simplex', record_history=False, verbose=0)


def _rows(res):
    # Per-slice scalars straight off the results (no pandas — a dev extra the
    # no-extras CI leg lacks); wall time excluded, it is not reproducible.
    keys = (
        "z",
        "init_n_neg",
        "init_min_T",
        "final_n_neg",
        "final_min_T",
        "feasible",
        "solver_used",
        "n_outer_iters",
        "notes",
    )
    return [{k: getattr(r, k) for k in keys} for r in res.slice_results]


@pytest.mark.parametrize('n_workers', [None, 2])
def test_dvfopt_fit_resumes_and_matches_cold_run(
    tmp_path, stop_after_marks, monkeypatch, n_workers
):
    phi = _volume()
    ref = DVFopt(DVFoptConfig(**_CFG, n_workers=n_workers)).fit(phi)
    ck = tmp_path / 'ck'
    cfg = DVFoptConfig(**_CFG, n_workers=n_workers, checkpoint_dir=str(ck))

    stop_after_marks(1)
    with pytest.raises(_Stop):
        DVFopt(cfg).fit(phi)
    state = json.loads((ck / 'state.json').read_text())
    assert len(state['done']) == 1 and state['stage'] == 'run'

    monkeypatch.undo()
    solved = []
    orig = DVFopt._run_slice
    monkeypatch.setattr(
        DVFopt, '_run_slice', lambda self, p, z: solved.append(z) or orig(self, p, z)
    )
    res = DVFopt(cfg).fit(phi)
    if not n_workers:  # the pool path solves in child processes — nothing to count here
        assert sorted(solved) == [z for z in range(3) if z not in state['done']]
    np.testing.assert_array_equal(res.corrected, ref.corrected)
    assert [s.z for s in res.slice_results] == [0, 1, 2]
    assert _rows(res) == _rows(ref)
    assert json.loads((ck / 'state.json').read_text())['stage'] == 'done'

    # A finished checkpoint reloads without solving anything.
    solved.clear()
    again = DVFopt(cfg).fit(phi)
    assert (
        not solved and np.array_equal(again.corrected, ref.corrected) and _rows(again) == _rows(ref)
    )


def test_dvfopt_checkpoint_mismatch_refuses_and_none_is_identical(tmp_path):
    phi = _volume(seed=3)
    ck = tmp_path / 'ck'
    a = DVFopt(DVFoptConfig(**_CFG)).fit(phi)
    b = DVFopt(DVFoptConfig(**_CFG, checkpoint_dir=str(ck))).fit(phi)
    np.testing.assert_array_equal(a.corrected, b.corrected)
    with pytest.raises(ValueError, match='threshold'):
        DVFopt(DVFoptConfig(**_CFG, checkpoint_dir=str(ck), threshold=0.02)).fit(phi)
    with pytest.raises(ValueError, match='input_sha256'):
        DVFopt(DVFoptConfig(**_CFG, checkpoint_dir=str(ck))).fit(phi + 1e-3)


def _planted_3d():
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, 6, 20, 20))
    phi[0, 2, 8:10, 8:10] = 1.5
    phi[0, 3, 8:10, 8:10] = -1.5
    return phi


def _stages(rep):
    return [{k: v for k, v in s.items() if k != 'wall_s'} for s in rep.stages]


def test_correct_dvf_3d_resumes_and_matches_cold_run(tmp_path, stop_after_marks, monkeypatch):
    phi = _planted_3d()
    ref, ref_rep = correct_dvf_3d(phi, threshold=0.01)
    ck = tmp_path / 'ck'

    stop_after_marks(1)  # interrupt right after the bulk stage
    with pytest.raises(_Stop):
        correct_dvf_3d(phi, threshold=0.01, checkpoint_dir=ck)
    assert json.loads((ck / 'state.json').read_text())['done'] == ['bulk']

    monkeypatch.undo()
    marks = stop_after_marks(0)  # count only: a stage is marked only when it runs
    out, rep = correct_dvf_3d(phi, threshold=0.01, checkpoint_dir=ck)
    assert len(marks) == len(json.loads((ck / 'state.json').read_text())['done']) - 1
    np.testing.assert_array_equal(out, ref)
    assert _stages(rep) == _stages(ref_rep)
    assert (rep.n_neg_out, rep.n_below_out, rep.feasible) == (
        ref_rep.n_neg_out,
        ref_rep.n_below_out,
        ref_rep.feasible,
    )
    assert json.loads((ck / 'state.json').read_text())['stage'] == 'done'

    out2, rep2 = correct_dvf_3d(phi, threshold=0.01, checkpoint_dir=ck)
    assert np.array_equal(out2, ref) and rep2.stages[-1]['stage'] == 'resumed'
    with pytest.raises(ValueError, match='does not match'):
        correct_dvf_3d(phi, threshold=0.02, checkpoint_dir=ck)


def test_correct_dvf_3d_no_checkpoint_is_byte_identical(tmp_path):
    phi = _planted_3d()
    a, _ = correct_dvf_3d(phi, threshold=0.01)
    b, _ = correct_dvf_3d(phi, threshold=0.01, checkpoint_dir=tmp_path / 'ck')
    np.testing.assert_array_equal(a, b)


def test_cli_slices_checkpoint_resumes(tmp_path, stop_after_marks, monkeypatch):
    phi = np.zeros((3, 3, 10, 10))
    phi[1:] = _volume(seed=5)[1:]
    p = tmp_path / 'vol.npy'
    np.save(p, phi)
    argv = ['correct', str(p), str(tmp_path / 'out.npy'), '--pipeline', 'slices']
    assert main([*argv, '--report-dir', str(tmp_path / 'rep0')]) == 0
    ref = np.load(tmp_path / 'out.npy')
    ck = tmp_path / 'ck'
    argv += ['--checkpoint', str(ck), '--report-dir', str(tmp_path / 'rep')]

    stop_after_marks(1)
    with pytest.raises(_Stop):
        main(argv)
    assert json.loads((ck / 'state.json').read_text())['done'] == [0]

    monkeypatch.undo()
    assert main(argv) == 0
    np.testing.assert_array_equal(np.load(tmp_path / 'out.npy'), ref)
    summary = json.loads((tmp_path / 'rep' / 'summary.json').read_text())
    ref_summary = json.loads((tmp_path / 'rep0' / 'summary.json').read_text())
    assert [r['z'] for r in summary['per_slice']] == [0, 1, 2]
    for s in (summary, ref_summary):
        s.pop('output')
        for r in s['per_slice']:
            r.pop('wall_time_s')
    assert summary == ref_summary
    assert json.loads((ck / 'state.json').read_text())['stage'] == 'done'
