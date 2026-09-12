import json

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2DBilinear, SimplexConstraint3D
from dvfopt.core.windowed import min_field, windowed_correct
from dvfopt.objectives import L2Objective
from tests.conftest import planted_fold, planted_fold_3d

pytest.importorskip('osqp')


def _run(phi, c, **kw):
    return windowed_correct(
        phi, 'isqp', constraint=c, objective=L2Objective(), threshold=0.01, verbose=0, **kw
    )


def test_checkpoint_writes_units_and_finished_run_reloads(tmp_path):
    phi = planted_fold(40, 44).astype(np.float64)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    out1, rep1 = _run(phi, c, checkpoint_dir=tmp_path)
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['stage'] == 'done'
    assert any(str(u).startswith('round:') for u in state['done'])
    assert (tmp_path / 'touched.npy').exists()
    out2, rep2 = _run(phi, c, checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(out1, out2)
    assert rep2.resumed_from == 'finished'
    assert rep2.folds_after == rep1.folds_after == 0 and rep2.damage == 0


def test_checkpoint_mismatch_is_refused(tmp_path):
    phi = planted_fold(40, 44).astype(np.float64)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    _run(phi, c, checkpoint_dir=tmp_path)
    with pytest.raises(ValueError, match='does not match'):
        _run(phi, c, checkpoint_dir=tmp_path, max_rounds=3)


def test_resume_after_a_round_keeps_the_invariants(tmp_path):
    """Simulate an interruption after round 1: rewrite the state to 'run' with only
    'round:1' done; the mirror already holds the corrected field, so the resumed run
    re-enters the loop, finds no fold, and finishes with damage 0 against the ORIGINAL."""
    phi = planted_fold(40, 44).astype(np.float64)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    out1, rep1 = _run(phi, c, checkpoint_dir=tmp_path)
    sp = tmp_path / 'state.json'
    state = json.loads(sp.read_text())
    state['stage'] = 'run'
    state['done'] = [u for u in state['done'] if u == 'round:1']
    sp.write_text(json.dumps(state))
    touched_out = np.zeros(phi.shape[1:], bool)
    out2, rep2 = _run(phi, c, checkpoint_dir=tmp_path, touched_out=touched_out)
    assert rep2.resumed_from == 'round:1'
    assert rep2.rounds == 1  # restored from the row; no new round ran on a fold-free field
    assert rep2.folds_after == 0 and rep2.damage == 0
    np.testing.assert_array_equal(out2, out1)
    assert touched_out.any()  # the restored touched mask reached the caller
    assert not rep2.windows  # restore path must not re-solve


def test_budget_expired_run_stays_resumable(tmp_path):
    """R7: a run cut short by ``time_budget_s`` must NOT stamp the checkpoint 'done' —
    otherwise the next call takes the 'finished' branch and returns the UNFINISHED
    field forever. ``time_budget_s=1e-6`` expires before the round loop's first check."""
    phi = planted_fold(40, 44).astype(np.float64)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    _run(phi, c, checkpoint_dir=tmp_path, time_budget_s=1e-6, coarse_to_fine=False)
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['stage'] == 'run'
    out2, rep2 = _run(phi, c, checkpoint_dir=tmp_path)
    assert rep2.folds_after == 0 and rep2.damage == 0
    state2 = json.loads((tmp_path / 'state.json').read_text())
    assert state2['stage'] == 'done'


def test_fold_free_input_checkpoint_round_trips(tmp_path):
    """A fold-free input (with the reseed stage off) never calls ``_mark`` — the round
    loop finds nothing, and mop/reanchor each guard on there being work to do — so
    ``touched.npy`` is never written even though the run completes and is marked
    'done'. Resuming must tolerate the missing file rather than raise."""
    phi = np.zeros((2, 12, 12), dtype=np.float64)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    out1, rep1 = _run(phi, c, checkpoint_dir=tmp_path, reseed_rounds=0)
    assert rep1.folds_after == 0
    assert not (tmp_path / 'touched.npy').exists()
    out2, rep2 = _run(phi, c, checkpoint_dir=tmp_path, reseed_rounds=0)
    assert rep2.resumed_from == 'finished'
    np.testing.assert_array_equal(out2, out1)


def test_touched_out_covers_every_moved_voxel_3d():
    phi = planted_fold_3d(8, 12, 12, depth=1.4)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    touched = np.zeros(phi.shape[1:], bool)
    out, rep = _run(phi, c, touched_out=touched)
    moved = np.any(out != phi, axis=0)
    assert rep.folds_after == 0 and rep.damage == 0
    assert not (moved & ~touched).any()
