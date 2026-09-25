"""dvfopt.pipeline_pins.correct_dvf_pins on a small synthetic Laplacian field."""

import numpy as np
import pytest

from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.laplacian.solver import solveLaplacianFromCorrespondences
from dvfopt.pipeline_pins import correct_dvf_pins

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason='osqp not installed')

SHAPE = (6, 24, 24)


def _field():
    """Smooth pins (a gentle affine warp + noise), ONE pin 12 px off its
    neighbour one voxel away (contradictory: dropped by the pairwise test) and a
    steep-but-consistent compressive pair (kept; its folds survive the re-fill,
    so the 2D and 2.5D stages have work to do)."""
    rng = np.random.default_rng(0)
    tgt = np.column_stack([rng.integers(2, s - 2, 24) for s in SHAPE]).astype(float)
    tgt = np.unique(tgt, axis=0)
    src = tgt.copy()
    src[:, 1:] += 0.05 * (tgt[:, 1:] - 12) + rng.normal(0, 0.2, (len(tgt), 2))
    bad_t = tgt[0] + [0, 0, 1]
    bad_s = src[0] + [0, 0, 1] + [0, 12.0, 0]
    steep_t = np.array([[3, 12, 10], [3, 12, 13]], float)  # 3 px apart, 2.8 px squeeze
    steep_s = steep_t + np.array([[0, 0, 1.4], [0, 0, -1.4]])
    tgt = np.vstack([tgt, bad_t, steep_t])
    src = np.vstack([src, bad_s, steep_s])
    phi = solveLaplacianFromCorrespondences(
        SHAPE, src, tgt, rtol=1e-8, maxiter=5000, log_fn=lambda _m: None
    )
    return phi, tuple(bad_t.astype(int))


def test_rejects_bad_input():
    phi, _ = _field()
    with pytest.raises(ValueError, match='shape'):
        correct_dvf_pins(phi[:, 0])
    phi[0, 1, 1, 1] = 0.5
    with pytest.raises(ValueError, match='dz'):
        correct_dvf_pins(phi)


def test_preprocessing_only_drops_the_contradictory_pin():
    phi, bad = _field()
    out, rep = correct_dvf_pins(phi, run_2d=False, run_25d=False)
    assert rep.n_below_in > rep.n_below_out > 0  # the steep pair's folds are left
    assert 1 <= rep.n_dropped <= 2 and rep.n_kept == rep.n_pins - rep.n_dropped
    assert abs(out[1][bad] - phi[1][bad]) > 5  # the 12 px spike was re-filled away
    assert [s for s, _ in rep.stages] == ['refill', 'census']


@needs_osqp
def test_chain_certifies_and_resumes(tmp_path):
    phi, _ = _field()
    before = phi.copy()
    out, rep = correct_dvf_pins(phi, checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(phi, before)  # input never mutated
    assert rep.feasible and rep.n_below_out == 0 and rep.n_neg_best_diag_out == 0
    assert rep.min_T_out >= 0.01 and rep.n_dropped >= 1
    assert np.all(out[0] == 0)
    again, rep2 = correct_dvf_pins(phi, checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(again, out)
    assert rep2.n_pins == rep.n_pins and rep2.feasible


@needs_osqp
def test_cli_pins_route(tmp_path):
    import json

    from dvfopt.cli import main

    phi, _ = _field()
    p, out, rep_dir = tmp_path / 'in.npy', tmp_path / 'out.npy', tmp_path / 'rep'
    np.save(p, phi)
    rc = main(['correct', str(p), str(out), '--pipeline', 'pins', '--report-dir', str(rep_dir)])
    assert rc == 0 and out.is_file()
    summary = json.loads((rep_dir / 'summary.json').read_text(encoding='utf-8'))
    assert summary['pipeline'] == 'pins' and summary['feasible'] and summary['n_dropped'] >= 1
