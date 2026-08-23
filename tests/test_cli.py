"""CLI tests — drive dvfopt.cli.main() directly (no subprocess)."""

import json

import numpy as np
import pytest

from dvfopt.cli import _parse_params, main
from tests.conftest import planted_fold


def _save_folded(tmp_path, name='in.npy'):
    phi = planted_fold(10, 10, seed=0, scale=0.4)
    p = tmp_path / name
    np.save(p, phi)
    return p, phi


def test_parse_params_literal_eval():
    got = _parse_params(['a=1', 'b=0.5', 'c=x', 'd=True'])
    assert got == {'a': 1, 'b': 0.5, 'c': 'x', 'd': True}


def test_parse_params_rejects_bare_key():
    with pytest.raises(SystemExit):
        _parse_params(['oops'])


def test_info_check_flags_folds(tmp_path, capsys):
    p, _ = _save_folded(tmp_path)
    rc = main(['info', str(p), '--check'])
    report = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert report['constraint'] == '2tri' and report['n_neg'] > 0


def test_info_feasible_field(tmp_path, capsys):
    p = tmp_path / 'flat.npy'
    np.save(p, np.zeros((2, 8, 8)))
    rc = main(['info', str(p), '--check'])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)['feasible'] is True


def test_info_ift_diagnostics(tmp_path, capsys):
    p, _ = _save_folded(tmp_path)
    rc = main(['info', str(p), '--ift'])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    inj = report['injectivity']
    # A 2tri-folded field must also fail the bilinear corner certificate
    # (each tracked triangle is one of the cell's corner Jdets).
    assert inj['cell_min_jdet'] <= 0 and inj['n_cells_nonpos'] >= 1
    assert inj['min_radius'] < 1.0


def test_correct_roundtrip_with_report(tmp_path, capsys):
    p, _ = _save_folded(tmp_path)
    out = tmp_path / 'out.npy'
    rep = tmp_path / 'rep'
    rc = main(['correct', str(p), str(out), '--report-dir', str(rep)])
    assert rc == 0
    capsys.readouterr()  # drop the correct-outcome line
    assert main(['info', str(out), '--check']) == 0
    summary = json.loads((rep / 'summary.json').read_text())
    assert summary['feasible'] is True and summary['final_n_neg'] == 0
    assert (rep / 'convergence.png').is_file()


def test_correct_unsupported_format(tmp_path):
    bad = tmp_path / 'in.txt'
    bad.write_text('nope')
    assert main(['correct', str(bad), str(tmp_path / 'out.npy')]) == 2


def test_correct_missing_input(tmp_path):
    assert main(['info', str(tmp_path / 'ghost.npy')]) == 2


def test_correct_slices_pipeline(tmp_path):
    rng = np.random.default_rng(5)
    phi = np.zeros((3, 2, 10, 10))
    phi[1:] = rng.normal(0, 0.4, size=(2, 2, 10, 10))
    p = tmp_path / 'vol.npy'
    np.save(p, phi)
    out = tmp_path / 'out.npy'
    rc = main(['correct', str(p), str(out), '--pipeline', 'slices'])
    assert rc == 0
    corrected = np.load(out)
    assert corrected.shape == phi.shape
    np.testing.assert_array_equal(corrected[0], phi[0])  # dz untouched


def test_correct_25d_pipeline_smoke(tmp_path):
    rng = np.random.default_rng(3)
    phi = np.zeros((3, 3, 8, 8))
    phi[1:] = rng.normal(0, 0.3, size=(2, 3, 8, 8))
    p = tmp_path / 'vol.npy'
    np.save(p, phi)
    out = tmp_path / 'out.npy'
    rc = main(['correct', str(p), str(out), '--pipeline', '25d'])
    assert out.is_file() and rc in (0, 1)  # tiny random volume may hit the geometric floor
