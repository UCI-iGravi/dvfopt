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
    assert report['constraint'] == 'simplex' and report['n_neg'] > 0


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


def test_info_ift_check_gates_on_bilinear_cells(tmp_path, capsys):
    # 2tri-feasible but bilinear-folded (fold on the untracked diagonal):
    # --check alone passes; --ift --check fails on the exact cell certificate.
    phi = np.zeros((2, 5, 5))
    phi[0, 2, 2] = -0.7
    phi[1, 2, 2] = +0.7
    p = tmp_path / 'diag.npy'
    np.save(p, phi)
    assert main(['info', str(p), '--check']) == 0
    capsys.readouterr()
    rc = main(['info', str(p), '--ift', '--check'])
    report = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert report['feasible'] is True
    assert report['injectivity']['n_cells_nonpos'] == 1


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


def test_n_workers_defaults_to_none():
    from dvfopt.cli import build_parser

    args = build_parser().parse_args(['correct', 'in.npy', 'out.npy'])
    assert args.n_workers is None
    assert build_parser().parse_args(['correct', 'i', 'o', '--n-workers', '2']).n_workers == 2


def test_correct_slices_n_workers_matches_serial(tmp_path):
    # 3 slices, 16x16 each: below SLPStrategy.cluster_pixel_threshold, so the
    # per-slice solves take the global (pool-free) path and stay quick.
    phi = np.zeros((3, 3, 16, 16))
    phi[1:] = np.stack([planted_fold(16, 16, seed=z, scale=0.4) for z in range(3)], axis=1)
    p = tmp_path / 'vol.npy'
    np.save(p, phi)

    outs, summaries = [], []
    for n in (1, 2):
        out, rep = tmp_path / f'out{n}.npy', tmp_path / f'rep{n}'
        argv = ['correct', str(p), str(out), '--pipeline', 'slices']
        argv += ['--n-workers', str(n), '--report-dir', str(rep)]
        assert main(argv) == 0
        outs.append(np.load(out))
        summaries.append(json.loads((rep / 'summary.json').read_text()))

    np.testing.assert_array_equal(outs[1], outs[0])
    for s in summaries:
        s.pop('output')
        for row in s['per_slice']:
            row.pop('wall_time_s')
    assert summaries[1] == summaries[0]
    assert summaries[0]['final_n_neg'] == 0


def test_correct_25d_pipeline_smoke(tmp_path):
    rng = np.random.default_rng(3)
    phi = np.zeros((3, 3, 8, 8))
    phi[1:] = rng.normal(0, 0.3, size=(2, 3, 8, 8))
    p = tmp_path / 'vol.npy'
    np.save(p, phi)
    out = tmp_path / 'out.npy'
    rc = main(['correct', str(p), str(out), '--pipeline', '25d'])
    assert out.is_file() and rc in (0, 1)  # tiny random volume may hit the geometric floor
