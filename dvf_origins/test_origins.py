"""Self-check: ``pytest dvf_origins`` (also appended to the CI test command)."""

import csv
import importlib.util
import json

import numpy as np
import pytest

from dvf_origins import CASES, DATA_TOKENS, MECHANISMS, build, learned, real, registered, synthetic
from dvf_origins.__main__ import main
from dvf_origins.morphology import COLUMNS, morphology

SHAPE = (96, 96)
SMALL = [
    (synthetic.interp_sparse, dict(warp_sigma=10, outlier_frac=0.1, n_collapse=2, collapse_size=5)),
    (synthetic.dense_weak_reg, dict(warp_sigma=8, attachment=60)),
    (synthetic.learned_proxy, dict(warp_sigma=8)),
    (synthetic.diffeo_discretized, dict(svf_sigma=4, svf_max=24, n_steps=6, decimate=2)),
]
_HAVE_COHORT = all(
    (real.COHORT / 'B0039' / 'laplacian_exterior' / f).is_file()
    for f in ('ants_warp_0.nii.gz', 'laplacian_deformation_field.npz')
)
_HAVE_BRAIN = registered.FIXED.is_file() and registered.MOVING.is_file()
_HAVE_TORCH = importlib.util.find_spec('torch') is not None


def _check_field(phi):
    assert phi.dtype == np.float64 and phi.ndim == 4 and phi.shape[:2] == (3, 1)
    assert np.isfinite(phi).all()
    assert not phi[0].any(), 'dz must be zero'


@pytest.mark.parametrize('fn,kw', SMALL, ids=[f.__name__ for f, _ in SMALL])
def test_synthetic_convention_determinism_folds(fn, kw):
    phi, meta = fn(shape=SHAPE, seed=1, **kw)
    phi2, _ = fn(shape=SHAPE, seed=1, **kw)
    _check_field(phi)
    np.testing.assert_array_equal(phi, phi2)
    assert meta['source'] == 'synthetic'
    m = morphology(phi)
    assert m['simplex_neg_cells'] > 0, f'{fn.__name__} at its test params should fold'


def test_control_case_is_nearly_clean():
    phi, meta = synthetic.interp_sparse(shape=SHAPE, seed=1, warp_sigma=10)
    corrupt, _ = synthetic.interp_sparse(shape=SHAPE, seed=1, warp_sigma=10, outlier_frac=0.1)
    assert morphology(phi)['simplex_neg_cells'] < morphology(corrupt)['simplex_neg_cells']
    assert meta['n_pts'] + meta['n_duplicate_pins_dropped'] == 200


def test_morphology_identity_and_columns():
    m = morphology(np.zeros((3, 1, 20, 30)))
    assert set(m) == set(COLUMNS)
    assert m['simplex_neg_cells'] == m['jdet_neg_px'] == m['bilinear_neg_cells'] == 0
    # simplex values are raw triangle areas (dvfopt's scale): 0.5 on the identity
    assert m['n_clusters'] == 0 and m['simplex_min'] == 0.5 and m['bilinear_min'] == 1.0
    with pytest.raises(ValueError):
        morphology(np.zeros((3, 2, 20, 30)))


def test_morphology_bilinear_only_cell():
    # Twisted cell: both BL-TR triangles keep positive area (simplex passes)
    # while the bilinear map folds inside the cell (min corner Jdet < 0).
    phi = np.zeros((3, 1, 2, 2))
    phi[1, 0] = [[0.0, 0.7], [-0.1, -0.3]]
    phi[2, 0] = [[0.1, 0.1], [0.0, 1.0]]
    m = morphology(phi)
    assert m['simplex_neg_cells'] == 0 and m['simplex_min'] > 0
    assert m['bilinear_neg_cells'] == 1 and m['bilinear_only_cells'] == 1


def test_registry_names_and_build_synthetic():
    assert {mech for mech, _, _ in CASES.values()} <= set(MECHANISMS)
    for name, (mech, _, _) in CASES.items():  # m<k>_<tool>_<data>_<variant>, k = mechanism
        parts = name.split('_')
        assert parts[0] == f'm{mech}' and len(parts) >= 4, name
        assert parts[2] in DATA_TOKENS, f'{name}: unknown <data> token {parts[2]!r}'
    phi, meta = build('m3_proxy_synthetic_mild')
    _check_field(phi)
    assert meta['case'] == 'm3_proxy_synthetic_mild' and meta['mechanism'] == 3 and meta['proxy']


def test_generate_layout_manifest_and_sweep(tmp_path):
    a, b = 'm3_proxy_synthetic_mild', 'm4_svf_synthetic_decimated'
    main(['generate', '--case', a, '--out', str(tmp_path)])
    main(['generate', '--case', b, '--out', str(tmp_path)])  # a second run must keep a's entry
    assert (tmp_path / 'm3_learned' / f'{a}.npy').is_file()
    assert (tmp_path / 'm3_learned' / f'{a}.json').is_file()
    assert (tmp_path / 'm4_diffeomorphic' / f'{b}.npy').is_file()
    man = json.loads((tmp_path / 'manifest.json').read_text())
    assert set(man) == {a, b}
    assert man[a]['file'] == f'm3_learned/{a}.npy' and man[b]['mechanism'] == 4
    assert man[a]['shape'] == [3, 1, 192, 192]
    # a stray field at a non-registered path is reported, never swept; the documented
    # external drop location is not a stray
    (tmp_path / 'm1_interpolation').mkdir()
    np.save(tmp_path / 'm1_interpolation' / f'{a}.npy', np.zeros((3, 1, 4, 4)))
    (tmp_path / 'external').mkdir()
    np.save(tmp_path / 'external' / 'learned.npy', np.zeros((3, 1, 4, 4)))
    main(['sweep', '--in', str(tmp_path), '--out', str(tmp_path / 'out'), '--latest'])
    (ts_dir,) = [p for p in (tmp_path / 'out').iterdir() if p.is_dir()]
    assert (ts_dir / 'results.csv').read_bytes() == (
        tmp_path / 'out' / 'results_latest.csv'
    ).read_bytes()
    with open(tmp_path / 'out' / 'results_latest.csv') as fh:
        rows = list(csv.DictReader(fh))
    assert {r['case'] for r in rows} == {a, b}
    assert all(int(r['mechanism']) == CASES[r['case']][0] for r in rows)
    assert 'simplex_neg_cells' in rows[0]
    # the manifest is a function of the tree: delete a field, sweep, and it is gone
    (tmp_path / 'm4_diffeomorphic' / f'{b}.npy').unlink()
    main(['sweep', '--in', str(tmp_path), '--out', str(tmp_path / 'out2'), '--no-latest'])
    assert set(json.loads((tmp_path / 'manifest.json').read_text())) == {a}
    assert not (tmp_path / 'out2' / 'results_latest.csv').exists()


def test_migrate_old_names_into_place(tmp_path):
    from dvf_origins import RENAMED
    from dvf_origins.__main__ import migrate

    assert set(RENAMED.values()) <= set(CASES)
    # a pre-layout flat file and an old name inside a mechanism dir both move
    np.save(tmp_path / 'm3_learned_proxy_mild.npy', np.zeros((3, 1, 4, 4)))
    (tmp_path / 'm3_learned_proxy_mild.json').write_text(json.dumps({'case': 'old', 'tool': 't'}))
    (tmp_path / 'm2_dense_optimization').mkdir()
    np.save(tmp_path / 'm2_dense_optimization' / 'm2_tvl1_brainpair.npy', np.zeros((3, 1, 4, 4)))
    assert len(migrate(tmp_path)) == 2
    new = tmp_path / 'm3_learned' / 'm3_proxy_synthetic_mild'
    assert (
        new.with_suffix('.npy').is_file() and not (tmp_path / 'm3_learned_proxy_mild.npy').exists()
    )
    meta = json.loads(new.with_suffix('.json').read_text())
    assert (
        meta['case'] == 'm3_proxy_synthetic_mild'
        and meta['renamed_from'] == 'm3_learned_proxy_mild'
    )
    assert meta['tool'] == 't'
    assert (tmp_path / 'm2_dense_optimization' / 'm2_tvl1_brainpair_a60.npy').is_file()
    assert migrate(tmp_path) == []  # idempotent


def test_slice2d_validation_and_missing_data():
    with pytest.raises(FileNotFoundError):
        real.saved_field('data/origins/external/does_not_exist.npy')
    if not _HAVE_TORCH:  # what `generate` classifies as a clean skip in the main venv
        with pytest.raises(ModuleNotFoundError):
            learned.transmorph(epochs=0)
    with pytest.raises(ValueError):
        real._slice2d(np.zeros((2, 8, 8)), 0)  # a raw (2, H, W) flow is not a field
    with pytest.raises(ValueError):
        real._slice2d(np.zeros((3, 4, 8, 8)), 4)  # z out of range
    phi, dz_max = real._slice2d(np.ones((3, 4, 8, 8)), 2)
    _check_field(phi)
    assert dz_max == 1.0


@pytest.mark.skipif(not _HAVE_BRAIN, reason='mouse_brain images absent (gitignored data)')
def test_registered_pair_and_demons():
    f, m = registered.load_pair(downsample=8)
    assert f.shape == m.shape and f.min() >= 0 and f.max() <= 1
    phi, meta = registered.demons(sigma=0.5, iterations=20, downsample=8)
    _check_field(phi)
    assert phi.shape[-2:] == f.shape and meta['source'] == 'registered'


@pytest.mark.parametrize('fn', [learned.voxelmorph, learned.transmorph], ids=['vxm', 'swin'])
def test_learned_generators_convention(fn):
    pytest.importorskip('torch', reason='learned generators need the torch venv (learned.py)')
    if fn is learned.voxelmorph:
        pytest.importorskip('voxelmorph')
    else:
        pytest.importorskip('timm')
    phi, meta = fn(seed=1, image_size=32, n_train=8, n_test_pairs=1, epochs=1, steps_per_epoch=3)
    _check_field(phi)
    assert phi.shape == (3, 1, 32, 32) and meta['source'] == 'learned'
    # the RETURNED field pull-back-warps the source onto the network's own warped output;
    # the swapped-channel warp must be clearly worse (the field is small but not zero here)
    assert meta['warp_rmse'] < 1e-4
    assert meta['warp_rmse_swapped'] > 10 * meta['warp_rmse']
    with pytest.raises(ValueError):
        fn(n_test_pairs=1, pair=1, epochs=0)  # validated before any training


def test_cohort_data_is_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(learned, 'REGTOOLS_COHORT', tmp_path / 'nowhere')
    with pytest.raises(FileNotFoundError):
        learned.cohort_data(cache=False)
    with pytest.raises(TypeError):
        learned._resolve_data('cohort', 64, 8, 1, 0, 0)  # strings are not accepted


def test_prep_plane_shape_range_and_guards():
    a = np.arange(320 * 456, dtype=np.float64).reshape(320, 456)
    p = learned._prep_plane(a, 3, (96, 128))
    assert p.shape == (96, 128) and p.dtype == np.float32 and p.min() >= 0 and p.max() <= 1
    with pytest.raises(ValueError):
        learned._prep_plane(a, 4, (96, 128))  # 80x114 after /4 is smaller than the crop
    with pytest.raises(ValueError):
        learned._prep_plane(a, 3, (96, 120))  # not a multiple of 32


def test_batches_pairs_brain_major_rows_with_the_template_planes():
    torch = pytest.importorskip('torch')
    n_tgt, n_brains = 5, 3
    tgt = np.arange(n_tgt, dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), np.float32)
    src = np.concatenate([tgt + 100 * b for b in range(n_brains)])
    data_d = dict(train=(src, tgt), test=(src[0], tgt[0]), paired=True, info={})
    sample, _ = learned._batches(data_d, 'cpu', torch)
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        s, t = sample(gen)
        assert float(s.mean()) % 100 == float(t.mean())  # source row i <-> template plane i % 5
    with pytest.raises(ValueError):
        learned._batches(dict(train=(src[:7], tgt), test=data_d['test'], paired=True), 'cpu', torch)


@pytest.mark.skipif(not _HAVE_COHORT, reason='brain cohort absent (gitignored data)')
def test_ants_slice_matches_laplacian_grid_and_is_fold_free():
    ants, meta = real.ants_slice('B0039', 264)
    lap, _ = real.laplacian_slice('B0039', 264)
    _check_field(ants)
    assert ants.shape == lap.shape, 'ANTs slice must land on the Laplacian grid'
    assert morphology(ants)['simplex_neg_cells'] == 0, 'a SyN warp must be fold-free in-plane'
    assert meta['dz_max_dropped'] > 0
