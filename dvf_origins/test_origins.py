"""Self-check: ``pytest dvf_origins`` (also appended to the CI test command)."""

import numpy as np
import pytest

from dvf_origins import CASES, MECHANISMS, build, real, registered, synthetic
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


def test_registry_and_build_synthetic():
    assert {mech for mech, _, _ in CASES.values()} <= set(MECHANISMS)
    phi, meta = build('m3_learned_proxy_mild')
    _check_field(phi)
    assert meta['case'] == 'm3_learned_proxy_mild' and meta['mechanism'] == 3 and meta['proxy']


def test_slice2d_validation_and_missing_data():
    with pytest.raises(FileNotFoundError):
        real.saved_field('data/origins/external/does_not_exist.npy')
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


@pytest.mark.skipif(not _HAVE_COHORT, reason='brain cohort absent (gitignored data)')
def test_ants_slice_matches_laplacian_grid_and_is_fold_free():
    ants, meta = real.ants_slice('B0039', 264)
    lap, _ = real.laplacian_slice('B0039', 264)
    _check_field(ants)
    assert ants.shape == lap.shape, 'ANTs slice must land on the Laplacian grid'
    assert morphology(ants)['simplex_neg_cells'] == 0, 'a SyN warp must be fold-free in-plane'
    assert meta['dz_max_dropped'] > 0
