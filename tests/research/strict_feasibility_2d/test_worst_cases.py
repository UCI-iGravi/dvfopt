"""Smoke-test the worst-case loaders."""

import numpy as np
import pytest

from research.strict_feasibility_2d.worst_cases import _load

# These loaders read large DVF data under data/dvfs/ (and a local synthetic/
# dir) that isn't committed and is absent in CI. Skip cleanly when the data
# isn't present; the tests still run wherever it is (e.g. local dev).
requires_canonical = pytest.mark.skipif(
    not _load._CANONICAL_DIR.exists(),
    reason='canonical 2-tri DVFs not present (data/dvfs/canonical_2tri_2d/)',
)
requires_adversarial = pytest.mark.skipif(
    not _load._ADVERSARIAL_DIR.exists(),
    reason='adversarial synthetic DVFs not present',
)
requires_b0039 = pytest.mark.skipif(
    not _load._B0039.exists(),
    reason='B0039 DVF not present (data/dvfs/b0039/)',
)


@requires_canonical
def test_load_synthetic_canonical():
    cases = _load.load_synthetic_canonical()
    keys = {name for name, phi, meta in cases}
    assert 'bowtie_7x7_shoelace' in keys
    assert '03c_20x20_opposite' in keys
    assert '03d_20x20_crossing' in keys


def test_synthetic_shapes():
    for name, phi, _meta in _load.load_synthetic_canonical():
        assert phi.ndim == 3 and phi.shape[0] == 2, name
        assert phi.dtype == np.float64, name


@requires_b0039
def test_load_b0039_z12():
    name, phi, meta = _load.load_b0039_slice(12)
    assert name == 'b0039_z012'
    assert phi.shape == (2, 320, 456)
    assert meta['init_n_neg'] > 0


@requires_b0039
def test_b0039_load_invalid_z_raises():
    with pytest.raises(IndexError):
        _load.load_b0039_slice(99999)


@requires_adversarial
def test_load_synthetic_includes_adversarial():
    cases = _load.load_synthetic_canonical()
    keys = {name for name, phi, meta in cases}
    assert 'dense_bowtie_cluster_15x15' in keys
    assert 'tiny_margin_10x10' in keys
