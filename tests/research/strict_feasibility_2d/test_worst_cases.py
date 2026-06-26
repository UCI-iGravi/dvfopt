"""Smoke-test the worst-case loaders."""
import numpy as np
import pytest

from research.strict_feasibility_2d.worst_cases import _load


def test_load_synthetic_canonical():
    cases = _load.load_synthetic_canonical()
    keys = {name for name, phi, meta in cases}
    assert 'bowtie_7x7_shoelace' in keys
    assert '03c_20x20_opposite' in keys
    assert '03d_20x20_crossing' in keys


def test_synthetic_shapes():
    for name, phi, meta in _load.load_synthetic_canonical():
        assert phi.ndim == 3 and phi.shape[0] == 2, name
        assert phi.dtype == np.float64, name


def test_load_b0039_z12():
    name, phi, meta = _load.load_b0039_slice(12)
    assert name == 'b0039_z012'
    assert phi.shape == (2, 320, 456)
    assert meta['init_n_neg'] > 0


def test_b0039_load_invalid_z_raises():
    with pytest.raises(IndexError):
        _load.load_b0039_slice(99999)


def test_load_synthetic_includes_adversarial():
    cases = _load.load_synthetic_canonical()
    keys = {name for name, phi, meta in cases}
    assert 'dense_bowtie_cluster_15x15' in keys
    assert 'tiny_margin_10x10' in keys
