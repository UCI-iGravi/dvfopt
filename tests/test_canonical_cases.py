"""Smoke tests for the canonical 2D 2-triangle benchmark suite.

The six cases promoted from notebook 14 are now the cross-solver
feasibility baseline; this test pins their stats so any change to the
correspondence-based laplacian interpolation that silently shifts the
init fold count will fail loudly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from dvfopt.testdata import (
    CANONICAL_2TRI_2D_KEYS,
    canonical_2tri_2d,
)

# The init-fold stats are pinned to what notebook 14 reports. If
# anyone changes the Laplacian solver / interpolation convention and
# the numbers shift, this test breaks loudly.
EXPECTED = {
    '01a_10x10_crossing': dict(
        shape=(10, 10), init_n_neg=24, init_min_T_lo=-0.75, init_min_T_hi=-0.74
    ),
    '01b_10x10_opposite': dict(
        shape=(10, 10), init_n_neg=10, init_min_T_lo=-0.59, init_min_T_hi=-0.58
    ),
    '03a_10x10_opposite': dict(
        shape=(10, 10), init_n_neg=23, init_min_T_lo=-0.82, init_min_T_hi=-0.81
    ),
    '03b_10x10_crossing': dict(
        shape=(10, 10), init_n_neg=28, init_min_T_lo=-0.71, init_min_T_hi=-0.69
    ),
    '03c_20x20_opposite': dict(
        shape=(20, 20), init_n_neg=58, init_min_T_lo=-0.81, init_min_T_hi=-0.80
    ),
    '03d_20x20_crossing': dict(
        shape=(20, 20), init_n_neg=72, init_min_T_lo=-0.75, init_min_T_hi=-0.73
    ),
}


def test_canonical_keys_match_expected():
    assert tuple(CANONICAL_2TRI_2D_KEYS) == tuple(EXPECTED.keys())


@pytest.mark.parametrize('expected_key', list(EXPECTED.keys()))
def test_canonical_case_stats(expected_key):
    cases = {name: (phi, meta) for (name, phi, meta) in canonical_2tri_2d()}
    phi, meta = cases[expected_key]
    exp = EXPECTED[expected_key]
    assert phi.shape == (2,) + exp['shape']
    assert phi.dtype == np.float64
    assert meta['shape'] == exp['shape']
    assert meta['init_n_neg'] == exp['init_n_neg']
    assert exp['init_min_T_lo'] <= meta['init_min_T'] <= exp['init_min_T_hi']


def test_canonical_no_meta_shape():
    pairs = canonical_2tri_2d(with_meta=False)
    assert len(pairs) == len(CANONICAL_2TRI_2D_KEYS)
    for item in pairs:
        assert len(item) == 2
        _name, phi = item
        assert phi.ndim == 3 and phi.shape[0] == 2


def test_benchmark_harness_identity_fn_reports_failure():
    """A no-op method that returns input unchanged should be reported
    as infeasible on every canonical case (since they all start folded)."""
    benchmarks_dir = Path(__file__).resolve().parents[1] / 'benchmarks'
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))
    import benchmark_utils

    rows = benchmark_utils.benchmark_canonical_2tri_2d(
        lambda phi: phi, label='identity', verbose=False
    )
    assert len(rows) == 6
    assert all(not r['feasible'] for r in rows)
    # No-op should preserve init stats.
    for r in rows:
        assert r['final_n_neg'] == r['init_n_neg']
        assert r['l1'] == 0.0


def test_benchmark_harness_catches_exceptions():
    benchmarks_dir = Path(__file__).resolve().parents[1] / 'benchmarks'
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))
    import benchmark_utils

    def broken(phi):
        raise RuntimeError('intentional')

    rows = benchmark_utils.benchmark_canonical_2tri_2d(broken, label='broken', verbose=False)
    assert all(r['error'].startswith('RuntimeError') for r in rows)
    assert all(not r['feasible'] for r in rows)
