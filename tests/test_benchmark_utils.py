"""Tests for benchmark notebook utility helpers."""

import sys
from pathlib import Path

import numpy as np

benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

import benchmark_utils  # noqa: E402 — requires sys.path patched above


def test_results_to_rows_keeps_flat_results_shape():
    results = {
        "case-a": {
            "n_neg_init": 4,
            "n_neg_final": 0,
            "min_jdet_init": -0.25,
            "min_jdet": 0.2,
            "l2_err": 1.23456789,
            "time": 0.3333333,
        }
    }

    rows, cols = benchmark_utils.results_to_rows(results)

    assert cols == [
        "case",
        "n_neg_init",
        "n_neg_final",
        "min_jdet_init",
        "min_jdet",
        "l2_err",
        "time",
    ]
    assert rows == [
        {
            "case": "case-a",
            "n_neg_init": 4,
            "n_neg_final": 0,
            "min_jdet_init": -0.25,
            "min_jdet": 0.2,
            "l2_err": 1.234568,
            "time": 0.333333,
        }
    ]


def test_results_to_rows_flattens_nested_method_results():
    results = {
        10: {
            "n_neg_init": 7,
            "jac_init": np.array([[[-0.5, 0.2], [0.1, 0.3]]]),
            "windowed": {
                "time": 1.23456789,
                "neg": 0,
                "min_jdet": 0.02,
                "l2": 3.456789,
            },
            "fullgrid": {
                "time": 9.87654321,
                "neg": 0,
                "min_jdet": 0.03,
                "l2": 1.234567,
                "n_vars": 200,
            },
        }
    }

    rows, cols = benchmark_utils.results_to_rows(results)

    assert cols == [
        "case",
        "method",
        "n_neg_init",
        "n_neg_final",
        "min_jdet_init",
        "min_jdet",
        "l2_err",
        "time",
    ]
    assert rows == [
        {
            "case": 10,
            "method": "windowed",
            "n_neg_init": 7,
            "n_neg_final": 0,
            "min_jdet_init": -0.5,
            "min_jdet": 0.02,
            "l2_err": 3.456789,
            "time": 1.234568,
        },
        {
            "case": 10,
            "method": "fullgrid",
            "n_neg_init": 7,
            "n_neg_final": 0,
            "min_jdet_init": -0.5,
            "min_jdet": 0.03,
            "l2_err": 1.234567,
            "time": 9.876543,
        },
    ]


# --- cohort loader (data is gitignored; skips cleanly where absent, e.g. CI) ---

import pytest  # noqa: E402


def test_cohort_dir_points_at_repo_data():
    d = benchmark_utils.cohort_dir()
    assert d.name == "brain25_cohort_corrected"
    assert d.parent.name == "dvfs"


def test_list_cohort_returns_pairs_or_empty():
    pairs = benchmark_utils.list_cohort()
    # Always a list; each entry is (brain, variant) with a known variant.
    assert isinstance(pairs, list)
    for b, v in pairs:
        assert isinstance(b, str) and v in benchmark_utils.COHORT_VARIANTS


def test_load_cohort_field_is_dvfopt_native_when_present():
    pairs = benchmark_utils.list_cohort()
    if not pairs:
        pytest.skip("cohort data not present (gitignored)")
    brain, variant = pairs[0]
    phi = benchmark_utils.load_cohort_field(brain, variant)
    assert phi.ndim == 4 and phi.shape[0] == 3  # (3, D, H, W)
    assert np.all(phi[0] == 0)  # dz == 0 (in-plane residual)
