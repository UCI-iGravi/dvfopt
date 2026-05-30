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
