"""Tests for the SQP-solver comparison harness (benchmarks/slsqp_variants.py)."""

import sys
from pathlib import Path

import numpy as np

benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

import slsqp_variants as sv  # noqa: E402


def _folded(h=20, w=20, seed=0, scale=0.8):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, scale, (h, w)), rng.normal(0, scale, (h, w))])


def test_available_solvers_always_has_scipy():
    av = sv.available_solvers()
    assert "scipy-slsqp" in av and "scipy-trust-constr" in av


def test_scipy_slsqp_reduces_true_folds():
    from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d

    patch = _folded()
    n0 = int((_numpy_jdet_2d(patch[0], patch[1]) < 0.0).sum())
    out, info = sv.full_grid_correct(patch, "scipy-slsqp", maxiter=100)
    assert info["folds_before"] > 0
    # eliminates all strictly-negative determinants
    assert int((_numpy_jdet_2d(out[0], out[1]) < 0.0).sum()) == 0
    assert n0 > 0


def test_nlopt_matches_scipy_when_available():
    if "nlopt-slsqp" not in sv.available_solvers():
        import pytest

        pytest.skip("nlopt not installed")
    patch = _folded(seed=1)
    xs, _ = sv.full_grid_correct(patch, "scipy-slsqp", maxiter=100)
    xn, _ = sv.full_grid_correct(patch, "nlopt-slsqp", maxiter=100)
    # same Kraft algorithm -> agree to solver tolerance
    assert np.abs(xs - xn).max() < 1e-3
