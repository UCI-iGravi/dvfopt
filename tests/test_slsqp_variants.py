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


def test_pyslsqp_eliminates_folds_when_available():
    if "pyslsqp" not in sv.available_solvers():
        import pytest

        pytest.skip("pyslsqp not installed (needs a py<=3.12 wheel)")
    from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d

    patch = _folded(seed=1)
    out, _ = sv.full_grid_correct(patch, "pyslsqp", maxiter=100)
    assert int((_numpy_jdet_2d(out[0], out[1]) < 0.0).sum()) == 0


def test_isqp_proto_eliminates_folds_when_available():
    if "isqp-proto" not in sv.available_solvers():
        import pytest

        pytest.skip("quadprog not installed")
    from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d

    # The elastic-QP prototype reaches feasibility on a small dense field without
    # SLSQP's bouncing (its whole point). Small size keeps the dense QP fast.
    patch = _folded(h=14, w=14, seed=2)
    out, info = sv.full_grid_correct(patch, "isqp-proto", maxiter=60)
    assert info["success"]  # no strictly-negative determinants remain
    assert int((_numpy_jdet_2d(out[0], out[1]) < 0.0).sum()) == 0


def test_isqp_osqp_eliminates_folds_when_available():
    if "isqp-osqp" not in sv.available_solvers():
        import pytest

        pytest.skip("osqp not installed")
    from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d

    # The optimized (sparse OSQP + CPR-coloured Jacobian + warm-start) variant.
    patch = _folded(h=16, w=16, seed=3)
    out, info = sv.full_grid_correct(patch, "isqp-osqp", maxiter=80)
    assert info["success"]
    assert int((_numpy_jdet_2d(out[0], out[1]) < 0.0).sum()) == 0


def test_colored_jacobian_matches_dense():
    """CPR-coloured Jacobian must equal the dense adjoint build (stride-3 is exact
    for the radius-1 Jdet stencil)."""
    from dvfopt.constraints import JdetConstraint2D

    h = w = 12
    c = JdetConstraint2D(shape=(h, w))
    rng = np.random.default_rng(4)
    flat0 = rng.normal(0, 0.5, c.n_variables)
    coloring = sv.jacobian_coloring(c, flat0)
    for _ in range(3):
        x = flat0 + rng.normal(0, 0.3, flat0.size)
        dense = sv.dense_jacobian(c, x)
        colored = sv.colored_jacobian(c, x, *coloring).toarray()
        assert np.abs(dense - colored).max() < 1e-10
