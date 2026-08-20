"""Tests for the windowed fold-correction driver (benchmarks/windowed_isqp.py)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

if importlib.util.find_spec("osqp") is None:
    pytest.skip("osqp not installed", allow_module_level=True)

import windowed_isqp as wi  # noqa: E402

from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d  # noqa: E402


def _sparse_folds(H=100, W=100, seed=3):
    """A mostly fold-free field with a few separated sharp fold blobs."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((2, H, W))
    for cy, cx in [(20, 20), (20, 75), (70, 25), (72, 72)]:
        phi[0, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (5, 5))
        phi[1, cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.normal(0, 1.5, (5, 5))
    return phi


def test_patch_matches_global_on_interior_rows():
    """Finite-difference correctness: the patch's Jdet on interior (central-diff)
    rows must equal the global field exactly."""
    rng = np.random.default_rng(0)
    phi = np.stack([rng.normal(0, 0.5, (60, 60)), rng.normal(0, 0.5, (60, 60))])
    jg = _numpy_jdet_2d(phi[0], phi[1])
    sub = wi.build_subproblem("jdet", phi, (20, 30, 25, 38), threshold=0.01)
    py0, py1, px0, px1 = sub.patch_box
    cvals = np.asarray(sub.constraint.values(sub.flat0)).reshape(py1 - py0, px1 - px0)
    diff = np.abs(cvals - jg[py0:py1, px0:px1])
    assert diff[1:-1, 1:-1].max() < 1e-10  # interior rows match global to machine eps


def test_no_damage_and_full_clear():
    """The invariant: windowing clears folds with ZERO damage outside every window."""
    phi = _sparse_folds()
    n0 = int((_numpy_jdet_2d(phi[0], phi[1]) < 0.01).sum())
    out, rep = wi.windowed_correct(phi, family="jdet", objective="l2")
    assert n0 > 0 and rep.folds_before == n0
    assert rep.damage == 0  # no fold created outside any window — the whole point
    assert rep.folds_after == 0  # fully cleared
    assert int((_numpy_jdet_2d(out[0], out[1]) < 0.01).sum()) == 0


def test_no_damage_holds_for_l1():
    phi = _sparse_folds(seed=5)
    _, rep = wi.windowed_correct(phi, family="jdet", objective="l1", eps=1e-2)
    assert rep.damage == 0
