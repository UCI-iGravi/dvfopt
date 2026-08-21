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


@pytest.mark.parametrize("family", ["jdet", "2tri"])
def test_schwarz_tiling_clears_giant_connected_region(family):
    """A large CONNECTED fold region exceeds max_window_area and is cleared by
    overlapping-tile Schwarz decomposition — still with zero damage."""
    rng = np.random.default_rng(7)
    H = W = 140
    phi = np.zeros((2, H, W))
    phi[0, 35:105, 35:105] = rng.normal(0, 1.2, (70, 70))
    phi[1, 35:105, 35:105] = rng.normal(0, 1.2, (70, 70))
    _, rep = wi.windowed_correct(phi, family=family, objective="l2")
    assert rep.giant_regions >= 1  # the region tripped the tiler
    assert rep.damage == 0
    assert rep.folds_after == 0  # tiling fully cleared it


@pytest.mark.parametrize("objective", ["l2", "l1"])
def test_2tri_no_damage_and_full_clear(objective):
    """The 2-triangle metric (cell grid, exact areas, ring=1) clears folds with
    zero damage, same invariant as Jdet."""
    phi = _sparse_folds(seed=3)
    n0 = int((wi.min_field("2tri", phi) < 0.01).sum())
    _, rep = wi.windowed_correct(phi, family="2tri", objective=objective, eps=1e-2)
    assert n0 > 0
    assert rep.damage == 0
    assert rep.folds_after == 0


@pytest.mark.parametrize("objective", ["l2", "l1"])
def test_finite_no_damage_and_full_clear(objective):
    """The forward-diff Jdet metric (cell grid, exact det, ring=1, one row/cell)
    clears folds with zero damage, same invariant as 2tri/Jdet."""
    phi = _sparse_folds(seed=3)
    n0 = int((wi.min_field("finite", phi) < 0.01).sum())
    _, rep = wi.windowed_correct(phi, family="finite", objective=objective, eps=1e-2)
    assert n0 > 0
    assert rep.damage == 0
    assert rep.folds_after == 0


@pytest.mark.parametrize("inner", ["scipy-slsqp", "scipy-slsqp+trust-constr"])
def test_pluggable_inner_preserves_no_damage(inner):
    """Swapping the inner solver must not break the no-damage invariant: every inner
    only moves the window's free pixels, so folds outside every window stay 0."""
    phi = _sparse_folds()
    _, rep = wi.windowed_correct(phi, family="jdet", objective="l2", inner=inner)
    assert rep.damage == 0


def test_border_folds_are_corrected():
    """Folds ON the image border must be fixed, not frozen. find_windows must not
    inset a side that reached the image edge (regression: it used to freeze the
    ring-wide border band, silently leaving border folds uncorrected)."""
    rng = np.random.default_rng(11)
    H = W = 60
    phi = np.zeros((2, H, W))
    phi[0, 0:5, 20:32] += rng.normal(0, 1.5, (5, 12))  # cluster touching top border
    phi[1, 0:5, 20:32] += rng.normal(0, 1.5, (5, 12))
    assert (wi.min_field("jdet", phi)[0] < 0.01).any()  # folds ON row 0 exist
    _, rep = wi.windowed_correct(phi, family="jdet", objective="l2")
    assert rep.damage == 0
    assert rep.folds_after == 0  # border folds cleared, not left frozen


def test_mop_only_helps_and_never_damages():
    """The terminal mop pass (large-margin re-window of the round-loop residual) must
    never increase folds or cause damage — it only clears the boundary-stuck residual
    a small window can't. Verified against the same field with the mop disabled."""
    rng = np.random.default_rng(21)
    H = W = 120
    phi = np.zeros((2, H, W))
    phi[0, 30:90, 30:90] = rng.normal(0, 2.2, (60, 60))  # large, high-amplitude region
    phi[1, 30:90, 30:90] = rng.normal(0, 2.2, (60, 60))
    _, no_mop = wi.windowed_correct(phi, family="jdet", objective="l2", mop_margin=0)
    _, with_mop = wi.windowed_correct(phi, family="jdet", objective="l2")  # default mop=25
    assert no_mop.damage == 0 and with_mop.damage == 0
    assert with_mop.folds_after <= no_mop.folds_after  # the mop only helps


def test_no_damage_on_severe_field():
    """The no-damage invariant holds even on a severe field where some windows may
    not fully clear — damage must be 0 regardless of feasibility (touched covers
    the full enforced footprint, not just the free box)."""
    rng = np.random.default_rng(13)
    H = W = 90
    phi = np.zeros((2, H, W))
    for cy, cx in [(25, 25), (25, 65), (65, 30), (66, 66)]:
        phi[0, cy - 3 : cy + 4, cx - 3 : cx + 4] += rng.normal(0, 4.0, (7, 7))  # amp 4 = hard
        phi[1, cy - 3 : cy + 4, cx - 3 : cx + 4] += rng.normal(0, 4.0, (7, 7))
    _, rep = wi.windowed_correct(phi, family="jdet", objective="l2")
    assert rep.damage == 0
