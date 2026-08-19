"""Tests for correspondence (Laplacian BC) diagnostics + outlier detection."""

import sys
from pathlib import Path

import numpy as np

benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

import correspondence_analysis as ca  # noqa: E402


def _grid_case(h=20, w=20):
    ys, xs = np.meshgrid(np.arange(2, 18, 3), np.arange(2, 18, 3), indexing="ij")
    fy, fx = ys.ravel(), xs.ravel()
    n = len(fy)
    pdy = np.ones(n, dtype=int)  # coherent dy=+1, dx=0
    pdx = np.zeros(n, dtype=int)
    my, mx = fy + pdy, fx + pdx
    fp = np.stack([np.zeros(n, int), fy, fx], 1)
    mp = np.stack([np.zeros(n, int), my, mx], 1)
    # honored field: field[:, fixed] = moving - fixed
    sec = np.zeros((3, 1, h, w))
    sec[1, 0, fy, fx] = my - fy
    sec[2, 0, fy, fx] = mx - fx
    return mp, fp, sec, (fy, fx)


def test_residual_zero_when_field_honors_correspondences():
    mp, fp, sec, _ = _grid_case()
    r = ca.analyze_slice(sec, sec, mp, fp)
    assert r["stats"]["n"] == len(mp)
    assert r["stats"]["mean_resid_before"] < 1e-6  # exactly honored


def test_flags_large_disp_and_high_residual():
    mp, fp, sec, (fy, fx) = _grid_case()
    # plant a large-displacement correspondence
    mp[0, 1] = fp[0, 1] + 15
    sec[1, 0, fy[0], fx[0]] = 15  # keep it honored so it's flagged only as large-disp
    sec_out = sec.copy()
    sec_out[1, 0, fy[1], fx[1]] = -5.0  # break one -> high residual after
    r = ca.analyze_slice(sec, sec_out, mp, fp)
    assert r["stats"]["n_large"] >= 1
    assert r["stats"]["n_high_resid"] >= 1
    assert any("large-disp" in o["types"] for o in r["outliers"])
    assert any("high-residual" in o["types"] for o in r["outliers"])


def test_empty_slice_returns_none():
    empty = np.empty((0, 3), dtype=int)
    assert ca.analyze_slice(np.zeros((3, 1, 5, 5)), np.zeros((3, 1, 5, 5)), empty, empty) is None


def test_slice_correspondences_filters_by_fixed_z():
    fp = np.array([[1, 2, 3], [2, 4, 5], [1, 6, 7]])
    mp = fp + 1
    mps, fps = ca.slice_correspondences(mp, fp, 1)
    assert len(fps) == 2 and set(fps[:, 0]) == {1}
    assert ca.slice_correspondences(None, None, 1) == (None, None)
