"""Tests for the residual-escape modes (benchmarks/windowed_escape.py).

The distance-minimising windowed solve can leave a fold in an objective-basin trap;
the escape modes clear it within a single objective. Every mode must (a) reach 0 folds
and (b) preserve the no-damage invariant (damage == 0).
"""

import sys
from pathlib import Path

import numpy as np

benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

import pytest  # noqa: E402
import windowed_escape as we  # noqa: E402
from windowed_isqp import min_field  # noqa: E402


def _one_inverted_cell(h=40, w=40):
    """A smooth (feasible) field with a single deeply-inverted interior cell: move one
    corner node across its cell so the 2-tri area flips negative, leaving the rest
    fold-free — the sparse-residual regime the escape modes target."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dy = 0.25 * np.sin(xx / 4.0)
    dx = 0.25 * np.cos(yy / 4.0)
    dy[21, 21] -= 3.0  # push node (21,21) far across cell (20,20) -> inverted quad
    dx[21, 21] -= 3.0
    return np.stack([dy, dx])


def test_fixture_has_isolated_fold():
    phi = _one_inverted_cell()
    n = int((min_field("2tri", phi) < 0.01).sum())
    assert 0 < n < 10  # a handful of folded cells, not a dense field


@pytest.mark.parametrize("mode", ["twophase", "weighted", "penalty"])
def test_escape_mode_clears_and_no_damage(mode):
    phi = _one_inverted_cell()
    before = int((min_field("2tri", phi) < 0.01).sum())
    out, rep = we.repair_residuals(phi, family="2tri", threshold=0.01, mode=mode, maxiter=400)
    assert rep["folds_before"] == before > 0
    assert rep["folds_after"] == 0, f"{mode} left {rep['folds_after']} folds"
    assert rep["damage"] == 0, f"{mode} violated no-damage: {rep['damage']}"


def test_unknown_mode_raises():
    phi = _one_inverted_cell()
    with pytest.raises(ValueError):
        we.repair_residuals(phi, mode="nope")


def test_repair_is_noop_on_clean_field():
    """A field with no folds is returned unchanged (no windows, no move)."""
    yy, xx = np.mgrid[0:30, 0:30].astype(float)
    phi = np.stack([0.1 * np.sin(xx / 5.0), 0.1 * np.cos(yy / 5.0)])
    assert int((min_field("2tri", phi) < 0.01).sum()) == 0
    out, rep = we.repair_residuals(phi, family="2tri", mode="weighted")
    assert rep["n_windows"] == 0 and rep["folds_after"] == 0
    assert np.array_equal(out, phi)
