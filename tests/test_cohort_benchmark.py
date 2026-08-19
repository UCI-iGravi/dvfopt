"""Tests for the cohort correction benchmark runner + HTML report.

Uses the ``fields=`` bypass with tiny synthetic volumes, so these run in CI
without the gitignored 17 GB cohort data.
"""

import json
import sys
from pathlib import Path

import numpy as np

benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

import cohort_benchmark as cb  # noqa: E402


def _folded(seed):
    rng = np.random.default_rng(seed)
    phi = np.zeros((3, 3, 10, 10))
    phi[1:] = rng.normal(0, 3.0, size=(2, 3, 10, 10))  # in-plane displacement, dz stays 0
    return phi


def test_run_writes_expected_artifacts(tmp_path):
    rd = cb.run_cohort_benchmark(
        corrector=lambda p: p,  # no-op
        fields={"A": _folded(0), "B": _folded(1)},
        run_name="t",
        out_base=str(tmp_path / "cohort"),
    )
    assert (rd / "results.csv").is_file()
    assert (rd / "summary.json").is_file()
    assert (rd / "report.html").is_file()
    figs = list((rd / "figures").glob("*.png"))
    assert len(figs) == 2


def test_summary_and_report_content(tmp_path):
    rd = cb.run_cohort_benchmark(
        corrector=lambda p: p,
        fields={"A": _folded(0), "B": _folded(1)},
        out_base=str(tmp_path / "cohort"),
    )
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert summ["n_fields"] == 2
    # no-op corrector leaves folds untouched
    assert summ["total_folds_after"] == summ["total_folds_before"] > 0
    doc = (rd / "report.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in doc
    assert "Cohort Correction Report" in doc
    assert "data:image/png;base64," in doc  # figures embedded, self-contained


def test_corrector_that_zeroes_is_feasible(tmp_path):
    rd = cb.run_cohort_benchmark(
        corrector=lambda p: np.zeros_like(p),  # identity field: zero folds
        fields={"A": _folded(2)},
        out_base=str(tmp_path / "cohort"),
    )
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert summ["n_feasible"] == 1
    assert summ["total_folds_after"] == 0


def _folded_section(seed):
    """A tiny (3, 1, H, W) folded 2D section (dz == 0)."""
    rng = np.random.default_rng(seed)
    sec = np.zeros((3, 1, 10, 10))
    sec[1:] = rng.normal(0, 3.0, size=(2, 1, 10, 10))
    return sec


def test_2d_sections_run(tmp_path, monkeypatch):
    import benchmark_utils as bu

    monkeypatch.setattr(bu, "load_cohort_section", lambda b, z, variant="x": _folded_section(z))
    rd = cb.run_cohort_2d_sections(
        corrector=lambda s: np.zeros_like(s),  # identity: zero folds
        sections=[("B0000", 1), ("B0000", 2)],
        out_base=str(tmp_path / "cohort2d"),
    )
    assert (rd / "results.csv").is_file()
    assert (rd / "report.html").is_file()
    assert len(list((rd / "figures").glob("B0000*.png"))) == 2
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert summ["n_fields"] == 2
    assert summ["total_folds_after"] == 0  # zeroed sections are fold-free
    assert summ["total_folds_before"] > 0


def test_2d_measure_matches_jacobian(tmp_path):
    from dvfopt import jacobian_det2D

    sec = _folded_section(7)
    m = cb._measure_2d(sec, sec, 0.0, 0.01)
    jac = jacobian_det2D(np.stack([sec[1, 0], sec[2, 0]]))
    assert m["n_neg_init"] == int((np.asarray(jac).squeeze() < 0.01).sum())
    assert m["n_neg_init"] == m["n_neg_final"]  # same field in and out
