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


def test_default_output_is_timestamped_subfolder_of_output(tmp_path, monkeypatch):
    # No out_base given -> a single timestamped subfolder under ./output/.
    monkeypatch.chdir(tmp_path)
    rd = cb.run_cohort_benchmark(
        corrector=lambda p: p, fields={"A": _folded_volume(0, d=3)}, make_figures=False
    )
    assert rd.parent.name == "output"  # timestamped folder sits directly under output/
    assert (tmp_path / rd / "summary.json").is_file()
    assert rd.name.split("_")[0].isdigit()  # folder name is <timestamp>_<run_name>


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


def _folded_volume(seed, d=4):
    """A tiny (3, D, H, W) volume with per-slice folds; dz == 0."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((3, d, 10, 10))
    vol[1:] = rng.normal(0, 3.0, size=(2, d, 10, 10))
    return vol


def test_2d_sections_run(tmp_path, monkeypatch):
    import benchmark_utils as bu

    # Explicit sections load each brain's field once (grouped) via load_cohort_field.
    monkeypatch.setattr(bu, "load_cohort_field", lambda b, variant="x": _folded_volume(0))
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


def test_2d_auto_selects_worst_slices(tmp_path, monkeypatch):
    import benchmark_utils as bu

    monkeypatch.setattr(bu, "load_cohort_field", lambda b, variant="x": _folded_volume(3, d=5))
    rd = cb.run_cohort_2d_sections(
        corrector=lambda s: s,  # no-op
        brains=["B0000"],
        n_worst=2,
        out_base=str(tmp_path / "cohort2d_auto"),
    )
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert summ["n_fields"] == 2  # two worst slices picked


def test_2d_measure_matches_jacobian(tmp_path):
    from dvfopt import jacobian_det2D

    sec = _folded_section(7)
    m = cb._measure_2d(sec, sec, 0.0, 0.01)
    jac = jacobian_det2D(np.stack([sec[1, 0], sec[2, 0]]))
    assert m["n_neg_init"] == int((np.asarray(jac).squeeze() < 0.01).sum())
    assert m["n_neg_init"] == m["n_neg_final"]  # same field in and out


def test_local_cond():
    rng = np.random.default_rng(0)
    dy = rng.normal(0, 0.8, (30, 30))
    dx = rng.normal(0, 0.8, (30, 30))
    cond = cb._local_cond(dy, dx, 0.01)
    assert cond is not None and 1.0 < cond < 1e4  # well-conditioned, finite
    # too-small fields return None (nothing meaningful)
    assert cb._local_cond(dy[:10, :10], dx[:10, :10], 0.01) is None


def test_tri_and_tet_stats():
    sec = _folded_section(3)
    n_tri, min_tri = cb._tri_stats_2d(sec, 0.01)
    assert n_tri > 0 and min_tri <= 0.01
    vol = _folded_volume(3, d=4)
    n_tet, min_tet = cb._tet_stats_3d(vol, 0.01)
    assert n_tet is not None and n_tet > 0
    # oversized volumes are skipped (returns None, None) rather than OOM
    assert cb._tet_stats_3d(vol, 0.01, max_voxels=1) == (None, None)


def test_2d_sections_interactive_report(tmp_path, monkeypatch):
    import benchmark_utils as bu

    monkeypatch.setattr(bu, "load_cohort_field", lambda b, variant="x": _folded_volume(0))
    rd = cb.run_cohort_2d_sections(
        corrector=lambda s: np.zeros_like(s),
        sections=[("B0", 1), ("B0", 2)],
        interactive=True,
        out_base=str(tmp_path / "int2d"),
    )
    doc = (rd / "report.html").read_text(encoding="utf-8")
    assert "data-viewer" in doc and "2-tri" in doc and "jdet_before" in doc
    header = (rd / "results.csv").read_text().splitlines()[0]
    assert "n_tri_init" in header


def test_2d_interactive_with_correspondences(tmp_path, monkeypatch):
    import benchmark_utils as bu

    vol = _folded_volume(0, d=4)
    # synthetic correspondences on slice z=1 (fixed z==1)
    rng = np.random.default_rng(1)
    n = 40
    fy = rng.integers(1, 9, n)
    fx = rng.integers(1, 9, n)
    fp = np.stack([np.ones(n, int), fy, fx], 1)
    mp = fp + rng.integers(-2, 3, (n, 3)) * np.array([0, 1, 1])  # keep z, move y/x
    monkeypatch.setattr(bu, "load_cohort_field", lambda b, variant="x": vol)
    monkeypatch.setattr(bu, "load_cohort_correspondences", lambda b, variant="x": (mp, fp))
    rd = cb.run_cohort_2d_sections(
        corrector=lambda s: s,
        sections=[("B0", 1)],
        interactive=True,
        out_base=str(tmp_path / "corr"),
    )
    doc = (rd / "report.html").read_text(encoding="utf-8")
    assert "data-act=corr" in doc and "corr_fx" in doc and "registration residual" in doc


def test_3d_interactive_report(tmp_path):
    rd = cb.run_cohort_benchmark(
        corrector=lambda p: p,
        fields={"A": _folded_volume(0, d=4), "B": _folded_volume(1, d=4)},
        interactive=True,
        out_base=str(tmp_path / "int3d"),
    )
    doc = (rd / "report.html").read_text(encoding="utf-8")
    assert "data-viewer" in doc and "6-tet" in doc and "worst z" in doc
    assert "n_tet_init" in (rd / "results.csv").read_text().splitlines()[0]


def test_empty_report_is_warn_not_false_success(tmp_path):
    meta = {"corrector": "x", "threshold": 0.01, "generated": "now", "total_time_s": 0.0}
    p = cb.build_cohort_report(tmp_path, meta, [])
    doc = p.read_text(encoding="utf-8")
    assert "No fields processed" in doc
    assert "All 0 fields feasible" not in doc


def test_fmt_handles_non_finite():
    # nan/inf must render as text, never crash the whole report via int(nan).
    assert cb._fmt(float("nan")) == "nan"
    assert cb._fmt(float("inf")) == "inf"


def test_process_section_worker():
    # The parallel worker: solve + measure + figure for one section.
    sec = _folded_section(5)
    brain, z, m, png = cb._process_section(lambda s: np.zeros_like(s), "B0", 3, sec, 0.01, True)
    assert (brain, z) == ("B0", 3)
    assert m["n_neg_init"] > 0 and m["n_neg_final"] == 0
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_jdet2d_corrector_is_picklable():
    # Trusted in-process round-trip (not untrusted data): ProcessPoolExecutor
    # pickles the corrector to each worker, so n_workers>1 needs this to hold.
    import pickle

    c = cb.make_jdet2d_corrector(threshold=0.01)
    assert pickle.loads(pickle.dumps(c)).label == c.label


# Module-level so it is importable/picklable by spawned worker processes.
class _ZeroCorrector:
    label = "zero"

    def __call__(self, section):
        return np.zeros_like(section)


def test_2d_sections_parallel_branch(tmp_path, monkeypatch):
    # Drives the n_workers>1 ProcessPoolExecutor path end-to-end. Field loading
    # happens in the parent (so the monkeypatch applies); workers only receive
    # the picklable corrector + section arrays.
    import benchmark_utils as bu

    monkeypatch.setattr(bu, "load_cohort_field", lambda b, variant="x": _folded_volume(1, d=4))
    rd = cb.run_cohort_2d_sections(
        corrector=_ZeroCorrector(),
        sections=[("B0", 1), ("B0", 2), ("B0", 3)],
        n_workers=2,
        make_figures=False,
        out_base=str(tmp_path / "par"),
    )
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert summ["n_fields"] == 3
    assert summ["total_folds_after"] == 0
    # rows must stay aligned to sections (submission order restored)
    import csv as _csv

    with open(rd / "results.csv") as f:
        labels = [r["label"] for r in _csv.DictReader(f)]
    assert labels == ["B0/z1", "B0/z2", "B0/z3"]
