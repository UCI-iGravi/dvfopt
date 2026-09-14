"""Tests for the canonical 2D benchmark driver.

Synthetic only: the gitignored payloads (origins / cohort / ANTs / crops) are
absent in CI and in a fresh worktree, so every real-data path must degrade to an
empty registry with a warning — which is itself asserted here.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

import canonical_2d as c2  # noqa: E402

from tests.conftest import planted_fold  # noqa: E402


def _field(H=10, W=10, seed=0, scale=0.4):
    return c2.as_field(planted_fold(H, W, seed=seed, scale=scale))


# ---------------------------------------------------------------------------
# as_field
# ---------------------------------------------------------------------------


def test_as_field_accepts_both_layouts():
    p2 = planted_fold(6, 7, seed=1)
    f = c2.as_field(p2)
    assert f.shape == (3, 1, 6, 7) and f.dtype == np.float64
    assert np.array_equal(f[0], np.zeros((1, 6, 7)))
    assert np.array_equal(f[1:, 0], p2)
    assert np.array_equal(c2.as_field(f), f)
    assert c2.as_field(np.zeros((3, 5, 4))).shape == (3, 1, 5, 4)
    with pytest.raises(ValueError):
        c2.as_field(np.zeros((4, 5, 6)))


# ---------------------------------------------------------------------------
# The metric block
# ---------------------------------------------------------------------------


def test_certificates_agree_with_fold_stats_on_the_same_maps():
    from dvfopt.core.primitives.tri import tri_areas_flat
    from dvfopt.core.windowed import min_field
    from dvfopt.jacobian.injectivity_radius import cell_min_jdet_2d
    from dvfopt.metrics import fold_stats

    phi = _field()
    dydx = phi[1:, 0]
    got = c2._certificates(phi, c2.THRESHOLD, "init")

    for name, ctor in c2._FAMILIES:
        m = min_field(ctor(shape=dydx.shape[1:]), dydx)
        s = fold_stats(m, c2.THRESHOLD, c2.ERR_TOL)
        assert got[f"{name}_n_neg_init"] == s.n_neg
        assert got[f"{name}_n_below_init"] == s.n_below
        assert got[f"{name}_min_init"] == pytest.approx(s.min_val)

    # the bilinear gauge IS the cell_min_jdet_2d certificate, at the solver's
    # (half-determinant) triangle-area scale
    bil = min_field(c2.SimplexConstraint2DBilinear(shape=dydx.shape[1:]), dydx)[:-1, :-1]
    assert np.allclose(2 * bil, cell_min_jdet_2d(dydx))
    # and the simplex gauge is the per-cell min of the two triangle rows
    H, W = dydx.shape[1:]
    areas = tri_areas_flat(np.concatenate([dydx[0].ravel(), dydx[1].ravel()]), H, W)
    simplex = min_field(c2.SimplexConstraint2D(shape=(H, W)), dydx)[:-1, :-1]
    assert simplex.min() == pytest.approx(areas.min())
    assert got["simplex_min_init"] == pytest.approx(areas.min())
    # strictness ordering (CLAUDE.md): central jdet < finite < simplex
    assert got["jdet_min_init"] >= got["finite_min_init"]
    assert got["finite_min_init"] <= got["simplex_min_init"]


def test_reg_stats_match_a_hand_computation():
    jac = np.array([[1.0, -2.0], [0.5, 0.0]])
    frac, sd = c2._reg_stats(jac)
    assert frac == pytest.approx(0.5)  # -2.0 and 0.0 are <= 0
    hand = np.std(np.log(np.array([1.0, 1e-3, 0.5, 1e-3])))
    assert sd == pytest.approx(hand)


def test_move_stats_match_numpy():
    phi_in = _field(seed=0)
    phi_out = phi_in.copy()
    phi_out[1, 0, 2, 3] += 0.25
    phi_out[2, 0, 5, 5] -= 0.5
    m = c2.metrics(phi_in, phi_out, res=None, elapsed=1.5)
    diff = phi_out - phi_in
    n_px = phi_in.shape[2] * phi_in.shape[3]
    assert m["l1_move"] == pytest.approx(np.abs(diff).sum())
    assert m["l2_move"] == pytest.approx(np.linalg.norm(diff.ravel()))
    assert m["l2_err"] == pytest.approx(m["l2_move"])
    assert m["max_move"] == pytest.approx(0.5)
    assert m["moved_frac"] == pytest.approx(2 / n_px)
    assert m["mean_move_moved"] == pytest.approx(0.75 / 2)
    assert m["time_s"] == pytest.approx(1.5)
    assert m["feasible"] is False and m["damage"] == -1  # no SolveResult -> -1 accounting


def test_metrics_identity_field_is_clean_and_unmoved():
    phi = np.zeros((3, 1, 8, 8))
    m = c2.metrics(phi, phi.copy())
    assert m["n_neg_init"] == 0 and m["n_neg_final"] == 0
    assert m["moved_frac"] == 0.0 and m["l1_move"] == 0.0
    assert m["frac_nonpos_jdet_init"] == 0.0
    assert m["sdlogj_init"] == pytest.approx(0.0)
    assert m["certified"] is True


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_synthetic_registry_enumerates_synthetic_cases():
    from dvfopt.testdata import RANDOM_DVF_CASES, SYNTHETIC_CASES

    got = c2.cases("synthetic")
    ids = [c.id for c in got]
    assert set(SYNTHETIC_CASES) <= set(ids)
    assert set(RANDOM_DVF_CASES) <= set(ids)
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert all(c.source == "synthetic" for c in got)
    assert c2.cases("synthetic", sample="smoke") == got[:2]


def test_synthetic_cases_load_as_fields():
    for case in c2.cases("synthetic", sample="smoke"):
        phi = c2.load_case(case)
        assert phi.ndim == 4 and phi.shape[:2] == (3, 1)
        assert np.array_equal(phi[0], np.zeros_like(phi[0]))


def test_missing_real_data_degrades_to_empty_registry(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(c2, "DVF_ROOT", tmp_path / "nothing")
    monkeypatch.setattr(c2.bu, "list_cohort", list)
    for src in ("origins", "cohort", "ants", "crops"):
        assert c2.cases(src) == []
    assert "WARNING" in capsys.readouterr().out


def test_unknown_source_and_sample_raise():
    with pytest.raises(ValueError):
        c2.cases("nope")
    with pytest.raises(ValueError):
        c2.cases("synthetic", sample="nope")


def test_work_list_applies_the_protocol_source_filter():
    cs = c2.cases("synthetic", sample="smoke")
    assert cs, "synthetic registry is never empty"
    every = {cfg for _, cfg in c2._work_list(["synthetic"], ("slp",), "smoke", False)}
    assert every == {"slp"}  # synthetic is a small source: the whole taxonomy applies
    assert len(c2._work_list(["synthetic"], ("slp", "isqp_none"), "smoke", False)) == 2 * len(cs)

    # a source outside the small set keeps only the two engine rows by default,
    # and exactly what was asked for when the configs are explicit
    assert c2._work_list(["cohort"], ("slp",), "smoke", False) == []
    assert {cfg for _, cfg in c2._work_list(["synthetic"], ("m14",), "smoke", True)} == {"m14"}


# ---------------------------------------------------------------------------
# Writers + end-to-end smoke
# ---------------------------------------------------------------------------


def test_results_csv_leads_with_the_existing_schema(tmp_path):
    recs = [
        {"label": "a__x", "case": "a", "source": "s", "config": "x", "n_neg_init": 1, "extra": 2},
        {"label": "b__x", "case": "b", "source": "s", "config": "x", "n_neg_init": 0, "other": 3},
    ]
    c2._write_results_csv(tmp_path, recs)
    with open(tmp_path / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0][:5] == ["label", "case", "source", "config", "n_neg_init"]
    assert set(rows[0][5:]) == {"extra", "other"}  # union of every row's keys
    assert len(rows) == 3


def test_smoke_run_writes_every_artifact(tmp_path, monkeypatch):
    pytest.importorskip("osqp", reason="the isqp inner needs the [solvers] extra")
    run_dir = tmp_path / "run"
    dvf_root = tmp_path / "dvfs"
    # keep the corrected DVFs inside tmp_path instead of data/dvfs/results/
    monkeypatch.setattr(c2, "DVF_ROOT", dvf_root)
    out = c2.run(
        ["synthetic"],
        ("isqp_none",),
        sample="smoke",
        run_dir=run_dir,
        explicit_configs=True,
        table=True,
    )
    assert out == run_dir
    for name in ("results.csv", "summary.json", "manifest.json", "table.md"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "report" / "report.html").is_file()

    n_cases = len(c2.cases("synthetic", sample="smoke"))
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_cases
    assert {r["config"] for r in rows} == {"isqp_none"}

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["fields"]) == n_cases
    saved = sorted((dvf_root / "results" / run_dir.name / "synthetic").glob("*.npz"))
    assert len(saved) == n_cases
    for entry, path in zip(manifest["fields"], saved):
        assert entry["sha256"] and entry["input_path"] is not None
        with np.load(path) as z:
            assert z["arr"].shape[:2] == (3, 1)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["time_budget_s"] is None
    assert summary["provenance"]["dvfopt_version"]
    assert summary["provenance"]["git_commit"]
    grp = summary["groups"]["synthetic/isqp_none"]
    assert grp["n"] == n_cases
    assert grp["max_damage"] == 0, "the windowed engine must never damage"
    assert grp["certified_rate"] == 1.0, "the robust recipe certifies the 10x10 cases"
    assert "median" in grp["time_s"]


def test_markdown_table_has_one_row_per_group():
    recs = [
        dict(
            source="crops",
            config="isqp_none",
            certified=True,
            feasible=True,
            time_s=1.0,
            l1_move=2.0,
            l2_move=3.0,
            sdlogj_init=0.5,
            sdlogj_final=0.4,
            frac_nonpos_jdet_init=0.1,
            frac_nonpos_jdet_final=0.0,
            damage=0,
        )
    ]
    md = c2.markdown_table(recs)
    assert md.count("\n") == 2  # header + separator + one row
    assert "TUNING SET" in md and "1/1" in md
