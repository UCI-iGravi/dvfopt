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


def _fake_res(strategy_name, phases=(), extras=None, feasible=True):
    """A stand-in SolveResult/SolveInfo for the accounting tests."""
    from dvfopt.solver import PhaseInfo, SolveInfo, SolveResult

    info = SolveInfo(
        strategy_name=strategy_name,
        phases=[PhaseInfo(name=n, n_iter=i) for n, i in phases],
        total_iter=sum(i for _, i in phases),
        extras=dict(extras or {}),
    )
    return SolveResult(np.zeros((3, 1, 4, 4)), 0, 0.0, 0, 0.0, feasible, 0.0, info)


def test_engine_stats_excludes_nested_giant_phases():
    # a giant phase is nested INSIDE its round, so summing both double-counts
    phases = [("round1", 40), ("giant", 25), ("round2", 10), ("mop", 5), ("final", 0)]
    res = _fake_res(
        "ISQPWindowedStrategy",
        phases,
        dict(damage=0, n_windows=7, giant_regions=1, mop_cleared=2),
    )
    got = c2._engine_stats(res)
    assert got["sqp_iters"] == 55  # 40 + 10 + 5, NOT total_iter's 80
    assert got["rounds"] == 2
    assert (got["damage"], got["n_windows"], got["giant_regions"], got["mop_cleared"]) == (
        0,
        7,
        1,
        2,
    )


def test_engine_stats_are_sentinels_off_the_windowed_engine():
    # barrier logs L-BFGS iterations and slp/m14 log named stages — neither is
    # an SQP count, so every engine column must be -1 rather than a wrong number
    for name, phases in (
        ("BarrierStrategy", [("penalty", 1), ("penalty", 2), ("barrier", 42)]),
        ("SLPStrategy", [("iters", 0), ("converged", 0)]),
        ("SLSQPWindowedStrategy", []),
    ):
        got = c2._engine_stats(_fake_res(name, phases))
        assert got == dict.fromkeys(c2.ENGINE_KEYS, -1), name
    assert c2._engine_stats(None) == dict.fromkeys(c2.ENGINE_KEYS, -1)
    # an `auto` run reports the RESOLVED class, so it still counts as windowed
    assert c2._engine_stats(_fake_res("ISQPWindowedStrategy", [("round1", 3)]))["rounds"] == 1


def test_corr_stats_match_analytic_residuals():
    # two correspondences on an 8x8 slice: fixed -> moving prescribes (dy, dx)
    sec = np.zeros((3, 1, 8, 8))
    fp = np.array([[0, 2, 3], [0, 5, 6]])
    mp = np.array([[0, 4, 3], [0, 5, 9]])  # prescribed (dy,dx) = (2,0) and (0,3)
    sec[1, 0, 2, 3], sec[2, 0, 2, 3] = 2.0, 0.0  # honored exactly -> residual 0
    sec[1, 0, 5, 6], sec[2, 0, 5, 6] = 0.0, 0.0  # missed by (0,3)  -> residual 3
    out = sec.copy()
    out[1, 0, 2, 3] = 5.0  # the correction breaks the first BC by 3
    got = c2._corr_stats(sec, out, (mp, fp))
    assert got["corr_n"] == 2
    assert got["corr_resid_med_init"] == pytest.approx(1.5)  # median of {0, 3}
    assert got["corr_resid_med_final"] == pytest.approx(3.0)  # median of {3, 3}
    assert got["corr_resid_mad_final"] == pytest.approx(0.0)
    assert c2._corr_stats(sec, out, None)["corr_n"] == -1  # sentinel off the cohort


def test_ants_volume_reorients_to_the_cohort_grid(monkeypatch):
    # a warp stored on the permuted grid: its spatial axes are the cohort's
    # (D, H, W) permuted, its channels in the order of its own axes
    monkeypatch.setattr(c2, "COHORT_SHAPE", (4, 3, 5))
    monkeypatch.setattr(c2, "COHORT_VARIANT", "v")
    small = np.zeros((3, 5, 3, 4))
    small[0] = 1.0  # tag the channel that must end up as the new axis-0 component
    monkeypatch.setattr(c2, "load_dvf", lambda p: small)
    c2._ants_volume.cache_clear()
    got = c2._ants_volume("B0000")
    assert got.shape == (3, 4, 3, 5)
    assert np.array_equal(got[2], np.ones((4, 3, 5)))  # channels permuted with the axes
    c2._ants_volume.cache_clear()
    monkeypatch.setattr(c2, "load_dvf", lambda p: np.zeros((3, 9, 9, 9)))
    with pytest.raises(ValueError, match="permutation"):
        c2._ants_volume("B0000")
    c2._ants_volume.cache_clear()


def test_ants_reorientation_keeps_a_fold_free_field_fold_free(monkeypatch):
    # a smooth, fold-free field stays fold-free through the reorientation:
    # a permutation of the axes with the matching component permutation is a
    # relabelling of the grid, not a deformation
    rng = np.random.default_rng(1)
    base = np.zeros((3, 4, 9, 11))
    base[1:] = rng.normal(0, 0.02, (2, 4, 9, 11))
    stored = np.transpose(base, (0, 3, 2, 1))[[2, 1, 0]]  # to the on-disk order
    monkeypatch.setattr(c2, "COHORT_SHAPE", (4, 9, 11))
    monkeypatch.setattr(c2, "load_dvf", lambda p: stored)
    c2._ants_volume.cache_clear()
    got = c2._ants_volume("B0000")
    assert np.allclose(got, base)
    for z in range(4):
        m = c2.metrics(c2.as_field(base[:, z]), c2.as_field(got[:, z]))
        assert m["bilinear_n_below_init"] == 0 and m["bilinear_n_below_final"] == 0
    c2._ants_volume.cache_clear()


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


def test_results_csv_header_leads_with_the_schema_and_rows_stream(tmp_path):
    keys = c2.record_keys()
    assert keys[:7] == c2._IDENT_KEYS
    assert keys[7:17] == c2._SCHEMA_COLS  # cohort_benchmark's schema, in its order
    case = c2.Case(id="a", source="s")
    with c2.results_csv(tmp_path) as append:
        # the header exists BEFORE any row — a crashed chain still leaves a CSV
        with open(tmp_path / "results.csv", newline="", encoding="utf-8") as f:
            assert list(csv.reader(f)) == [list(keys)]
        append(c2.sentinel_record(case, "x", "Boom: nope"))
        append({**c2.sentinel_record(case, "y", ""), "n_neg_init": 7, "stray": 1})
    with open(tmp_path / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["label"] == "a__x" and rows[0]["error"] == "Boom: nope"
    assert rows[1]["n_neg_init"] == "7"
    assert "stray" not in rows[0]  # unknown keys never widen the header mid-stream


def test_sentinel_record_is_a_full_minus_one_row():
    rec = c2.sentinel_record(c2.Case(id="a", source="origins", tool="t"), "isqp_none", "IOError: x")
    assert set(rec) == set(c2.record_keys())
    assert rec["feasible"] is False and rec["certified"] is False and rec["hit_cap"] is False
    assert rec["error"] == "IOError: x" and rec["out_path"] == "" and rec["shape"] == ""
    for k in ("time_s", "n_neg_init", "n_neg_final", "l2_move", "damage", "bilinear_n_below_final"):
        assert rec[k] == -1, k
    # it counts against every rate but contributes to no distribution
    grp = c2._group_summary([rec])
    assert grp["n"] == 1 and grp["certified_rate"] == 0.0 and grp["errors"] == 1
    assert grp["max_damage"] is None and grp["time_s"] is None


def test_load_failure_becomes_a_row(tmp_path, monkeypatch):
    bad = c2.Case(id="ghost", source="origins", path="nowhere.npy", key="file:nowhere.npy")
    monkeypatch.setattr(c2, "DVF_ROOT", tmp_path / "dvfs")
    monkeypatch.setattr(c2, "_work_list", lambda *a, **k: [(bad, "isqp_none")])
    run_dir = tmp_path / "run"
    c2.run(["origins"], ("isqp_none",), run_dir=run_dir, explicit_configs=True)
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # a dropped case is the one outcome the protocol forbids
    assert rows[0]["case"] == "ghost" and rows[0]["feasible"] == "False"
    assert rows[0]["error"] and rows[0]["out_path"] == ""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["fields"]) == 1 and manifest["fields"][0]["file"] == ""
    assert manifest["fields"][0]["error"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    grp = summary["groups"]["origins/isqp_none"]
    assert grp["n"] == 1 and grp["certified_rate"] == 0.0 and grp["max_damage"] is None
    assert any("sentinel" in n for n in summary["notes"])
    assert summary["gauges"]["bilinear"]["scale"].endswith("cell_min_jdet_2d / 2")


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
    lines = md.splitlines()
    assert len(lines) == 4  # gauge legend + header + separator + one row
    assert lines[0].startswith("<!-- certificate gauges:") and "det/2" in lines[0]
    assert "TUNING SET" in md and "1/1" in md
    # a group with no windowed row reads n/a, never "-1"
    md2 = c2.markdown_table([{**recs[0], "damage": -1, "time_s": -1}])
    assert md2.splitlines()[-1].endswith("| n/a |") and "| -1 " not in md2
