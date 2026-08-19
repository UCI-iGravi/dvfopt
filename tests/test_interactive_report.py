"""Tests for the interactive HTML report module (pure-Python pieces)."""

import base64
import sys
from pathlib import Path

import numpy as np

benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

import interactive_report as ir  # noqa: E402


def test_b64_floats_roundtrips_float16():
    # Embedded field arrays are display-only float16 (halves report size).
    a = np.array([[1.5, -2.0], [0.01, 300.0]], dtype=np.float64)
    back = np.frombuffer(base64.b64decode(ir.b64_floats(a)), dtype="<f2").reshape(a.shape)
    assert np.allclose(a, back, atol=0.2)  # ~3-4 sig digits
    assert len(base64.b64decode(ir.b64_floats(a))) == a.size * 2  # 2 bytes/value


def test_b64_floats_clips_instead_of_overflowing_to_inf():
    big = np.array([70000.0, -70000.0])
    back = np.frombuffer(base64.b64decode(ir.b64_floats(big)), dtype="<f2")
    assert np.isfinite(back).all()  # clipped to +-65504, not inf


def test_b64_uint16_coords_are_exact():
    # Correspondence coords must be exact (float16 would snap >= 2048).
    coords = np.array([0, 456, 2049, 5000, 60000], dtype=np.float32)
    back = np.frombuffer(base64.b64decode(ir.b64_uint16(coords)), dtype="<u2")
    assert np.array_equal(back, coords.astype(np.uint16))


def test_fold_clusters_2d_ranks_by_severity():
    jac = np.ones((12, 12))
    jac[2:5, 2:5] = -3.0  # big, deep cluster
    jac[9, 9] = -0.2  # tiny shallow cluster
    cl = ir.fold_clusters_2d(jac, 0.01)
    assert len(cl) == 2
    assert cl[0]["rank"] == 1 and cl[0]["neg_vol"] >= cl[1]["neg_vol"]
    assert {"y", "x", "size", "min_jdet", "neg_vol", "bbox"} <= set(cl[0])


def test_fold_clusters_3d_has_z():
    jac = np.ones((4, 8, 8))
    jac[1, 3:5, 3:5] = -1.0
    cl = ir.fold_clusters_3d(jac, 0.01)
    assert cl and "z" in cl[0] and cl[0]["z"] == 1


def test_fold_clusters_empty_when_none():
    assert ir.fold_clusters_2d(np.ones((5, 5)), 0.01) == []


def test_build_report_is_self_contained_and_has_viewer(tmp_path):
    payload = {
        "id": "f0",
        "label": "T/z0",
        "w": 4,
        "h": 4,
        "threshold": 0.01,
        "vmax": 2.0,
        "jdet_before": ir.b64_floats(np.full((4, 4), -1.0)),
        "jdet_after": ir.b64_floats(np.ones((4, 4))),
        "dy_before": ir.b64_floats(np.zeros((4, 4))),
        "dx_before": ir.b64_floats(np.zeros((4, 4))),
        "dy_after": ir.b64_floats(np.zeros((4, 4))),
        "dx_after": ir.b64_floats(np.zeros((4, 4))),
        "rois": ir.fold_clusters_2d(np.full((4, 4), -1.0), 0.01),
        "families": [("Jdet", 16, 0, -1.0, 1.0)],
    }
    p = ir.build_interactive_report(tmp_path / "report.html", {"threshold": 0.01}, [payload])
    doc = p.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in doc
    assert "data-viewer" in doc and "initViewers" in doc
    assert "http://" not in doc and "https://" not in doc  # no external assets
    assert "T/z0" in doc


def test_build_report_never_raises_on_bad_payload(tmp_path):
    # Missing keys must degrade to the stub, not raise.
    p = ir.build_interactive_report(tmp_path / "r.html", {}, [{"id": "x"}])
    assert p.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_build_report_empty_is_warn(tmp_path):
    p = ir.build_interactive_report(tmp_path / "e.html", {"threshold": 0.01}, [])
    assert "No fields processed" in p.read_text(encoding="utf-8")


def _base_payload(**extra):
    p = {
        "id": "f0",
        "label": "T/z0",
        "w": 4,
        "h": 4,
        "threshold": 0.01,
        "vmax": 2.0,
        "jdet_before": ir.b64_floats(np.full((4, 4), -1.0)),
        "jdet_after": ir.b64_floats(np.ones((4, 4))),
        "dy_before": ir.b64_floats(np.zeros((4, 4))),
        "dx_before": ir.b64_floats(np.zeros((4, 4))),
        "dy_after": ir.b64_floats(np.zeros((4, 4))),
        "dx_after": ir.b64_floats(np.zeros((4, 4))),
        "rois": [],
        "families": [("Jdet", 16, 0, -1.0, 1.0)],
    }
    p.update(extra)
    return p


def test_report_with_traj_has_play_controls(tmp_path):
    frames = [ir.b64_floats(np.full((4, 4), v)) for v in (-1.0, 0.2, 1.0)]
    payload = _base_payload(traj=frames, traj_labels=["input", "iter 1", "final"])
    doc = ir.build_interactive_report(
        tmp_path / "r.html", {"threshold": 0.01}, [payload]
    ).read_text(encoding="utf-8")
    assert "Play iterations" in doc and "class=traj" in doc  # play button + slider
    assert all(f in doc for f in frames)  # frames embedded
    assert "http://" not in doc and "https://" not in doc  # still self-contained


def test_report_without_traj_has_no_play(tmp_path):
    # The button/slider live only in the field block; `data-act=play` also
    # appears in the shared JS, so key on the field-block-only markup.
    doc = ir.build_interactive_report(
        tmp_path / "r.html", {"threshold": 0.01}, [_base_payload()]
    ).read_text(encoding="utf-8")
    assert "Play iterations" not in doc and "class=traj" not in doc


def test_js_has_trajectory_logic():
    # Guard the viewer wiring so a refactor can't silently drop the animation.
    for token in ("curImg", "imgTraj", "trajF32", "playTimer"):
        assert token in ir._JS
