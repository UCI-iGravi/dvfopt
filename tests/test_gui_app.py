"""Offscreen widget-level tests for ``dvfopt_gui.app``.

These construct a real ``LiveSolverWindow`` under the ``offscreen`` Qt
platform (no display needed) and exercise the wiring that the pure
unit tests in ``test_gui_logic.py`` can't reach: the history-scrub
controller's state machine, the method-id dispatch contract, and the
diff/auto-level view paths.

Skipped wholesale if PySide6 isn't installed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip('PySide6', reason='dvfopt_gui requires the [gui] extra (PySide6)')

# Must be set before the first QApplication is created.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtWidgets

from dvfopt_gui.app import (
    VIEW_DIFF,
    LiveSolverWindow,
    _compose_method_id,
    _default_roi_geometry,
    _quiver_lines,
)
from dvfopt_gui.persistence import LoadedRun
from dvfopt_gui.worker import StateSnapshot


def _snap(phi, **over):
    base = dict(
        phi=phi,
        window_y0=0,
        window_y1=0,
        window_x0=0,
        window_x1=0,
        opt_y0=0,
        opt_y1=0,
        opt_x0=0,
        opt_x1=0,
        is_padded=False,
        neg_y=0,
        neg_x=0,
        per_index_iter=0,
        outer_iter=0,
        n_neg=0,
        min_T=0.0,
    )
    base.update(over)
    return StateSnapshot(**base)


def _loaded_run(n, *, history_total=None, H=5, W=5):
    vol = np.zeros((3, 1, H, W))
    snaps = [_snap(np.full((2, H, W), float(i)), n_neg=i, outer_iter=i) for i in range(n)]
    return LoadedRun(
        volume=vol,
        z=0,
        snapshots=snaps,
        history_total=history_total if history_total is not None else n,
    )


# ---------------------------------------------------------------------------
# method-id dispatch contract
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ROI geometry (pure helper — no widgets)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('H,W', [(7, 7), (3, 3), (5, 9), (456, 320)])
def test_default_roi_geometry_always_fits(H, W):
    x, y, w, h = _default_roi_geometry(H, W)
    assert x >= 0 and y >= 0
    assert w >= 3 and h >= 3
    assert x + w <= W and y + h <= H  # never overhangs the field


def test_default_roi_geometry_quarter_on_large_field():
    x, y, w, h = _default_roi_geometry(320, 456)
    assert (w, h) == (456 // 4, 320 // 4)


# ---------------------------------------------------------------------------
# method-id dispatch contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'algo,constraint',
    [('m14', '2tri'), ('m14_schwarz', '2tri'), ('barrier', 'jdet'), ('slsqp_windowed', '2tri')],
)
def test_compose_method_id_roundtrips_via_rpartition(algo, constraint):
    mid = _compose_method_id(algo, constraint)
    # Worker.run splits on the LAST underscore so multi-word algos survive.
    recovered_algo, _, kind = mid.rpartition('_')
    assert recovered_algo == algo
    assert kind == constraint


# ---------------------------------------------------------------------------
# HistoryController via the real window
# ---------------------------------------------------------------------------


def test_loaded_run_parks_on_final_step(qapp):
    win = LiveSolverWindow()
    win._apply_loaded_run(_loaded_run(5))
    assert not win._history.is_live()
    assert win._history_slider.maximum() == 4
    assert win._history_slider.value() == 4
    assert win._latest.n_neg == 4  # final snapshot rendered


def test_manual_scrub_drops_live_and_renders(qapp):
    win = LiveSolverWindow()
    win._apply_loaded_run(_loaded_run(5))
    win._history.set_live(True)  # pretend we were live
    win._history_slider.setValue(2)  # user scrub
    assert not win._history.is_live()
    assert win._latest.n_neg == 2


def test_prev_next_step(qapp):
    win = LiveSolverWindow()
    win._apply_loaded_run(_loaded_run(5))
    win._history_slider.setValue(3)
    win._history_prev_btn.click()
    assert win._history_slider.value() == 2
    assert win._latest.n_neg == 2
    win._history_next_btn.click()
    assert win._history_slider.value() == 3


def test_relive_snaps_to_latest(qapp):
    win = LiveSolverWindow()
    win._apply_loaded_run(_loaded_run(5))
    win._history_slider.setValue(1)
    assert win._latest.n_neg == 1
    win._live_check.setChecked(True)  # re-engage Live
    assert win._history_slider.value() == 4
    assert win._latest.n_neg == 4


def test_spinbox_absolute_steps_with_aged_out_entries(qapp):
    # 4 retained snapshots but 10 emitted total → absolute steps 6..9.
    win = LiveSolverWindow()
    win._apply_loaded_run(_loaded_run(4, history_total=10))
    assert win._history_spin.minimum() == 6
    assert win._history_spin.maximum() == 9
    # Slider parked at last buffer idx (3) → absolute step 9.
    assert win._history_spin.value() == 9
    # Type absolute step 6 → buffer index 0.
    win._history_spin.setValue(6)
    assert win._history_slider.value() == 0
    assert win._latest.n_neg == 0


def test_loaded_run_restores_input_volume_as_baseline(qapp):
    # A saved run carries its pre-correction input; loading must restore
    # it as the pristine baseline (not the corrected phi_full_volume), so
    # Revert / a fresh Run operate on the true input.
    H = W = 5
    inp = np.zeros((3, 1, H, W))
    inp[1, 0, 2, 2] = 0.7
    cur = np.zeros((3, 1, H, W))
    cur[1, 0, 2, 2] = 0.1
    win = LiveSolverWindow()
    win._apply_loaded_run(LoadedRun(volume=cur, z=0, input_volume=inp))
    assert win._original_volume[1, 0, 2, 2] == pytest.approx(0.7)
    assert win._volume[1, 0, 2, 2] == pytest.approx(0.1)


def test_loaded_run_without_input_volume_falls_back_to_loaded(qapp):
    cur = np.zeros((3, 1, 5, 5))
    cur[1, 0, 1, 1] = 0.3
    win = LiveSolverWindow()
    win._apply_loaded_run(LoadedRun(volume=cur, z=0))  # input_volume None
    assert win._original_volume[1, 0, 1, 1] == pytest.approx(0.3)


def test_section_bounds_initialized_to_none(qapp):
    win = LiveSolverWindow()
    assert win._section_bounds is None


def test_view_only_load_attaches_no_worker(qapp):
    # The user's "just visualize a volume" path: load → inspect, no solve.
    win = LiveSolverWindow()
    win._load_array(np.zeros((3, 3, 6, 6)))  # 3D volume
    assert win._worker is None
    assert win._save_btn.isEnabled()
    assert win._revert_btn.isEnabled()
    assert win._run_all_btn.isEnabled()  # enabled for 3D


def test_menubar_exposes_action_groups(qapp):
    win = LiveSolverWindow()
    titles = [a.text() for a in win.menuBar().actions()]
    assert '&File' in titles
    assert '&Run' in titles
    assert '&Help' in titles


def test_idle_stats_shows_infeasible_line(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    assert 'infeasible' in win._format_stats(None)


def test_run_all_pushes_single_undo_entry(qapp, monkeypatch):
    # Run-all coalesces into ONE undo entry (pushed up front), not one
    # per slice — so a single Undo reverts the whole batch.
    win = LiveSolverWindow(np.zeros((3, 3, 6, 6)))
    monkeypatch.setattr(win, '_start_worker', lambda *a, **k: None)
    win._on_run_all()
    assert len(win._undo_stack) == 1
    assert win._run_all_remaining == [1, 2]  # slice 0 popped & launched


def test_fresh_dvf_resets_history(qapp):
    win = LiveSolverWindow()
    win._apply_loaded_run(_loaded_run(5))
    # Now load a bare DVF (no history) — controller must reset.
    win._load_array(np.zeros((2, 5, 5)))
    assert win._history_slider.maximum() == 0
    assert not win._history_slider.isEnabled()
    assert win._history.is_live()


# ---------------------------------------------------------------------------
# view paths
# ---------------------------------------------------------------------------


def test_diff_view_renders(qapp):
    vol = np.zeros((3, 1, 7, 7))
    vol[2, 0, 3, 3] = 1.2
    vol[2, 0, 3, 4] = -1.2
    win = LiveSolverWindow(vol)
    idx = win._view_combo.findData(VIEW_DIFF)
    win._view_combo.setCurrentIndex(idx)
    assert win._img.isVisible()
    assert win._cbar.isVisible()


def test_levels_fixed_vs_autoscale(qapp):
    from dvfopt.jacobian.numpy_jdet import jacobian_det2D

    # Smooth expansion so Jdet clearly leaves the ±1 band.
    H, W = 7, 7
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    vol = np.zeros((3, 1, H, W))
    vol[1, 0] = 0.6 * yy  # dy grows with y → expansion
    vol[2, 0] = 0.6 * xx  # dx grows with x
    win = LiveSolverWindow(vol)
    # Levels only apply in a heatmap view (grid view has no image).
    from dvfopt_gui.app import VIEW_JDET

    win._view_combo.setCurrentIndex(win._view_combo.findData(VIEW_JDET))

    # Auto off → fixed ±1 regardless of data.
    win._autolevel_check.setChecked(False)
    win._refresh_display_from_volume()
    assert tuple(win._img.levels) == (-1.0, 1.0)

    # Auto on → symmetric about zero, matching the data extent.
    win._autolevel_check.setChecked(True)
    win._refresh_display_from_volume()
    lo, hi = win._img.levels
    expected = float(np.max(np.abs(jacobian_det2D(vol[1:, 0])[0])))
    assert lo == pytest.approx(-hi)
    assert hi == pytest.approx(expected)
    assert hi > 1.0  # this field really does exceed the fixed band


def test_2tri_menu_has_fullgrid_and_schwarz(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._select_combo_data(win._constraint_combo, '2tri')
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'slsqp_fullgrid' in algos
    assert 'schwarz' in algos


def test_run_all_disabled_for_2d(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    assert not win._run_all_btn.isEnabled()


def test_run_all_enabled_for_3d(qapp):
    win = LiveSolverWindow(np.zeros((3, 3, 6, 6)))
    assert win._run_all_btn.isEnabled()


# ---------------------------------------------------------------------------
# convergence chart
# ---------------------------------------------------------------------------


def test_convergence_populates_and_tracks_cursor(qapp):
    win = LiveSolverWindow()
    win._apply_loaded_run(_loaded_run(5))
    assert win._conv_len == 5
    # Parked on final step → cursor at step 4 (offset 0).
    assert win._conv_plot._cursor.value() == pytest.approx(4)
    win._history_slider.setValue(1)
    assert win._conv_len == 5  # length unchanged on scrub
    assert win._conv_plot._cursor.value() == pytest.approx(1)


def test_convergence_clears_on_fresh_load(qapp):
    win = LiveSolverWindow()
    win._apply_loaded_run(_loaded_run(5))
    win._load_array(np.zeros((2, 5, 5)))
    assert win._conv_len == -1


# ---------------------------------------------------------------------------
# richer stats + progress
# ---------------------------------------------------------------------------


def test_stats_shows_delta_and_displacement(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._input_n_neg = 7
    snap = _snap(np.full((2, 6, 6), 0.5), n_neg=3, outer_iter=2)
    s = win._format_stats(snap)
    assert 'vs input' in s
    assert '7 → 3' in s
    assert 'max |disp|' in s


def test_max_abs_disp():
    phi = np.zeros((2, 4, 4))
    phi[0, 1, 1] = 3.0  # dy
    phi[1, 1, 1] = 4.0  # dx → magnitude 5
    assert LiveSolverWindow._max_abs_disp(phi) == pytest.approx(5.0)


def test_finalize_clears_progress(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._active_method_id = 'm14_2tri'
    win._progress.setFormat('60s / 60s')
    win._finalize_run_ui()
    assert win._active_method_id is None
    assert win._progress.format() == ''


class _RunningStub:
    def isRunning(self):
        return True


def test_progress_slsqp_is_iter_fraction(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._worker = _RunningStub()
    win._run_elapsed.restart()
    win._active_method_id = 'slsqp_windowed_jdet'
    win._max_iter_spin.setValue(50)
    win._latest = _snap(np.zeros((2, 6, 6)), outer_iter=10)
    win._update_progress()
    assert win._progress.value() == 20  # 10 / 50
    assert 'iter 10 / 50' in win._progress.format()


def test_progress_wallbreaker_is_time_budget(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._worker = _RunningStub()
    win._run_elapsed.restart()
    win._active_method_id = 'm14_2tri'
    win._budget_spin.setValue(60.0)
    win._update_progress()
    assert '/ 60s' in win._progress.format()


def test_progress_other_methods_indeterminate(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._worker = _RunningStub()
    win._run_elapsed.restart()
    win._active_method_id = 'barrier_jdet'
    win._update_progress()
    # Busy indicator: range collapsed to (0, 0).
    assert win._progress.minimum() == 0
    assert win._progress.maximum() == 0


# ---------------------------------------------------------------------------
# displacement-arrow overlay
# ---------------------------------------------------------------------------


def test_quiver_lines_empty_for_zero_field():
    xs, ys = _quiver_lines(np.zeros((2, 5, 5)))
    assert xs.size == 0 and ys.size == 0


def test_quiver_lines_one_arrow_has_segments():
    phi = np.zeros((2, 5, 5))
    phi[1, 2, 2] = 1.0  # dx
    phi[0, 2, 2] = 0.5  # dy
    xs, ys = _quiver_lines(phi, stride=1)
    assert xs.size > 0
    finite = xs[np.isfinite(xs)]
    # Shaft base (2) and tip (3) present among the finite x-coords.
    assert finite.min() == pytest.approx(2.0)
    assert finite.max() == pytest.approx(3.0)


def test_arrows_toggle_shows_overlay(qapp):
    vol = np.zeros((3, 1, 7, 7))
    vol[2, 0, 3, 3] = 1.2
    win = LiveSolverWindow(vol)
    assert not win._quiver_curve.isVisible()
    win._arrows_check.setChecked(True)
    assert win._quiver_curve.isVisible()
    win._arrows_check.setChecked(False)
    assert not win._quiver_curve.isVisible()


# ---------------------------------------------------------------------------
# undo / redo
# ---------------------------------------------------------------------------


def test_undo_redo_roundtrip(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    assert not win._undo_btn.isEnabled()
    win._volume[1, 0, 0, 0] = 9.0  # pre-run marker
    win._push_undo_state()
    assert win._undo_btn.isEnabled()
    win._volume[1, 0, 0, 0] = 1.0  # "corrected"
    win._on_undo()
    assert win._volume[1, 0, 0, 0] == 9.0
    assert win._redo_btn.isEnabled()
    win._on_redo()
    assert win._volume[1, 0, 0, 0] == 1.0


def test_push_undo_clears_redo(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._push_undo_state()
    win._on_undo()
    assert win._redo_stack  # redo available after an undo
    win._push_undo_state()  # a new correction invalidates redo
    assert not win._redo_stack
    assert not win._redo_btn.isEnabled()


def test_undo_cap(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    for _ in range(win._UNDO_MAX + 5):
        win._push_undo_state()
    assert len(win._undo_stack) == win._UNDO_MAX


def test_load_clears_undo_redo(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._push_undo_state()
    win._on_undo()
    assert win._undo_stack or win._redo_stack
    win._load_array(np.zeros((2, 5, 5)))
    assert not win._undo_stack
    assert not win._redo_stack
    assert not win._undo_btn.isEnabled()


# ---------------------------------------------------------------------------
# session persistence (isolated to a temp INI — never touches the registry)
# ---------------------------------------------------------------------------


def test_settings_roundtrip(qapp, tmp_path, monkeypatch):
    ini = str(tmp_path / 'settings.ini')

    def _fake_settings():
        return QtCore.QSettings(ini, QtCore.QSettings.IniFormat)

    monkeypatch.setattr(LiveSolverWindow, '_settings', staticmethod(_fake_settings))

    w1 = LiveSolverWindow()
    w1._select_combo_data(w1._constraint_combo, 'jdet')
    w1._select_combo_data(w1._method_combo, 'nmvf')
    w1._objective_combo.setCurrentIndex(w1._objective_combo.findData('l2'))
    w1._budget_spin.setValue(123.0)
    w1._autolevel_check.setChecked(True)
    w1._save_settings()

    w2 = LiveSolverWindow()  # _restore_settings runs in __init__
    assert w2._constraint_combo.currentData() == 'jdet'
    assert w2._method_combo.currentData() == 'nmvf'
    assert w2._objective_combo.currentData() == 'l2'
    assert w2._budget_spin.value() == pytest.approx(123.0)
    assert w2._autolevel_check.isChecked()


def test_demo_3d_loaders_and_initial_constraint(qapp):
    # The demo's 3D fixture loads + the launch() initial_constraint path
    # opens straight into 3D mode.
    from dvfopt_gui.demo import _synthetic_3d_volume
    from dvfopt_gui.worker import _metric_counts_3d

    vol = _synthetic_3d_volume()
    assert vol.shape == (3, 4, 16, 16)
    assert _metric_counts_3d(vol, 'tet3d')[0] > 0  # genuinely folded

    win = LiveSolverWindow(vol)
    win._select_combo_data(win._constraint_combo, 'tet3d')  # what launch() does
    assert win._is_3d_run
    assert win._constraint_combo.currentData() == 'tet3d'
    assert win._run_roi_btn.isEnabled()  # 3D ROI now supported


def test_3d_end_to_end_run_through_window(qapp):
    # Real integration: load folded volume -> simplex (3D) 3D -> M14Tet ->
    # Run full -> the QThread solves the whole volume -> finished signal
    # splices the corrected volume back -> 3D fold count drops.
    from dvfopt_gui.demo import _synthetic_3d_volume
    from dvfopt_gui.worker import _metric_counts_3d

    vol = _synthetic_3d_volume()
    n_before, _ = _metric_counts_3d(vol, 'tet3d')
    assert n_before > 0

    win = LiveSolverWindow(vol)
    win.start()  # render timer drains worker snapshots
    win._select_combo_data(win._constraint_combo, 'tet3d')
    win._select_combo_data(win._method_combo, 'm14')
    assert win._is_3d_run
    win._budget_spin.setValue(30.0)

    win._on_run(use_roi=False)
    worker = win._worker
    assert worker is not None and worker.isRunning()

    waited = 0
    while worker.isRunning() and waited < 90_000:
        QtWidgets.QApplication.processEvents()
        worker.wait(50)
        waited += 50
    assert not worker.isRunning(), 'M14Tet 3D run did not finish in time'
    # Let the queued finishedWithResult slot (_on_finished) run.
    for _ in range(50):
        QtWidgets.QApplication.processEvents()

    # The corrected (3, D, H, W) volume was spliced back; folds dropped.
    assert win._volume.shape == vol.shape
    n_after, _ = _metric_counts_3d(win._volume, 'tet3d')
    assert n_after < n_before
    # Per-run history carries full-volume 3D snapshots (ndim 4).
    assert worker.history_len() >= 2
    assert worker.history_get(0).phi.ndim == 4


def test_initial_params_override_saved_max_iter(qapp, tmp_path, monkeypatch):
    ini = str(tmp_path / 's.ini')
    monkeypatch.setattr(
        LiveSolverWindow,
        '_settings',
        staticmethod(lambda: QtCore.QSettings(ini, QtCore.QSettings.IniFormat)),
    )
    w1 = LiveSolverWindow()
    w1._max_iter_spin.setValue(77)
    w1._save_settings()
    # Demo-style explicit override wins over the saved value.
    w2 = LiveSolverWindow(initial_params={'max_iterations': 999})
    assert w2._max_iter_spin.value() == 999


# ---------------------------------------------------------------------------
# 3D mode: constraint gating + run-control gating
# ---------------------------------------------------------------------------


def test_3d_constraints_gated_by_volume_depth(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))  # 2D section
    tet_idx = win._constraint_combo.findData('tet3d')
    assert tet_idx >= 0
    model = win._constraint_combo.model()
    assert not model.item(tet_idx).isEnabled()  # disabled for D == 1
    win._load_array(np.zeros((3, 4, 6, 6)))  # 3D volume
    assert model.item(win._constraint_combo.findData('tet3d')).isEnabled()


def test_selecting_3d_constraint_enters_3d_mode_and_gates_runs(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    assert win._is_3d_run
    assert win._run_roi_btn.isEnabled()  # 3D ROI now supported
    assert not win._run_all_btn.isEnabled()
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert (
        'm14' in algos and 'm14_schwarz' in algos and 'm10' in algos and 'slsqp_fullgrid' in algos
    )


def test_3d_menu_has_experimental_strategies(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'active_band' in algos
    assert 'coupled_kring' in algos


def test_run_all_stays_disabled_in_3d_after_run_finishes(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    assert not win._run_all_btn.isEnabled()  # gated on entering 3D mode
    win._finalize_run_ui()  # simulate a run completing
    assert not win._run_all_btn.isEnabled()  # must stay disabled in 3D mode
    assert win._run_roi_btn.isEnabled()  # 3D ROI now supported


def test_run_all_in_3d_routes_to_full_volume_run(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    captured = {}
    monkeypatch.setattr(
        win, '_start_worker', lambda def_i, *a, **k: captured.setdefault('shape', def_i.shape)
    )
    win._on_run_all()
    # 3D Run-all must hand the worker the FULL (3,D,H,W) volume, not a (3,1,H,W) slice.
    assert captured['shape'] == (3, 4, 6, 6)
    assert win._run_all_remaining is None  # did not enter the per-slice batch


# ---------------------------------------------------------------------------
# 3D rendering, stats, inspector
# ---------------------------------------------------------------------------


def test_3d_render_and_stats(qapp):
    from dvfopt_gui.worker import _volume_snapshot

    vol = np.zeros((3, 4, 8, 8))
    vol[2, :, 3:5, 3:5] = 1.4
    win = LiveSolverWindow(vol)
    win._select_combo_data(win._constraint_combo, 'tet3d')
    snap = _volume_snapshot(vol, n_neg=5, min_T=-0.2, outer_iter=1)
    win._render_snapshot(snap)  # must not raise on a 4-D phi
    assert win._img.isVisible() or win._grid_curve.isVisible()
    s = win._format_stats(snap)
    assert 'min_T' in s
    # Idle 3D stats mention the volume shape.
    idle = win._format_stats(None)
    assert '4×8×8' in idle


def test_3d_zslider_reslices_without_dropping_worker(qapp):
    from dvfopt_gui.worker import ReplayHistory, _volume_snapshot

    vol = np.zeros((3, 4, 8, 8))
    win = LiveSolverWindow(vol)
    win._select_combo_data(win._constraint_combo, 'tet3d')
    snap = _volume_snapshot(vol, n_neg=0, min_T=0.16, outer_iter=1)
    win._worker = ReplayHistory([snap], 1)
    win._latest = snap
    win._z_slider.setValue(2)  # re-slice
    assert win._worker is not None  # not reset in 3D mode


def test_progress_3d_wallbreaker_is_time_budget(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._worker = _RunningStub()
    win._run_elapsed.restart()
    win._active_method_id = 'm14_tet3d'
    win._budget_spin.setValue(60.0)
    win._update_progress()
    assert '/ 60s' in win._progress.format()


def test_progress_3d_fullgrid_is_busy(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._worker = _RunningStub()
    win._run_elapsed.restart()
    win._active_method_id = 'slsqp_fullgrid_tet3d'
    win._update_progress()
    assert win._progress.maximum() == 0  # busy indicator


def test_input_n_neg_uses_3d_metric_in_3d_mode(qapp, monkeypatch):
    from dvfopt_gui.worker import _metric_counts_3d

    vol = np.zeros((3, 4, 8, 8))
    vol[2, :, 3:5, 3:5] = 1.4  # a few folded cells across all z
    win = LiveSolverWindow(vol)
    win._select_combo_data(win._constraint_combo, 'tet3d')
    # Don't actually launch the solver thread.
    monkeypatch.setattr('dvfopt_gui.worker.SolverWorker.start', lambda self: None)
    win._start_worker(win._original_volume.copy())
    expected, _ = _metric_counts_3d(vol, 'tet3d')
    assert win._input_n_neg == expected
    assert expected > 0  # the field really does have 3D folds


# ---------------------------------------------------------------------------
# SLP default + Auto strategy picker (menu wiring)
# ---------------------------------------------------------------------------


def test_slp_is_first_and_default_2tri(qapp, tmp_path, monkeypatch):
    ini = str(tmp_path / 'fresh.ini')
    monkeypatch.setattr(
        LiveSolverWindow,
        '_settings',
        staticmethod(lambda: QtCore.QSettings(ini, QtCore.QSettings.IniFormat)),
    )
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))  # fresh settings -> defaults
    win._select_combo_data(win._constraint_combo, '2tri')
    assert win._method_combo.itemData(0) == 'slp'
    assert win._method_combo.currentData() == 'slp'
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'auto' in algos
    win._select_combo_data(win._constraint_combo, 'jdet')
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'auto' in algos


# ---------------------------------------------------------------------------
# user-editable feasibility threshold
# ---------------------------------------------------------------------------


def test_threshold_spinbox_feeds_params_and_stats(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    assert win._thr_spin.value() == pytest.approx(0.01)
    win._thr_spin.setValue(0.05)
    captured = {}
    monkeypatch.setattr(
        'dvfopt_gui.worker.SolverWorker.start', lambda self: captured.setdefault('p', self._params)
    )
    win._on_run(use_roi=False)
    assert captured['p']['threshold'] == pytest.approx(0.05)
    # Idle stats use the spinbox threshold, not the module constant.
    assert '0.05' in win._format_stats(None)


# ---------------------------------------------------------------------------
# 3D whole-volume metric cache (z-scrub / hover must not re-run the kernel)
# ---------------------------------------------------------------------------


def test_3d_metric_cached_across_zscrub_and_hover(qapp, monkeypatch):
    import dvfopt_gui.app as A
    from dvfopt_gui.worker import _metric_field_3d as real_field
    from dvfopt_gui.worker import _volume_snapshot

    calls = {'n': 0}

    def counting(phi3d, kind):
        calls['n'] += 1
        return real_field(phi3d, kind)

    monkeypatch.setattr(A, '_metric_field_3d', counting)
    vol = np.zeros((3, 5, 8, 8))
    vol[2, :, 3:5, 3:5] = 1.4
    win = LiveSolverWindow(vol)
    win._select_combo_data(win._constraint_combo, 'tet3d')
    snap = _volume_snapshot(vol, n_neg=5, min_T=-0.2, outer_iter=1)
    win._render_snapshot(snap)
    baseline = calls['n']
    assert baseline >= 1
    # z-scrub + inspector hover on the SAME field: zero new kernel runs.
    win._z_slider.setValue(2)
    win._z_slider.setValue(3)
    win._format_inspector((2, 2))
    win._format_inspector((3, 3))
    assert calls['n'] == baseline
    # A new snapshot invalidates and recomputes.
    win._render_snapshot(_volume_snapshot(vol * 0.5, n_neg=0, min_T=0.1, outer_iter=2))
    assert calls['n'] > baseline


# ---------------------------------------------------------------------------
# Undo byte budget + non-finite load rejection
# ---------------------------------------------------------------------------


def test_undo_stack_byte_budget(qapp, monkeypatch):
    import dvfopt_gui.app as A

    win = LiveSolverWindow(np.zeros((3, 1, 64, 64)))
    entry_bytes = win._volume.nbytes
    # Budget that fits exactly two entries.
    monkeypatch.setattr(A, 'UNDO_MAX_BYTES', int(entry_bytes * 2.5))
    for _ in range(5):
        win._push_undo_state()
    assert len(win._undo_stack) == 2
    assert sum(v.nbytes for v in win._undo_stack) <= entry_bytes * 2.5


def test_undo_budget_keeps_at_least_one(qapp, monkeypatch):
    import dvfopt_gui.app as A

    win = LiveSolverWindow(np.zeros((3, 1, 64, 64)))
    monkeypatch.setattr(A, 'UNDO_MAX_BYTES', 1)  # smaller than one entry
    win._push_undo_state()
    assert len(win._undo_stack) == 1


def test_nonfinite_load_rejected(qapp, monkeypatch):
    from dvfopt_gui.app import validate_finite
    from dvfopt_gui.persistence import LoadedRun

    bad = np.zeros((3, 1, 5, 5))
    bad[2, 0, 2, 2] = np.nan
    msg = validate_finite(bad)
    assert msg is not None and 'non-finite' in msg
    assert validate_finite(np.zeros((3, 1, 4, 4))) is None

    win = LiveSolverWindow()
    seen = {}
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        'critical',
        staticmethod(lambda *a, **k: seen.setdefault('called', True)),
    )
    prev = win._volume
    win._apply_loaded_run(LoadedRun(volume=bad))
    assert seen.get('called')
    assert win._volume is prev  # rejected load leaves state untouched


def test_rejected_load_shows_no_success_message(qapp, tmp_path, monkeypatch):
    bad = np.zeros((3, 1, 5, 5))
    bad[1, 0, 1, 1] = np.inf
    npy = tmp_path / 'bad.npy'
    np.save(npy, bad)
    win = LiveSolverWindow()
    monkeypatch.setattr(
        QtWidgets.QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(npy), ''))
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical', staticmethod(lambda *a, **k: None))
    win._on_load()
    # Load now runs on a QThread; the result is delivered asynchronously
    # via a queued signal, so wait for the worker then pump the event loop.
    win._load_worker.wait(10_000)
    for _ in range(50):
        QtWidgets.QApplication.processEvents()
    assert win._volume is None  # rejected: nothing loaded
    assert 'Loaded' not in win.statusBar().currentMessage()


def test_export_writes_npy(qapp, tmp_path, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 2, 5, 5)))
    out = tmp_path / 'corr.npy'
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **k: (str(out), 'NumPy array (*.npy)')),
    )
    win._on_export()
    assert out.exists() and np.load(out).shape == (3, 2, 5, 5)


def test_load_worker_path_used_by_on_load(qapp, tmp_path, monkeypatch):
    # _on_load must go through LoadWorker (GUI thread does no np.load).
    npy = tmp_path / 'f.npy'
    np.save(npy, np.zeros((3, 1, 6, 6)))
    win = LiveSolverWindow()
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getOpenFileName',
        staticmethod(lambda *a, **k: (str(npy), '')),
    )
    win._on_load()
    win._load_worker.wait(10_000)
    for _ in range(50):
        QtWidgets.QApplication.processEvents()
    assert win._volume is not None and win._volume.shape == (3, 1, 6, 6)
    assert win._load_btn.isEnabled()


# ---------------------------------------------------------------------------
# tet3d menu: full-3D pipeline entry + torch-barrier gating
# ---------------------------------------------------------------------------


def test_tet3d_menu_pipeline_and_torch_gating(qapp, monkeypatch):
    import dvfopt_gui.app as A

    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'pipeline3d' in algos
    assert 'barrier_torch' in algos
    # Torch missing -> the item is disabled (greyed), still listed.
    monkeypatch.setattr(A, '_torch_available', lambda: False)
    win._repopulate_method_combo('tet3d')
    idx = win._method_combo.findData('barrier_torch')
    assert idx >= 0
    assert not win._method_combo.model().item(idx).isEnabled()


# ---------------------------------------------------------------------------
# Pipeline ▾: 2.5D marching + one-click full pipeline (2D + 2.5D)
# ---------------------------------------------------------------------------


def test_pipeline_button_exists_and_gates(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))  # 2D: disabled
    assert hasattr(win, '_pipeline_btn')
    assert not win._pipeline_btn.isEnabled()
    win._load_array(np.zeros((3, 4, 6, 6)))  # volume: enabled
    assert win._pipeline_btn.isEnabled()


def test_run_25d_rejects_nonzero_dz(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._volume[0, 1, 2, 2] = 0.5  # nonzero dz
    asked = {}
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        'question',
        staticmethod(lambda *a, **k: asked.setdefault('q', True) and QtWidgets.QMessageBox.No),
    )
    started = {}
    monkeypatch.setattr(win, '_start_worker', lambda *a, **k: started.setdefault('s', True))
    win._on_run_25d()
    assert asked.get('q'), 'dz violation must prompt'
    assert not started.get('s'), 'declined prompt must not start a run'


def test_run_25d_zero_dz_consent_runs_pipeline(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._volume[0, 1, 2, 2] = 0.5  # nonzero dz
    win._original_volume[0, 1, 2, 2] = 0.5
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        'question',
        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes),
    )
    monkeypatch.setattr(win, '_start_worker', lambda *a, **k: None)
    win._on_run_25d()
    # Consent -> dz zeroed on BOTH volumes, inside one undo entry, pipeline armed.
    assert float(np.abs(win._volume[0]).max()) == 0.0
    assert float(np.abs(win._original_volume[0]).max()) == 0.0
    assert win._pipeline_active and win._pipeline_after_run_all
    assert len(win._undo_stack) == 1
    # Undo restores the pre-zero dz.
    win._pipeline_active = False  # simulate pipeline over
    win._run_all_remaining = None
    win._on_undo()
    assert win._volume[0, 1, 2, 2] == pytest.approx(0.5)


def test_run_25d_starts_marching_on_current_volume(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._volume[2, 1, 2, 2] = 0.3  # differs from _original_volume
    captured = {}

    def fake_start(def_i, method_id=None):
        captured['shape'] = def_i.shape
        captured['mid'] = method_id
        captured['val'] = float(def_i[2, 1, 2, 2])

    monkeypatch.setattr(win, '_start_worker', fake_start)
    win._on_run_25d()
    assert captured['mid'] == 'marching25d_tet3d'
    assert captured['shape'] == (3, 4, 6, 6)
    assert captured['val'] == pytest.approx(0.3)  # CURRENT volume, not original


def test_full_pipeline_chains_25d_after_batch(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 3, 6, 6)))
    monkeypatch.setattr(win, '_start_worker', lambda *a, **k: None)
    win._on_run_pipeline_full()
    assert win._pipeline_active and win._pipeline_after_run_all
    assert len(win._undo_stack) == 1  # exactly one entry for the whole pipeline
    # Simulate the batch draining to completion.
    started = {}
    monkeypatch.setattr(win, '_start_marching_25d', lambda: started.setdefault('m', True))
    win._run_all_remaining = []
    win._run_all_step()
    assert started.get('m'), '2.5D stage must start after the batch'
    assert not win._pipeline_after_run_all


def test_pipeline_report_message_is_final(qapp):
    # _on_finished must leave a pipeline run's report summary as the LAST
    # status-bar message, not overwrite it with the generic 'Run finished.'
    # Uses a plain fake worker + a direct call (not a real signal emission)
    # -- see the ``sender()`` note on ``_on_finished``: a direct call always
    # has ``sender() is None``, so the guard trusts ``self._worker`` as given
    # rather than requiring it to equal the (nonexistent) signal sender.
    win = LiveSolverWindow(np.zeros((3, 2, 4, 4)))

    class _FakeReport:
        n_neg_in = 5
        n_neg_out = 0
        feasible = True
        wall_s = 1.23

    class _FakeWorker:
        pipeline_report = _FakeReport()

        def history_len(self):
            return 0

    win._worker = _FakeWorker()
    vol4d = np.zeros((3, 2, 4, 4))
    win._on_finished(vol4d, None)
    msg = win.statusBar().currentMessage()
    assert 'Pipeline:' in msg
    assert 'Run finished' not in msg


# ---------------------------------------------------------------------------
# per-slice fold overview strip
# ---------------------------------------------------------------------------


def test_overview_strip_counts_and_click(qapp):
    from dvfopt_gui.overview import OverviewWorker, SliceOverviewStrip
    from dvfopt_gui.worker import _metric_counts

    vol = np.zeros((3, 4, 8, 8))
    vol[2, 2, 3, 3] = 1.2  # slice 2 has simplex (2D) folds
    vol[2, 2, 3, 4] = -1.2

    # Worker computes per-slice simplex (2D) fold counts (run synchronously).
    got = {}
    w = OverviewWorker(vol)
    w.chunkReady.connect(lambda start, arr: got.setdefault(start, np.asarray(arr)))
    w.run()
    counts = np.concatenate([got[k] for k in sorted(got)])
    assert counts.shape == (4,)
    assert counts[2] == _metric_counts(vol[1:, 2], '2tri')[0] > 0
    assert counts[0] == 0

    strip = SliceOverviewStrip()
    clicks = []
    strip.sliceClicked.connect(clicks.append)
    strip.set_counts(counts)
    strip.set_current(1)
    strip._emit_click_at(2.4)  # test hook: x-coordinate -> slice index
    assert clicks == [2]


def test_overview_strip_wired_into_window(qapp):
    vol = np.zeros((3, 4, 8, 8))
    vol[2, 1, 3, 3] = 1.2
    vol[2, 1, 3, 4] = -1.2
    win = LiveSolverWindow(vol)
    assert win._overview_strip.isVisibleTo(win)
    win._overview_worker.wait(10_000)
    for _ in range(50):
        QtWidgets.QApplication.processEvents()
    assert win._overview_counts is not None and win._overview_counts[1] > 0
    win._overview_strip.sliceClicked.emit(3)
    assert win._z_slider.value() == 3
    # 2D single-slice field hides the strip.
    win._load_array(np.zeros((3, 1, 6, 6)))
    assert not win._overview_strip.isVisibleTo(win)


# ---------------------------------------------------------------------------
# auto-generated per-strategy parameter panel
# ---------------------------------------------------------------------------


def test_params_dialog_strategy_tab_and_persistence(qapp, tmp_path, monkeypatch):
    ini = str(tmp_path / 's.ini')
    monkeypatch.setattr(
        LiveSolverWindow,
        '_settings',
        staticmethod(lambda: QtCore.QSettings(ini, QtCore.QSettings.IniFormat)),
    )
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._strategy_overrides['slp'] = {'cluster_pixel_threshold': 99}
    win._save_settings()
    win2 = LiveSolverWindow()
    assert win2._strategy_overrides.get('slp') == {'cluster_pixel_threshold': 99}
    # Overrides reach the worker params.
    captured = {}
    monkeypatch.setattr(
        'dvfopt_gui.worker.SolverWorker.start',
        lambda self: captured.setdefault('p', self._params),
    )
    win._select_combo_data(win._constraint_combo, '2tri')
    win._select_combo_data(win._method_combo, 'slp')
    win._on_run(use_roi=False)
    assert captured['p']['strategy_overrides'] == {'cluster_pixel_threshold': 99}


def test_strategy_params_no_spurious_float_overrides(qapp):
    # Opening the tab and reading values() without touching anything must
    # be a no-op — even for strategies with sub-1e-6 float defaults.
    from dvfopt_gui.strategy_params import StrategyParamsTab

    for algo in ('slsqp_fullgrid', 'slsqp_fullgrid@tet3d', 'coupled_kring@tet3d'):
        tab = StrategyParamsTab()
        tab.build(algo, {})
        assert tab.values() == {}, f'{algo}: spurious overrides {tab.values()}'


def test_params_algo_key_family_accurate(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._select_combo_data(win._constraint_combo, 'jdet3d')
    win._select_combo_data(win._method_combo, 'barrier')
    assert win._current_params_algo() == 'barrier@jdet3d'
    win._select_combo_data(win._constraint_combo, 'tet3d')
    assert win._current_params_algo().endswith('@tet3d')


# ---------------------------------------------------------------------------
# 3D sub-volume ROI (Rect ROI for y/x + z-range spinboxes)
# ---------------------------------------------------------------------------


def test_3d_roi_spinboxes_and_run_section(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 6, 20, 20)))
    assert not win._z0_spin.isVisibleTo(win)  # hidden in 2D mode
    win._select_combo_data(win._constraint_combo, 'tet3d')
    assert win._z0_spin.isVisibleTo(win) and win._z1_spin.isVisibleTo(win)
    assert win._run_roi_btn.isEnabled()  # 3D ROI now supported
    assert (win._z0_spin.value(), win._z1_spin.value()) == (0, 5)

    win._section_roi.setPos(4, 4)
    win._section_roi.setSize([10, 10])
    win._z0_spin.setValue(1)
    win._z1_spin.setValue(4)
    captured = {}
    monkeypatch.setattr(
        win,
        '_start_worker',
        lambda def_i, method_id=None: captured.setdefault('shape', def_i.shape),
    )
    win._on_run(use_roi=True)
    assert captured['shape'] == (3, 4, 10, 10)
    assert win._section_bounds_3d == (1, 5, 4, 14, 4, 14)


def test_3d_roi_splice_back(qapp):
    win = LiveSolverWindow(np.zeros((3, 6, 20, 20)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    win._worker = None  # sender() is None -> guard passes
    win._section_bounds_3d = (1, 5, 4, 14, 4, 14)
    sub = np.full((3, 4, 10, 10), 2.0)
    win._on_finished(sub, None)
    assert win._volume[1, 2, 5, 5] == pytest.approx(2.0)
    assert win._volume[1, 0, 5, 5] == 0.0  # outside the box untouched


# ---------------------------------------------------------------------------
# final-review fixes: upfront dz check on the direct pipeline path; gated
# overview restart during Run-all / pipeline batches
# ---------------------------------------------------------------------------


def test_pipeline_full_checks_dz_upfront(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._volume[0, 1, 2, 2] = 0.5
    win._original_volume[0, 1, 2, 2] = 0.5
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        'question',
        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No),
    )
    started = {}
    monkeypatch.setattr(win, '_start_worker', lambda *a, **k: started.setdefault('s', True))
    win._on_run_pipeline_full()
    assert not started.get('s'), 'declined dz consent must not start the batch'
    assert not win._pipeline_active


def test_overview_restart_skipped_per_slice(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 3, 6, 6)))
    calls = {'n': 0}
    monkeypatch.setattr(win, '_restart_overview', lambda: calls.update(n=calls['n'] + 1))
    # Mid-batch, ``_on_finished`` chains into ``_run_all_step``, which pops
    # slice 2 and starts its worker for real -- stub that out too.
    monkeypatch.setattr(win, '_start_worker', lambda *a, **k: None)
    win._worker = None
    win._run_all_remaining = [2]  # mid-batch
    win._on_finished(np.zeros((2, 6, 6)), None)  # per-slice finish
    assert calls['n'] == 0, 'no overview restart mid-batch'


# ---------------------------------------------------------------------------
# GUI Minors B1: worker lifecycle & guards
# ---------------------------------------------------------------------------


class TestMinorsSweepLifecycle:
    def _win(self, qapp, D=1):
        vol = np.zeros((3, D, 8, 8), dtype=np.float64)
        win = LiveSolverWindow(np.zeros((3, 1, 8, 8)))
        win._apply_loaded_run(LoadedRun(volume=vol))
        return win

    def test_load_reentry_guarded_and_controls_reenabled(self, qapp, monkeypatch):
        win = self._win(qapp)
        monkeypatch.setattr(QtWidgets.QMessageBox, 'critical', staticmethod(lambda *a, **k: None))

        class FakeWorker:
            def isRunning(self):
                return True

        win._load_worker = FakeWorker()
        # A second Ctrl+O while a load is in flight must return before
        # even opening the file dialog.
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            'getOpenFileName',
            staticmethod(lambda *a, **k: pytest.fail('dialog opened during in-flight load')),
        )
        win._on_load()
        # Finish/fail paths re-enable BOTH the toolbar button and menu action.
        win._load_btn.setEnabled(False)
        win._load_action.setEnabled(False)
        win._on_load_failed('boom')
        assert win._load_btn.isEnabled()
        assert win._load_action.isEnabled()

    def test_redo_stack_byte_budgeted(self, qapp, monkeypatch):
        import dvfopt_gui.app as app_mod

        win = self._win(qapp)
        vol_bytes = win._volume.nbytes
        monkeypatch.setattr(app_mod, 'UNDO_MAX_BYTES', 3 * vol_bytes)
        stack = [win._volume.copy() for _ in range(6)]
        win._cap_stack(stack)
        assert len(stack) <= 3
        assert sum(v.nbytes for v in stack) <= 3 * vol_bytes

    def test_undo_pushes_capped_redo(self, qapp, monkeypatch):
        import dvfopt_gui.app as app_mod

        win = self._win(qapp)
        monkeypatch.setattr(app_mod, 'UNDO_MAX_BYTES', 2 * win._volume.nbytes)
        win._undo_stack = [win._volume.copy() for _ in range(4)]
        win._redo_stack = [win._volume.copy(), win._volume.copy()]
        win._on_undo()
        assert sum(v.nbytes for v in win._redo_stack) <= 2 * win._volume.nbytes

    def test_thr_spin_repaints_after_run_finished(self, qapp, monkeypatch):
        win = self._win(qapp)

        class DoneWorker:
            def isRunning(self):
                return False

        win._worker = DoneWorker()  # run finished, ref not cleared
        calls = []
        monkeypatch.setattr(win, '_refresh_display_from_volume', lambda: calls.append(1))
        win._on_threshold_changed(0.02)
        assert calls, 'threshold change after a finished run must repaint'

    def test_thr_spin_noop_while_running(self, qapp, monkeypatch):
        win = self._win(qapp)

        class LiveWorker:
            def isRunning(self):
                return True

        win._worker = LiveWorker()
        calls = []
        monkeypatch.setattr(win, '_refresh_display_from_volume', lambda: calls.append(1))
        win._on_threshold_changed(0.02)
        assert not calls, 'must not repaint mid-run (stream owns the display)'

    def test_inspector_3d_idle_readout(self, qapp):
        win = self._win(qapp, D=4)
        assert win._latest is None
        html = win._format_inspector((2, 2))
        assert '3D' in html and 'min simplex (3D)' in html, (
            f'idle 3D volume must get the 3D readout, got: {html}'
        )

    def test_latest_cleared_on_finish(self, qapp):
        win = self._win(qapp)
        win._latest = _snap(np.zeros((2, 8, 8)))
        win._on_finished(np.zeros((2, 8, 8)), None)
        assert win._latest is None


class TestMinorsSweepPolish:
    def test_validate_finite_reports_first_index(self):
        from dvfopt_gui.app import validate_finite

        vol = np.zeros((3, 2, 4, 4))
        vol[1, 1, 2, 3] = np.nan
        msg = validate_finite(vol)
        assert msg is not None and '(1, 1, 2, 3)' in msg
        assert validate_finite(np.zeros((3, 1, 2, 2))) is None

    def test_choice_field_default_mismatch_asserts(self):
        import dataclasses

        from dvfopt_gui.strategy_params import editable_fields

        @dataclasses.dataclass
        class Bogus:
            accuracy: str = 'warp-speed'  # not in _CHOICE_FIELDS['accuracy']

        with pytest.raises(AssertionError, match='bare field name'):
            editable_fields(Bogus)

    def test_build_sanitizes_bad_overrides(self, qapp):
        from dvfopt_gui.strategy_params import (
            StrategyParamsTab,
            editable_fields,
            strategy_class_for,
        )

        tab = StrategyParamsTab()
        # Brief used 'slp', but SLPStrategy has no float-kind field (only
        # int/str/choice), so the shared-kind assert below can't be
        # satisfied on it. Substituted 'slsqp_fullgrid'
        # (SLSQPFullGridStrategy), which has both float (warm_ftol,
        # warm_sigma) and int (max_iter, warm_max_iter, warm_seed) fields.
        algo = 'slsqp_fullgrid'
        cls = strategy_class_for(algo)
        fields = editable_fields(cls)
        float_fields = [n for n, k, _ in fields if k == 'float']
        int_fields = [n for n, k, _ in fields if k == 'int']
        assert float_fields and int_fields, f'test needs one field of each kind on {algo}'
        bad = {float_fields[0]: float('nan'), int_fields[0]: 'abc'}
        tab.build(algo, bad)  # must not crash
        vals = tab.values()
        # Sanitized fields fall back to defaults -> values() reports no override.
        assert float_fields[0] not in vals
        assert int_fields[0] not in vals

    def test_overview_stale_chunk_rejected(self, qapp):
        from dvfopt_gui.overview import OverviewWorker

        vol = np.zeros((3, 4, 8, 8), dtype=np.float64)
        win = LiveSolverWindow(np.zeros((3, 1, 8, 8)))
        win._apply_loaded_run(LoadedRun(volume=vol))
        win._overview_counts = np.zeros(4, dtype=np.int64)
        w_old = OverviewWorker(vol, parent=win)
        w_new = OverviewWorker(vol, parent=win)
        win._overview_worker = w_new
        # Neither thread is started: emitting from the test thread delivers
        # synchronously and sender() is the emitting worker inside the slot.
        w_old.chunkReady.connect(win._on_overview_chunk)
        w_new.chunkReady.connect(win._on_overview_chunk)
        w_old.chunkReady.emit(0, np.array([7, 7], dtype=np.int64))
        assert not win._overview_counts.any(), 'stale worker chunk must be rejected'
        w_new.chunkReady.emit(0, np.array([5, 5], dtype=np.int64))
        assert list(win._overview_counts[:2]) == [5, 5], 'current worker chunk must land'
