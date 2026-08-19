"""Tests for the GUI integration round: 3D SLP/Auto menu entries, the
injectivity-gap view, optional-float strategy params, phase-aware
convergence (stage snapshots + markers + report), and the solver log dock.

Offscreen widget tests where a real window/widget is needed; pure logic
tests otherwise. Skipped wholesale without PyQt5.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip('PyQt5', reason='dvfopt_gui requires the [gui] extra (PyQt5)')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets

from dvfopt_gui._shared import (
    _METHOD_SPECS_JDET3D,
    _METHOD_SPECS_TET3D,
    VIEW_INJ,
    _min_gap_2d,
)
from dvfopt_gui.worker import (
    SolverWorker,
    _metric_field_3d,
    _volume_snapshot,
)
from tests.conftest import planted_fold_3d as _folded_volume


def _folded_slice(H=8, W=8):
    """(3, 1, H, W) [dz, dy, dx] with a central fold."""
    rng = np.random.default_rng(0)
    d = rng.normal(0, 0.05, (3, 1, H, W))
    d[0] = 0.0
    d[1, 0, 3:5, 3:5] -= 1.3
    d[2, 0, 3:5, 3:5] -= 1.3
    return d


# ---------------------------------------------------------------------------
# 3D method menu entries + worker dispatch
# ---------------------------------------------------------------------------


class TestNew3DMethods:
    def test_menus_contain_slp_and_auto(self):
        tet_algos = [a for a, _ in _METHOD_SPECS_TET3D]
        jdet_algos = [a for a, _ in _METHOD_SPECS_JDET3D]
        assert 'slp' in tet_algos and 'auto' in tet_algos
        assert 'auto' in jdet_algos

    def test_slp_tet3d_builds_slp_strategy(self):
        from dvfopt import SLPStrategy

        w = SolverWorker(deformation_i=_folded_volume(), method_id='slp_tet3d')
        assert isinstance(w._build_strategy(), SLPStrategy)

    def test_auto_tet3d_resolves_and_records_label(self):
        from dvfopt.strategies.base import _STRATEGY_REGISTRY

        w = SolverWorker(deformation_i=_folded_volume(), method_id='auto_tet3d')
        strategy = w._build_strategy()
        assert w.resolved_strategy_label in _STRATEGY_REGISTRY
        assert type(strategy) is _STRATEGY_REGISTRY[w.resolved_strategy_label]

    def test_auto_jdet3d_mild_routes_to_barrier(self):
        rng = np.random.default_rng(1)
        vol = rng.normal(0, 0.01, (3, 4, 6, 6))  # mild — no dense folds
        w = SolverWorker(deformation_i=vol, method_id='auto_jdet3d')
        w._build_strategy()
        assert w.resolved_strategy_label in ('barrier', 'slsqp_windowed')

    def test_params_mapping_has_slp_tet3d(self):
        from dvfopt import SLPStrategy
        from dvfopt_gui.strategy_params import strategy_class_for

        assert strategy_class_for('slp@tet3d') is SLPStrategy


# ---------------------------------------------------------------------------
# Injectivity-gap view
# ---------------------------------------------------------------------------


class TestInjectivityGapView:
    def test_min_gap_2d_identity_is_unity(self):
        q = _min_gap_2d(np.zeros((2, 5, 6)))
        assert np.allclose(q, 1.0)

    def test_min_gap_2d_flags_crossing(self):
        phi = np.zeros((2, 5, 6))
        phi[1, :, 1] = -1.5  # deformed X col 1 crosses col 0
        q = _min_gap_2d(phi)
        assert (q[:, 0] < 0).all() and (q[:, 1] < 0).all()
        assert np.allclose(q[:, 4], 1.0)

    def test_metric_field_3d_inj_kind(self):
        from dvfopt.jacobian.monotonicity import injectivity_quality_3d

        vol = _folded_volume()
        np.testing.assert_array_equal(_metric_field_3d(vol, 'inj3d'), injectivity_quality_3d(vol))

    def test_view_combo_contains_inj(self, qapp):
        from dvfopt_gui.app import LiveSolverWindow

        win = LiveSolverWindow()
        try:
            datas = [win._view_combo.itemData(i) for i in range(win._view_combo.count())]
            assert VIEW_INJ in datas
        finally:
            win.close()


# ---------------------------------------------------------------------------
# Optional-float / filtered / disabled strategy params
# ---------------------------------------------------------------------------


class TestStrategyParamsNewKinds:
    def test_injectivity_threshold_is_optfloat(self):
        from dvfopt import SLSQPWindowedStrategy
        from dvfopt_gui.strategy_params import editable_fields

        kinds = {name: kind for name, kind, _d in editable_fields(SLSQPWindowedStrategy)}
        assert kinds['injectivity_threshold'] == 'optfloat'
        assert kinds['enforce_injectivity'] == 'bool'

    def test_windowed_2d_tab_shows_only_toggles(self, qapp):
        from dvfopt_gui.strategy_params import StrategyParamsTab

        tab = StrategyParamsTab()
        tab.build('slsqp_windowed', {})
        assert set(tab._widgets) == {
            'enforce_shoelace',
            'enforce_injectivity',
            'injectivity_threshold',
        }

    def test_optfloat_roundtrip(self, qapp):
        from dvfopt_gui.strategy_params import StrategyParamsTab

        tab = StrategyParamsTab()
        tab.build('slsqp_windowed', {'injectivity_threshold': 0.15})
        _kind, w = tab._widgets['injectivity_threshold']
        assert w._opt_check.isChecked()
        assert w._opt_spin.value() == pytest.approx(0.15)
        assert tab.values()['injectivity_threshold'] == pytest.approx(0.15)
        # Unchecking returns to None = default -> no override emitted.
        w._opt_check.setChecked(False)
        assert 'injectivity_threshold' not in tab.values()

    def test_shoelace_disabled_on_jdet3d(self, qapp):
        from dvfopt_gui.strategy_params import StrategyParamsTab

        tab = StrategyParamsTab()
        tab.build('slsqp_windowed@jdet3d', {})
        _kind, w = tab._widgets['enforce_shoelace']
        assert not w.isEnabled()

    def test_optfloat_override_validation(self):
        from dvfopt_gui.strategy_params import _valid_override

        assert _valid_override('optfloat', 'injectivity_threshold', None)
        assert _valid_override('optfloat', 'injectivity_threshold', 0.15)
        assert not _valid_override('optfloat', 'injectivity_threshold', 'nope')
        assert not _valid_override('optfloat', 'injectivity_threshold', float('nan'))


# ---------------------------------------------------------------------------
# Stage snapshots + solve_info + persistence round-trip
# ---------------------------------------------------------------------------


class TestStagesAndSolveInfo:
    def test_snapshot_stage_defaults_none(self):
        snap = _volume_snapshot(np.zeros((3, 3, 4, 4)), n_neg=0, min_T=1.0, outer_iter=0)
        assert snap.stage is None
        snap2 = _volume_snapshot(
            np.zeros((3, 3, 4, 4)), n_neg=0, min_T=1.0, outer_iter=1, stage='alm'
        )
        assert snap2.stage == 'alm'

    def test_worker_records_solve_info_and_stages(self):
        """A tiny Solver-path run records SolveInfo phases and marks the
        input/final snapshots with stage names."""
        w = SolverWorker(
            deformation_i=_folded_slice(),
            method_id='barrier_2tri',
            params={'threshold': 0.01, 'objective_id': 'l2'},
        )
        w.run()  # synchronous call — no thread needed for the tiny field
        assert w.solve_info is not None and len(w.solve_info.phases) > 0
        assert w.history_get(0).stage == 'input'

    def test_persistence_roundtrip_keeps_stage(self, tmp_path):
        from dvfopt_gui.persistence import build_save_payload, parse_loaded

        vol = _folded_volume()
        snaps = [
            _volume_snapshot(vol, n_neg=5, min_T=-0.2, outer_iter=0, stage='input'),
            _volume_snapshot(vol, n_neg=1, min_T=0.001, outer_iter=1, stage='harmonic'),
            _volume_snapshot(vol, n_neg=0, min_T=0.02, outer_iter=2, stage='final'),
        ]
        payload = build_save_payload(
            phi_active=np.stack([vol[1, 0], vol[2, 0]]),
            full_volume=vol,
            z=0,
            constraint='tet3d',
            method='m14',
            objective='l1',
            time_budget_s=60.0,
            max_iterations=80,
            history_max_size=100,
            history_snaps=snaps,
            history_total=3,
            input_volume=vol,
            dim=3,
        )
        path = tmp_path / 'run.npz'
        np.savez_compressed(path, **payload)
        loaded = np.load(path, allow_pickle=False)
        try:
            run = parse_loaded(loaded)
        finally:
            loaded.close()
        assert [s.stage for s in run.snapshots] == ['input', 'harmonic', 'final']


# ---------------------------------------------------------------------------
# Convergence plot: markers + threshold
# ---------------------------------------------------------------------------


class TestConvergenceMarkers:
    def test_stage_markers_and_threshold(self, qapp):
        from dvfopt_gui.convergence import ConvergencePlot

        plot = ConvergencePlot()
        plot.set_data([0, 1, 2], [5, 2, 0], [-0.5, 0.001, 0.02])
        plot.set_threshold(0.01)
        plot.set_stage_markers([1, 2], ['harmonic', 'final'])
        assert len(plot._stage_lines) == 2
        assert plot._thr_line.isVisible()
        # Replacing markers clears the old ones; clear_data clears all.
        plot.set_stage_markers([2], ['final'])
        assert len(plot._stage_lines) == 1
        plot.clear_data()
        assert len(plot._stage_lines) == 0


# ---------------------------------------------------------------------------
# Log dock
# ---------------------------------------------------------------------------


class TestLogDock:
    def test_attach_receives_log_lines(self, qapp):
        from dvfopt._logging import log_info, log_warning
        from dvfopt_gui.logdock import LogDock

        dock = LogDock()
        dock.attach()
        try:
            dock._level_combo.setCurrentIndex(1)  # Info
            log_info('hello from the solver')
            log_warning('cluster solve FAILED: boom')
            qapp.processEvents()
            text = dock._text.toPlainText()
            assert 'hello from the solver' in text
            assert 'cluster solve FAILED: boom' in text
        finally:
            dock.detach()

    def test_warning_level_filters_info(self, qapp):
        from dvfopt._logging import log_info, log_warning
        from dvfopt_gui.logdock import LogDock

        dock = LogDock()
        dock.attach()
        try:
            dock._level_combo.setCurrentIndex(0)  # Warnings
            log_info('quiet info line')
            log_warning('loud warning line')
            qapp.processEvents()
            text = dock._text.toPlainText()
            assert 'quiet info line' not in text
            assert 'loud warning line' in text
        finally:
            dock.detach()

    def test_level_maps_to_worker_verbose(self, qapp):
        from dvfopt_gui.logdock import LogDock

        dock = LogDock()
        seen = []
        dock.verboseChanged.connect(seen.append)
        dock._level_combo.setCurrentIndex(2)  # Debug
        assert dock.worker_verbose == 2
        dock._level_combo.setCurrentIndex(0)  # Warnings
        assert dock.worker_verbose == 0
        assert seen == [2, 0]

    def test_detach_idempotent(self, qapp):
        from dvfopt_gui.logdock import LogDock

        dock = LogDock()
        dock.attach()
        dock.detach()
        dock.detach()  # no-op


# ---------------------------------------------------------------------------
# Window wiring: report action + log verbose threading
# ---------------------------------------------------------------------------


class TestWindowWiring:
    def test_report_action_enables_and_saves(self, qapp, tmp_path, monkeypatch):
        from dvfopt.solver import PhaseInfo, SolveInfo
        from dvfopt_gui.app import LiveSolverWindow

        win = LiveSolverWindow()
        try:
            assert not win._report_action.isEnabled()
            info = SolveInfo(
                strategy_name='BarrierStrategy',
                phases=[
                    PhaseInfo(name='penalty', n_iter=3, wall_s=0.1, n_neg=2, min_T=-0.1),
                    PhaseInfo(name='barrier', n_iter=2, wall_s=0.2, n_neg=0, min_T=0.02),
                ],
            )
            win._last_solve_info = info
            out = tmp_path / 'report.png'
            monkeypatch.setattr(
                QtWidgets.QFileDialog,
                'getSaveFileName',
                staticmethod(lambda *a, **k: (str(out), 'Images (*.png *.pdf)')),
            )
            win._on_save_report()
            assert out.exists() and out.stat().st_size > 0
        finally:
            win.close()

    def test_log_verbose_reaches_worker_params(self, qapp):
        from dvfopt_gui.app import LiveSolverWindow

        win = LiveSolverWindow()
        try:
            win._log_dock._level_combo.setCurrentIndex(1)  # Info -> verbose 1
            win._load_array(_folded_slice())
            win._start_worker(win._current_slice())
            try:
                assert win._worker._params.get('verbose') == 1
                win._worker.request_stop()
            finally:
                win._worker.wait(5000)
        finally:
            win.close()


class TestReviewRegressions:
    """Pinned fixes from the branch review."""

    def test_jdet3d_windowed_tab_keeps_iteration_knobs(self, qapp):
        """The 2D whitelist must NOT bleed onto the 3D windowed tab —
        for the 3D path the Params tab is the only route to the
        iteration knobs (the toolbar spinbox only reaches the 2D path)."""
        from dvfopt_gui.strategy_params import StrategyParamsTab

        tab = StrategyParamsTab()
        tab.build('slsqp_windowed@jdet3d', {})
        assert 'max_iterations' in tab._widgets
        assert 'max_minimize_iter' in tab._widgets

    def test_disabled_field_never_emitted_and_sanitized(self, qapp):
        """Greying is not enough: values() must skip disabled widgets and
        the window-side sanitizer must strip persisted overrides for
        them (stale QSettings bypass the dialog entirely)."""
        from dvfopt_gui._win_run import _sanitized_overrides
        from dvfopt_gui.strategy_params import StrategyParamsTab

        tab = StrategyParamsTab()
        # A stale persisted override for the disabled field...
        tab.build('slsqp_windowed@jdet3d', {'enforce_shoelace': True})
        # ...is never re-emitted by the dialog...
        assert 'enforce_shoelace' not in tab.values()
        # ...and is stripped at the single override-application point.
        out = _sanitized_overrides(
            'slsqp_windowed@jdet3d', {'enforce_shoelace': True, 'max_iterations': 50}
        )
        assert out == {'max_iterations': 50}
        # Algos without a disabled set pass through untouched.
        assert _sanitized_overrides('m14', {'margin': 1e-3}) == {'margin': 1e-3}

    def test_optfloat_detected_by_annotation_not_name(self):
        """float|None knobs render as optfloat via their dataclass
        annotation — a new such knob needs no registry entry."""
        from dvfopt import ActiveBandALM3DStrategy
        from dvfopt_gui.strategy_params import editable_fields

        kinds = {name: kind for name, kind, _d in editable_fields(ActiveBandALM3DStrategy)}
        assert kinds['band_threshold'] == 'optfloat'


class TestReviewRound3:
    """Pinned fixes from review rounds 2-3 (lifecycle, staleness, coercion)."""

    def test_logdock_attach_detach_restores_logger_state(self, qapp):
        import logging

        from dvfopt._logging import logger as dlog
        from dvfopt_gui.logdock import LogDock

        prev_level, prev_prop = dlog.level, dlog.propagate
        dock = LogDock()
        dock.attach()
        assert dlog.level == logging.DEBUG and dlog.propagate is False
        dock.detach()
        assert dlog.level == prev_level and dlog.propagate == prev_prop
        # Re-attach after detach must still deliver records (the handler
        # is NOT closed on detach).
        from dvfopt._logging import log_warning

        dock.attach()
        try:
            log_warning('re-attach works')
            qapp.processEvents()
            assert 're-attach works' in dock._text.toPlainText()
        finally:
            dock.detach()

    def test_logdock_coalesces_bursts(self, qapp):
        """A burst of records produces at most one outstanding signal —
        and every line still lands after a single drain."""
        from dvfopt._logging import log_info
        from dvfopt_gui.logdock import LogDock

        dock = LogDock()
        dock.attach()
        try:
            dock._level_combo.setCurrentIndex(1)  # Info
            for i in range(50):
                log_info(f'burst line {i}')
            qapp.processEvents()
            text = dock._text.toPlainText()
            assert 'burst line 0' in text and 'burst line 49' in text
        finally:
            dock.detach()

    def test_clear_data_hides_threshold_line(self, qapp):
        from dvfopt_gui.convergence import ConvergencePlot

        plot = ConvergencePlot()
        plot.set_threshold(0.01)
        assert plot._thr_line.isVisible()
        plot.clear_data()
        assert not plot._thr_line.isVisible()

    def test_stage_marker_braces_do_not_crash(self, qapp):
        from dvfopt_gui.convergence import ConvergencePlot

        plot = ConvergencePlot()
        plot.set_stage_markers([1], ['bulk:{m14}'])  # format-template braces
        assert len(plot._stage_lines) == 1

    def test_optfloat_rejects_bool_and_str(self):
        from dvfopt_gui.strategy_params import _valid_override

        assert not _valid_override('optfloat', 'injectivity_threshold', True)
        assert not _valid_override('optfloat', 'injectivity_threshold', '0.01')
        assert _valid_override('optfloat', 'injectivity_threshold', 0.01)

    def test_auto_resolved_strategy_honors_time_budget(self):
        """Auto runs must honor the toolbar budget like explicit menu
        entries do (the wallbreaker would otherwise self-terminate at
        its dataclass default regardless of the spinbox)."""
        import dataclasses

        vol = _folded_volume()
        vol[1, 1:3, 2:4, 2:4] -= 30.0  # extreme -> wallbreaker tier
        w = SolverWorker(
            deformation_i=vol,
            method_id='auto_tet3d',
            params={'time_budget_s': 123.0, 'objective_id': 'l1'},
        )
        strategy = w._build_strategy()
        if dataclasses.is_dataclass(strategy) and any(
            f.name == 'time_budget_s' for f in dataclasses.fields(strategy)
        ):
            assert strategy.time_budget_s == 123.0

    def test_windowed_overrides_type_coerced(self, monkeypatch):
        """Bool/str values from a hand-edited settings file must not
        reach iterative_serial as the injectivity threshold."""
        captured = {}

        def fake_serial(deformation, **kwargs):
            captured.update(kwargs)
            return np.stack([deformation[1, 0], deformation[2, 0]])

        import dvfopt.core.slsqp.iterative as it_mod

        monkeypatch.setattr(it_mod, 'iterative_serial', fake_serial)
        w = SolverWorker(
            deformation_i=_folded_slice(),
            method_id='slsqp_windowed_2tri',
            params={
                'threshold': 0.01,
                'strategy_overrides': {
                    'enforce_injectivity': True,
                    'injectivity_threshold': True,  # poisoned value
                },
            },
        )
        w._run_windowed_slsqp(enforce_triangles=True)
        assert captured.get('enforce_injectivity') is True
        assert 'injectivity_threshold' not in captured

    def test_report_resets_on_new_load(self, qapp):
        from dvfopt.solver import PhaseInfo, SolveInfo
        from dvfopt_gui.app import LiveSolverWindow

        win = LiveSolverWindow()
        try:
            win._last_solve_info = SolveInfo(
                strategy_name='X', phases=[PhaseInfo(name='p', wall_s=0.1)]
            )
            win._report_action.setEnabled(True)
            win._load_array(_folded_slice())
            assert win._last_solve_info is None
            assert not win._report_action.isEnabled()
        finally:
            win.close()
