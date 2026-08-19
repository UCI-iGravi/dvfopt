"""Run/stop/pipeline orchestration for LiveSolverWindow.

Extracted verbatim from ``app.py``'s ``LiveSolverWindow`` — pure code
motion. The mixin holds no state of its own; every attribute it touches
is created in ``LiveSolverWindow.__init__``.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from dvfopt_gui._shared import (
    CONSTRAINT_TET3D,
    DEFAULT_CONSTRAINT,
    _compose_method_id,
)
from dvfopt_gui.worker import (
    SolverWorker,
    _metric_counts,
    _metric_counts_3d,
)


class RunActionsMixin:
    # ----- run buttons -------------------------------------------------------

    def _on_run(self, *, use_roi: bool):
        if self._volume is None:
            QtWidgets.QMessageBox.information(self, 'No DVF', 'Load a DVF first via "Load DVF…".')
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, 'Already running', 'Stop the current run first.'
            )
            return

        if self._is_3d_run:
            if use_roi:
                D, H, W = self._volume.shape[1:]
                x, y = self._section_roi.pos()
                w, h = self._section_roi.size()
                y0, x0 = max(0, round(y)), max(0, round(x))
                y1, x1 = min(H, round(y + h)), min(W, round(x + w))
                z0, z1 = int(self._z0_spin.value()), int(self._z1_spin.value())
                if z1 < z0:
                    z0, z1 = z1, z0
                z1ex = z1 + 1
                if (z1ex - z0) < 3 or (y1 - y0) < 3 or (x1 - x0) < 3:
                    QtWidgets.QMessageBox.warning(
                        self, 'Section too small', 'The 3D section must be at least 3×3×3.'
                    )
                    return
                self._section_bounds_3d = (z0, z1ex, y0, y1, x0, x1)
                self.statusBar().showMessage(
                    'Run section (3D): solving the sub-volume — check the box '
                    'boundary for seam folds after it completes.',
                    6_000,
                )
                sub = self._original_volume[:, z0:z1ex, y0:y1, x0:x1].copy()
                self._start_worker(sub)
                return
            self._section_bounds_3d = None
            self._section_bounds = None
            self._start_worker(self._original_volume.copy())
            return

        deformation_i = self._current_slice()
        H, W = deformation_i.shape[2:]
        if use_roi:
            x, y = self._section_roi.pos()
            w, h = self._section_roi.size()
            y0 = max(0, round(y))
            x0 = max(0, round(x))
            y1 = min(H, round(y + h))
            x1 = min(W, round(x + w))
            if y1 - y0 < 3 or x1 - x0 < 3:
                QtWidgets.QMessageBox.warning(
                    self, 'Section too small', 'The ROI must be at least 3x3.'
                )
                return
            self._section_bounds = (y0, y1, x0, x1)
            self._section_bounds_3d = None
            sub = deformation_i[:, :, y0:y1, x0:x1].copy()
            # The ROI is solved in isolation (frozen edges) then spliced
            # back, so new folds can appear at the seam where the patched
            # region meets the untouched field — most likely for
            # context-dependent methods. Flag it rather than let a
            # surprise boundary fold look like a solver failure.
            self.statusBar().showMessage(
                'Run section: solving the ROI with frozen edges — check the '
                'patch boundary for new seam folds after it completes.',
                6_000,
            )
            self._start_worker(sub)
        else:
            self._section_bounds = None
            self._section_bounds_3d = None
            self._start_worker(deformation_i)

    def _start_worker(self, deformation_i: np.ndarray, method_id: str | None = None):
        objective_id = self._objective_combo.currentData()
        if method_id is None:
            algo = self._method_combo.currentData()
            constraint = self._constraint_combo.currentData()
            method_id = _compose_method_id(algo, constraint)
        else:
            constraint = self._constraint_combo.currentData()

        # Baseline fold count of *this run's* input (full slice or ROI),
        # counted with the SAME metric the run's trajectory uses (Jdet for
        # the windowed-SLSQP path, else the constraint's own metric), so
        # the before→after delta lines up with the live n_neg readout.
        if self._is_3d_run:
            # 3D run: count folds over the whole volume with the run's 3D
            # metric, matching the per-step snapshot's n_neg (the "after").
            kind = 'tet3d' if constraint == CONSTRAINT_TET3D else 'jdet3d'
            self._input_n_neg, _ = _metric_counts_3d(deformation_i, kind)
        else:
            phi_in = np.stack([deformation_i[1, 0], deformation_i[2, 0]])
            metric_kind = 'jdet' if method_id.startswith('slsqp_windowed') else constraint
            self._input_n_neg, _ = _metric_counts(phi_in, metric_kind)
        self._active_method_id = method_id
        self._auto_label_shown = False
        self._run_elapsed.restart()
        params = {
            'time_budget_s': float(self._budget_spin.value()),
            'max_iterations': int(self._max_iter_spin.value()),
            'threshold': self._display_threshold(),
            'objective_id': objective_id,
            'method_name': self._slsqp_method_name,
            'strategy_overrides': self._strategy_overrides.get(self._current_params_algo(), {}),
            # Log-dock level → solver verbose (0 = warnings only).
            'verbose': int(getattr(self, '_log_verbose', 0)),
        }
        if self._max_per_index_iter is not None:
            params['max_per_index_iter'] = int(self._max_per_index_iter)
        self._worker = SolverWorker(
            deformation_i=deformation_i,
            method_id=method_id,
            params=params,
            history_max_size=self._history_max_size,
            parent=self,
        )
        self._worker.finishedWithResult.connect(self._on_finished)
        self._worker.errored.connect(self._on_error)
        self._stop_btn.setEnabled(True)
        self._run_full_btn.setEnabled(False)
        self._run_roi_btn.setEnabled(False)
        self._run_all_btn.setEnabled(False)
        self._pipeline_btn.setEnabled(False)
        self._undo_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        # Freeze the z-slider during a run: switching slices mid-solve
        # would orphan the in-flight worker against a different slice.
        self._z_slider.setEnabled(False)
        self._fps_label.setText(f'starting {method_id}…')
        self._last_count = 0
        self._last_tick.restart()
        # Reset the history widgets for the new run. Re-engage Live so
        # the first snapshots from the new worker auto-track.
        self._history.begin_run()
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._stop_btn.setText('Stopping…')
            # The stop flag is only checked when the solver next fires its
            # step_callback (between SLSQP sub-windows / at the next
            # wallbreaker stage boundary), so there can be a one-checkpoint
            # delay — say so rather than leaving the user with a frozen
            # "Stopping…" button.
            self.statusBar().showMessage(
                'Stop requested — will halt at the next solver checkpoint…', 0
            )

    def _on_run_all(self):
        """Solve every z-slice of a 3D volume in sequence with the
        current method. The chain is driven from ``_on_finished``: each
        slice's result splices in, then the next slice starts."""
        if self._volume is None:
            QtWidgets.QMessageBox.information(self, 'No DVF', 'Load a DVF first via "Load DVF…".')
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, 'Already running', 'Stop the current run first.'
            )
            return
        if self._is_3d_run:
            # 3D solves the whole volume in one run; "Run all z" maps to a
            # single full-volume run.
            self._on_run(use_roi=False)
            return
        D = self._volume.shape[1]
        if D <= 1:
            # Single slice — just run it normally.
            self._on_run(use_roi=False)
            return
        # One undo entry for the whole batch: snapshot the pre-batch
        # volume now and suppress the per-slice pushes in ``_on_finished``,
        # so a single Ctrl+Z reverts the entire Run-all (not just its last
        # slice).
        self._push_undo_state()
        self._begin_run_all_batch()

    def _begin_run_all_batch(self) -> None:
        """Start the per-slice batch WITHOUT pushing an undo entry (callers
        own the undo semantics: Run-all pushes one; the full pipeline pushes
        one covering both stages)."""
        D = self._volume.shape[1]
        self._run_all_remaining = list(range(D))
        self._run_all_step()

    def _run_all_step(self):
        """Start the next queued slice in a Run-all batch, or finish the
        batch if the queue is empty."""
        if not self._run_all_remaining and self._pipeline_after_run_all:
            self._run_all_remaining = None
            self._pipeline_after_run_all = False
            self.statusBar().showMessage('Pipeline: 2.5D marching…', 0)
            self._start_marching_25d()
            return
        if not self._run_all_remaining:
            self._run_all_remaining = None
            self._finalize_run_ui()
            # The per-slice gate in ``_on_finished`` skipped every restart
            # during the batch (``_run_all_remaining`` was non-None then) —
            # this drain branch is the batch's actual end, so it owns the
            # one restart that reflects the whole batch's splices. The
            # pipeline chain-trigger branch above needs no equivalent call:
            # it hands off to the 2.5D marching worker, whose own finish
            # later runs with ``_run_all_remaining`` already None, so the
            # gate in ``_on_finished`` fires naturally for it.
            self._restart_overview()
            self.statusBar().showMessage('Run all z finished.', 10_000)
            return
        z = self._run_all_remaining.pop(0)
        D = self._volume.shape[1]
        self._z = z
        self._z_label.setText(f'{z} / {D - 1}')
        self._section_bounds = None
        self._section_bounds_3d = None
        remaining = len(self._run_all_remaining)
        self.statusBar().showMessage(f'Run all z: solving slice {z} ({D - remaining}/{D})…', 0)
        # Solve from the pristine input for this slice.
        self._start_worker(self._original_volume[:, z : z + 1].copy())

    def _on_run_25d(self):
        """Run 2.5D marching on the CURRENT volume (which must be per-slice
        corrected: dz == 0)."""
        if self._volume is None:
            QtWidgets.QMessageBox.information(self, 'No DVF', 'Load a DVF first via "Load DVF…".')
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, 'Already running', 'Stop the current run first.'
            )
            return
        if self._volume.shape[1] <= 1:
            QtWidgets.QMessageBox.information(
                self, '2.5D needs a volume', '2.5D marching needs a (3, D>1, H, W) volume.'
            )
            return
        max_dz = float(np.abs(self._volume[0]).max())
        if max_dz > 1e-9:
            ans = QtWidgets.QMessageBox.question(
                self,
                'dz is not zero',
                '2.5D marching requires dz == 0 (its input is per-slice '
                f'2D-corrected data). This volume has max |dz| = {max_dz:.3g}.\n\n'
                'Zero the dz channel and run the full pipeline (per-slice 2D, '
                'then 2.5D)? The change is one undoable operation.',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ans == QtWidgets.QMessageBox.Yes:
                self._on_run_pipeline_full(zero_dz=True)
            return
        self._start_marching_25d()

    def _start_marching_25d(self) -> None:
        """Launch the 2.5D worker on the CURRENT volume (the deliberate
        exception to the runs-read-the-pristine-original rule — the 2.5D
        input IS the per-slice-corrected state)."""
        self._select_combo_data(self._constraint_combo, CONSTRAINT_TET3D)
        self._section_bounds = None
        self._section_bounds_3d = None
        self.statusBar().showMessage('2.5D marching…', 0)
        self._start_worker(self._volume.copy(), method_id='marching25d_tet3d')

    def _on_run_pipeline_full(self, *, zero_dz: bool = False):
        """One-click production workflow: per-slice 2D (selected method) →
        2.5D marching, as a single undoable operation.

        ``zero_dz`` is the explicit-consent path from ``_on_run_25d``'s
        dz-violation dialog (the user agreed to zero dz so the 2.5D
        stage's ``dz == 0`` precondition holds). Keyword-only: this slot
        is wired directly to ``QAction.triggered`` (Pipeline ▾ menu and
        the Run menu below), which passes a positional ``checked: bool`` —
        a positional ``zero_dz`` would silently receive that bool instead
        of its default.
        """
        if self._volume is None:
            QtWidgets.QMessageBox.information(self, 'No DVF', 'Load a DVF first via "Load DVF…".')
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, 'Already running', 'Stop the current run first.'
            )
            return
        if self._volume.shape[1] <= 1:
            QtWidgets.QMessageBox.information(
                self, 'Pipeline needs a volume', 'The full pipeline needs a (3, D>1, H, W) volume.'
            )
            return
        if not zero_dz and float(np.abs(self._volume[0]).max()) > 1e-9:
            ans = QtWidgets.QMessageBox.question(
                self,
                'dz is not zero',
                'The 2.5D stage requires dz == 0, but this volume has a '
                'nonzero dz channel — the pipeline would fail after the '
                'per-slice 2D stage.\n\nZero the dz channel first (one '
                'undoable operation together with the pipeline)?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
            zero_dz = True
        if self._is_3d_run:
            # Per-slice stage needs a 2D method; drop back to the 2-tri family.
            self._select_combo_data(self._constraint_combo, DEFAULT_CONSTRAINT)
        self._push_undo_state()
        if zero_dz:
            # Zero BOTH: the per-slice 2D stage below reads
            # ``_original_volume`` per slice (see ``_run_all_step``) and
            # the 2.5D stage reads ``self._volume`` (see
            # ``_start_marching_25d``). ``_original_volume`` is
            # intentionally zeroed too -- the pipeline's semantic input is
            # the dz-stripped field, not the as-loaded one. The undo entry
            # just pushed above snapshotted the pre-zero ``self._volume``,
            # so Ctrl+Z after this restores the displayed volume's dz;
            # ``_original_volume`` isn't on the undo stack and stays
            # zeroed, matching the dialog's "one undoable operation".
            self._volume[0] = 0.0
            self._original_volume[0] = 0.0
        self._pipeline_active = True
        self._pipeline_after_run_all = True
        self.statusBar().showMessage('Pipeline: per-slice 2D…', 0)
        self._begin_run_all_batch()

    def _on_finished(self, phi_out, info):
        # Ignore late signals from a worker we've already replaced /
        # discarded (e.g. user loaded a new DVF mid-run). ``sender()`` is
        # only non-None inside a slot invoked via an actual signal
        # emission — a direct Python call (e.g. from a test) always sees
        # None, in which case there is no "other" worker to guard against
        # and we trust ``self._worker`` as given. A real, stale signal
        # (sender() is some past worker, no longer ``self._worker``) is
        # still rejected: without that, the old worker's phi_out would
        # get spliced into the *new* volume.
        sender = self.sender()
        if sender is not None and sender is not self._worker:
            return
        # Capture the run's recorded SolveInfo (Solver-path runs record
        # phase history) for the View → Save convergence report action.
        if self._worker is not None:
            solve_info = getattr(self._worker, 'solve_info', None)
            if solve_info is not None and getattr(solve_info, 'phases', None):
                self._last_solve_info = solve_info
                self._report_action.setEnabled(True)
        # Splice the result back into the volume so subsequent runs /
        # view toggles see the corrected state.
        if phi_out is not None and self._volume is not None:
            # Snapshot the pre-splice volume for Undo — but only for a
            # standalone run. A Run-all batch pushes one undo entry up
            # front (see ``_on_run_all``) so the whole batch undoes as a
            # unit rather than one slice at a time; the full pipeline
            # (``_on_run_pipeline_full``) likewise pushes ONE entry
            # covering both the per-slice batch and the 2.5D stage, so
            # every worker finish along the way must skip this too.
            if self._run_all_remaining is None and not self._pipeline_active:
                self._push_undo_state()
            phi_out = np.asarray(phi_out)
            if phi_out.ndim == 4:  # full-volume or 3D-ROI result [dz,dy,dx]
                sb3 = self._section_bounds_3d
                if sb3 is not None:
                    z0, z1ex, y0, y1, x0, x1 = sb3
                    self._volume[:, z0:z1ex, y0:y1, x0:x1] = phi_out
                else:
                    self._volume[...] = phi_out
            else:
                sb = self._section_bounds
                if sb is not None:
                    y0, y1, x0, x1 = sb
                    self._volume[1, self._z, y0:y1, x0:x1] = phi_out[0]
                    self._volume[2, self._z, y0:y1, x0:x1] = phi_out[1]
                else:
                    self._volume[1, self._z] = phi_out[0]
                    self._volume[2, self._z] = phi_out[1]
            self._refresh_display_from_volume()
            # The run is over and its result is spliced into the volume —
            # drop the last streamed snapshot so idle-path readers
            # (inspector, view toggles, thr-spin repaints) all see the
            # volume, exactly like after load/undo (which also set None).
            self._latest = None
            # Skip the (expensive, full-volume) overview recompute for a
            # per-slice finish mid-batch (Run-all / the pipeline's per-slice
            # stage) — ``_run_all_remaining`` is a list, not None, for every
            # finish except a standalone run or the 2.5D marching stage's
            # own finish. The batch's own end-of-run restarts are triggered
            # explicitly: see ``_run_all_step``'s drain branch.
            if self._run_all_remaining is None:
                self._restart_overview()
        report = getattr(self._worker, 'pipeline_report', None)
        if report is not None:
            self.statusBar().showMessage(
                f'Pipeline: {report.n_neg_in} → {report.n_neg_out} folds, '
                f'feasible={report.feasible}, {report.wall_s:.0f}s',
                15_000,
            )
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText('Stop')

        # Run-all chaining: a clean finish advances to the next slice; a
        # stop/interrupt (info set) aborts the remaining batch.
        if self._run_all_remaining is not None:
            if info is not None:
                self._run_all_remaining = None
                self._pipeline_active = False
                self._pipeline_after_run_all = False
                self._finalize_run_ui()
                self.statusBar().showMessage(f'Run all z stopped: {info}.', 10_000)
            else:
                self._run_all_step()
            return

        self._finalize_run_ui()
        if info is not None:
            self.statusBar().showMessage(f'Run stopped: {info}.', 10_000)
        elif report is None:
            # When a pipeline_report was surfaced above, that message IS
            # the finish message for this run — don't clobber it with the
            # generic one.
            self.statusBar().showMessage('Run finished.', 10_000)

    def _on_error(self, err: str):
        sender = self.sender()
        if sender is not None and sender is not self._worker:
            return
        self._run_all_remaining = None
        self._pipeline_active = False
        self._pipeline_after_run_all = False
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText('Stop')
        self._finalize_run_ui()
        QtWidgets.QMessageBox.critical(self, 'Solver error', err)
        self._fps_label.setText('errored')

    def _finalize_run_ui(self) -> None:
        """Restore the toolbar to its idle state after a run (or batch)
        ends. Re-enables the run buttons + z-slider as appropriate."""
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText('Stop')
        self._run_full_btn.setEnabled(True)
        D = self._volume.shape[1] if self._volume is not None else 1
        self._z_slider.setEnabled(D > 1)
        self._apply_mode_gating()
        self._update_undo_redo_enabled()
        self._fps_label.setText('idle')
        # Clear the progress bar (unless a Run-all batch is still going —
        # the next slice's first tick will repaint it).
        if self._run_all_remaining is None:
            self._active_method_id = None
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setFormat('')
            # The pipeline (if any) is over once its last worker finalizes
            # with no batch left to run. ``_pipeline_after_run_all`` is
            # cleared separately, right where the chain either fires (see
            # ``_run_all_step``) or the run/batch is aborted (see above
            # and ``_on_error``) — by the time we get here it only needs
            # to have lived long enough for THIS call's undo-suppression
            # check above, which already ran.
            self._pipeline_active = False

    def _update_progress(self) -> None:
        """Repaint the progress bar for the active run. Wallbreakers show
        elapsed/budget, SLSQP shows outer-iter/max, the rest show a busy
        indicator with elapsed time."""
        mid = self._active_method_id
        worker = self._worker
        if mid is None or worker is None or not worker.isRunning():
            return
        elapsed = self._run_elapsed.elapsed() / 1000.0
        if mid == 'marching25d_tet3d':
            prog = getattr(worker, 'marching_progress', None)
            if prog is not None:
                phase, index, total, n_neg = prog
                self._progress.setRange(0, max(1, int(total)))
                self._progress.setValue(int(index))
                self._progress.setFormat(f'{phase} {index}/{total} · n_neg {n_neg}')
            else:
                self._progress.setRange(0, 0)
                self._progress.setFormat(f'{elapsed:.0f}s')
            return
        if mid == 'pipeline3d_tet3d':
            self._progress.setRange(0, 0)
            self._progress.setFormat(f'{elapsed:.0f}s')
            return
        if mid.startswith(('m10', 'm14')):
            budget = float(self._budget_spin.value())
            frac = min(1.0, elapsed / budget) if budget > 0 else 0.0
            self._progress.setRange(0, 100)
            self._progress.setValue(int(frac * 100))
            self._progress.setFormat(f'{elapsed:.0f}s / {budget:.0f}s')
        elif mid.startswith('slsqp_windowed'):
            mx = int(self._max_iter_spin.value())
            cur = self._latest.outer_iter if self._latest is not None else 0
            frac = min(1.0, cur / mx) if mx > 0 else 0.0
            self._progress.setRange(0, 100)
            self._progress.setValue(int(frac * 100))
            self._progress.setFormat(f'iter {cur} / {mx}  ·  {elapsed:.0f}s')
        else:
            # barrier / nmvf / slsqp_fullgrid: busy indicator + elapsed.
            self._progress.setRange(0, 0)
            self._progress.setFormat(f'{elapsed:.0f}s')
