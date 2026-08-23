"""File I/O + undo/redo for LiveSolverWindow.

Extracted verbatim from ``app.py``'s ``LiveSolverWindow`` — pure code
motion. The mixin holds no state of its own; every attribute it touches
is created in ``LiveSolverWindow.__init__``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtWidgets

from dvfopt_gui._shared import (
    DEFAULT_CONSTRAINT,
    _default_roi_geometry,
    validate_finite,
)
from dvfopt_gui.persistence import (
    LoadedRun,
    build_save_payload,
    normalise_to_volume,
)
from dvfopt_gui.worker import (
    LoadWorker,
    ReplayHistory,
)


class FileIOMixin:
    # ----- DVF loading -------------------------------------------------------

    def _on_load(self):
        # A load is already decoding on the worker thread — ignore re-entry
        # (the controls are disabled, but the Ctrl+O shortcut still fires).
        lw = self._load_worker
        if lw is not None and getattr(lw, 'isRunning', lambda: False)():
            return
        flt = 'DVF files (*.npy *.npz'
        from dvfopt.io.fields import SITK_EXTENSIONS, sitk_available

        if sitk_available():
            flt += ' ' + ' '.join(f'*{e}' for e in SITK_EXTENSIONS)
        flt += ');;NumPy arrays (*.npy);;NumPy compressed (*.npz)'
        if sitk_available():
            flt += ';;Medical images (' + ' '.join(f'*{e}' for e in SITK_EXTENSIONS) + ')'
        flt += ';;All files (*)'
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Load DVF (.npy or .npz)',
            self._last_dir,
            flt,
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        # Loading dispatches to a QThread (LoadWorker): GB-scale np.load +
        # float64 conversion + sitk decode no longer block the GUI thread.
        self._load_btn.setEnabled(False)
        self._load_action.setEnabled(False)
        self.statusBar().showMessage(f'Loading {Path(path).name}…', 0)
        self._load_worker = LoadWorker(path, parent=self)
        self._load_worker.loadedRun.connect(lambda run: self._on_load_finished(path, run))
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.start()

    def _on_load_finished(self, path: str, run) -> None:
        self._load_btn.setEnabled(True)
        self._load_action.setEnabled(True)
        if not self._apply_loaded_run(run):
            self.statusBar().clearMessage()
            return
        n_hist = len(run.snapshots)
        suffix = f'  ({n_hist} history step(s))' if n_hist else ''
        self.statusBar().showMessage(f'Loaded {path}{suffix}', 5_000)

    def _on_load_failed(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._load_action.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Load failed', msg)

    def _on_save(self):
        """Open a save dialog and write the current DVF + run history
        to a compressed NPZ. Schema documented in
        :meth:`_build_save_payload`.
        """
        if self._volume is None:
            QtWidgets.QMessageBox.information(
                self, 'Nothing to save', 'Load a DVF first via "Load DVF…".'
            )
            return
        # Suggest a filename that hints at the slice + method used.
        algo = self._method_combo.currentData() or 'noalgo'
        constraint = self._constraint_combo.currentData() or 'noconstraint'
        suggested = f'dvfopt_run_{algo}_{constraint}_z{self._z}.npz'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save DVF + optimization run (.npz)',
            str(Path(self._last_dir) / suggested),
            'NumPy compressed (*.npz);;All files (*)',
        )
        if not path:
            return
        if not path.lower().endswith('.npz'):
            path = path + '.npz'
        self._last_dir = str(Path(path).parent)
        try:
            payload = self._build_save_payload()
            np.savez_compressed(path, **payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Save failed', f'{type(exc).__name__}: {exc}')
            return
        n_steps = int(payload.get('n_history_steps', np.array(0)))
        self.statusBar().showMessage(
            f'Saved {Path(path).name}  ({n_steps} history step(s))', 10_000
        )

    def _on_export(self):
        """Write just the corrected volume (no run history) as .npy or, when
        SimpleITK is available, .nii.gz — for interop with the rest of the
        registration pipeline."""
        if self._volume is None:
            QtWidgets.QMessageBox.information(
                self, 'Nothing to export', 'Load a DVF first via "Load DVF…".'
            )
            return
        from dvfopt.io.fields import save_dvf_sitk, sitk_available

        filters = 'NumPy array (*.npy)'
        if sitk_available():
            filters += ';;NIfTI (*.nii.gz)'
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export corrected DVF', str(Path(self._last_dir) / 'corrected_dvf.npy'), filters
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        try:
            if 'NIfTI' in chosen or path.lower().endswith(('.nii', '.nii.gz')):
                if not path.lower().endswith(('.nii', '.nii.gz')):
                    path += '.nii.gz'
                save_dvf_sitk(path, self._volume)
            else:
                if not path.lower().endswith('.npy'):
                    path += '.npy'
                np.save(path, self._volume)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Export failed', f'{type(exc).__name__}: {exc}')
            return
        self.statusBar().showMessage(f'Exported {path}', 8_000)

    def _on_revert(self):
        """Discard all corrections: restore the originally-loaded volume
        and clear the run history. No-op (with a hint) if a solve is
        running or nothing is loaded."""
        if self._original_volume is None:
            QtWidgets.QMessageBox.information(
                self, 'Nothing to revert', 'Load a DVF first via "Load DVF…".'
            )
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, 'Run in progress', 'Stop the current run before reverting.'
            )
            return
        self._volume = self._original_volume.copy()
        self._worker = None
        self._latest = None
        self._latest_jacobian = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_undo_redo_enabled()
        self._history.reset()
        self._history.set_live(True)
        self._refresh_display_from_volume()
        self.statusBar().showMessage('Reverted to the loaded DVF.', 5_000)

    # ----- undo / redo -------------------------------------------------------

    def _cap_stack(self, stack: list) -> None:
        """Enforce the shared count + byte budget on an undo/redo stack
        (evicts oldest first; always keeps at least one entry)."""
        if len(stack) > self._UNDO_MAX:
            stack.pop(0)
        # Resolve through the app module at call time so app-level
        # monkeypatching of UNDO_MAX_BYTES (test contract) still works.
        from dvfopt_gui import app as _app

        while len(stack) > 1 and sum(v.nbytes for v in stack) > _app.UNDO_MAX_BYTES:
            stack.pop(0)

    def _push_undo_state(self) -> None:
        """Snapshot the current volume onto the undo stack (capped) and
        invalidate the redo stack. Called just before a run splices its
        result in."""
        if self._volume is None:
            return
        self._undo_stack.append(self._volume.copy())
        self._cap_stack(self._undo_stack)
        self._redo_stack.clear()
        self._update_undo_redo_enabled()

    def _on_undo(self):
        """Restore the volume to before the last applied correction."""
        if not self._undo_stack or (self._worker is not None and self._worker.isRunning()):
            return
        self._redo_stack.append(self._volume.copy())
        self._cap_stack(self._redo_stack)
        self._volume = self._undo_stack.pop()
        self._after_undo_redo('Undid last correction.')

    def _on_redo(self):
        """Re-apply the most recently undone correction."""
        if not self._redo_stack or (self._worker is not None and self._worker.isRunning()):
            return
        self._undo_stack.append(self._volume.copy())
        self._cap_stack(self._undo_stack)
        self._volume = self._redo_stack.pop()
        self._after_undo_redo('Redid correction.')

    def _after_undo_redo(self, message: str) -> None:
        """Shared tail for undo/redo: the restored volume has no live run,
        so drop the worker + history and repaint from the volume."""
        self._worker = None
        self._latest = None
        self._latest_jacobian = None
        self._history.reset()
        self._history.set_live(True)
        self._update_undo_redo_enabled()
        self._refresh_display_from_volume()
        self.statusBar().showMessage(message, 5_000)

    def _update_undo_redo_enabled(self) -> None:
        self._undo_btn.setEnabled(bool(self._undo_stack))
        self._redo_btn.setEnabled(bool(self._redo_stack))

    def _build_save_payload(self) -> dict:
        """Assemble the NPZ payload from the current window + worker state.

        Thin Qt adapter over :func:`dvfopt_gui.persistence.build_save_payload`
        — it just reads widget/worker state and hands plain values to the
        headless builder (which owns the schema; see that module's
        docstring).
        """
        worker = self._worker
        if worker is not None and worker.history_len() > 0:
            snaps = [worker.history_get(i) for i in range(worker.history_len())]
            history_total = worker.history_total
        else:
            snaps = []
            history_total = 0
        method = self._method_combo.currentData() or ''
        if method == 'auto' and getattr(worker, 'resolved_strategy_label', None):
            method = f'auto:{worker.resolved_strategy_label}'
        return build_save_payload(
            phi_active=self._volume[1:, self._z],
            full_volume=self._volume,
            z=self._z,
            constraint=self._constraint_combo.currentData() or '',
            method=method,
            objective=self._objective_combo.currentData() or '',
            time_budget_s=self._budget_spin.value(),
            max_iterations=self._max_iter_spin.value(),
            history_max_size=self._history_max_size,
            history_snaps=snaps,
            history_total=history_total,
            input_volume=self._original_volume,
            dim=3 if getattr(self, '_is_3d_run', False) else 2,
        )

    def _load_array(self, arr: np.ndarray) -> None:
        """Accept any of: ``(2, H, W)``, ``(3, H, W)``, ``(3, 1, H, W)``,
        ``(3, D, H, W)``. Normalises to a ``(3, D, H, W)`` volume and
        loads it as a fresh (history-less) DVF.

        Raises ``ValueError`` on any other shape.
        """
        self._apply_loaded_run(LoadedRun(volume=normalise_to_volume(arr)))

    def _apply_loaded_run(self, run: LoadedRun) -> bool:
        """Install a parsed :class:`LoadedRun` into the window. Returns
        False (state left untouched) if ``run.volume`` is rejected for
        non-finite values, else True on success.

        Handles both a bare DVF (``run.snapshots`` empty) and a full
        saved run — in the latter case the per-step snapshots are loaded
        into a :class:`ReplayHistory` so the scrub slider can replay the
        run, and the saved constraint/method/objective selections are
        restored to the toolbar.
        """
        msg = validate_finite(np.asarray(run.volume))
        if msg is not None:
            QtWidgets.QMessageBox.critical(self, 'Invalid DVF', msg)
            return False
        # A new dataset invalidates the previous run's phase history —
        # the report action must not render volume A's phases for B.
        self._last_solve_info = None
        self._report_action.setEnabled(False)
        self._volume = np.asarray(run.volume, dtype=np.float64)
        # Pristine copy of what was loaded — every Run reads its input
        # from here, never from ``self._volume`` (which is mutated by
        # ``_on_finished`` for the post-run display). Without this,
        # clicking Run twice would optimize the already-optimized
        # volume — history[0] would equal history[-1] and the scrub
        # slider would show "the same DVF" at both ends.
        #
        # A saved run carries its original pre-correction field as
        # ``input_volume``; prefer it so Revert and a fresh Run after
        # loading restore the *input*, not the already-corrected
        # ``phi_full_volume``. Bare DVFs / older archives fall back to the
        # loaded field itself.
        if run.input_volume is not None and run.input_volume.shape == self._volume.shape:
            self._original_volume = np.asarray(run.input_volume, dtype=np.float64)
        else:
            self._original_volume = self._volume.copy()
        D = self._volume.shape[1]
        self._z = max(0, min(D - 1, int(run.z)))
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(max(0, D - 1))
        self._z_slider.setValue(self._z)
        self._z_slider.setEnabled(D > 1)
        self._z_slider.blockSignals(False)
        self._z_label.setText(f'{self._z} / {D - 1}' if D > 1 else '0 / 0 (2D)')
        self._latest = None
        self._latest_jacobian = None
        self._picked_yx = None
        # A freshly loaded field starts a new correction lineage.
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_undo_redo_enabled()

        # Restore the toolbar selections a saved run carried (constraint
        # first, since it repopulates the method combo).
        if run.constraint:
            self._select_combo_data(self._constraint_combo, run.constraint)
        if run.method:
            self._select_combo_data(self._method_combo, run.method)
        if run.objective:
            self._select_combo_data(self._objective_combo, run.objective)

        # Show the ROI rectangle now that we have geometry to drag on.
        # Geometry is clamped to the field so it never overhangs (small
        # fields like the 7×7 bowtie default used to spill past the grid).
        H, W = self._volume.shape[2:]
        roi_x, roi_y, roi_w, roi_h = _default_roi_geometry(H, W)
        self._section_roi.setPos(roi_x, roi_y)
        self._section_roi.setSize([roi_w, roi_h])
        self._section_roi.setVisible(True)
        # Save is meaningful as soon as a DVF is loaded — even before
        # any solver run (you'd just get phi + minimal metadata).
        self._save_btn.setEnabled(True)
        self._revert_btn.setEnabled(True)
        self._run_all_btn.setEnabled(D > 1)
        self._update_3d_constraint_enabled()
        # A freshly-loaded D==1 field can't stay in a 3D constraint.
        if self._is_3d_run and D <= 1:
            self._select_combo_data(self._constraint_combo, DEFAULT_CONSTRAINT)
        self._apply_mode_gating()

        if run.snapshots:
            # Re-loaded run: wire the snapshots into a read-only history so
            # the slider can scrub them. The controller freezes Live and
            # parks the slider on the final step; we render that step.
            self._worker = ReplayHistory(run.snapshots, run.history_total)
            self._history.load_finished_run(len(run.snapshots))
            self._render_snapshot(run.snapshots[-1])
        else:
            # Fresh DVF: no prior run. Drop any worker reference so the
            # slider can't scrub stale snapshots, and reset to pristine.
            self._worker = None
            self._history.reset()
            self._history.set_live(True)
            self._refresh_display_from_volume()
        self._restart_overview()
        return True
