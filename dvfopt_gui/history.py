"""History-scrub controller for the live-viz window.

Owns the run-history scrub widgets (slider, ◀/▶ step buttons, absolute
step spinbox, total label, and the **Live** checkbox) and the small
state machine that keeps them consistent — including the
programmatic-vs-user-move guard that prevents the slider↔spinbox sync
from looping and the Live-mode auto-track from fighting a manual scrub.

Factored out of :class:`dvfopt_gui.app.LiveSolverWindow` so that state
machine lives in one cohesive place. The controller talks to the rest
of the window through two callbacks:

* ``get_worker()`` → the current worker (a live ``SolverWorker`` or a
  loaded :class:`~dvfopt_gui.worker.ReplayHistory`), or ``None``. It
  must expose ``history_len()``, ``history_get(i)`` and
  ``history_total``.
* ``render_snapshot(snap)`` → push a snapshot to the plot/panels.

The controller never touches the volume, the solver, or the plot
directly — it only drives its own widgets and asks the window to render
a chosen snapshot.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt5 import QtWidgets


class HistoryController:
    def __init__(
        self,
        *,
        slider: QtWidgets.QSlider,
        spin: QtWidgets.QSpinBox,
        prev_btn: QtWidgets.QToolButton,
        next_btn: QtWidgets.QToolButton,
        total_label: QtWidgets.QLabel,
        live_check: QtWidgets.QCheckBox,
        get_worker: Callable[[], object],
        render_snapshot: Callable[[object], None],
    ):
        self._slider = slider
        self._spin = spin
        self._prev_btn = prev_btn
        self._next_btn = next_btn
        self._total_label = total_label
        self._live = live_check
        self._get_worker = get_worker
        self._render_snapshot = render_snapshot

        # Distinguishes user-driven valueChanged (dragging / typing /
        # clicking) from programmatic ones (auto-track advance, slider↔
        # spinbox cross-sync). Without it, every auto-advance would
        # re-fire the user-grab path and uncheck Live in a loop.
        self._programmatic = False

        slider.valueChanged.connect(self._on_slider)
        slider.sliderPressed.connect(self._on_grab)
        prev_btn.clicked.connect(self._on_prev)
        next_btn.clicked.connect(self._on_next)
        spin.valueChanged.connect(self._on_spin)
        live_check.toggled.connect(self._on_live_toggled)

    # ----- Live state --------------------------------------------------------

    def is_live(self) -> bool:
        return self._live.isChecked()

    def set_live(self, on: bool) -> None:
        self._live.setChecked(on)

    # ----- lifecycle hooks ---------------------------------------------------

    def begin_run(self) -> None:
        """Reset for a freshly-started run and re-engage Live so the
        first incoming snapshots auto-track."""
        self.reset()
        self.set_live(True)

    def load_finished_run(self, n: int) -> None:
        """Configure the widgets for a loaded (already-finished) run of
        ``n`` snapshots: freeze Live and park the slider on the final
        step. The caller renders the final snapshot."""
        self.set_live(False)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(max(0, n - 1))
        self._slider.blockSignals(False)
        self.sync()

    def on_tick(self) -> None:
        """Per render-tick: extend the slider range to everything emitted
        so far and (if Live) advance to the latest frame. Mirrors the
        history half of the window's render timer."""
        worker = self._get_worker()
        if worker is None:
            return
        hist_len = worker.history_len()
        if hist_len > 0:
            self._slider.blockSignals(True)
            self._slider.setMaximum(hist_len - 1)
            if self._live.isChecked():
                self._programmatic = True
                self._slider.setValue(hist_len - 1)
                self._programmatic = False
            self._slider.blockSignals(False)
            self.sync()

    # ----- widget state ------------------------------------------------------

    def reset(self) -> None:
        """Put the slider / buttons / spinbox / total label back into
        their pristine, no-history state."""
        self._slider.blockSignals(True)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setEnabled(False)
        self._slider.blockSignals(False)
        self._spin.blockSignals(True)
        self._spin.setRange(0, 0)
        self._spin.setValue(0)
        self._spin.setEnabled(False)
        self._spin.blockSignals(False)
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._total_label.setText('/ —')

    def sync(self) -> None:
        """Make the slider/spinbox/buttons/label consistent with the
        worker's current history. Programmatic moves are guarded against
        the spinbox↔slider feedback loop via ``_programmatic``.
        """
        worker = self._get_worker()
        if worker is None or worker.history_len() == 0:
            self.reset()
            return
        n = worker.history_len()
        total = worker.history_total
        idx = self._slider.value()
        offset = total - n  # absolute step at buffer index 0
        abs_step = idx + offset
        abs_max = total - 1
        # Slider already has its value/max set elsewhere; just enable it.
        self._slider.setEnabled(True)
        # Prev/next: only enabled when there's somewhere to step.
        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setEnabled(idx < n - 1)
        # Spinbox: absolute range + value. Guard the back-edge of the
        # sync loop via blockSignals + the programmatic flag.
        self._programmatic = True
        self._spin.blockSignals(True)
        self._spin.setRange(offset, abs_max)
        self._spin.setValue(abs_step)
        self._spin.setEnabled(True)
        self._spin.blockSignals(False)
        self._programmatic = False
        # Total label. The leading slash echoes "step <N> / <max>" so the
        # spinbox + label read naturally side-by-side.
        self._total_label.setText(f'/ {abs_max}')

    # ----- signal handlers ---------------------------------------------------

    def _on_grab(self) -> None:
        """User started dragging the slider — drop out of live mode so
        the next auto-advance doesn't fight the user's selection."""
        self._live.setChecked(False)

    def _on_slider(self, idx: int) -> None:
        worker = self._get_worker()
        if worker is None:
            return
        snap = worker.history_get(int(idx))
        if snap is None:
            return
        # A user-driven valueChanged means active scrubbing — disengage
        # Live so the next auto-tick doesn't snap back to the end.
        # Programmatic moves (auto-track) set the flag to skip this.
        if not self._programmatic and self._live.isChecked():
            self._live.setChecked(False)
        self._render_snapshot(snap)
        self.sync()

    def _on_prev(self) -> None:
        """Step back by one — counts as a user action, so drop Live."""
        worker = self._get_worker()
        if worker is None or worker.history_len() == 0:
            return
        self._live.setChecked(False)
        self._slider.setValue(max(0, self._slider.value() - 1))

    def _on_next(self) -> None:
        """Step forward by one."""
        worker = self._get_worker()
        if worker is None or worker.history_len() == 0:
            return
        n = worker.history_len()
        self._live.setChecked(False)
        self._slider.setValue(min(n - 1, self._slider.value() + 1))

    def _on_spin(self, abs_step: int) -> None:
        """User typed an absolute step into the spinbox. Convert to the
        slider's buffer index and let the slider valueChanged path do the
        render/sync. Skipped for programmatic edits (the sync feedback
        edge)."""
        if self._programmatic:
            return
        worker = self._get_worker()
        if worker is None or worker.history_len() == 0:
            return
        n = worker.history_len()
        total = worker.history_total
        offset = total - n  # absolute step at buffer index 0
        buf_idx = max(0, min(n - 1, int(abs_step) - offset))
        if buf_idx != self._slider.value():
            self._live.setChecked(False)
            self._slider.setValue(buf_idx)

    def _on_live_toggled(self, on: bool) -> None:
        """Re-checking Live snaps the view back to the latest step."""
        if not on:
            return
        worker = self._get_worker()
        if worker is None:
            return
        hist_len = worker.history_len()
        if hist_len > 0:
            self._programmatic = True
            self._slider.setValue(hist_len - 1)
            self._programmatic = False
            snap = worker.history_get(hist_len - 1)
            if snap is not None:
                self._render_snapshot(snap)
            self.sync()
