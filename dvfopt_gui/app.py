"""Main PyQtGraph window — live Jdet/2-tri heatmap + window overlay +
pixel inspector + stats panel."""

from __future__ import annotations

import sys

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from dvfopt_gui.worker import SolverWorker, StateSnapshot

# ---------------------------------------------------------------------------
# Colour map — diverging, red = negative (folded), blue = positive
# ---------------------------------------------------------------------------


def _jdet_colormap():
    """Diverging Jdet colormap; matches the dvfopt theme convention.

    Centre at 0 separates folded (red) from feasible (blue/green)
    cells; symmetric out to ±1 covers the typical Jdet range on
    synthetic + B0039 slices."""
    stops = np.array([0.0, 0.49, 0.5, 0.51, 1.0])
    colors = np.array(
        [
            [180, 0, 0, 255],  # deep red (very negative)
            [255, 200, 180, 255],  # pale red near zero
            [240, 240, 240, 255],  # white at zero
            [200, 220, 255, 255],  # pale blue just positive
            [0, 90, 200, 255],  # deep blue (very positive)
        ],
        dtype=np.uint8,
    )
    return pg.ColorMap(stops, colors)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class LiveSolverWindow(QtWidgets.QMainWindow):
    """PyQtGraph window showing the solver state in real time.

    Layout::

        ┌──────────────────────────────┬──────────────────────┐
        │                              │  Stats               │
        │   Jdet heatmap (+ window     │   phase / iter /     │
        │   overlay rect, target dot)  │   n_neg / min_T /    │
        │                              │   wall_s             │
        │                              │                      │
        │   click any pixel →          │  Pixel inspector     │
        │   inspector updates          │   (y, x), Jdet       │
        │                              │   (T1/T2 — v2 TODO)  │
        │                              │                      │
        └──────────────────────────────┴──────────────────────┘
        [Stop]
    """

    def __init__(self, deformation_i, *, solver_kwargs=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('dvfopt — live solver visualisation')
        self.resize(1400, 800)

        self._deformation_i = deformation_i
        self._latest: StateSnapshot | None = None

        # --- central widget: two-column split ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        split = QtWidgets.QHBoxLayout()
        outer.addLayout(split, stretch=1)

        # Left: image view
        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        # imshow-style: y increases downward
        self._plot.invertY(True)
        self._plot.setLabels(left='y', bottom='x')
        split.addWidget(self._plot, stretch=3)

        self._img = pg.ImageItem(axisOrder='row-major')
        cmap = _jdet_colormap()
        self._img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self._img.setLevels((-1.0, 1.0))
        self._plot.addItem(self._img)

        # Active-window overlay rect (yellow). Sized via setRect on each update.
        self._window_rect = pg.QtWidgets.QGraphicsRectItem()
        pen = pg.mkPen(color=(255, 220, 60), width=2)
        self._window_rect.setPen(pen)
        self._plot.addItem(self._window_rect)

        # Padded-window overlay rect (cyan, dashed).
        self._opt_rect = pg.QtWidgets.QGraphicsRectItem()
        pen2 = pg.mkPen(color=(80, 220, 255), width=1, style=QtCore.Qt.DashLine)
        self._opt_rect.setPen(pen2)
        self._plot.addItem(self._opt_rect)

        # Target-pixel marker (the current worst-Jdet pixel).
        self._target_marker = pg.ScatterPlotItem(
            symbol='o',
            size=12,
            pen=pg.mkPen('y', width=2),
            brush=pg.mkBrush(None),
        )
        self._plot.addItem(self._target_marker)

        # Right: stats + inspector panel
        right = QtWidgets.QVBoxLayout()
        split.addLayout(right, stretch=1)

        self._stats_label = QtWidgets.QLabel(self._format_stats(None))
        self._stats_label.setFont(QtGui.QFont('Consolas', 10))
        self._stats_label.setTextFormat(QtCore.Qt.RichText)
        self._stats_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        right.addWidget(self._stats_label, stretch=1)

        right.addWidget(QtWidgets.QLabel('<i>Click any pixel for inspector readout</i>'))
        self._inspector_label = QtWidgets.QLabel(self._format_inspector(None))
        self._inspector_label.setFont(QtGui.QFont('Consolas', 10))
        self._inspector_label.setTextFormat(QtCore.Qt.RichText)
        self._inspector_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        right.addWidget(self._inspector_label, stretch=1)

        # --- bottom: control buttons ---
        bar = QtWidgets.QHBoxLayout()
        outer.addLayout(bar)
        self._stop_btn = QtWidgets.QPushButton('Stop solver')
        self._stop_btn.clicked.connect(self._on_stop)
        bar.addWidget(self._stop_btn)
        bar.addStretch(1)
        self._fps_label = QtWidgets.QLabel('0 callbacks · 0 fps')
        bar.addWidget(self._fps_label)

        # --- worker thread ---
        self._worker = SolverWorker(
            deformation_i=self._deformation_i,
            solver_kwargs=solver_kwargs or {},
            parent=self,
        )
        self._worker.errored.connect(self._on_error)
        self._worker.finishedWithResult.connect(self._on_finished)

        # Mouse pick
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_click)
        self._picked_yx: tuple[int, int] | None = None

        # Render timer — drain the worker queue at 30 Hz
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self._on_render_tick)

        # Callback-rate display — polled in ``_on_render_tick`` against
        # the worker's atomic counter. We deliberately don't connect a
        # per-callback Qt signal: queued cross-thread signals would
        # accumulate in the event loop and defeat the bounded queue.
        self._last_count = 0
        self._last_tick = QtCore.QElapsedTimer()
        self._last_tick.start()

    # ----- public ------------------------------------------------------------

    def start(self):
        self._render_timer.start()
        self._worker.start()

    # ----- formatting --------------------------------------------------------

    def _format_stats(self, snap: StateSnapshot | None) -> str:
        if snap is None:
            return '<b>Stats</b><br>(waiting for solver…)'
        H, W = snap.jacobian.shape
        return (
            '<b>Stats</b><br>'
            f'outer iter . . {snap.outer_iter}<br>'
            f'per-pixel . . . {snap.per_index_iter}<br>'
            f'n_neg . . . . . {snap.n_neg}<br>'
            f'min_T . . . . . {snap.min_T:+.5f}<br>'
            f'window . . . . ({snap.window_y0}–{snap.window_y1}, '
            f'{snap.window_x0}–{snap.window_x1})  '
            f'{snap.window_y1 - snap.window_y0}×{snap.window_x1 - snap.window_x0}<br>'
            f'padded . . . . {snap.is_padded}<br>'
            f'target pixel . (y={snap.neg_y}, x={snap.neg_x})<br>'
            f'grid . . . . . {H}×{W}'
        )

    def _format_inspector(self, yx: tuple[int, int] | None) -> str:
        if yx is None or self._latest is None:
            return '<b>Pixel inspector</b><br>(click a pixel)'
        y, x = yx
        snap = self._latest
        if not (0 <= y < snap.jacobian.shape[0] and 0 <= x < snap.jacobian.shape[1]):
            return '<b>Pixel inspector</b><br>(out of bounds)'
        # Triangle areas on the LIVE state. The solver passes us
        # ``jacobian`` directly; for T1/T2 we'd need access to the
        # current phi, which the snapshot doesn't carry by design. So
        # we report Jdet only — the live T1/T2 readout requires the
        # phi snapshot which is omitted for memory reasons in v1.
        return (
            '<b>Pixel inspector</b><br>'
            f'(y={y}, x={x})<br>'
            f'Jdet . . . {snap.jacobian[y, x]:+.5f}<br>'
            '<i>(T1/T2 needs phi snapshot — TODO in v2)</i>'
        )

    # ----- render loop -------------------------------------------------------

    def _on_render_tick(self):
        snap = self._worker.take_latest()
        if snap is not None:
            self._latest = snap
            self._img.setImage(snap.jacobian, autoLevels=False)
            self._window_rect.setRect(
                snap.window_x0,
                snap.window_y0,
                snap.window_x1 - snap.window_x0,
                snap.window_y1 - snap.window_y0,
            )
            if snap.is_padded and (
                snap.opt_x0 != snap.window_x0
                or snap.opt_y0 != snap.window_y0
                or snap.opt_x1 != snap.window_x1
                or snap.opt_y1 != snap.window_y1
            ):
                self._opt_rect.setRect(
                    snap.opt_x0,
                    snap.opt_y0,
                    snap.opt_x1 - snap.opt_x0,
                    snap.opt_y1 - snap.opt_y0,
                )
                self._opt_rect.setVisible(True)
            else:
                self._opt_rect.setVisible(False)
            self._target_marker.setData(x=[snap.neg_x], y=[snap.neg_y])
            self._stats_label.setText(self._format_stats(snap))
            if self._picked_yx is not None:
                self._inspector_label.setText(self._format_inspector(self._picked_yx))

        # FPS / cb-rate update once per second. ``callback_count`` is
        # an atomic int read from the worker thread (single CPython
        # 64-bit store is thread-safe, no GIL gymnastics required).
        if self._last_tick.elapsed() >= 1000:
            dt_s = self._last_tick.restart() / 1000.0
            cb_count = self._worker.callback_count
            delta = cb_count - self._last_count
            self._last_count = cb_count
            self._fps_label.setText(f'{cb_count} callbacks · {delta / dt_s:.1f} cb/s')

    # ----- mouse pick --------------------------------------------------------

    def _on_mouse_click(self, ev):
        if not self._plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        mouse_point = self._plot.plotItem.vb.mapSceneToView(ev.scenePos())
        x = round(mouse_point.x())
        y = round(mouse_point.y())
        self._picked_yx = (y, x)
        self._inspector_label.setText(self._format_inspector((y, x)))

    # ----- lifecycle ---------------------------------------------------------

    def _on_stop(self):
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText('Stopping…')
        self._worker.request_stop()

    def _on_finished(self, phi_out, info):
        self._render_timer.stop()
        self._stop_btn.setText('Solver finished')
        self._stop_btn.setEnabled(False)
        msg = 'Solver completed.'
        if isinstance(info, str):
            msg = f'Solver stopped: {info}.'
        self.statusBar().showMessage(msg, 10_000)

    def _on_error(self, err: str):
        self._render_timer.stop()
        self._stop_btn.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, 'Solver error', err)

    def closeEvent(self, ev):
        if self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(2000)
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
# Top-level launch helper
# ---------------------------------------------------------------------------


def launch(deformation_i, *, solver_kwargs=None) -> int:
    """Open the live-viz window for ``deformation_i`` (shape ``(3, 1, H, W)``)
    and start the solver. Blocks until the window is closed.

    Returns Qt exit code."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = LiveSolverWindow(deformation_i, solver_kwargs=solver_kwargs)
    win.show()
    win.start()
    return app.exec_()
