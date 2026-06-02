"""Main PyQtGraph window — DVF loader, view-mode toggles, section ROIs,
live overlay rect, pixel inspector, stats panel.

Features
--------

* **Load DVF...** — pick a ``.npy`` from disk. Both ``(3, D, H, W)``
  3D volumes (each z-slice runnable independently) and
  ``(3, 1, H, W)`` / ``(2, H, W)`` single 2D slices are supported.
* **View modes** — radio selector switches the central image between:
    * **Jdet (CD)** — central-difference Jacobian determinant per pixel
    * **2-tri (min T1, T2)** — minimum signed triangle area per cell
      (catches sub-pixel folds the Jdet stencil misses)
    * **Deformation grid** — warped wireframe of the displacement field
* **Slice slider** (3D only) — scrub z to switch the visible slice.
* **Section ROI** — drag the dashed rectangle on the heatmap to mark
  a sub-region; "Run section" then solves only inside the ROI
  (cropping → solving → splicing back in place).
* **Run full / Run section / Stop** — kick off / interrupt the solver
  with the appropriate scope.
"""

from __future__ import annotations

import sys

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from dvfopt_gui.worker import SolverWorker, StateSnapshot

# ---------------------------------------------------------------------------
# Helpers — colourmaps, view-mode math, grid wireframe geometry
# ---------------------------------------------------------------------------


def _jdet_colormap():
    """Diverging Jdet colormap; red = negative (folded), blue = positive."""
    stops = np.array([0.0, 0.49, 0.5, 0.51, 1.0])
    colors = np.array(
        [
            [180, 0, 0, 255],  # deep red
            [255, 200, 180, 255],  # pale red near zero
            [240, 240, 240, 255],  # white at zero
            [200, 220, 255, 255],  # pale blue
            [0, 90, 200, 255],  # deep blue
        ],
        dtype=np.uint8,
    )
    return pg.ColorMap(stops, colors)


def _min_tri_from_phi(phi_2hw: np.ndarray) -> np.ndarray:
    """Compute per-cell ``min(T1, T2)`` from a ``(2, H, W)`` field.

    Returns ``(H-1, W-1)`` array padded back to ``(H, W)`` by appending
    a row + column of NaN so the image registers in the same coordinate
    system as the Jdet heatmap. Lifts the ``_triangle_areas_2d``
    primitive from :mod:`dvfopt.jacobian.triangle_sign`.
    """
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    min_T = np.minimum(T1, T2)
    H, W = phi_2hw.shape[1:]
    out = np.full((H, W), np.nan, dtype=np.float64)
    out[: H - 1, : W - 1] = min_T
    return out


def _grid_lines(phi_2hw: np.ndarray, stride: int = 1):
    """Return ``(xs, ys)`` arrays for a connected line series tracing
    every row + column of the warped grid.

    Uses NaN separators between rows so a single ``PlotDataItem`` can
    render the entire wireframe in one draw call. Adapts the
    matplotlib logic from :func:`dvfopt.viz.grids.plot_grid`.
    """
    dy, dx = phi_2hw[0], phi_2hw[1]
    H, W = dy.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    # Warped pixel locations.
    Y = yy + dy
    X = xx + dx

    Y_sub = Y[::stride, ::stride]
    X_sub = X[::stride, ::stride]
    Hs, Ws = Y_sub.shape

    xs_list = []
    ys_list = []
    nan = np.array([np.nan])
    # Horizontal lines (one per row).
    for r in range(Hs):
        xs_list.append(X_sub[r])
        ys_list.append(Y_sub[r])
        xs_list.append(nan)
        ys_list.append(nan)
    # Vertical lines (one per column).
    for c in range(Ws):
        xs_list.append(X_sub[:, c])
        ys_list.append(Y_sub[:, c])
        xs_list.append(nan)
        ys_list.append(nan)
    return np.concatenate(xs_list), np.concatenate(ys_list)


def _folded_cells_path(phi_2hw: np.ndarray, max_cells: int = 10_000):
    """Build a ``QPainterPath`` outlining every cell where
    ``min(T1, T2) <= 0`` (i.e. at least one of the cell's two
    sign-area triangles has flipped). Returned with the warped-corner
    quad geometry so we can fill in red over the wireframe.

    Caps at ``max_cells`` folded cells to keep the draw call bounded
    on dense fields (e.g., a 320×456 B0039 slice can have ~5000 folded
    cells, well under the cap). When the cap is exceeded the loudest
    folds (by ``min(T1, T2)``) are kept and the rest dropped.
    """
    from PyQt5.QtGui import QPainterPath

    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

    dy, dx = phi_2hw[0], phi_2hw[1]
    T1, T2 = _triangle_areas_2d(dy, dx)
    cell_min = np.minimum(T1, T2)
    folded_mask = cell_min <= 0
    if not folded_mask.any():
        return QPainterPath()

    folded_yx = np.argwhere(folded_mask)
    if len(folded_yx) > max_cells:
        # Keep the deepest folds.
        vals = cell_min[folded_mask]
        order = np.argsort(vals)[:max_cells]
        folded_yx = folded_yx[order]

    H, W = dy.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    Y = yy + dy
    X = xx + dx

    path = QPainterPath()
    for r, c in folded_yx:
        # Quad corners: top-left → top-right → bottom-right → bottom-left
        # (using row-major (y, x) indexing on the (H, W) grid).
        path.moveTo(float(X[r, c]), float(Y[r, c]))
        path.lineTo(float(X[r, c + 1]), float(Y[r, c + 1]))
        path.lineTo(float(X[r + 1, c + 1]), float(Y[r + 1, c + 1]))
        path.lineTo(float(X[r + 1, c]), float(Y[r + 1, c]))
        path.closeSubpath()
    return path


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


VIEW_JDET = 'jdet'
VIEW_2TRI = '2tri'
VIEW_GRID = 'grid'


# Method dropdown specs. ``label`` shows in the combo; ``id`` is what
# the worker dispatches on. Keep the wallbreaker family first since
# they're the only options that genuinely correct 2-tri folds — the
# bowtie + most B0039 slices need a 2-tri-aware solver.
_METHOD_SPECS = [
    ('m14_2tri', 'M14 — 2-tri (Harmonic + ALM + L2 refine + repair + polish)'),
    ('m14_schwarz_2tri', 'M14-Schwarz — 2-tri (cluster decomposition + global polish)'),
    ('m10_2tri', 'M10 — 2-tri (Harmonic + ALM + barrier polish)'),
    ('barrier_2tri', 'Barrier — 2-tri (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed_2tri', 'SLSQP windowed — 2-tri (live progress)'),
    ('slsqp_windowed_jdet', 'SLSQP windowed — Jdet only (live progress; misses 2-tri folds)'),
]
DEFAULT_METHOD_ID = 'm14_2tri'


class LiveSolverWindow(QtWidgets.QMainWindow):
    """Live-viz window for the windowed-SLSQP solver.

    Construct with an optional starting ``deformation_i`` (any of
    ``(3, D, H, W)``, ``(3, 1, H, W)``, or ``(2, H, W)``) — pass
    ``None`` to start empty and use **Load DVF...** to pick a file.
    """

    def __init__(self, deformation_i=None, *, parent=None):
        super().__init__(parent)
        self.setWindowTitle('dvfopt — live solver visualisation')
        self.resize(1500, 900)

        # ---- state -----------------------------------------------------
        # ``_volume`` is the full 3D field, shape ``(3, D, H, W)``; ``_z``
        # is the slice index currently displayed. For 2D inputs we store
        # them with D=1.
        self._volume: np.ndarray | None = None
        self._z = 0
        self._view_mode = VIEW_JDET
        self._latest: StateSnapshot | None = None
        self._worker: SolverWorker | None = None
        self._picked_yx: tuple[int, int] | None = None

        # ---- toolbar (top) ---------------------------------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()
        outer.addLayout(bar)

        load_btn = QtWidgets.QPushButton('Load DVF…')
        load_btn.clicked.connect(self._on_load)
        bar.addWidget(load_btn)

        bar.addWidget(QtWidgets.QLabel('View:'))
        self._view_combo = QtWidgets.QComboBox()
        self._view_combo.addItem('Jdet (CD)', VIEW_JDET)
        self._view_combo.addItem('2-tri (min T1, T2)', VIEW_2TRI)
        self._view_combo.addItem('Deformation grid', VIEW_GRID)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        bar.addWidget(self._view_combo)

        bar.addWidget(QtWidgets.QLabel('z:'))
        self._z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.setEnabled(False)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        bar.addWidget(self._z_slider, stretch=1)
        self._z_label = QtWidgets.QLabel('—')
        bar.addWidget(self._z_label)

        self._run_full_btn = QtWidgets.QPushButton('Run full')
        self._run_full_btn.clicked.connect(lambda: self._on_run(use_roi=False))
        bar.addWidget(self._run_full_btn)
        self._run_roi_btn = QtWidgets.QPushButton('Run section')
        self._run_roi_btn.clicked.connect(lambda: self._on_run(use_roi=True))
        bar.addWidget(self._run_roi_btn)
        self._stop_btn = QtWidgets.QPushButton('Stop')
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        bar.addWidget(self._stop_btn)

        # ---- second toolbar row: method + parameters -------------------
        method_bar = QtWidgets.QHBoxLayout()
        outer.addLayout(method_bar)
        method_bar.addWidget(QtWidgets.QLabel('Method:'))
        self._method_combo = QtWidgets.QComboBox()
        for mid, label in _METHOD_SPECS:
            self._method_combo.addItem(label, mid)
        # Default to a 2-tri wallbreaker so the bowtie + most B0039
        # slices visibly correct on the first run.
        default_idx = next(
            (i for i, (mid, _) in enumerate(_METHOD_SPECS) if mid == DEFAULT_METHOD_ID),
            0,
        )
        self._method_combo.setCurrentIndex(default_idx)
        method_bar.addWidget(self._method_combo, stretch=1)

        method_bar.addWidget(QtWidgets.QLabel('time_budget_s:'))
        self._budget_spin = QtWidgets.QDoubleSpinBox()
        self._budget_spin.setRange(1.0, 3600.0)
        self._budget_spin.setSingleStep(10.0)
        self._budget_spin.setValue(60.0)
        self._budget_spin.setToolTip(
            'Wall-clock budget for the wallbreaker family '
            '(M10, M14, Schwarz, Barrier). Ignored by SLSQP-windowed.'
        )
        method_bar.addWidget(self._budget_spin)

        method_bar.addWidget(QtWidgets.QLabel('max_iter:'))
        self._max_iter_spin = QtWidgets.QSpinBox()
        self._max_iter_spin.setRange(1, 100_000)
        self._max_iter_spin.setSingleStep(10)
        self._max_iter_spin.setValue(200)
        self._max_iter_spin.setToolTip(
            'Outer-iteration cap for SLSQP-windowed. Ignored by '
            'wallbreaker methods (they use time_budget_s instead).'
        )
        method_bar.addWidget(self._max_iter_spin)

        # ---- split: left image, right info panel -----------------------
        split = QtWidgets.QHBoxLayout()
        outer.addLayout(split, stretch=1)

        # White background so heatmap text + dark grid lines are
        # legible. The default pyqtgraph black bg made the deformation-
        # grid wireframe look "faded" (dark lines on dark bg).
        self._plot = pg.PlotWidget(background='w')
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)
        self._plot.setLabels(left='y', bottom='x')
        split.addWidget(self._plot, stretch=3)

        self._img = pg.ImageItem(axisOrder='row-major')
        cmap = _jdet_colormap()
        self._img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self._img.setLevels((-1.0, 1.0))
        self._plot.addItem(self._img)

        self._grid_curve = pg.PlotDataItem(pen=pg.mkPen(color=(0, 0, 0), width=2), connect='finite')
        self._grid_curve.setVisible(False)
        self._plot.addItem(self._grid_curve)

        # Folded-cell overlay (deformation-grid view only). Filled red
        # with a darker red outline so flipped cells stand out against
        # the gray wireframe.
        self._fold_overlay = pg.QtWidgets.QGraphicsPathItem()
        self._fold_overlay.setBrush(pg.mkBrush(220, 30, 30, 180))
        self._fold_overlay.setPen(pg.mkPen(color=(160, 0, 0), width=1))
        self._fold_overlay.setVisible(False)
        self._plot.addItem(self._fold_overlay)

        # Section-selection ROI — used by "Run section". Hidden until a
        # DVF is loaded; user drags handles to outline the region.
        self._section_roi = pg.RectROI(
            [0, 0],
            [10, 10],
            pen=pg.mkPen(color=(0, 200, 100), width=2, style=QtCore.Qt.DashLine),
            movable=True,
            resizable=True,
        )
        self._section_roi.setVisible(False)
        self._plot.addItem(self._section_roi)

        self._window_rect = pg.QtWidgets.QGraphicsRectItem()
        self._window_rect.setPen(pg.mkPen(color=(255, 220, 60), width=2))
        self._plot.addItem(self._window_rect)
        self._opt_rect = pg.QtWidgets.QGraphicsRectItem()
        self._opt_rect.setPen(pg.mkPen(color=(80, 220, 255), width=1, style=QtCore.Qt.DashLine))
        self._plot.addItem(self._opt_rect)
        self._target_marker = pg.ScatterPlotItem(
            symbol='o', size=12, pen=pg.mkPen('y', width=2), brush=pg.mkBrush(None)
        )
        self._plot.addItem(self._target_marker)

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

        # ---- bottom status row -----------------------------------------
        statusbar = QtWidgets.QHBoxLayout()
        outer.addLayout(statusbar)
        statusbar.addStretch(1)
        self._fps_label = QtWidgets.QLabel('idle')
        statusbar.addWidget(self._fps_label)

        # Mouse pick
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_click)

        # Render timer — drain the worker queue at 30 Hz; idle if no worker.
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self._on_render_tick)
        self._last_count = 0
        self._last_tick = QtCore.QElapsedTimer()
        self._last_tick.start()

        # Initial DVF if supplied.
        if deformation_i is not None:
            self._load_array(np.asarray(deformation_i))

    # ----- public ------------------------------------------------------------

    def start(self):
        """Open the window and (if a DVF was passed at construction) start
        the solver on the full field. Backward-compatible with the v1
        ``launch()`` entry point."""
        self._render_timer.start()
        if self._volume is not None and self._worker is None:
            self._on_run(use_roi=False)

    # ----- DVF loading -------------------------------------------------------

    def _on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load DVF (.npy)', '', 'NumPy arrays (*.npy);;All files (*)'
        )
        if not path:
            return
        try:
            arr = np.load(path).astype(np.float64)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Load failed', f'{type(exc).__name__}: {exc}')
            return
        try:
            self._load_array(arr)
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, 'Bad shape', str(exc))
            return
        self.statusBar().showMessage(f'Loaded {path}', 5_000)

    def _load_array(self, arr: np.ndarray) -> None:
        """Accept any of: ``(2, H, W)``, ``(3, H, W)``, ``(3, 1, H, W)``,
        ``(3, D, H, W)``. Normalises to a ``(3, D, H, W)`` volume."""
        if arr.ndim == 3 and arr.shape[0] == 2:
            # (2, H, W) — 2D, channels [dy, dx]. Pad dz=0 + D=1.
            H, W = arr.shape[1:]
            vol = np.zeros((3, 1, H, W), dtype=np.float64)
            vol[1, 0] = arr[0]
            vol[2, 0] = arr[1]
        elif arr.ndim == 3 and arr.shape[0] == 3:
            # (3, H, W) — 2D with dz channel. Pad D=1.
            vol = arr[:, None, :, :]
        elif arr.ndim == 4 and arr.shape[0] == 3:
            # (3, D, H, W) — already in canonical layout.
            vol = arr
        else:
            raise ValueError(f'expected (2,H,W), (3,H,W), (3,1,H,W), or (3,D,H,W); got {arr.shape}')
        self._volume = vol.astype(np.float64)
        D = vol.shape[1]
        self._z = 0
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(max(0, D - 1))
        self._z_slider.setValue(0)
        self._z_slider.setEnabled(D > 1)
        self._z_slider.blockSignals(False)
        self._z_label.setText(f'0 / {D - 1}' if D > 1 else '0 / 0 (2D)')
        self._latest = None
        self._picked_yx = None
        self._refresh_display_from_volume()
        # Show the ROI rectangle now that we have geometry to drag on.
        H, W = vol.shape[2:]
        roi_w, roi_h = max(8, W // 4), max(8, H // 4)
        self._section_roi.setPos((W - roi_w) // 2, (H - roi_h) // 2)
        self._section_roi.setSize([roi_w, roi_h])
        self._section_roi.setVisible(True)

    def _current_slice(self) -> np.ndarray:
        """Return the active ``(3, 1, H, W)`` slice for the solver."""
        if self._volume is None:
            raise RuntimeError('no DVF loaded')
        return self._volume[:, self._z : self._z + 1].copy()

    # ----- rendering ---------------------------------------------------------

    def _refresh_display_from_volume(self):
        """When the volume / z-slice / view-mode changes (but no live
        solver state is available), recompute the image from the
        volume itself and clear overlays."""
        if self._volume is None:
            return
        phi_2hw = self._volume[1:, self._z]  # (2, H, W)
        from dvfopt.jacobian.numpy_jdet import jacobian_det2D

        jac = jacobian_det2D(phi_2hw)[0]
        self._set_view(phi_2hw, jac)
        self._window_rect.setRect(0, 0, 0, 0)
        self._opt_rect.setVisible(False)
        self._target_marker.setData(x=[], y=[])
        self._stats_label.setText(self._format_stats(None))
        self._inspector_label.setText(self._format_inspector(None))

    def _set_view(self, phi_2hw: np.ndarray, jacobian: np.ndarray) -> None:
        """Update the central plot to reflect the current view mode."""
        mode = self._view_mode
        if mode == VIEW_JDET:
            self._img.setImage(jacobian, autoLevels=False)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_2TRI:
            self._img.setImage(_min_tri_from_phi(phi_2hw), autoLevels=False)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_GRID:
            # Pure grid view: hide the Jdet heatmap entirely and draw
            # only the warped wireframe. Folded cells (min(T1,T2) <= 0)
            # are overlaid with a translucent red fill so they pop out
            # against the gray grid without obscuring its geometry.
            self._img.setVisible(False)
            stride = max(1, min(phi_2hw.shape[1:]) // 40)
            xs, ys = _grid_lines(phi_2hw, stride=stride)
            self._grid_curve.setData(xs, ys)
            self._grid_curve.setVisible(True)
            self._fold_overlay.setPath(_folded_cells_path(phi_2hw))
            self._fold_overlay.setVisible(True)

    def _on_view_changed(self, idx: int):
        self._view_mode = self._view_combo.itemData(idx)
        if self._latest is not None:
            self._set_view(self._latest.phi, self._latest.jacobian)
        else:
            self._refresh_display_from_volume()

    def _on_z_changed(self, value: int):
        self._z = int(value)
        D = self._volume.shape[1] if self._volume is not None else 1
        self._z_label.setText(f'{self._z} / {D - 1}')
        self._refresh_display_from_volume()

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
            sub = deformation_i[:, :, y0:y1, x0:x1].copy()
            self._start_worker(sub)
        else:
            self._section_bounds = None
            self._start_worker(deformation_i)

    def _start_worker(self, deformation_i: np.ndarray):
        method_id = self._method_combo.currentData()
        params = {
            'time_budget_s': float(self._budget_spin.value()),
            'max_iterations': int(self._max_iter_spin.value()),
        }
        self._worker = SolverWorker(
            deformation_i=deformation_i,
            method_id=method_id,
            params=params,
            parent=self,
        )
        self._worker.finishedWithResult.connect(self._on_finished)
        self._worker.errored.connect(self._on_error)
        self._stop_btn.setEnabled(True)
        self._run_full_btn.setEnabled(False)
        self._run_roi_btn.setEnabled(False)
        self._fps_label.setText(f'starting {method_id}…')
        self._last_count = 0
        self._last_tick.restart()
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._stop_btn.setText('Stopping…')

    def _on_finished(self, phi_out, info):
        # Splice the result back into the volume so subsequent runs /
        # view toggles see the corrected state.
        if phi_out is not None and self._volume is not None:
            phi_out = np.asarray(phi_out)
            sb = getattr(self, '_section_bounds', None)
            if sb is not None:
                y0, y1, x0, x1 = sb
                self._volume[1, self._z, y0:y1, x0:x1] = phi_out[0]
                self._volume[2, self._z, y0:y1, x0:x1] = phi_out[1]
            else:
                self._volume[1, self._z] = phi_out[0]
                self._volume[2, self._z] = phi_out[1]
            self._refresh_display_from_volume()
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText('Stop')
        self._run_full_btn.setEnabled(True)
        self._run_roi_btn.setEnabled(True)
        msg = 'Run finished.' if info is None else f'Run stopped: {info}.'
        self.statusBar().showMessage(msg, 10_000)
        self._fps_label.setText('idle')

    def _on_error(self, err: str):
        self._stop_btn.setEnabled(False)
        self._run_full_btn.setEnabled(True)
        self._run_roi_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, 'Solver error', err)
        self._fps_label.setText('errored')

    # ----- render loop -------------------------------------------------------

    def _on_render_tick(self):
        if self._worker is None:
            return
        snap = self._worker.take_latest()
        if snap is not None:
            self._latest = snap
            self._set_view(snap.phi, snap.jacobian)
            self._window_rect.setRect(
                snap.window_x0,
                snap.window_y0,
                snap.window_x1 - snap.window_x0,
                snap.window_y1 - snap.window_y0,
            )
            if snap.is_padded:
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

        # cb-rate once per second
        if self._last_tick.elapsed() >= 1000 and self._worker is not None:
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

    # ----- formatters --------------------------------------------------------

    def _format_stats(self, snap: StateSnapshot | None) -> str:
        if snap is None:
            if self._volume is None:
                return '<b>Stats</b><br>(no DVF loaded — click "Load DVF…")'
            H, W = self._volume.shape[2:]
            D = self._volume.shape[1]
            return f'<b>Stats</b><br>volume {D}×{H}×{W}<br>view: {self._view_mode}<br>(idle)'
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
        if yx is None:
            return '<b>Pixel inspector</b><br>(click a pixel)'
        y, x = yx
        # Prefer the live snapshot's phi; fall back to the volume.
        if self._latest is not None:
            phi = self._latest.phi
            jac = self._latest.jacobian
        elif self._volume is not None:
            phi = self._volume[1:, self._z]
            from dvfopt.jacobian.numpy_jdet import jacobian_det2D

            jac = jacobian_det2D(phi)[0]
        else:
            return '<b>Pixel inspector</b><br>(no DVF loaded)'
        H, W = jac.shape
        if not (0 <= y < H and 0 <= x < W):
            return '<b>Pixel inspector</b><br>(out of bounds)'
        # Per-cell T1/T2 — only defined for (y < H-1 and x < W-1) since
        # they index the cell anchored at the (y, x) top-left.
        t1_str = t2_str = '—'
        if y < H - 1 and x < W - 1:
            from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            t1_str = f'{T1[y, x]:+.5f}'
            t2_str = f'{T2[y, x]:+.5f}'
        return (
            '<b>Pixel inspector</b><br>'
            f'(y={y}, x={x})<br>'
            f'Jdet . . . {jac[y, x]:+.5f}<br>'
            f'T1 . . . . {t1_str}<br>'
            f'T2 . . . . {t2_str}'
        )

    # ----- lifecycle ---------------------------------------------------------

    def closeEvent(self, ev):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(2000)
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
# Top-level launch helper
# ---------------------------------------------------------------------------


def launch(deformation_i=None, *, solver_kwargs=None) -> int:
    """Open the live-viz window.

    Parameters
    ----------
    deformation_i : ndarray, optional
        Any of ``(2, H, W)``, ``(3, H, W)``, ``(3, 1, H, W)``, or
        ``(3, D, H, W)``. When ``None`` (default), the window starts
        empty — use **Load DVF…** to pick a file.
    solver_kwargs : dict, optional
        Ignored. The choice of solver + its parameters now lives in
        the toolbar (Method dropdown + the ``time_budget_s`` /
        ``max_iter`` spinboxes). Kept in the signature for back-
        compat with v1 callers.

    Returns Qt exit code."""
    del solver_kwargs  # kept for back-compat; choice is in-GUI now
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = LiveSolverWindow(deformation_i)
    win.show()
    win.start()
    return app.exec_()
