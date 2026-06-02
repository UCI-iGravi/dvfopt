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
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from dvfopt_gui.worker import DEFAULT_HISTORY_MAX, SolverWorker, StateSnapshot

# Repo root used to anchor the default file-dialog directory. The GUI
# can be launched from anywhere, but ``data/dvfs/`` is the project's
# canonical DVF folder — pointing the dialog there saves a few clicks.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DVF_DIR = str(_REPO_ROOT / 'data' / 'dvfs')

# ---------------------------------------------------------------------------
# Helpers — colourmaps, view-mode math, grid wireframe geometry
# ---------------------------------------------------------------------------


def _jdet_colormap():
    """Diverging Jdet colormap; **red = positive (feasible)**,
    **blue = negative (folded)**. White separates the two at zero.

    (Note: this is the opposite of the standard "red = bad" convention
    used in :mod:`dvfopt.viz`; chosen here per user request to match
    their preferred reading.)"""
    stops = np.array([0.0, 0.49, 0.5, 0.51, 1.0])
    colors = np.array(
        [
            [0, 90, 200, 255],  # deep blue at Jdet=-1
            [200, 220, 255, 255],  # pale blue near zero
            [240, 240, 240, 255],  # white at zero
            [255, 200, 180, 255],  # pale red just positive
            [180, 0, 0, 255],  # deep red at Jdet=+1
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


# Constraint families. The worker dispatches on ``method_id`` which is
# always ``<algo>_<constraint>`` so the dispatch table stays flat.
CONSTRAINT_2TRI = '2tri'
CONSTRAINT_JDET = 'jdet'
_CONSTRAINT_SPECS = [
    (CONSTRAINT_2TRI, '2-tri (full-coverage; catches sub-pixel folds)'),
    (CONSTRAINT_JDET, 'Jdet (central-diff; blind to sub-pixel folds)'),
]
DEFAULT_CONSTRAINT = CONSTRAINT_2TRI

# Per-constraint method specs. Wallbreakers are 2-tri-only by design
# (HarmonicALMBarrier/RefineRepair internally depend on the 2-tri
# adjoint). Jdet gets the legacy windowed-SLSQP, the penalty→barrier
# path, and the NMVF heuristic smoother.
_METHOD_SPECS_2TRI = [
    ('m14', 'M14 (Harmonic + ALM + L2 refine + repair + polish)'),
    ('m14_schwarz', 'M14-Schwarz (cluster decomposition + global polish)'),
    ('m10', 'M10 (Harmonic + ALM + barrier polish)'),
    ('barrier', 'Barrier (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed', 'SLSQP windowed (live progress)'),
]
_METHOD_SPECS_JDET = [
    ('barrier', 'Barrier (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed', 'SLSQP windowed (live progress)'),
    ('nmvf', 'NMVF (heuristic neighborhood-mean smoother)'),
]
_METHOD_SPECS_BY_CONSTRAINT = {
    CONSTRAINT_2TRI: _METHOD_SPECS_2TRI,
    CONSTRAINT_JDET: _METHOD_SPECS_JDET,
}
DEFAULT_METHOD_BY_CONSTRAINT = {
    CONSTRAINT_2TRI: 'm14',
    CONSTRAINT_JDET: 'slsqp_windowed',
}

# Objective families. The L-BFGS-based strategies (Barrier, M10, M14,
# Schwarz) accept an Objective instance via ``Solver``; SLSQP-windowed
# has its own internal L1 and ignores this choice (we still pass it
# through for metadata bookkeeping in saved runs).
OBJECTIVE_L1 = 'l1'
OBJECTIVE_L2 = 'l2'
OBJECTIVE_NONE = 'none'
_OBJECTIVE_SPECS = [
    (OBJECTIVE_L1, 'L1  (smooth |∇phi|, eps=1e-4)'),
    (OBJECTIVE_L2, 'L2  (½ ‖∇phi‖²)'),
    (OBJECTIVE_NONE, 'None  (no smoothness penalty)'),
]
DEFAULT_OBJECTIVE = OBJECTIVE_L1


def _compose_method_id(algo: str, constraint: str) -> str:
    """Combine ``algo`` + ``constraint`` into the worker dispatch key."""
    return f'{algo}_{constraint}'


class ParamsDialog(QtWidgets.QDialog):
    """Modal dialog for editing window-level parameters.

    Organised as a ``QTabWidget`` so new param groups can be added as
    additional tabs without crowding any single page. The current
    state is read from the parent window's instance attrs on open,
    and written back on accept — there's no settings file (yet).

    Tabs
    ----
    * **History** — buffer size for the scrub slider (per-worker;
      changes apply to the next run only, since ``collections.deque``
      can't be resized in place).
    """

    def __init__(self, parent, *, history_max_size: int):
        super().__init__(parent)
        self.setWindowTitle('Params')
        self.setModal(True)
        self._history_max_size = int(history_max_size)

        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # --- History tab ----------------------------------------------------
        history_tab = QtWidgets.QWidget()
        history_form = QtWidgets.QFormLayout(history_tab)
        self._hist_max_spin = QtWidgets.QSpinBox()
        # 2 floor so the slider always has at least an init+final to scrub
        # between; 100000 ceiling is a runaway-safety cap (at 256² that's
        # ~100 GB worst case — well past any practical research run).
        self._hist_max_spin.setRange(2, 100_000)
        self._hist_max_spin.setSingleStep(50)
        self._hist_max_spin.setValue(self._history_max_size)
        self._hist_max_spin.setToolTip(
            'Max snapshots retained for the History slider. Each snapshot '
            'is ~24·H·W bytes (one copy of phi per step). Default 500 ≈ '
            '500 MB worst case at 256². Lower if you hit memory pressure '
            'on long SLSQP runs; raise if you want full scrub fidelity '
            'on >500-step runs.\n\n'
            "Takes effect on the NEXT solver run — Python's deque can't "
            'be resized in place.'
        )
        history_form.addRow('History buffer size (snapshots):', self._hist_max_spin)
        info = QtWidgets.QLabel(
            '<i>Applies to the next run. The current run keeps its '
            'original buffer size.</i>'
        )
        info.setWordWrap(True)
        history_form.addRow(info)
        tabs.addTab(history_tab, 'History')

        # --- OK / Cancel ----------------------------------------------------
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_values(self) -> dict:
        """Return the user's edits as a plain dict — only valid after
        the dialog has been accepted."""
        return {
            'history_max_size': int(self._hist_max_spin.value()),
        }


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
        # Pristine copy of the loaded DVF — survives in-place splice
        # mutations of ``self._volume`` so subsequent Runs always
        # restart from the loaded data. See ``_current_slice``.
        self._original_volume: np.ndarray | None = None
        self._z = 0
        # Default to the deformation-grid view: it makes folded cells
        # immediately visible via the magenta overlay, regardless of
        # whether the field has any neg-Jdet pixels (the canonical
        # bowtie fixture has 0 neg-Jdet pixels and would otherwise
        # display as uniformly "feasible" red in the Jdet heatmap).
        self._view_mode = VIEW_GRID
        self._latest: StateSnapshot | None = None
        # Frame-level cache of ``jacobian_det2D(self._latest.phi)`` —
        # refreshed exactly once per ``_render_snapshot`` call. The
        # snapshot itself no longer carries the jacobian (saves 33% of
        # per-snapshot memory in the history deque), so we recompute it
        # on render and share that single copy across the heatmap, the
        # inspector, and any stats reads in the same frame.
        self._latest_jacobian: np.ndarray | None = None
        self._worker: SolverWorker | None = None
        self._picked_yx: tuple[int, int] | None = None
        # Window-level params editable via the Params dialog. New
        # workers pick these up at construction; in-flight workers
        # retain whatever they were started with.
        self._history_max_size: int = DEFAULT_HISTORY_MAX

        # ---- toolbar (top) ---------------------------------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()
        outer.addLayout(bar)

        load_btn = QtWidgets.QPushButton('Load DVF…')
        load_btn.clicked.connect(self._on_load)
        bar.addWidget(load_btn)

        self._save_btn = QtWidgets.QPushButton('Save…')
        self._save_btn.setToolTip(
            'Save the current DVF + per-step optimization history as a '
            'compressed .npz. Enabled once a DVF is loaded.'
        )
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        bar.addWidget(self._save_btn)

        bar.addWidget(QtWidgets.QLabel('View:'))
        self._view_combo = QtWidgets.QComboBox()
        self._view_combo.addItem('Jdet (CD)', VIEW_JDET)
        self._view_combo.addItem('2-tri (min T1, T2)', VIEW_2TRI)
        self._view_combo.addItem('Deformation grid', VIEW_GRID)
        # Keep the dropdown in sync with the default ``_view_mode``.
        # The grid view is the only one that always makes folds visible
        # (Jdet view is uniformly red when min Jdet > 0, even with
        # 2-tri folds present — that's the canonical "looks already
        # optimized" trap).
        _default_idx = self._view_combo.findData(self._view_mode)
        if _default_idx >= 0:
            self._view_combo.setCurrentIndex(_default_idx)
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

        # ---- second toolbar row: constraint + method + parameters ------
        method_bar = QtWidgets.QHBoxLayout()
        outer.addLayout(method_bar)
        method_bar.addWidget(QtWidgets.QLabel('Constraint:'))
        self._constraint_combo = QtWidgets.QComboBox()
        for cid, label in _CONSTRAINT_SPECS:
            self._constraint_combo.addItem(label, cid)
        _default_c_idx = self._constraint_combo.findData(DEFAULT_CONSTRAINT)
        if _default_c_idx >= 0:
            self._constraint_combo.setCurrentIndex(_default_c_idx)
        self._constraint_combo.setToolTip(
            '2-tri: full-coverage triangle areas (catches sub-pixel folds the '
            'Jdet central-diff stencil misses — e.g. the bowtie default). '
            'Jdet: per-pixel central-diff determinant (legacy / cheaper).'
        )
        # Signal hooked AFTER _method_combo is built (see below) so the
        # initial repopulation doesn't race with the method combo's
        # construction.
        method_bar.addWidget(self._constraint_combo, stretch=1)

        method_bar.addWidget(QtWidgets.QLabel('Method:'))
        self._method_combo = QtWidgets.QComboBox()
        # Initial population for the default constraint. The combo
        # gets re-filled whenever the constraint changes.
        self._repopulate_method_combo(DEFAULT_CONSTRAINT)
        self._constraint_combo.currentIndexChanged.connect(self._on_constraint_changed)
        method_bar.addWidget(self._method_combo, stretch=2)

        method_bar.addWidget(QtWidgets.QLabel('Objective:'))
        self._objective_combo = QtWidgets.QComboBox()
        for oid, label in _OBJECTIVE_SPECS:
            self._objective_combo.addItem(label, oid)
        _default_o_idx = self._objective_combo.findData(DEFAULT_OBJECTIVE)
        if _default_o_idx >= 0:
            self._objective_combo.setCurrentIndex(_default_o_idx)
        self._objective_combo.setToolTip(
            'Smoothness penalty applied during the polish stages of '
            'the wallbreaker / barrier strategies. Ignored by '
            'SLSQP-windowed (it uses its own internal L1).'
        )
        method_bar.addWidget(self._objective_combo, stretch=1)

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

        # Spacer + Params button — opens the tabbed settings dialog for
        # window-level params that don't belong in the per-run toolbar
        # (e.g. ``history_max_size``).
        self._params_btn = QtWidgets.QPushButton('Params…')
        self._params_btn.setToolTip('Edit window-level parameters (history buffer size, …)')
        self._params_btn.clicked.connect(self._on_open_params)
        method_bar.addWidget(self._params_btn)

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

        # Folded-cell overlay (deformation-grid view only). Filled
        # magenta with a dark-magenta outline so flipped cells stand
        # out against the black wireframe. Magenta (not red) so the
        # highlight reads distinctly from the red-=-positive heatmap.
        self._fold_overlay = pg.QtWidgets.QGraphicsPathItem()
        self._fold_overlay.setBrush(pg.mkBrush(220, 30, 200, 200))
        self._fold_overlay.setPen(pg.mkPen(color=(120, 0, 110), width=1))
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

        # ---- history scrub row -----------------------------------------
        # Every snapshot the worker emits lands in ``worker._history``
        # (in addition to the bounded live queue). The slider scrubs
        # that history. "Live" auto-tracks the latest step while the
        # solver runs; dragging the slider drops out of live mode so
        # the user's chosen frame doesn't get yanked away by the next
        # incoming snapshot.
        history_bar = QtWidgets.QHBoxLayout()
        outer.addLayout(history_bar)
        history_bar.addWidget(QtWidgets.QLabel('History:'))

        # ◀ step-back button — nudges the slider by one.
        # No keyboard shortcut: ←/→ are already handled by the QSlider
        # when it has focus, and an explicit shortcut here would steal
        # the keystroke from the spinbox text editor.
        self._history_prev_btn = QtWidgets.QToolButton()
        self._history_prev_btn.setArrowType(QtCore.Qt.LeftArrow)
        self._history_prev_btn.setEnabled(False)
        self._history_prev_btn.setToolTip('Previous step')
        self._history_prev_btn.clicked.connect(self._on_history_prev)
        history_bar.addWidget(self._history_prev_btn)

        self._history_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._history_slider.setMinimum(0)
        self._history_slider.setMaximum(0)
        self._history_slider.setEnabled(False)
        self._history_slider.setToolTip(
            'Scrub through every snapshot the solver emitted for the '
            'current run. The leftmost position (step 0) is the input '
            'field before any optimization.'
        )
        self._history_slider.valueChanged.connect(self._on_history_slider)
        # Pressing on the slider counts as a user grab — disengage Live.
        self._history_slider.sliderPressed.connect(self._on_history_grab)
        history_bar.addWidget(self._history_slider, stretch=1)

        # ▶ step-forward button.
        self._history_next_btn = QtWidgets.QToolButton()
        self._history_next_btn.setArrowType(QtCore.Qt.RightArrow)
        self._history_next_btn.setEnabled(False)
        self._history_next_btn.setToolTip('Next step')
        self._history_next_btn.clicked.connect(self._on_history_next)
        history_bar.addWidget(self._history_next_btn)

        # Editable step number. Shows the *absolute* step index (so
        # what the user types matches the "step N / M" they read) —
        # we convert to the slider's buffer index in the handler.
        history_bar.addWidget(QtWidgets.QLabel('step'))
        self._history_spin = QtWidgets.QSpinBox()
        self._history_spin.setRange(0, 0)
        self._history_spin.setEnabled(False)
        self._history_spin.setToolTip(
            'Jump to a specific step by typing its index. '
            'Mirrors the slider position.'
        )
        self._history_spin.valueChanged.connect(self._on_history_spin)
        history_bar.addWidget(self._history_spin)
        self._history_total_label = QtWidgets.QLabel('/ —')
        self._history_total_label.setFont(QtGui.QFont('Consolas', 9))
        self._history_total_label.setMinimumWidth(60)
        history_bar.addWidget(self._history_total_label)

        self._live_check = QtWidgets.QCheckBox('Live')
        self._live_check.setChecked(True)
        self._live_check.setToolTip(
            'Auto-track the latest solver step. Uncheck (drag the '
            'slider, click ◀/▶, or type a step) to freeze.'
        )
        self._live_check.toggled.connect(self._on_live_toggled)
        history_bar.addWidget(self._live_check)
        # Internal flag — distinguishes user-driven moves (valueChanged
        # with the user dragging / typing / clicking) from programmatic
        # ones (auto-track advancing, slider↔spinbox cross-sync). Without
        # it, every auto-advance would re-fire the user-grab path and
        # uncheck Live in an infinite loop.
        self._history_programmatic = False

        # ---- bottom status row -----------------------------------------
        statusbar = QtWidgets.QHBoxLayout()
        outer.addLayout(statusbar)
        statusbar.addStretch(1)
        self._fps_label = QtWidgets.QLabel('idle')
        statusbar.addWidget(self._fps_label)

        # Mouse pick
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_click)

        # Render timer — drain the worker queue at 10 Hz; idle if no
        # worker. 100 ms (10 Hz) instead of 33 ms (30 Hz): at the
        # higher rate, large fields with many folded cells exhaust the
        # GUI thread building ``_folded_cells_path`` (pure-Python
        # QPainterPath construction at thousands of cells per tick).
        # 10 Hz still feels live and cuts per-second GUI work by 3×.
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(100)
        self._render_timer.timeout.connect(self._on_render_tick)
        self._last_count = 0
        self._last_tick = QtCore.QElapsedTimer()
        self._last_tick.start()
        # Pixel-count threshold above which a field counts as "big"
        # and switches to fast-render during live updates: drop the
        # expensive fold-overlay rebuild while the solver is running.
        # ~50 K pixels ≈ 224×224 — below the B0039 slice size, so the
        # protection kicks in for typical large research slices.
        self._fast_render_pixel_threshold = 50_000

        # Initial DVF if supplied.
        if deformation_i is not None:
            self._load_array(np.asarray(deformation_i))

    # ----- public ------------------------------------------------------------

    def start(self):
        """Open the window. The render timer starts so any in-progress
        worker (if one is ever attached programmatically) gets drained,
        but we deliberately do **not** auto-run the solver: the boot
        state must show the *input* field so the user can verify what
        they loaded before kicking off a solve. Auto-running on launch
        (the v1 behavior) raced the first paint and made fast solvers
        like M14 look like the input was already feasible."""
        self._render_timer.start()

    # ----- DVF loading -------------------------------------------------------

    def _on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Load DVF (.npy or .npz)',
            _DEFAULT_DVF_DIR,
            'DVF files (*.npy *.npz);;NumPy arrays (*.npy);;NumPy compressed (*.npz);;All files (*)',
        )
        if not path:
            return
        try:
            loaded = np.load(path, allow_pickle=False)
            # NPZ archives carry multiple arrays; the canonical
            # data/dvfs/ schema uses the key ``phi`` for the field.
            # Fall back to the first array if ``phi`` isn't present.
            if isinstance(loaded, np.lib.npyio.NpzFile):
                if 'phi' in loaded.files:
                    arr = loaded['phi']
                else:
                    arr = loaded[loaded.files[0]]
                loaded.close()
            else:
                arr = loaded
            arr = np.asarray(arr).astype(np.float64)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Load failed', f'{type(exc).__name__}: {exc}')
            return
        try:
            self._load_array(arr)
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, 'Bad shape', str(exc))
            return
        self.statusBar().showMessage(f'Loaded {path}', 5_000)

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
            str(Path(_DEFAULT_DVF_DIR) / suggested),
            'NumPy compressed (*.npz);;All files (*)',
        )
        if not path:
            return
        if not path.lower().endswith('.npz'):
            path = path + '.npz'
        try:
            payload = self._build_save_payload()
            np.savez_compressed(path, **payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Save failed', f'{type(exc).__name__}: {exc}'
            )
            return
        n_steps = int(payload.get('n_history_steps', np.array(0)))
        self.statusBar().showMessage(
            f'Saved {Path(path).name}  ({n_steps} history step(s))', 10_000
        )

    def _build_save_payload(self) -> dict:
        """Assemble the NPZ payload from the current window + worker state.

        Schema (all keys present unless noted):

        * ``phi`` — ``(2, H, W)`` float64, the current (possibly
          corrected) field for the active z-slice.
        * ``phi_full_volume`` — ``(3, D, H, W)`` float64, the full
          volume (with the ``dz`` channel). Provenance for multi-slice
          datasets.
        * ``z`` — 0-d int, the active slice index.
        * ``constraint``, ``method``, ``objective`` — 0-d strings (the
          dropdown selections at save time).
        * ``time_budget_s``, ``max_iterations`` — 0-d floats/ints.
        * ``history_max_size`` — 0-d int, the cap that bounded the run's
          history buffer.

        When a worker exists (i.e. at least one run happened), also:

        * ``n_history_steps`` — 0-d int, ``= history_len``.
        * ``history_phi`` — ``(N, 2, H, W)`` float64, every snapshot's
          phi (init → final). Largest array; ``savez_compressed`` keeps
          it manageable.
        * ``history_n_neg``, ``history_min_T`` — ``(N,)`` arrays of the
          running fold-count and worst-area trajectory.
        * ``history_outer_iter``, ``history_per_index_iter`` — ``(N,)``
          int arrays of solver bookkeeping (mostly meaningful for the
          SLSQP-windowed path).
        * ``history_total`` — 0-d int, total snapshots ever emitted
          (may exceed ``n_history_steps`` if some aged out of the cap).
        """
        payload: dict = {}
        # Active slice + full volume.
        phi_active = self._volume[1:, self._z].astype(np.float64)
        payload['phi'] = phi_active
        payload['phi_full_volume'] = self._volume.astype(np.float64)
        payload['z'] = np.int64(self._z)

        # Current dropdown selections (truth at save time, not run time).
        payload['constraint'] = np.asarray(self._constraint_combo.currentData() or '')
        payload['method'] = np.asarray(self._method_combo.currentData() or '')
        payload['objective'] = np.asarray(self._objective_combo.currentData() or '')
        payload['time_budget_s'] = np.float64(self._budget_spin.value())
        payload['max_iterations'] = np.int64(self._max_iter_spin.value())
        payload['history_max_size'] = np.int64(self._history_max_size)

        # Per-slice fold stats, computed fresh on save.
        from dvfopt.jacobian.numpy_jdet import jacobian_det2D

        jac = jacobian_det2D(phi_active)[0]
        payload['final_min_jdet'] = np.float64(jac.min())
        payload['final_n_neg_jdet'] = np.int64((jac < 0).sum())

        # History (if a worker has populated any).
        worker = self._worker
        if worker is not None and worker.history_len() > 0:
            n = worker.history_len()
            H, W = phi_active.shape[1:]
            phi_hist = np.empty((n, 2, H, W), dtype=np.float64)
            n_neg_arr = np.empty(n, dtype=np.int64)
            min_T_arr = np.empty(n, dtype=np.float64)
            outer_arr = np.empty(n, dtype=np.int64)
            sub_arr = np.empty(n, dtype=np.int64)
            for i in range(n):
                snap = worker.history_get(i)
                phi_hist[i] = snap.phi
                n_neg_arr[i] = snap.n_neg
                min_T_arr[i] = snap.min_T
                outer_arr[i] = snap.outer_iter
                sub_arr[i] = snap.per_index_iter
            payload['n_history_steps'] = np.int64(n)
            payload['history_phi'] = phi_hist
            payload['history_n_neg'] = n_neg_arr
            payload['history_min_T'] = min_T_arr
            payload['history_outer_iter'] = outer_arr
            payload['history_per_index_iter'] = sub_arr
            payload['history_total'] = np.int64(worker.history_total)
        else:
            payload['n_history_steps'] = np.int64(0)

        return payload

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
        # Pristine copy of what was loaded — every Run reads its input
        # from here, never from ``self._volume`` (which is mutated by
        # ``_on_finished`` for the post-run display). Without this,
        # clicking Run twice would optimize the already-optimized
        # volume — history[0] would equal history[-1] and the scrub
        # slider would show "the same DVF" at both ends.
        self._original_volume = self._volume.copy()
        D = vol.shape[1]
        self._z = 0
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(max(0, D - 1))
        self._z_slider.setValue(0)
        self._z_slider.setEnabled(D > 1)
        self._z_slider.blockSignals(False)
        self._z_label.setText(f'0 / {D - 1}' if D > 1 else '0 / 0 (2D)')
        self._latest = None
        self._latest_jacobian = None
        self._picked_yx = None
        # A new DVF invalidates any prior run's history. Drop the worker
        # reference so the slider can't scrub stale snapshots, and reset
        # the widget back to its pristine state.
        self._worker = None
        self._reset_history_widgets()
        self._live_check.setChecked(True)
        self._refresh_display_from_volume()
        # Show the ROI rectangle now that we have geometry to drag on.
        H, W = vol.shape[2:]
        roi_w, roi_h = max(8, W // 4), max(8, H // 4)
        self._section_roi.setPos((W - roi_w) // 2, (H - roi_h) // 2)
        self._section_roi.setSize([roi_w, roi_h])
        self._section_roi.setVisible(True)
        # Save is meaningful as soon as a DVF is loaded — even before
        # any solver run (you'd just get phi + minimal metadata).
        self._save_btn.setEnabled(True)

    def _current_slice(self) -> np.ndarray:
        """Return the active ``(3, 1, H, W)`` slice for the solver.

        Reads from ``self._original_volume`` (the as-loaded snapshot),
        not ``self._volume`` (which gets spliced with each run's
        output). This is the key fix for "scrub between 0 and 1 shows
        the same DVF" — without it, a second Run would use the
        already-corrected ``self._volume`` as its input and history[0]
        would no longer be the loaded data.
        """
        if self._original_volume is None:
            raise RuntimeError('no DVF loaded')
        return self._original_volume[:, self._z : self._z + 1].copy()

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

    def _set_view(
        self, phi_2hw: np.ndarray, jacobian: np.ndarray, *, fast: bool = False
    ) -> None:
        """Update the central plot to reflect the current view mode.

        ``fast=True`` skips the per-frame fold-overlay rebuild in
        VIEW_GRID — the live-render tick uses this on big fields to
        avoid the pure-Python QPainterPath construction that scales
        with folded-cell count. The wireframe (grid curves) is still
        updated; only the magenta overlay is dropped until the next
        scrub or live tick falls back to ``fast=False``.
        """
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
            # are overlaid with a translucent magenta fill — unless
            # ``fast`` is set, in which case we drop the overlay for
            # this frame (the QPainterPath construction is the bottleneck
            # on big fields with many folds).
            self._img.setVisible(False)
            stride = max(1, min(phi_2hw.shape[1:]) // 40)
            xs, ys = _grid_lines(phi_2hw, stride=stride)
            self._grid_curve.setData(xs, ys)
            self._grid_curve.setVisible(True)
            if fast:
                self._fold_overlay.setVisible(False)
            else:
                self._fold_overlay.setPath(_folded_cells_path(phi_2hw))
                self._fold_overlay.setVisible(True)

    def _on_view_changed(self, idx: int):
        self._view_mode = self._view_combo.itemData(idx)
        if self._latest is not None and self._latest_jacobian is not None:
            self._set_view(self._latest.phi, self._latest_jacobian)
        else:
            self._refresh_display_from_volume()

    def _repopulate_method_combo(self, constraint: str) -> None:
        """Refill the method combo with the algorithms valid for the
        chosen constraint. Tries to preserve the previously-selected
        algo if it exists in the new list."""
        prev_algo = self._method_combo.currentData() if self._method_combo.count() else None
        self._method_combo.blockSignals(True)
        self._method_combo.clear()
        for algo, label in _METHOD_SPECS_BY_CONSTRAINT[constraint]:
            self._method_combo.addItem(label, algo)
        # Keep the prior algo selected if the new constraint also supports
        # it (e.g. switching constraint while "barrier" is selected keeps
        # barrier); otherwise fall back to the per-constraint default.
        target = (
            prev_algo
            if prev_algo and self._method_combo.findData(prev_algo) >= 0
            else DEFAULT_METHOD_BY_CONSTRAINT[constraint]
        )
        idx = self._method_combo.findData(target)
        if idx >= 0:
            self._method_combo.setCurrentIndex(idx)
        self._method_combo.blockSignals(False)

    def _on_constraint_changed(self, idx: int):
        constraint = self._constraint_combo.itemData(idx)
        self._repopulate_method_combo(constraint)

    def _on_z_changed(self, value: int):
        self._z = int(value)
        D = self._volume.shape[1] if self._volume is not None else 1
        self._z_label.setText(f'{self._z} / {D - 1}')
        self._refresh_display_from_volume()

    def _on_open_params(self):
        """Open the Params dialog. On accept, write the edited values
        back to the window's instance attrs; on cancel, discard."""
        dlg = ParamsDialog(self, history_max_size=self._history_max_size)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            vals = dlg.result_values()
            new_hms = int(vals['history_max_size'])
            if new_hms != self._history_max_size:
                self._history_max_size = new_hms
                self.statusBar().showMessage(
                    f'history_max_size set to {new_hms} (takes effect on next run)',
                    8_000,
                )

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
        algo = self._method_combo.currentData()
        constraint = self._constraint_combo.currentData()
        objective_id = self._objective_combo.currentData()
        method_id = _compose_method_id(algo, constraint)
        params = {
            'time_budget_s': float(self._budget_spin.value()),
            'max_iterations': int(self._max_iter_spin.value()),
            'objective_id': objective_id,
        }
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
        self._fps_label.setText(f'starting {method_id}…')
        self._last_count = 0
        self._last_tick.restart()
        # Reset the history widgets for the new run. Re-engage Live so
        # the first snapshots from the new worker auto-track.
        self._reset_history_widgets()
        self._live_check.setChecked(True)
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._stop_btn.setText('Stopping…')

    def _on_finished(self, phi_out, info):
        # Ignore late signals from a worker we've already replaced /
        # discarded (e.g. user loaded a new DVF mid-run). Without this
        # guard the old worker's phi_out would get spliced into the
        # *new* volume.
        if self.sender() is not self._worker:
            return
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
        if self.sender() is not self._worker:
            return
        self._stop_btn.setEnabled(False)
        self._run_full_btn.setEnabled(True)
        self._run_roi_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, 'Solver error', err)
        self._fps_label.setText('errored')

    # ----- render loop -------------------------------------------------------

    def _render_snapshot(self, snap, *, fast: bool = False) -> None:
        """Push ``snap`` to the plot + overlays + stats panels. Shared
        between the live-render path (auto-track latest) and the history
        slider (replay a past step). Recomputes the jacobian from
        ``snap.phi`` once per call (the snapshot no longer caches it)
        and stashes it on ``self._latest_jacobian`` so the inspector
        and any subsequent reads in this frame don't recompute.

        ``fast=True`` is set by the live-render tick on big fields —
        it skips the fold-overlay rebuild in VIEW_GRID (pure-Python
        QPainterPath construction that scales with folded-cell count
        and dominates render time on large slices). The slider scrub
        path always uses ``fast=False`` for full fidelity.
        """
        from dvfopt.jacobian.numpy_jdet import jacobian_det2D

        self._latest = snap
        self._latest_jacobian = jacobian_det2D(snap.phi)[0]
        self._set_view(snap.phi, self._latest_jacobian, fast=fast)
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

    def _on_render_tick(self):
        if self._worker is None:
            return
        snap = self._worker.take_latest()
        # Update the history slider's range to cover everything emitted
        # so far, and (if Live is on) advance to the latest frame.
        hist_len = self._worker.history_len()
        if hist_len > 0:
            self._history_slider.blockSignals(True)
            self._history_slider.setMaximum(hist_len - 1)
            if self._live_check.isChecked():
                self._history_programmatic = True
                self._history_slider.setValue(hist_len - 1)
                self._history_programmatic = False
            self._history_slider.blockSignals(False)
            self._sync_history_widgets()
        # Render the latest snapshot only when Live is on. In freeze mode
        # the slider handler controls what's shown.
        if snap is not None and self._live_check.isChecked():
            # Big-field protection: skip the fold overlay rebuild during
            # live ticks once H·W exceeds the threshold. Scrubbing the
            # slider (post-run, or when paused) still gets the full
            # overlay — see ``_on_history_slider`` which passes ``fast=False``.
            H, W = snap.phi.shape[1:]
            fast = (H * W) > self._fast_render_pixel_threshold
            self._render_snapshot(snap, fast=fast)

        # cb-rate once per second
        if self._last_tick.elapsed() >= 1000 and self._worker is not None:
            dt_s = self._last_tick.restart() / 1000.0
            cb_count = self._worker.callback_count
            delta = cb_count - self._last_count
            self._last_count = cb_count
            self._fps_label.setText(f'{cb_count} callbacks · {delta / dt_s:.1f} cb/s')

    def _on_history_grab(self):
        """User started dragging the slider — drop out of live mode so
        the next auto-advance doesn't fight the user's selection."""
        self._live_check.setChecked(False)

    def _on_history_slider(self, idx: int):
        if self._worker is None:
            return
        snap = self._worker.history_get(int(idx))
        if snap is None:
            return
        # A user-driven valueChanged means they're actively scrubbing —
        # disengage Live so the next auto-tick doesn't snap them back to
        # the end. Programmatic moves (the auto-track path in
        # ``_on_render_tick``) set the flag to skip this.
        if not self._history_programmatic and self._live_check.isChecked():
            self._live_check.setChecked(False)
        self._render_snapshot(snap)
        self._sync_history_widgets()

    def _on_history_prev(self):
        """Step the slider back by one. Counts as a user action — drops
        Live mode so auto-track doesn't immediately undo the step."""
        if self._worker is None or self._worker.history_len() == 0:
            return
        self._live_check.setChecked(False)
        new_val = max(0, self._history_slider.value() - 1)
        self._history_slider.setValue(new_val)

    def _on_history_next(self):
        """Step the slider forward by one."""
        if self._worker is None or self._worker.history_len() == 0:
            return
        n = self._worker.history_len()
        self._live_check.setChecked(False)
        new_val = min(n - 1, self._history_slider.value() + 1)
        self._history_slider.setValue(new_val)

    def _on_history_spin(self, abs_step: int):
        """User typed a step into the spinbox. Convert the absolute
        step index back to the slider's buffer index and let the slider
        valueChanged path do the actual render/sync.

        Skipped when the spinbox change came from
        :meth:`_sync_history_widgets` (which sets ``_history_programmatic``
        before updating) — without that guard the slider→spinbox sync
        would re-fire the spinbox→slider sync indefinitely.
        """
        if self._history_programmatic:
            return
        if self._worker is None or self._worker.history_len() == 0:
            return
        n = self._worker.history_len()
        total = self._worker.history_total
        offset = total - n  # absolute step at buffer index 0
        buf_idx = max(0, min(n - 1, int(abs_step) - offset))
        if buf_idx != self._history_slider.value():
            self._live_check.setChecked(False)
            self._history_slider.setValue(buf_idx)

    def _on_live_toggled(self, on: bool):
        """Re-checking Live snaps the view back to the latest step."""
        if on and self._worker is not None:
            hist_len = self._worker.history_len()
            if hist_len > 0:
                self._history_programmatic = True
                self._history_slider.setValue(hist_len - 1)
                self._history_programmatic = False
                snap = self._worker.history_get(hist_len - 1)
                if snap is not None:
                    self._render_snapshot(snap)
                self._sync_history_widgets()

    def _reset_history_widgets(self) -> None:
        """Put the slider / buttons / spinbox / total label back into
        their pristine, no-history state."""
        self._history_slider.blockSignals(True)
        self._history_slider.setMaximum(0)
        self._history_slider.setValue(0)
        self._history_slider.setEnabled(False)
        self._history_slider.blockSignals(False)
        self._history_spin.blockSignals(True)
        self._history_spin.setRange(0, 0)
        self._history_spin.setValue(0)
        self._history_spin.setEnabled(False)
        self._history_spin.blockSignals(False)
        self._history_prev_btn.setEnabled(False)
        self._history_next_btn.setEnabled(False)
        self._history_total_label.setText('/ —')

    def _sync_history_widgets(self) -> None:
        """Make the slider/spinbox/buttons/label consistent with the
        worker's current history. Called after any history-affecting
        change (auto-track advance, user scrub, slider arrow, spinbox
        edit). Programmatic moves are guarded against the spinbox
        ↔ slider feedback loop via ``_history_programmatic``.
        """
        if self._worker is None or self._worker.history_len() == 0:
            self._reset_history_widgets()
            return
        n = self._worker.history_len()
        total = self._worker.history_total
        idx = self._history_slider.value()
        offset = total - n  # absolute step at buffer index 0
        abs_step = idx + offset
        abs_max = total - 1
        # Slider already has its value/max set elsewhere; just enable it.
        self._history_slider.setEnabled(True)
        # Prev/next: only enabled when there's somewhere to step.
        self._history_prev_btn.setEnabled(idx > 0)
        self._history_next_btn.setEnabled(idx < n - 1)
        # Spinbox: absolute range + value. Guard the back-edge of the
        # sync loop via blockSignals + the programmatic flag.
        self._history_programmatic = True
        self._history_spin.blockSignals(True)
        self._history_spin.setRange(offset, abs_max)
        self._history_spin.setValue(abs_step)
        self._history_spin.setEnabled(True)
        self._history_spin.blockSignals(False)
        self._history_programmatic = False
        # Total label. The leading slash echoes "step <N> / <max>" so
        # the spinbox + label read naturally side-by-side.
        self._history_total_label.setText(f'/ {abs_max}')

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
            # Compute fold counts straight from the current slice so the
            # idle panel never looks like the field is feasible when it
            # isn't (the Jdet heatmap is uniformly red for fields whose
            # min Jdet is positive, even with sub-pixel 2-tri folds — the
            # bowtie default is exactly that case: 0 Jdet folds but 2
            # 2-tri folds).
            from dvfopt.jacobian.numpy_jdet import jacobian_det2D

            phi_2hw = self._volume[1:, self._z]
            jac = jacobian_det2D(phi_2hw)[0]
            min_tri = _min_tri_from_phi(phi_2hw)
            n_neg_jdet = int((jac < 0).sum())
            n_neg_tri = int((min_tri < 0).sum())
            # ``_min_tri_from_phi`` returns NaN at the boundary (no
            # cell-anchor exists past H-1, W-1). Use nanmin so the
            # idle readout shows the real interior minimum.
            return (
                '<b>Stats</b><br>'
                f'volume . . . {D}×{H}×{W}<br>'
                f'view . . . . {self._view_mode}<br>'
                f'min Jdet . . {jac.min():+.4f}<br>'
                f'Jdet folds . {n_neg_jdet}<br>'
                f'min T1/T2  . {np.nanmin(min_tri):+.4f}<br>'
                f'2-tri folds  {n_neg_tri}<br>'
                '(idle — press <i>Run full</i> to start)'
            )
        H, W = snap.phi.shape[1:]
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
            # ``self._latest_jacobian`` is populated by ``_render_snapshot``
            # alongside ``self._latest`` — fall back to a fresh compute
            # only if it somehow got out of sync.
            if self._latest_jacobian is not None:
                jac = self._latest_jacobian
            else:
                from dvfopt.jacobian.numpy_jdet import jacobian_det2D

                jac = jacobian_det2D(phi)[0]
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
