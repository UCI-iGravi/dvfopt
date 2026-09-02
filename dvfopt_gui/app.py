"""Main PyQtGraph window — DVF loader, view-mode toggles, section ROIs,
live overlay rect, pixel inspector, stats panel.

Features
--------

* **Load DVF...** — pick a ``.npy`` from disk. Both ``(3, D, H, W)``
  3D volumes (each z-slice runnable independently) and
  ``(3, 1, H, W)`` / ``(2, H, W)`` single 2D slices are supported.
* **View modes** — radio selector switches the central image between:
    * **Jdet (CD)** — central-difference Jacobian determinant per pixel
    * **Simplex (min T1, T2)** — minimum signed triangle area per cell
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

import json
import sys

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt_gui._shared import (  # noqa: F401  (re-exported for back-compat)
    _CONSTRAINT_SPECS,
    _DEFAULT_DVF_DIR,
    _METHOD_SPECS_2TRI,
    _METHOD_SPECS_BY_CONSTRAINT,
    _METHOD_SPECS_JDET,
    _METHOD_SPECS_JDET3D,
    _METHOD_SPECS_TET3D,
    _OBJECTIVE_SPECS,
    _REPO_ROOT,
    CONSTRAINT_2TRI,
    CONSTRAINT_JDET,
    CONSTRAINT_JDET3D,
    CONSTRAINT_TET3D,
    DEFAULT_CONSTRAINT,
    DEFAULT_METHOD_BY_CONSTRAINT,
    DEFAULT_OBJECTIVE,
    OBJECTIVE_L1,
    OBJECTIVE_L2,
    OBJECTIVE_NONE,
    UNDO_MAX_BYTES,
    VIEW_2TRI,
    VIEW_DIFF,
    VIEW_GRID,
    VIEW_INJ,
    VIEW_JDET,
    _compose_method_id,
    _default_roi_geometry,
    _folded_cells_path,
    _grid_lines,
    _jdet_colormap,
    _min_tri_from_phi,
    _osqp_available,
    _quiver_lines,
    _toolbar_separator,
    _torch_available,
    validate_finite,
)
from dvfopt_gui._win_fileio import FileIOMixin
from dvfopt_gui._win_render import RenderMixin
from dvfopt_gui._win_run import RunActionsMixin
from dvfopt_gui.convergence import ConvergencePlot
from dvfopt_gui.history import HistoryController
from dvfopt_gui.logdock import LogDock
from dvfopt_gui.overview import OverviewWorker, SliceOverviewStrip
from dvfopt_gui.worker import (  # noqa: F401  (re-exported for back-compat)
    DEFAULT_HISTORY_MAX,
    FEASIBILITY_THRESHOLD,
    LoadWorker,
    ReplayHistory,
    SolverWorker,
    StateSnapshot,
    _infeasible_count,
    _metric_counts,
    _metric_counts_3d,
    _metric_field_3d,
)


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
    * **Strategy** — auto-generated form of the current method's Strategy
      dataclass fields (see :mod:`dvfopt_gui.strategy_params`); "no
      editable parameters" for non-dataclass methods (auto / pipelines /
      marching).
    """

    def __init__(
        self, parent, *, history_max_size: int, strategy_algo: str, strategy_overrides: dict
    ):
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
            '<i>Applies to the next run. The current run keeps its original buffer size.</i>'
        )
        info.setWordWrap(True)
        history_form.addRow(info)
        tabs.addTab(history_tab, 'History')

        # --- Strategy tab -----------------------------------------------------
        from dvfopt_gui.strategy_params import StrategyParamsTab

        self._strategy_tab = StrategyParamsTab()
        self._strategy_tab.build(strategy_algo, dict(strategy_overrides))
        reset_row = QtWidgets.QWidget()
        reset_lay = QtWidgets.QVBoxLayout(reset_row)
        reset_lay.setContentsMargins(0, 0, 0, 0)
        reset_lay.addWidget(self._strategy_tab)
        reset_btn = QtWidgets.QPushButton('Reset to defaults')
        reset_btn.clicked.connect(lambda: self._strategy_tab.build(strategy_algo, {}))
        reset_lay.addWidget(reset_btn)
        tabs.addTab(reset_row, 'Strategy')

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
            'strategy_overrides': self._strategy_tab.values(),
        }


class LiveSolverWindow(FileIOMixin, RenderMixin, RunActionsMixin, QtWidgets.QMainWindow):
    """Live-viz window for the windowed-SLSQP solver.

    Construct with an optional starting ``deformation_i`` (any of
    ``(3, D, H, W)``, ``(3, 1, H, W)``, or ``(2, H, W)``) — pass
    ``None`` to start empty and use **Load DVF...** to pick a file.
    """

    def __init__(self, deformation_i=None, *, initial_params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('dvfopt — live solver visualisation')
        self.resize(1500, 900)
        # Floor below which the two dense toolbar rows start clipping
        # their controls; keeps every button reachable on small displays.
        self.setMinimumSize(1100, 640)

        # Extra windowed-SLSQP knobs that have no toolbar widget but are
        # accepted by ``iterative_serial`` (scipy method name, per-pixel
        # sub-iteration cap). Seeded from ``initial_params`` (e.g. the
        # demo's CLI flags) and forwarded to the worker on each run.
        self._initial_params = dict(initial_params or {})
        self._slsqp_method_name = str(self._initial_params.get('method_name', 'SLSQP'))
        self._max_per_index_iter = self._initial_params.get('max_per_index_iter', None)

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
        # Inspector T1/T2 cache: ``_triangle_areas_2d`` over the whole
        # slice is recomputed at most once per displayed field, so
        # hovering doesn't recompute it on every mouse-move — cheap at
        # 7×7, but O(H·W) per move otherwise. Invalidated whenever the
        # displayed field changes.
        self._inspector_tri: tuple[np.ndarray, np.ndarray] | None = None
        # Whole-volume 3D metric field cache: ``kind -> ndarray``, cleared
        # by ``_invalidate_metric_caches``. See ``_metric3d_field``.
        self._metric3d_cache: dict = {}
        self._worker: SolverWorker | None = None
        # One-shot load QThread (LoadWorker) — set on each Load DVF click,
        # checked by ``_on_load`` to guard against re-entry (Ctrl+O firing
        # again while a decode is still in flight) and drained by
        # ``closeEvent`` so it can't outlive the window.
        self._load_worker: LoadWorker | None = None
        self._picked_yx: tuple[int, int] | None = None
        # Active "Run section" crop bounds ``(y0, y1, x0, x1)`` or None for
        # a full-slice run. Set per-run; initialised here so any read
        # (e.g. ``_on_finished``) before the first run is well-defined.
        self._section_bounds: tuple[int, int, int, int] | None = None
        # Active 3D "Run section" crop bounds ``(z0, z1ex, y0, y1, x0, x1)``
        # (``z1ex`` exclusive) or None for a full-volume 3D run. Mirrors
        # ``_section_bounds`` for the 3D ROI path; set per-run.
        self._section_bounds_3d: tuple | None = None
        # True when the selected constraint is a whole-volume 3D family
        # (tet3d / jdet3d). In 3D mode: Run-all z is disabled; Run-full
        # passes the entire (3, D, H, W) volume; Run-section solves a
        # (z0:z1, y0:y1, x0:x1) sub-volume (see ``_section_bounds_3d``).
        self._is_3d_run = False
        # When non-None, a "Run all z" batch is in flight; holds the
        # z-slice indices still to be solved (current one already popped
        # and running). Drives the sequential chain in ``_on_finished``.
        self._run_all_remaining: list[int] | None = None
        # Full-pipeline (per-slice 2D -> 2.5D marching) state. `_pipeline_active`
        # suppresses the per-run undo push (ONE entry covers the whole
        # pipeline); `_pipeline_after_run_all` arms the 2.5D stage to start
        # when the Run-all batch drains.
        self._pipeline_active = False
        self._pipeline_after_run_all = False
        # Active run bookkeeping for the progress bar / ETA and the
        # before→after stats delta.
        self._active_method_id: str | None = None
        # True once this run's "Auto → <label>" status-bar note has been
        # shown (or there's nothing to show yet). Starts True so the
        # getattr default in ``_on_render_tick`` never fires mid-run
        # before the first ``_start_worker`` call; reset to False there.
        self._auto_label_shown: bool = True
        self._run_elapsed = QtCore.QElapsedTimer()
        self._input_n_neg: int | None = None
        # Undo/redo of corrections: each entry is a full ``(3, D, H, W)``
        # volume snapshot. A run pushes the pre-run volume before splicing
        # its result; Undo/Redo swap between them. Capped to bound memory.
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._UNDO_MAX = 30
        # Window-level params editable via the Params dialog. New
        # workers pick these up at construction; in-flight workers
        # retain whatever they were started with.
        self._history_max_size: int = DEFAULT_HISTORY_MAX
        # Per-method Strategy dataclass-field overrides, keyed by the algo
        # tag (family-qualified as ``f'{algo}@tet3d'`` or ``f'{algo}@jdet3d'``
        # in 3D mode — see ``_current_params_algo``). Edited via the Params dialog's
        # Strategy tab; merged into the worker's constructed Strategy at
        # ``_build_strategy`` time. Persisted to QSettings as JSON.
        self._strategy_overrides: dict[str, dict] = {}
        # Starting directory for the load/save dialogs — seeded from the
        # canonical DVF folder, then remembered across sessions (and
        # updated to the last file's folder) via QSettings.
        self._last_dir: str = _DEFAULT_DVF_DIR

        # ---- toolbar (top) ---------------------------------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()
        outer.addLayout(bar)

        self._load_btn = QtWidgets.QPushButton('Load DVF…')
        self._load_btn.setShortcut('Ctrl+O')
        self._load_btn.setToolTip('Load a .npy DVF or a saved .npz run (Ctrl+O).')
        self._load_btn.clicked.connect(self._on_load)
        bar.addWidget(self._load_btn)

        self._save_btn = QtWidgets.QPushButton('Save…')
        self._save_btn.setShortcut('Ctrl+S')
        self._save_btn.setToolTip(
            'Save the current DVF + per-step optimization history as a '
            'compressed .npz (Ctrl+S). Enabled once a DVF is loaded.'
        )
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        bar.addWidget(self._save_btn)

        self._revert_btn = QtWidgets.QPushButton('Revert')
        self._revert_btn.setToolTip(
            'Discard all corrections and restore the originally-loaded '
            'DVF (and clear the run history). Enabled once a DVF is loaded.'
        )
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self._on_revert)
        bar.addWidget(self._revert_btn)

        self._undo_btn = QtWidgets.QPushButton('Undo')
        self._undo_btn.setShortcut('Ctrl+Z')
        self._undo_btn.setToolTip('Undo the last correction (Ctrl+Z).')
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._on_undo)
        bar.addWidget(self._undo_btn)

        self._redo_btn = QtWidgets.QPushButton('Redo')
        self._redo_btn.setShortcut('Ctrl+Y')
        self._redo_btn.setToolTip('Redo the last undone correction (Ctrl+Y).')
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self._on_redo)
        bar.addWidget(self._redo_btn)

        bar.addWidget(_toolbar_separator())
        bar.addWidget(QtWidgets.QLabel('View:'))
        self._view_combo = QtWidgets.QComboBox()
        self._view_combo.addItem('Jdet (CD)', VIEW_JDET)
        self._view_combo.addItem('Simplex (min T1, T2)', VIEW_2TRI)
        self._view_combo.addItem('Deformation grid', VIEW_GRID)
        self._view_combo.addItem('Δ Jdet vs input', VIEW_DIFF)
        self._view_combo.addItem('Injectivity gap (min axial)', VIEW_INJ)
        self._view_combo.setToolTip(
            'Central image. "Δ Jdet vs input" shows the current minus the '
            'originally-loaded per-pixel Jdet (red = increased, blue = '
            'decreased) — pair it with Auto levels to read the change.'
        )
        # Keep the dropdown in sync with the default ``_view_mode``.
        # The grid view is the only one that always makes folds visible
        # (Jdet view is uniformly red when min Jdet > 0, even with
        # simplex (2D) folds present — that's the canonical "looks already
        # optimized" trap).
        _default_idx = self._view_combo.findData(self._view_mode)
        if _default_idx >= 0:
            self._view_combo.setCurrentIndex(_default_idx)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        bar.addWidget(self._view_combo)

        # Auto-levels toggle for the heatmap colour scale. Off → fixed
        # ±1 levels (the historical default, good for reading Jdet as
        # feasible/folded). On → per-frame symmetric autoscale so fields
        # whose values exceed ±1 don't saturate to flat blue/red.
        self._autolevel_check = QtWidgets.QCheckBox('Auto levels')
        self._autolevel_check.setChecked(False)
        self._autolevel_check.setToolTip(
            'Heatmap colour scale. Off: fixed ±1 (white = 0). On: '
            'per-frame symmetric autoscale to the displayed extent, so '
            'large-magnitude fields stay readable. (No effect in the '
            'deformation-grid view.)'
        )
        self._autolevel_check.toggled.connect(self._on_autolevel_toggled)
        bar.addWidget(self._autolevel_check)

        # Displacement-arrow overlay toggle — draws per-pixel arrows
        # (grid point → warped point) on top of any view.
        self._arrows_check = QtWidgets.QCheckBox('Arrows')
        self._arrows_check.setChecked(False)
        self._arrows_check.setToolTip(
            'Overlay per-pixel displacement arrows (grid point → warped '
            'point) on the current view. Subsampled on large fields.'
        )
        self._arrows_check.toggled.connect(self._on_arrows_toggled)
        bar.addWidget(self._arrows_check)

        bar.addWidget(_toolbar_separator())
        bar.addWidget(QtWidgets.QLabel('z:'))
        self._z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.setEnabled(False)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        bar.addWidget(self._z_slider, stretch=1)
        self._z_label = QtWidgets.QLabel('—')
        bar.addWidget(self._z_label)

        # 3D-mode sub-volume z-range (hidden in 2D). Pairs with the Rect ROI
        # (which supplies y/x) for "Run section" on a sub-volume.
        self._z0_label = QtWidgets.QLabel('z0:')
        self._z0_spin = QtWidgets.QSpinBox()
        self._z1_label = QtWidgets.QLabel('z1:')
        self._z1_spin = QtWidgets.QSpinBox()
        for wdg in (self._z0_label, self._z0_spin, self._z1_label, self._z1_spin):
            bar.addWidget(wdg)
            wdg.setVisible(False)

        bar.addWidget(_toolbar_separator())
        self._run_full_btn = QtWidgets.QPushButton('Run full')
        self._run_full_btn.setShortcut('F5')
        self._run_full_btn.setToolTip('Solve the full current slice (F5).')
        self._run_full_btn.clicked.connect(lambda: self._on_run(use_roi=False))
        bar.addWidget(self._run_full_btn)
        self._run_roi_btn = QtWidgets.QPushButton('Run section')
        self._run_roi_btn.setShortcut('Ctrl+R')
        self._run_roi_btn.setToolTip(
            'Solve only inside the ROI rectangle (Ctrl+R). In 3D mode also '
            'uses the z0/z1 spinboxes to crop a sub-volume.'
        )
        self._run_roi_btn.clicked.connect(lambda: self._on_run(use_roi=True))
        bar.addWidget(self._run_roi_btn)
        self._run_all_btn = QtWidgets.QPushButton('Run all z')
        self._run_all_btn.setToolTip(
            'Solve every z-slice of a 3D volume in sequence with the '
            'current method (disabled for single-slice 2D fields).'
        )
        self._run_all_btn.setEnabled(False)
        self._run_all_btn.clicked.connect(self._on_run_all)
        bar.addWidget(self._run_all_btn)
        self._pipeline_btn = QtWidgets.QToolButton()
        self._pipeline_btn.setText('Pipeline ▾')
        self._pipeline_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._pipeline_btn.setToolTip(
            'Volume workflows: 2.5D marching (fold prevention; needs dz == 0, '
            'i.e. per-slice-corrected input) or the full pipeline (per-slice '
            '2D with the selected method, then 2.5D marching).'
        )
        pipe_menu = QtWidgets.QMenu(self._pipeline_btn)
        self._act_run_25d = pipe_menu.addAction('Run 2.5D marching', self._on_run_25d)
        self._act_run_pipeline = pipe_menu.addAction(
            'Full pipeline (2D + 2.5D)', self._on_run_pipeline_full
        )
        self._pipeline_btn.setMenu(pipe_menu)
        self._pipeline_btn.setEnabled(False)
        bar.addWidget(self._pipeline_btn)
        self._stop_btn = QtWidgets.QPushButton('Stop')
        self._stop_btn.setShortcut('Esc')
        self._stop_btn.setToolTip(
            'Request the running solve to stop (Esc). In 3D the wallbreaker '
            'methods (M10Tet/M14Tet/M14-Schwarz3D) stop at the next phase '
            'boundary; SLSQP-fullgrid-3D / Barrier run to completion '
            '(bound them with time_budget_s / max_iter).'
        )
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
            'Simplex (2D): full-coverage triangle areas (catches sub-pixel folds the '
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
        # Disable 3D constraint entries until a D>1 volume is loaded.
        self._update_3d_constraint_enabled()
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
        _init_max_iter = self._initial_params.get('max_iterations', None)
        self._max_iter_spin.setValue(int(_init_max_iter) if _init_max_iter else 200)
        self._max_iter_spin.setToolTip(
            'Outer-iteration cap for SLSQP-windowed. Ignored by '
            'wallbreaker methods (they use time_budget_s instead).'
        )
        method_bar.addWidget(self._max_iter_spin)

        method_bar.addWidget(QtWidgets.QLabel('thr:'))
        self._thr_spin = QtWidgets.QDoubleSpinBox()
        self._thr_spin.setDecimals(4)
        self._thr_spin.setRange(0.0, 1.0)
        self._thr_spin.setSingleStep(0.005)
        self._thr_spin.setValue(FEASIBILITY_THRESHOLD)
        self._thr_spin.setToolTip(
            'Solver feasibility threshold: every constraint is enforced as '
            'C(phi) >= thr. Also drives the stats panel\'s infeasible(<thr) '
            'counts. Default 0.01 (package default).'
        )
        method_bar.addWidget(self._thr_spin)
        self._thr_spin.valueChanged.connect(self._on_threshold_changed)

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

        # Colour-scale legend for the heatmap, docked to the right of the
        # plot. Non-interactive — the GUI drives its levels (fixed ±1 or
        # per-frame autoscale) via ``_apply_levels``. Hidden in the
        # deformation-grid view (no heatmap to scale).
        self._cbar = pg.ColorBarItem(values=(-1.0, 1.0), colorMap=cmap, interactive=False, width=14)
        self._cbar.setImageItem(self._img, insert_in=self._plot.getPlotItem())

        self._grid_curve = pg.PlotDataItem(pen=pg.mkPen(color=(0, 0, 0), width=2), connect='finite')
        self._grid_curve.setVisible(False)
        self._plot.addItem(self._grid_curve)

        # Displacement-arrow overlay (toggled by the Arrows checkbox).
        # Green so it reads over both the heatmap and the black wireframe.
        self._quiver_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=(0, 140, 0), width=1), connect='finite'
        )
        self._quiver_curve.setVisible(False)
        self._plot.addItem(self._quiver_curve)

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

        # Live convergence chart — fold count + worst area vs step, with a
        # cursor tracking the history slider. Populated from the worker's
        # recorded trajectories; see ``_refresh_convergence``.
        right.addWidget(QtWidgets.QLabel('<b>Convergence</b>'))
        self._conv_plot = ConvergencePlot()
        self._conv_plot.setMinimumHeight(150)

        # Live solver-log dock (hidden until toggled via the View menu).
        # Attaching its handler makes it the process's single log sink
        # (suppresses _logging's stdout auto-install). The dock's level
        # combo is read LIVE at worker start (worker_verbose) — no cached
        # copy to drift.
        self._log_dock = LogDock(self)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._log_dock)
        self._log_dock.hide()
        self._log_dock.attach()
        # SolveInfo of the last finished Solver-path run (for the report).
        self._last_solve_info = None
        right.addWidget(self._conv_plot, stretch=2)
        # Number of history entries last plotted — lets us rebuild the
        # curve only when it grows (the cursor still moves every frame).
        self._conv_len = -1

        # Per-slice fold overview (volumes only): computed in the background,
        # click to jump z.
        self._overview_strip = SliceOverviewStrip()
        self._overview_strip.setVisible(False)
        self._overview_strip.sliceClicked.connect(self._z_slider.setValue)
        outer.addWidget(self._overview_strip)
        self._overview_worker: OverviewWorker | None = None
        self._overview_counts: np.ndarray | None = None

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
        history_bar.addWidget(self._history_slider, stretch=1)

        # ▶ step-forward button.
        self._history_next_btn = QtWidgets.QToolButton()
        self._history_next_btn.setArrowType(QtCore.Qt.RightArrow)
        self._history_next_btn.setEnabled(False)
        self._history_next_btn.setToolTip('Next step')
        history_bar.addWidget(self._history_next_btn)

        # Editable step number. Shows the *absolute* step index (so
        # what the user types matches the "step N / M" they read) —
        # we convert to the slider's buffer index in the handler.
        history_bar.addWidget(QtWidgets.QLabel('step'))
        self._history_spin = QtWidgets.QSpinBox()
        self._history_spin.setRange(0, 0)
        self._history_spin.setEnabled(False)
        self._history_spin.setToolTip(
            'Jump to a specific step by typing its index. Mirrors the slider position.'
        )
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
        history_bar.addWidget(self._live_check)

        # The history-scrub state machine (slider/spin/buttons/Live sync)
        # lives in its own controller; it reaches back for the current
        # worker and renders chosen snapshots through the window.
        self._history = HistoryController(
            slider=self._history_slider,
            spin=self._history_spin,
            prev_btn=self._history_prev_btn,
            next_btn=self._history_next_btn,
            total_label=self._history_total_label,
            live_check=self._live_check,
            get_worker=lambda: self._worker,
            render_snapshot=self._render_snapshot,
        )

        # ---- bottom status row -----------------------------------------
        statusbar = QtWidgets.QHBoxLayout()
        outer.addLayout(statusbar)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setMaximumWidth(280)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat('')
        self._progress.setToolTip(
            'Run progress: elapsed/budget for the wallbreaker family, '
            'outer-iter/max for SLSQP, or a busy indicator otherwise.'
        )
        statusbar.addWidget(self._progress)
        statusbar.addStretch(1)
        self._fps_label = QtWidgets.QLabel('idle')
        statusbar.addWidget(self._fps_label)

        # Mouse pick — click pins a pixel; hover tracks the cursor so the
        # inspector reads out live without requiring a click. The hover
        # signal is rate-limited (≤30 Hz) via a SignalProxy because
        # ``_format_inspector`` recomputes triangle areas over the whole
        # field per call — unthrottled mouse-move would thrash big slices.
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_click)
        self._hover_proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved
        )

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

        # Menu bar mirroring the toolbar actions — pure discoverability
        # (the toolbar buttons own the keyboard shortcuts; the menu just
        # surfaces them as hint text, plus a shortcuts/about page).
        self._build_menus()

        # Restore window geometry + last selections from the previous
        # session (before loading any initial DVF so the restored view
        # mode / levels apply to it).
        self._restore_settings()

        # Initial DVF if supplied.
        if deformation_i is not None:
            self._load_array(np.asarray(deformation_i))

    # ----- menus -------------------------------------------------------------

    def _build_menus(self) -> None:
        """Populate the menu bar from the existing action handlers.

        Shortcut keys stay owned by the toolbar buttons (which set them);
        the menu items only *display* the key as hint text (via a ``\\t``
        in the label) so we don't double-register a shortcut and trigger
        Qt's ambiguous-overload warning.
        """
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        self._load_action = file_menu.addAction('Load DVF…\tCtrl+O', self._on_load)
        file_menu.addAction('New random folded field…', self._on_new_random)
        file_menu.addAction('Save…\tCtrl+S', self._on_save)
        file_menu.addAction('Export corrected DVF…', self._on_export)
        file_menu.addAction('Revert', self._on_revert)
        file_menu.addSeparator()
        # Quit owns its own shortcut (no toolbar button competes for it).
        quit_act = file_menu.addAction('Quit', self.close)
        quit_act.setShortcut('Ctrl+Q')

        view_menu = menubar.addMenu('&View')
        log_toggle = self._log_dock.toggleViewAction()
        log_toggle.setText('Solver log')
        view_menu.addAction(log_toggle)
        self._report_action = view_menu.addAction('Save convergence report…', self._on_save_report)
        self._report_action.setEnabled(False)

        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction('Undo\tCtrl+Z', self._on_undo)
        edit_menu.addAction('Redo\tCtrl+Y', self._on_redo)
        edit_menu.addSeparator()
        edit_menu.addAction('Params…', self._on_open_params)

        run_menu = menubar.addMenu('&Run')
        run_menu.addAction('Run full\tF5', lambda: self._on_run(use_roi=False))
        run_menu.addAction('Run section\tCtrl+R', lambda: self._on_run(use_roi=True))
        run_menu.addAction('Run all z', self._on_run_all)
        run_menu.addAction('Run 2.5D marching', self._on_run_25d)
        run_menu.addAction('Full pipeline (2D + 2.5D)', self._on_run_pipeline_full)
        run_menu.addAction('Stop\tEsc', self._on_stop)

        help_menu = menubar.addMenu('&Help')
        help_menu.addAction('Keyboard shortcuts…', self._show_shortcuts)
        help_menu.addAction('About', self._show_about)

    def _on_save_report(self) -> None:
        """Render the last run's SolveInfo phases via
        :func:`dvfopt.viz.plot_solve_info` and save as an image."""
        info = self._last_solve_info
        if info is None or not getattr(info, 'phases', None):
            QtWidgets.QMessageBox.information(
                self,
                'No solve history',
                'No recorded phase history yet — run a solver first.',
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save convergence report', 'convergence_report.png', 'Images (*.png *.pdf)'
        )
        if not path:
            return
        import sys

        import matplotlib

        # Only pick the Agg backend while pyplot is still unloaded: a
        # forced switch in an embedding session (Jupyter %matplotlib)
        # would close every open figure and hijack the user's backend.
        # savefig works on any backend, so an already-loaded one is fine.
        if 'matplotlib.pyplot' not in sys.modules:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        from dvfopt.viz import plot_solve_info

        fig = plot_solve_info(info, threshold=self._display_threshold(), save_path=path)
        plt.close(fig)
        self.statusBar().showMessage(f'Saved convergence report: {path}', 5000)

    def _show_shortcuts(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            'Keyboard shortcuts',
            '<b>Keyboard shortcuts</b>'
            '<pre>'
            'Ctrl+O   Load DVF (.npy / .npz)\n'
            'Ctrl+S   Save run (.npz)\n'
            'Ctrl+Z   Undo correction\n'
            'Ctrl+Y   Redo correction\n'
            'F5       Run full slice\n'
            'Ctrl+R   Run section (ROI)\n'
            'Esc      Stop the running solve\n'
            'Ctrl+Q   Quit\n'
            '←  / →   Step history (slider focused)'
            '</pre>',
        )

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            'About dvfopt GUI',
            '<b>dvfopt — live solver visualisation</b><br><br>'
            'Load a 2D section or 3D volume (.npy / .npz) to inspect its '
            'Jacobian-determinant / 2-triangle fold structure, or run a '
            'correction solver and scrub its per-step history.<br><br>'
            'Loading is view-only until you press Run — no solve happens '
            'on open.',
        )

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

    @staticmethod
    def _select_combo_data(combo: QtWidgets.QComboBox, data) -> None:
        """Set ``combo`` to the entry whose userData equals ``data``
        (no-op if absent)."""
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

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

    def _repopulate_method_combo(self, constraint: str) -> None:
        """Refill the method combo with the algorithms valid for the
        chosen constraint. Tries to preserve the previously-selected
        algo if it exists in the new list."""
        prev_algo = self._method_combo.currentData() if self._method_combo.count() else None
        self._method_combo.blockSignals(True)
        self._method_combo.clear()
        for algo, label in _METHOD_SPECS_BY_CONSTRAINT[constraint]:
            self._method_combo.addItem(label, algo)
        # GPU barrier needs torch; keep it visible but disabled when absent.
        idx = self._method_combo.findData('barrier_torch')
        if idx >= 0 and not _torch_available():
            self._method_combo.model().item(idx).setEnabled(False)
        # ISQP windowed needs osqp; same visible-but-disabled treatment.
        idx = self._method_combo.findData('isqp_windowed')
        if idx >= 0 and not _osqp_available():
            self._method_combo.model().item(idx).setEnabled(False)
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

    def _constraint_is_3d(self, tag: str) -> bool:
        return tag in (CONSTRAINT_TET3D, CONSTRAINT_JDET3D)

    def _update_3d_constraint_enabled(self) -> None:
        """Enable the 3D constraint entries only for D>1 volumes."""
        D = self._volume.shape[1] if self._volume is not None else 1
        model = self._constraint_combo.model()
        for tag in (CONSTRAINT_TET3D, CONSTRAINT_JDET3D):
            idx = self._constraint_combo.findData(tag)
            if idx >= 0:
                model.item(idx).setEnabled(D > 1)

    def _apply_mode_gating(self) -> None:
        """Reflect 2D/3D mode in the run controls."""
        D = self._volume.shape[1] if self._volume is not None else 1
        self._run_roi_btn.setEnabled(self._volume is not None)
        self._run_all_btn.setEnabled((not self._is_3d_run) and D > 1)
        self._pipeline_btn.setEnabled(D > 1)
        self._section_roi.setVisible(self._volume is not None)
        show_z = self._is_3d_run and D > 1
        for wdg in (self._z0_label, self._z0_spin, self._z1_label, self._z1_spin):
            wdg.setVisible(show_z)
        if show_z:
            self._z0_spin.setRange(0, D - 1)
            self._z1_spin.setRange(0, D - 1)
            if self._z1_spin.value() == 0:
                self._z1_spin.setValue(D - 1)

    def _on_constraint_changed(self, idx: int):
        constraint = self._constraint_combo.itemData(idx)
        self._is_3d_run = self._constraint_is_3d(constraint)
        self._repopulate_method_combo(constraint)
        self._apply_mode_gating()

    def _on_z_changed(self, value: int):
        self._z = int(value)
        D = self._volume.shape[1] if self._volume is not None else 1
        self._z_label.setText(f'{self._z} / {D - 1}')
        if self._is_3d_run:
            # In 3D the run spans the whole volume; changing z only
            # re-slices the view — keep the worker/history.
            if self._latest is not None and self._latest.phi.ndim == 4:
                self._render_snapshot(self._latest)
            else:
                self._refresh_display_from_volume()
            self._overview_strip.set_current(self._z)
            return
        # A run's history belongs to the slice it was solved on. Switching
        # z invalidates it — drop the worker reference and reset the scrub
        # widgets so the slider can't replay another slice's snapshots
        # over this one.
        self._worker = None
        self._latest = None
        self._latest_jacobian = None
        self._history.reset()
        self._history.set_live(True)
        self._refresh_display_from_volume()
        self._overview_strip.set_current(self._z)

    def _current_params_algo(self) -> str:
        """Key for strategy-override storage: the algo tag, family-qualified
        in 3D mode (the 3D strategy classes have different knobs)."""
        algo = self._method_combo.currentData() or ''
        if not self._is_3d_run:
            return algo
        family = self._constraint_combo.currentData()
        return f'{algo}@{family}'  # '@tet3d' or '@jdet3d'

    def _on_open_params(self):
        """Open the Params dialog. On accept, write the edited values
        back to the window's instance attrs; on cancel, discard."""
        algo = self._current_params_algo()
        dlg = ParamsDialog(
            self,
            history_max_size=self._history_max_size,
            strategy_algo=algo,
            strategy_overrides=self._strategy_overrides.get(algo, {}),
        )
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            vals = dlg.result_values()
            new_hms = int(vals['history_max_size'])
            if new_hms != self._history_max_size:
                self._history_max_size = new_hms
                self.statusBar().showMessage(
                    f'history_max_size set to {new_hms} (takes effect on next run)',
                    8_000,
                )
            strategy_overrides = vals['strategy_overrides']
            if strategy_overrides:
                self._strategy_overrides[algo] = strategy_overrides
            else:
                self._strategy_overrides.pop(algo, None)

    # ----- formatters --------------------------------------------------------

    def _format_stats(self, snap: StateSnapshot | None) -> str:
        if snap is None:
            if self._volume is None:
                return '<b>Stats</b><br>(no DVF loaded — click "Load DVF…")'
            H, W = self._volume.shape[2:]
            D = self._volume.shape[1]
            if self._is_3d_run:
                kind = (
                    'tet3d'
                    if self._constraint_combo.currentData() == CONSTRAINT_TET3D
                    else 'jdet3d'
                )
                field = self._metric3d_field(self._volume, kind)
                n_neg = int((field <= 0).sum())
                min_T = float(field.min())
                thr = self._display_threshold()
                infeas = int((field < thr).sum())
                return (
                    '<b>Stats (3D)</b><br>'
                    f'volume . . . . {D}×{H}×{W}<br>'
                    f'metric . . . . {kind}<br>'
                    f'3D folds . . . {n_neg}<br>'
                    f'min signed . . {min_T:+.5f}<br>'
                    f'infeasible(&lt;{thr:g}) {infeas}<br>'
                    '(idle — press <i>Run full</i> to start)'
                )
            # Compute fold counts straight from the current slice so the
            # idle panel never looks like the field is feasible when it
            # isn't (the Jdet heatmap is uniformly red for fields whose
            # min Jdet is positive, even with sub-pixel simplex (2D) folds — the
            # bowtie default is exactly that case: 0 Jdet folds but 2
            # simplex (2D) folds).
            phi_2hw = self._volume[1:, self._z]
            jac = jacobian_det2D(phi_2hw)[0]
            min_tri = _min_tri_from_phi(phi_2hw)
            # Fold counts (metric <= 0) share the worker's convention so
            # the idle panel matches the running n_neg readout. The
            # solver, however, targets ``>= threshold`` (user-editable via
            # the thr: spinbox, default 0.01) — surface the stricter
            # "still infeasible" counts too, so a field with 0 folds but
            # min in (0, thr) doesn't read as "done".
            thr = self._display_threshold()
            n_neg_jdet, _ = _metric_counts(phi_2hw, 'jdet')
            n_neg_tri, _ = _metric_counts(phi_2hw, '2tri')
            infeas_jdet = _infeasible_count(phi_2hw, 'jdet', thr)
            infeas_tri = _infeasible_count(phi_2hw, '2tri', thr)
            interior = max(1, (H - 1) * (W - 1))
            # ``_min_tri_from_phi`` returns NaN at the boundary (no
            # cell-anchor exists past H-1, W-1). Use nanmin so the
            # idle readout shows the real interior minimum.
            return (
                '<b>Stats</b><br>'
                f'volume . . . . {D}×{H}×{W}<br>'
                f'view . . . . . {self._view_mode}<br>'
                f'max |disp| . . {self._max_abs_disp(phi_2hw):.3f}<br>'
                f'min Jdet . . . {jac.min():+.4f}<br>'
                f'Jdet folds . . {n_neg_jdet}<br>'
                f'min T1/T2  . . {np.nanmin(min_tri):+.4f}<br>'
                f'simplex folds  {n_neg_tri}  ({100 * n_neg_tri / interior:.1f}%)<br>'
                f'infeasible(&lt;{thr:g}) Jdet {infeas_jdet} · simplex {infeas_tri}<br>'
                '(idle — press <i>Run full</i> to start)'
            )
        if snap.phi.ndim == 4:  # 3D volume snapshot
            _, D, H, W = snap.phi.shape
            thr = self._display_threshold()
            feas_flag = '' if snap.min_T >= thr else f'  (&lt;{thr:g})'
            delta = ''
            if self._input_n_neg is not None:
                delta = f'vs input . . . {self._input_n_neg} → {snap.n_neg}<br>'
            return (
                '<b>Stats (3D)</b><br>'
                f'outer iter . . {snap.outer_iter}<br>'
                f'volume . . . . {D}×{H}×{W}<br>'
                f'n_neg . . . . . {snap.n_neg}<br>'
                f'{delta}'
                f'min_T . . . . . {snap.min_T:+.5f}{feas_flag}'
            )
        H, W = snap.phi.shape[1:]
        interior = max(1, (H - 1) * (W - 1))
        delta = ''
        if self._input_n_neg is not None:
            delta = f'vs input . . . {self._input_n_neg} → {snap.n_neg}<br>'
        # Flag when the worst cell is positive but still inside the
        # solver's feasibility margin — folds==0 yet not solver-feasible.
        thr = self._display_threshold()
        feas_flag = '' if snap.min_T >= thr else f'  (&lt;{thr:g})'
        return (
            '<b>Stats</b><br>'
            f'outer iter . . {snap.outer_iter}<br>'
            f'per-pixel . . . {snap.per_index_iter}<br>'
            f'n_neg . . . . . {snap.n_neg}  ({100 * snap.n_neg / interior:.1f}%)<br>'
            f'{delta}'
            f'min_T . . . . . {snap.min_T:+.5f}{feas_flag}<br>'
            f'max |disp| . . {self._max_abs_disp(snap.phi):.3f}<br>'
            f'window . . . . ({snap.window_y0}–{snap.window_y1}, '
            f'{snap.window_x0}–{snap.window_x1})  '
            f'{snap.window_y1 - snap.window_y0}×{snap.window_x1 - snap.window_x0}<br>'
            f'padded . . . . {snap.is_padded}<br>'
            f'target pixel . (y={snap.neg_y}, x={snap.neg_x})<br>'
            f'grid . . . . . {H}×{W}'
        )

    def _display_threshold(self) -> float:
        """The user-selected feasibility threshold (spinbox), used for both
        solving and the stats panel's infeasible counts."""
        return float(self._thr_spin.value())

    @staticmethod
    def _max_abs_disp(phi_2hw: np.ndarray) -> float:
        """Largest per-pixel displacement magnitude ``√(dy²+dx²)``."""
        return float(np.sqrt(phi_2hw[0] ** 2 + phi_2hw[1] ** 2).max())

    def _format_inspector(self, yx: tuple[int, int] | None) -> str:
        if yx is None:
            return '<b>Pixel inspector</b><br>(click a pixel)'
        y, x = yx
        phi3d = None
        if self._latest is not None and self._latest.phi.ndim == 4:
            phi3d = self._latest.phi
        elif self._latest is None and self._volume is not None and self._volume.shape[1] > 1:
            # Idle with a true-3D volume loaded: read the volume directly
            # instead of falling through to the 2D single-slice readout.
            phi3d = self._volume
        if phi3d is not None:
            z = min(self._z, phi3d.shape[1] - 1)
            mv = self._metric3d_field(phi3d, 'tet3d')
            Dm, Hm, Wm = mv.shape
            if not (0 <= y < Hm and 0 <= x < Wm):
                return '<b>Pixel inspector</b><br>(out of bounds)'
            zz = min(z, Dm - 1)
            return (
                '<b>Pixel inspector (3D)</b><br>'
                f'(z={zz}, y={y}, x={x})<br>'
                f'min simplex V . {mv[zz, y, x]:+.5f}'
            )
        # Prefer the live snapshot's phi; fall back to the volume.
        if self._latest is not None:
            phi = self._latest.phi
            # ``self._latest_jacobian`` is populated by ``_render_snapshot``
            # alongside ``self._latest`` — fall back to a fresh compute
            # only if it somehow got out of sync.
            if self._latest_jacobian is not None:
                jac = self._latest_jacobian
            else:
                jac = jacobian_det2D(phi)[0]
        elif self._volume is not None:
            phi = self._volume[1:, self._z]
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
            T1, T2 = self._triangle_areas_cached(phi)
            t1_str = f'{T1[y, x]:+.5f}'
            t2_str = f'{T2[y, x]:+.5f}'
        return (
            '<b>Pixel inspector</b><br>'
            f'(y={y}, x={x})<br>'
            f'Jdet . . . {jac[y, x]:+.5f}<br>'
            f'T1 . . . . {t1_str}<br>'
            f'T2 . . . . {t2_str}'
        )

    # ----- session persistence -----------------------------------------------

    @staticmethod
    def _settings() -> QtCore.QSettings:
        return QtCore.QSettings('dvfopt', 'dvfopt_gui')

    def _restore_settings(self) -> None:
        """Restore window geometry + toolbar selections from the previous
        session. Anything the demo passed via ``initial_params`` wins over
        the saved value (it's a deliberate per-launch override)."""
        s = self._settings()
        geo = s.value('geometry')
        if geo is not None:
            self.restoreGeometry(geo)
        self._last_dir = s.value('last_dir', self._last_dir, type=str)
        # Constraint first — it repopulates the method combo.
        constraint = s.value('constraint', '', type=str)
        if constraint:
            self._select_combo_data(self._constraint_combo, constraint)
        method = s.value('method', '', type=str)
        if method:
            self._select_combo_data(self._method_combo, method)
        objective = s.value('objective', '', type=str)
        if objective:
            self._select_combo_data(self._objective_combo, objective)
        view = s.value('view_mode', '', type=str)
        if view:
            idx = self._view_combo.findData(view)
            if idx >= 0:
                self._view_combo.setCurrentIndex(idx)
        self._autolevel_check.setChecked(s.value('auto_levels', False, type=bool))
        tb = s.value('time_budget_s', 0.0, type=float)
        if tb:
            self._budget_spin.setValue(tb)
        if 'max_iterations' not in self._initial_params:
            mi = s.value('max_iter', 0, type=int)
            if mi:
                self._max_iter_spin.setValue(mi)
        # ``if thr:`` would silently skip a legitimately-saved 0.0 —
        # presence, not truthiness, decides whether to restore.
        if s.contains('threshold'):
            self._thr_spin.setValue(s.value('threshold', 0.0, type=float))
        hms = s.value('history_max_size', 0, type=int)
        if hms:
            self._history_max_size = hms
        raw = s.value('strategy_overrides', '', type=str)
        if raw:
            try:
                self._strategy_overrides = {k: dict(v) for k, v in json.loads(raw).items()}
            except (ValueError, TypeError, AttributeError):
                self._strategy_overrides = {}

    def _save_settings(self) -> None:
        """Persist window geometry + toolbar selections for next launch."""
        s = self._settings()
        s.setValue('geometry', self.saveGeometry())
        s.setValue('last_dir', self._last_dir)
        s.setValue('constraint', self._constraint_combo.currentData() or '')
        s.setValue('method', self._method_combo.currentData() or '')
        s.setValue('objective', self._objective_combo.currentData() or '')
        s.setValue('view_mode', self._view_mode)
        s.setValue('auto_levels', self._autolevel_check.isChecked())
        s.setValue('time_budget_s', float(self._budget_spin.value()))
        s.setValue('max_iter', int(self._max_iter_spin.value()))
        s.setValue('threshold', self._display_threshold())
        s.setValue('history_max_size', int(self._history_max_size))
        s.setValue('strategy_overrides', json.dumps(self._strategy_overrides))

    # ----- lifecycle ---------------------------------------------------------

    def closeEvent(self, ev):
        self._save_settings()
        # Cancel any in-flight run and wait for the worker to actually
        # exit before tearing down — otherwise the QThread can outlive
        # the window. ``request_stop`` is only honoured at the next
        # solver checkpoint, so we wait in slices (pumping the event loop
        # so a final snapshot signal can drain) up to a generous cap,
        # then fall back to ``terminate`` as a last resort since the
        # process is exiting anyway.
        worker = self._worker
        if worker is not None and getattr(worker, 'isRunning', lambda: False)():
            worker.request_stop()
            waited_ms = 0
            cap_ms = 30_000
            while worker.isRunning() and waited_ms < cap_ms:
                QtWidgets.QApplication.processEvents()
                worker.wait(100)
                waited_ms += 100
            if worker.isRunning():
                # Stuck inside an uninterruptible solve — force it down.
                worker.terminate()
                worker.wait(2000)
        # Same reasoning for the background overview-strip worker: a bare
        # ``cancel()`` only flips a flag checked between per-slice metric
        # computations, so wait (bounded) for it to actually exit before
        # the window (its parent) is torn down.
        ow = getattr(self, '_overview_worker', None)
        if ow is not None and ow.isRunning():
            ow.cancel()
            ow.wait(2_000)
            if ow.isRunning():
                # Per-slice cancel checks make this near-impossible, but a
                # still-running thread at teardown can crash Qt — force it.
                ow.terminate()
                ow.wait(1_000)
        # The load worker has no cancel flag (it is a one-shot decode);
        # give it a bounded wait so it cannot outlive the window, then
        # force it down — the process is exiting anyway.
        lw = self._load_worker
        if lw is not None and getattr(lw, 'isRunning', lambda: False)():
            lw.wait(5_000)
            if lw.isRunning():
                lw.terminate()
                lw.wait(1_000)
        # Detach AFTER the worker drains: removing the dock handler while
        # the solver thread still logs would trigger _logging's stdout
        # auto-install mid-shutdown (console spew + a leaked handler).
        self._log_dock.detach()
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
# Top-level launch helper
# ---------------------------------------------------------------------------


def launch(deformation_i=None, *, solver_kwargs=None, initial_constraint=None) -> int:
    """Open the live-viz window.

    Parameters
    ----------
    deformation_i : ndarray, optional
        Any of ``(2, H, W)``, ``(3, H, W)``, ``(3, 1, H, W)``, or
        ``(3, D, H, W)``. When ``None`` (default), the window starts
        empty — use **Load DVF…** to pick a file.
    solver_kwargs : dict, optional
        Seeds the windowed-SLSQP parameters that the toolbar / worker
        honour: ``max_iterations`` and ``max_per_index_iter`` (the
        ``max_iterations`` value pre-fills the ``max_iter`` spinbox) and
        the scipy ``method_name``. The choice of *which* solver to run
        still lives in the toolbar; these only seed its parameters.
    initial_constraint : str, optional
        Pre-select a constraint tag in the toolbar after the DVF loads —
        e.g. ``'tet3d'`` to open straight into true-3D mode for a
        ``(3, D, H, W)`` volume (no-op if the tag is disabled for the
        loaded field, e.g. a 3D tag on a 2D section).

    Returns Qt exit code."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = LiveSolverWindow(deformation_i, initial_params=solver_kwargs or {})
    # Applied after construction so the DVF is already loaded and the 3D
    # constraint entries have been enabled (they gate on D > 1).
    if initial_constraint:
        win._select_combo_data(win._constraint_combo, initial_constraint)
    win.show()
    win.start()
    return app.exec()
