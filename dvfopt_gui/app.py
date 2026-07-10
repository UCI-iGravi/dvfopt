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

from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt_gui.convergence import ConvergencePlot
from dvfopt_gui.history import HistoryController
from dvfopt_gui.persistence import (
    LoadedRun,
    build_save_payload,
    normalise_to_volume,
)
from dvfopt_gui.worker import (
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


def _quiver_lines(phi_2hw: np.ndarray, stride: int = 1, head_frac: float = 0.3):
    """Return ``(xs, ys)`` for a displacement-arrow field as one
    NaN-separated line series (shaft + a small two-segment arrowhead per
    sample), drawable by a single ``PlotDataItem``.

    Each arrow runs from the grid point ``(x, y)`` to the warped point
    ``(x + dx, y + dy)`` — the per-pixel displacement. ``stride``
    subsamples the grid; ``head_frac`` scales the arrowhead relative to
    each arrow's length.
    """
    dy, dx = phi_2hw[0], phi_2hw[1]
    H, W = dy.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    bx = xx[::stride, ::stride].ravel().astype(np.float64)
    by = yy[::stride, ::stride].ravel().astype(np.float64)
    vx = dx[::stride, ::stride].ravel()
    vy = dy[::stride, ::stride].ravel()
    tx = bx + vx
    ty = by + vy

    # Arrowhead: two short segments swept ±25° back from the tip along the
    # reversed arrow direction. Zero-length arrows get no head.
    ang = np.arctan2(vy, vx)
    length = np.hypot(vx, vy)
    hl = head_frac * length
    left = ang + np.deg2rad(180 - 25)
    right = ang + np.deg2rad(180 + 25)
    hlx, hrx = tx + hl * np.cos(left), tx + hl * np.cos(right)
    hly, hry = ty + hl * np.sin(left), ty + hl * np.sin(right)

    nan = np.nan
    xs_list = []
    ys_list = []
    for i in range(bx.size):
        if length[i] == 0:
            continue
        # shaft, then left head segment, then right head segment.
        xs_list.extend((bx[i], tx[i], nan, hlx[i], tx[i], hrx[i], nan))
        ys_list.extend((by[i], ty[i], nan, hly[i], ty[i], hry[i], nan))
    if not xs_list:
        return np.array([]), np.array([])
    return np.asarray(xs_list), np.asarray(ys_list)


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
VIEW_DIFF = 'diff'


# Constraint families. The worker dispatches on ``method_id`` which is
# always ``<algo>_<constraint>`` so the dispatch table stays flat.
CONSTRAINT_2TRI = '2tri'
CONSTRAINT_JDET = 'jdet'
CONSTRAINT_TET3D = 'tet3d'
CONSTRAINT_JDET3D = 'jdet3d'
_CONSTRAINT_SPECS = [
    (CONSTRAINT_2TRI, '2-tri (full-coverage; catches sub-pixel folds)'),
    (CONSTRAINT_JDET, 'Jdet (central-diff; blind to sub-pixel folds)'),
    (CONSTRAINT_TET3D, '6-tet (3D; whole-volume true 3D)'),
    (CONSTRAINT_JDET3D, 'Jdet (3D; whole-volume central-diff)'),
]
DEFAULT_CONSTRAINT = CONSTRAINT_2TRI

# Per-constraint method specs. Wallbreakers are 2-tri-only by design
# (HarmonicALMBarrier/RefineRepair internally depend on the 2-tri
# adjoint). Jdet gets the legacy windowed-SLSQP, the penalty→barrier
# path, and the NMVF heuristic smoother.
_METHOD_SPECS_2TRI = [
    ('slp', 'SLP (champion: cluster trust-region SLP + HiGHS L1)'),
    ('m14', 'M14 (Harmonic + ALM + L2 refine + repair + polish)'),
    ('m14_schwarz', 'M14-Schwarz (cluster decomposition + global polish)'),
    ('m10', 'M10 (Harmonic + ALM + barrier polish)'),
    ('barrier', 'Barrier (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed', 'SLSQP windowed (live progress)'),
    ('slsqp_fullgrid', 'SLSQP full-grid (2-tri; KKT, smallest L1 on mild folds)'),
    ('schwarz', 'Schwarz (2-tri; overlapping-tile decomposition)'),
    ('auto', 'Auto (pick by fold stats)'),
]
_METHOD_SPECS_JDET = [
    ('barrier', 'Barrier (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed', 'SLSQP windowed (live progress)'),
    ('nmvf', 'NMVF (heuristic neighborhood-mean smoother)'),
    ('auto', 'Auto (pick by fold stats)'),
]
_METHOD_SPECS_TET3D = [
    ('m14', 'M14Tet (harmonic + ALM + L2 refine + repair + polish)'),
    ('m14_schwarz', 'M14-Schwarz3D (cluster decomposition + global polish)'),
    ('m10', 'M10Tet (harmonic + ALM + barrier polish)'),
    ('slsqp_fullgrid', 'SLSQP full-grid 3D (KKT)'),
    ('active_band', 'ActiveBandALM3D (banded M10Tet recovery; research)'),
    ('coupled_kring', 'CoupledKRing3D (k-ring SLSQP attractor escape; research)'),
]
_METHOD_SPECS_JDET3D = [
    ('barrier', 'Barrier (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed', 'SLSQP windowed 3D'),
]
_METHOD_SPECS_BY_CONSTRAINT = {
    CONSTRAINT_2TRI: _METHOD_SPECS_2TRI,
    CONSTRAINT_JDET: _METHOD_SPECS_JDET,
    CONSTRAINT_TET3D: _METHOD_SPECS_TET3D,
    CONSTRAINT_JDET3D: _METHOD_SPECS_JDET3D,
}
DEFAULT_METHOD_BY_CONSTRAINT = {
    CONSTRAINT_2TRI: 'slp',
    CONSTRAINT_JDET: 'slsqp_windowed',
    CONSTRAINT_TET3D: 'm14',
    CONSTRAINT_JDET3D: 'barrier',
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


def _default_roi_geometry(H: int, W: int) -> tuple[int, int, int, int]:
    """Return ``(x, y, w, h)`` for the initial section ROI on an ``H×W``
    field — a centred rectangle ~¼ the field, but **clamped to the field
    and never negative**.

    Without the clamp, small fields produced an oversized ROI that
    overhung the image: e.g. the default 7×7 bowtie fixture gave
    ``max(8, 7//4) = 8`` → an 8×8 ROI at position ``(7-8)//2 = -1``, a
    dashed rectangle spilling past the grid (the very first thing the
    demo shows). Clamping keeps the handles on the image for any size.
    """
    # Target ~¼ the field (min 8, the historical default), then clamp to
    # [3 (the "Run section" floor), field size] so it always fits.
    roi_w = max(3, min(max(8, W // 4), W))
    roi_h = max(3, min(max(8, H // 4), H))
    x = max(0, (W - roi_w) // 2)
    y = max(0, (H - roi_h) // 2)
    return x, y, roi_w, roi_h


# Byte budget for the undo stack. Full-volume snapshots are cheap for 2D
# slices but ~1.8 GB each for a B0039-scale float64 volume — a count cap
# alone (30) would allow ~55 GB. Oldest entries are evicted past this
# budget; the most recent entry is always retained so Undo keeps working.
UNDO_MAX_BYTES = 2 * 1024**3


def validate_finite(vol: np.ndarray) -> str | None:
    """Return an error message if ``vol`` contains NaN/Inf, else None."""
    bad = ~np.isfinite(vol)
    n = int(bad.sum())
    if n == 0:
        return None
    first = tuple(int(i) for i in np.argwhere(bad)[0])
    return (
        f'The loaded field contains {n} non-finite value(s) (NaN/Inf); '
        f'first at index {first}. Fix the field before loading — solvers '
        'and fold metrics are undefined on non-finite data.'
    )


def _toolbar_separator() -> QtWidgets.QFrame:
    """A thin vertical divider for visually grouping toolbar widgets."""
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.VLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


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
            '<i>Applies to the next run. The current run keeps its original buffer size.</i>'
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
        self._picked_yx: tuple[int, int] | None = None
        # Active "Run section" crop bounds ``(y0, y1, x0, x1)`` or None for
        # a full-slice run. Set per-run; initialised here so any read
        # (e.g. ``_on_finished``) before the first run is well-defined.
        self._section_bounds: tuple[int, int, int, int] | None = None
        # True when the selected constraint is a whole-volume 3D family
        # (tet3d / jdet3d). In 3D mode: Run-section and Run-all z are
        # disabled; Run-full passes the entire (3, D, H, W) volume.
        self._is_3d_run = False
        # When non-None, a "Run all z" batch is in flight; holds the
        # z-slice indices still to be solved (current one already popped
        # and running). Drives the sequential chain in ``_on_finished``.
        self._run_all_remaining: list[int] | None = None
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
        self._view_combo.addItem('2-tri (min T1, T2)', VIEW_2TRI)
        self._view_combo.addItem('Deformation grid', VIEW_GRID)
        self._view_combo.addItem('Δ Jdet vs input', VIEW_DIFF)
        self._view_combo.setToolTip(
            'Central image. "Δ Jdet vs input" shows the current minus the '
            'originally-loaded per-pixel Jdet (red = increased, blue = '
            'decreased) — pair it with Auto levels to read the change.'
        )
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

        bar.addWidget(_toolbar_separator())
        self._run_full_btn = QtWidgets.QPushButton('Run full')
        self._run_full_btn.setShortcut('F5')
        self._run_full_btn.setToolTip('Solve the full current slice (F5).')
        self._run_full_btn.clicked.connect(lambda: self._on_run(use_roi=False))
        bar.addWidget(self._run_full_btn)
        self._run_roi_btn = QtWidgets.QPushButton('Run section')
        self._run_roi_btn.setShortcut('Ctrl+R')
        self._run_roi_btn.setToolTip('Solve only inside the ROI rectangle (Ctrl+R).')
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
        # The metric FIELD is threshold-independent (thr only affects the
        # reductions computed over it), so no cache invalidation is needed
        # here — just repaint the idle stats panel with the new threshold.
        self._thr_spin.valueChanged.connect(
            lambda _v: self._refresh_display_from_volume() if self._worker is None else None
        )

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
        right.addWidget(self._conv_plot, stretch=2)
        # Number of history entries last plotted — lets us rebuild the
        # curve only when it grows (the cursor still moves every frame).
        self._conv_len = -1

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
        file_menu.addAction('Load DVF…\tCtrl+O', self._on_load)
        file_menu.addAction('Save…\tCtrl+S', self._on_save)
        file_menu.addAction('Export corrected DVF…', self._on_export)
        file_menu.addAction('Revert', self._on_revert)
        file_menu.addSeparator()
        # Quit owns its own shortcut (no toolbar button competes for it).
        quit_act = file_menu.addAction('Quit', self.close)
        quit_act.setShortcut('Ctrl+Q')

        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction('Undo\tCtrl+Z', self._on_undo)
        edit_menu.addAction('Redo\tCtrl+Y', self._on_redo)
        edit_menu.addSeparator()
        edit_menu.addAction('Params…', self._on_open_params)

        run_menu = menubar.addMenu('&Run')
        run_menu.addAction('Run full\tF5', lambda: self._on_run(use_roi=False))
        run_menu.addAction('Run section\tCtrl+R', lambda: self._on_run(use_roi=True))
        run_menu.addAction('Run all z', self._on_run_all)
        run_menu.addAction('Stop\tEsc', self._on_stop)

        help_menu = menubar.addMenu('&Help')
        help_menu.addAction('Keyboard shortcuts…', self._show_shortcuts)
        help_menu.addAction('About', self._show_about)

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

    # ----- DVF loading -------------------------------------------------------

    def _on_load(self):
        flt = 'DVF files (*.npy *.npz'
        from dvfopt_gui.io_formats import SITK_EXTENSIONS, sitk_available

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
        self.statusBar().showMessage(f'Loading {Path(path).name}…', 0)
        self._load_worker = LoadWorker(path, parent=self)
        self._load_worker.loadedRun.connect(lambda run: self._on_load_finished(path, run))
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.start()

    def _on_load_finished(self, path: str, run) -> None:
        self._load_btn.setEnabled(True)
        if not self._apply_loaded_run(run):
            self.statusBar().clearMessage()
            return
        n_hist = len(run.snapshots)
        suffix = f'  ({n_hist} history step(s))' if n_hist else ''
        self.statusBar().showMessage(f'Loaded {path}{suffix}', 5_000)

    def _on_load_failed(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
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
        from dvfopt_gui.io_formats import save_dvf_sitk, sitk_available

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

    def _push_undo_state(self) -> None:
        """Snapshot the current volume onto the undo stack (capped) and
        invalidate the redo stack. Called just before a run splices its
        result in."""
        if self._volume is None:
            return
        self._undo_stack.append(self._volume.copy())
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack.pop(0)
        while (
            len(self._undo_stack) > 1 and sum(v.nbytes for v in self._undo_stack) > UNDO_MAX_BYTES
        ):
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_redo_enabled()

    def _on_undo(self):
        """Restore the volume to before the last applied correction."""
        if not self._undo_stack or (self._worker is not None and self._worker.isRunning()):
            return
        self._redo_stack.append(self._volume.copy())
        self._volume = self._undo_stack.pop()
        self._after_undo_redo('Undid last correction.')

    def _on_redo(self):
        """Re-apply the most recently undone correction."""
        if not self._redo_stack or (self._worker is not None and self._worker.isRunning()):
            return
        self._undo_stack.append(self._volume.copy())
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
        return True

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

    # ----- rendering ---------------------------------------------------------

    def _refresh_display_from_volume(self):
        """When the volume / z-slice / view-mode changes (but no live
        solver state is available), recompute the image from the
        volume itself and clear overlays."""
        if self._volume is None:
            return
        self._invalidate_metric_caches()
        if self._is_3d_run:
            self._set_view_3d(self._volume, fast=False)
            self._window_rect.setRect(0, 0, 0, 0)
            self._opt_rect.setVisible(False)
            self._target_marker.setData(x=[], y=[])
            self._stats_label.setText(self._format_stats(None))
            self._inspector_label.setText(self._format_inspector(None))
            self._refresh_convergence()
            return
        phi_2hw = self._volume[1:, self._z]  # (2, H, W)
        jac = jacobian_det2D(phi_2hw)[0]
        self._set_view(phi_2hw, jac)
        self._window_rect.setRect(0, 0, 0, 0)
        self._opt_rect.setVisible(False)
        self._target_marker.setData(x=[], y=[])
        self._stats_label.setText(self._format_stats(None))
        self._inspector_label.setText(self._format_inspector(None))
        self._refresh_convergence()

    def _set_view(self, phi_2hw: np.ndarray, jacobian: np.ndarray, *, fast: bool = False) -> None:
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
            self._apply_levels(jacobian)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_2TRI:
            min_tri = _min_tri_from_phi(phi_2hw)
            self._img.setImage(min_tri, autoLevels=False)
            self._apply_levels(min_tri)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_DIFF:
            # Current minus originally-loaded per-pixel Jdet. Positive
            # (red) = Jdet rose toward feasible; negative (blue) = fell.
            diff = jacobian - self._input_jacobian()
            self._img.setImage(diff, autoLevels=False)
            self._apply_levels(diff)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_GRID:
            self._cbar.setVisible(False)
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

        # Arrow overlay sits on top of whichever view is active.
        self._update_quiver(phi_2hw)

    def _heatmap_slice_3d(self, phi3d: np.ndarray) -> np.ndarray:
        """The per-slice 3D fold field for the current z (default 6-tet
        min volume; Jdet3D when that constraint is selected). Padded to
        (H, W) with NaN at the trailing row/col so it lines up with the
        grid (the tet field is (D-1, H-1, W-1)).

        Reads through ``_metric3d_field`` — the expensive whole-volume
        kernel runs at most once per displayed field; z only changes
        which slice of the cached field gets returned."""
        z = min(self._z, phi3d.shape[1] - 1)
        if self._constraint_combo.currentData() == CONSTRAINT_JDET3D:
            field = self._metric3d_field(phi3d, 'jdet3d')  # (D, H, W)
            return field[z]
        mv = self._metric3d_field(phi3d, 'tet3d')  # (D-1, H-1, W-1)
        H, W = phi3d.shape[2:]
        out = np.full((H, W), np.nan)
        zz = min(z, mv.shape[0] - 1)
        out[: H - 1, : W - 1] = mv[zz]
        return out

    def _set_view_3d(self, phi3d: np.ndarray, *, fast: bool = False) -> None:
        """3D heatmap: the fold-metric slice at the current z. The grid /
        2-tri / Jdet views fall back to the (dy,dx) of the current slice."""
        z = min(self._z, phi3d.shape[1] - 1)
        slice_2hw = phi3d[1:, z]  # (2, H, W) [dy, dx]
        mode = self._view_mode
        if mode == VIEW_GRID:
            self._img.setVisible(False)
            self._cbar.setVisible(False)
            stride = max(1, min(slice_2hw.shape[1:]) // 40)
            xs, ys = _grid_lines(slice_2hw, stride=stride)
            self._grid_curve.setData(xs, ys)
            self._grid_curve.setVisible(True)
            if not fast:
                self._fold_overlay.setPath(_folded_cells_path(slice_2hw))
                self._fold_overlay.setVisible(True)
            else:
                self._fold_overlay.setVisible(False)
        else:
            field = self._heatmap_slice_3d(phi3d)
            self._img.setImage(field, autoLevels=False)
            self._apply_levels(field)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        self._update_quiver(slice_2hw)

    def _update_quiver(self, phi_2hw: np.ndarray) -> None:
        """Refresh the displacement-arrow overlay for ``phi_2hw`` (or hide
        it when the Arrows toggle is off)."""
        if not self._arrows_check.isChecked():
            self._quiver_curve.setVisible(False)
            return
        stride = max(1, min(phi_2hw.shape[1:]) // 30)
        xs, ys = _quiver_lines(phi_2hw, stride=stride)
        self._quiver_curve.setData(xs, ys)
        self._quiver_curve.setVisible(True)

    def _on_arrows_toggled(self, _on: bool):
        """Re-render the current frame so the overlay appears/clears now."""
        if self._latest is not None and self._latest_jacobian is not None:
            if self._latest.phi.ndim == 4:
                self._set_view_3d(self._latest.phi)
            else:
                self._set_view(self._latest.phi, self._latest_jacobian)
        else:
            self._refresh_display_from_volume()

    def _refresh_convergence(self) -> None:
        """Sync the convergence chart with the current worker history.

        Rebuilds the curve only when the history grows (cheap during long
        SLSQP runs) but always re-positions the step cursor to the slider.
        Clears the chart when there's no run to show.
        """
        worker = self._worker
        if worker is None or worker.history_len() == 0:
            self._conv_plot.clear_data()
            self._conv_len = -1
            return
        n = worker.history_len()
        total = worker.history_total
        offset = total - n  # absolute step at buffer index 0
        if n != self._conv_len:
            steps = np.arange(offset, offset + n)
            n_neg = np.fromiter(
                (worker.history_get(i).n_neg for i in range(n)), dtype=float, count=n
            )
            min_T = np.fromiter(
                (worker.history_get(i).min_T for i in range(n)), dtype=float, count=n
            )
            self._conv_plot.set_data(steps, n_neg, min_T)
            self._conv_len = n
        self._conv_plot.set_cursor(offset + self._history_slider.value())

    def _input_jacobian(self) -> np.ndarray:
        """Per-pixel Jdet of the originally-loaded field for the active
        slice — the baseline for the ``Δ Jdet vs input`` view. Falls back
        to the current volume if no pristine copy exists."""
        base = self._original_volume if self._original_volume is not None else self._volume
        return jacobian_det2D(base[1:, self._z])[0]

    def _apply_levels(self, arr: np.ndarray) -> None:
        """Set the heatmap (and its colorbar) levels for ``arr``.

        Fixed ±1 when Auto-levels is off; otherwise a symmetric
        autoscale to the array's extent so the diverging colormap stays
        centred on zero (white) and large-magnitude fields don't clip.
        Driving the colorbar's levels also updates the linked image.
        """
        if self._autolevel_check.isChecked():
            finite = arr[np.isfinite(arr)]
            m = float(np.max(np.abs(finite))) if finite.size else 1.0
            if m <= 0:
                m = 1.0
            levels = (-m, m)
        else:
            levels = (-1.0, 1.0)
        self._cbar.setLevels(levels)
        self._img.setLevels(levels)

    def _on_autolevel_toggled(self, _on: bool):
        """Re-render the current frame so the new level policy takes
        effect immediately."""
        if self._latest is not None and self._latest_jacobian is not None:
            if self._latest.phi.ndim == 4:
                self._set_view_3d(self._latest.phi)
            else:
                self._set_view(self._latest.phi, self._latest_jacobian)
        else:
            self._refresh_display_from_volume()

    def _on_view_changed(self, idx: int):
        self._view_mode = self._view_combo.itemData(idx)
        if self._latest is not None and self._latest_jacobian is not None:
            if self._latest.phi.ndim == 4:
                self._set_view_3d(self._latest.phi)
            else:
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
        self._run_roi_btn.setEnabled(not self._is_3d_run)
        self._run_all_btn.setEnabled((not self._is_3d_run) and D > 1)
        self._section_roi.setVisible((not self._is_3d_run) and self._volume is not None)

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

        if self._is_3d_run:
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
            self._start_worker(deformation_i)

    def _start_worker(self, deformation_i: np.ndarray):
        algo = self._method_combo.currentData()
        constraint = self._constraint_combo.currentData()
        objective_id = self._objective_combo.currentData()
        method_id = _compose_method_id(algo, constraint)

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
        self._run_all_remaining = list(range(D))
        self._run_all_step()

    def _run_all_step(self):
        """Start the next queued slice in a Run-all batch, or finish the
        batch if the queue is empty."""
        if not self._run_all_remaining:
            self._run_all_remaining = None
            self._finalize_run_ui()
            self.statusBar().showMessage('Run all z finished.', 10_000)
            return
        z = self._run_all_remaining.pop(0)
        D = self._volume.shape[1]
        self._z = z
        self._z_label.setText(f'{z} / {D - 1}')
        self._section_bounds = None
        remaining = len(self._run_all_remaining)
        self.statusBar().showMessage(f'Run all z: solving slice {z} ({D - remaining}/{D})…', 0)
        # Solve from the pristine input for this slice.
        self._start_worker(self._original_volume[:, z : z + 1].copy())

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
            # Snapshot the pre-splice volume for Undo — but only for a
            # standalone run. A Run-all batch pushes one undo entry up
            # front (see ``_on_run_all``) so the whole batch undoes as a
            # unit rather than one slice at a time.
            if self._run_all_remaining is None:
                self._push_undo_state()
            phi_out = np.asarray(phi_out)
            if phi_out.ndim == 4:  # full-volume 3D result [dz,dy,dx]
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
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText('Stop')

        # Run-all chaining: a clean finish advances to the next slice; a
        # stop/interrupt (info set) aborts the remaining batch.
        if self._run_all_remaining is not None:
            if info is not None:
                self._run_all_remaining = None
                self._finalize_run_ui()
                self.statusBar().showMessage(f'Run all z stopped: {info}.', 10_000)
            else:
                self._run_all_step()
            return

        self._finalize_run_ui()
        msg = 'Run finished.' if info is None else f'Run stopped: {info}.'
        self.statusBar().showMessage(msg, 10_000)

    def _on_error(self, err: str):
        if self.sender() is not self._worker:
            return
        self._run_all_remaining = None
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

    def _update_progress(self) -> None:
        """Repaint the progress bar for the active run. Wallbreakers show
        elapsed/budget, SLSQP shows outer-iter/max, the rest show a busy
        indicator with elapsed time."""
        mid = self._active_method_id
        worker = self._worker
        if mid is None or worker is None or not worker.isRunning():
            return
        elapsed = self._run_elapsed.elapsed() / 1000.0
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
        # 3D z-scrub (``_on_z_changed``) re-renders this exact SAME
        # snapshot object just to re-slice the display at a new z —
        # comparing identity against the previously-rendered snapshot
        # (captured *before* ``self._latest`` is overwritten below) lets
        # that path skip the whole-volume metric invalidation so
        # ``_heatmap_slice_3d`` hits ``_metric3d_field``'s cache instead
        # of re-running the 6-tet/Jdet3D kernel on every tick. A genuinely
        # new snapshot (different object) still invalidates as before.
        same_field = snap is self._latest
        self._latest = snap
        if snap.phi.ndim == 4:  # 3D volume snapshot
            if not same_field:
                self._invalidate_metric_caches()
            self._latest_jacobian = self._heatmap_slice_3d(snap.phi)
            self._set_view_3d(snap.phi, fast=fast)
            self._window_rect.setRect(0, 0, 0, 0)
            self._opt_rect.setVisible(False)
            self._target_marker.setData(x=[], y=[])
            self._stats_label.setText(self._format_stats(snap))
            self._refresh_convergence()
            return
        self._latest_jacobian = jacobian_det2D(snap.phi)[0]
        self._invalidate_metric_caches()
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
        self._refresh_convergence()

    def _on_render_tick(self):
        if self._worker is None:
            return
        snap = self._worker.take_latest()
        # Update the history slider's range to cover everything emitted
        # so far, and (if Live is on) advance to the latest frame.
        self._history.on_tick()
        # Render the latest snapshot only when Live is on. In freeze mode
        # the slider handler controls what's shown.
        if snap is not None and self._history.is_live():
            # Big-field protection: skip the fold overlay rebuild during
            # live ticks once H·W exceeds the threshold. Scrubbing the
            # slider (post-run, or when paused) still gets the full
            # overlay — HistoryController renders with the default
            # ``fast=False``.
            H, W = snap.phi.shape[-2:]
            fast = self._fast_render_pixel_threshold < (H * W)
            self._render_snapshot(snap, fast=fast)
        else:
            # Frozen view: still extend the convergence curve as the run
            # progresses (cursor stays where the user parked it).
            self._refresh_convergence()

        # Live progress bar / ETA for the active run.
        self._update_progress()

        # One-time "Auto → <label>" note once the worker resolves it.
        if (
            not getattr(self, '_auto_label_shown', True)
            and self._worker is not None
            and getattr(self._worker, 'resolved_strategy_label', None)
        ):
            self._auto_label_shown = True
            self.statusBar().showMessage(f'Auto → {self._worker.resolved_strategy_label}', 8_000)

        # cb-rate once per second — only while a solve is actually
        # running. A loaded run is backed by a non-running ReplayHistory
        # (callback_count == 0); updating here would clobber "idle" with a
        # misleading "0 callbacks · 0.0 cb/s" for a static viewed run.
        if (
            self._last_tick.elapsed() >= 1000
            and self._worker is not None
            and self._worker.isRunning()
        ):
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

    def _on_mouse_moved(self, evt):
        """Hover readout — track the cursor so the inspector updates
        without a click. ``evt`` is the ``(scenePos,)`` tuple delivered
        by the rate-limiting :class:`pg.SignalProxy`. Updating
        ``_picked_yx`` (not a separate field) means the live-render path
        keeps showing whatever the cursor last hovered."""
        if self._volume is None:
            return
        scene_pos = evt[0]
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            return
        mouse_point = self._plot.plotItem.vb.mapSceneToView(scene_pos)
        x = round(mouse_point.x())
        y = round(mouse_point.y())
        if self._picked_yx == (y, x):
            return
        self._picked_yx = (y, x)
        self._inspector_label.setText(self._format_inspector((y, x)))

    def _triangle_areas_cached(self, phi: np.ndarray):
        """Return ``(T1, T2)`` for the currently-displayed ``phi``.

        The cache is invalidated explicitly whenever the displayed field
        changes (see ``_invalidate_metric_caches`` calls in the render
        / refresh paths), so repeated hovers over the same frame reuse
        one computation instead of an O(H·W) triangle-area recompute per
        mouse-move. (The volume-path ``phi`` is a fresh view object each
        call, so identity-keying wouldn't hit — hence explicit
        invalidation.)"""
        if self._inspector_tri is None:
            self._inspector_tri = _triangle_areas_2d(phi[0], phi[1])
        return self._inspector_tri

    def _metric3d_field(self, phi3d: np.ndarray, kind: str) -> np.ndarray:
        """Whole-volume 3D metric field, cached per kind until the displayed
        field changes (``_invalidate_metric_caches``). Counts are cheap numpy
        reductions over this array; only the kernel is expensive."""
        field = self._metric3d_cache.get(kind)
        if field is None:
            field = _metric_field_3d(phi3d, kind)
            self._metric3d_cache[kind] = field
        return field

    def _invalidate_metric_caches(self) -> None:
        """Drop cached per-field metrics (2D T1/T2 and 3D volume metric) —
        call whenever the displayed field changes."""
        self._inspector_tri = None
        self._metric3d_cache = {}

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
            # min Jdet is positive, even with sub-pixel 2-tri folds — the
            # bowtie default is exactly that case: 0 Jdet folds but 2
            # 2-tri folds).
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
                f'2-tri folds  . {n_neg_tri}  ({100 * n_neg_tri / interior:.1f}%)<br>'
                f'infeasible(&lt;{thr:g}) Jdet {infeas_jdet} · 2-tri {infeas_tri}<br>'
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
        if self._latest is not None and self._latest.phi.ndim == 4:
            phi3d = self._latest.phi
            z = min(self._z, phi3d.shape[1] - 1)
            mv = self._metric3d_field(phi3d, 'tet3d')
            Dm, Hm, Wm = mv.shape
            if not (0 <= y < Hm and 0 <= x < Wm):
                return '<b>Pixel inspector</b><br>(out of bounds)'
            zz = min(z, Dm - 1)
            return (
                '<b>Pixel inspector (3D)</b><br>'
                f'(z={zz}, y={y}, x={x})<br>'
                f'min 6-tet V . {mv[zz, y, x]:+.5f}'
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
        thr = s.value('threshold', 0.0, type=float)
        if thr:
            self._thr_spin.setValue(thr)
        hms = s.value('history_max_size', 0, type=int)
        if hms:
            self._history_max_size = hms

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
    return app.exec_()
