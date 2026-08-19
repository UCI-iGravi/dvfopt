"""Pure module-level helpers + constants shared by the LiveSolverWindow
mixins and :mod:`dvfopt_gui.app`.

Extracted verbatim from ``app.py`` so the window mixins
(:mod:`dvfopt_gui._win_fileio`, :mod:`dvfopt_gui._win_render`,
:mod:`dvfopt_gui._win_run`) can import them without a circular import.
``app.py`` re-exports every name for back-compat.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

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


def _min_gap_2d(phi_2hw: np.ndarray) -> np.ndarray:
    """Per-pixel min axial monotonicity gap (the 2D injectivity-gap view).

    2D analogue of :func:`dvfopt.jacobian.injectivity_quality_3d`
    (axial-only): each h/v deformed-coordinate gap is spread to both
    endpoint pixels and the element-wise minimum taken. Unit gaps (1.0)
    everywhere on the identity field; negative where deformed columns/
    rows cross.
    """
    from dvfopt.jacobian.monotonicity import _monotonicity_diffs_2d

    dy, dx = phi_2hw[0], phi_2hw[1]
    h, v = _monotonicity_diffs_2d(dy, dx)
    q = np.full(dy.shape, np.inf)
    q[:, :-1] = np.minimum(q[:, :-1], h)
    q[:, 1:] = np.minimum(q[:, 1:], h)
    q[:-1] = np.minimum(q[:-1], v)
    q[1:] = np.minimum(q[1:], v)
    return q


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
VIEW_INJ = 'inj'


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
    ('slp', 'SLP-3D (cluster trust-region SLP + HiGHS L1; m10 seed)'),
    ('m14', 'M14Tet (harmonic + ALM + L2 refine + repair + polish)'),
    ('m14_schwarz', 'M14-Schwarz3D (cluster decomposition + global polish)'),
    ('m10', 'M10Tet (harmonic + ALM + barrier polish)'),
    ('slsqp_fullgrid', 'SLSQP full-grid 3D (KKT)'),
    ('active_band', 'ActiveBandALM3D (banded M10Tet recovery; research)'),
    ('coupled_kring', 'CoupledKRing3D (k-ring SLSQP attractor escape; research)'),
    ('pipeline3d', 'Full 3D pipeline (bulk auto + k-ring escape)'),
    ('barrier_torch', 'Barrier GPU (torch; CPU fallback)'),
    ('auto', 'Auto (pick by fold stats)'),
]
_METHOD_SPECS_JDET3D = [
    ('barrier', 'Barrier (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed', 'SLSQP windowed 3D'),
    ('auto', 'Auto (pick by fold stats)'),
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


def _torch_available() -> bool:
    """True when PyTorch is importable (gates the GPU-barrier menu item)."""

    return importlib.util.find_spec('torch') is not None


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
    # argmax short-circuits to the first True in C-order — same index as
    # argwhere(bad)[0] without materialising every bad coordinate.
    first = tuple(int(i) for i in np.unravel_index(int(bad.argmax()), vol.shape))
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
