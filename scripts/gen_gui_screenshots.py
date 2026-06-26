"""Generate dvfopt-GUI screenshots for the manuscript ``fig:gui``.

Renders :class:`dvfopt_gui.app.LiveSolverWindow` in several modes and
grabs each as a PNG. Runs headless under the ``offscreen`` Qt platform.

Usage::

    python scripts/gen_gui_screenshots.py [OUTDIR]   # default: docs/gui_screenshots

Caveat: the ``offscreen`` Qt platform does not render QPushButton/label
*text*, so toolbars appear blank in the grabs. The plot panels (heatmaps,
deformation grid + fold overlay, convergence chart, history scrubber)
render fully. For a publication figure, re-grab on a real display, or
crop to the plot panels.
"""

import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt5 import QtWidgets

from dvfopt_gui.app import VIEW_2TRI, VIEW_GRID, VIEW_JDET, LiveSolverWindow
from dvfopt_gui.demo import _bowtie_fixture, _synthetic_3d_volume

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pump(ms=400):
    """Process the Qt event loop for ``ms`` so the widget paints."""
    end = time.time() + ms / 1000.0
    while time.time() < end:
        _app.processEvents()


def _grab(win, outdir, name):
    _pump()
    path = os.path.join(outdir, name)
    win.grab().save(path)
    print('wrote', path, flush=True)


def _set_view(win, view):
    win._view_combo.setCurrentIndex(win._view_combo.findData(view))


def _dense_folded_2d(h=24, w=24, n=8, seed=0):
    """A ``(3, 1, h, w)`` field with ``n`` scattered dx-swap fold pairs."""
    rng = np.random.default_rng(seed)
    dy = np.zeros((h, w))
    dx = np.zeros((h, w))
    for _ in range(n):
        y, x = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
        dx[y, x] += 1.3
        dx[y, x + 1] -= 1.3
    vol = np.zeros((3, 1, h, w))
    vol[1, 0] = dy
    vol[2, 0] = dx
    return vol


def _run_to_completion(win, timeout_ms=60_000):
    worker = win._worker
    waited = 0
    while worker is not None and worker.isRunning() and waited < timeout_ms:
        _app.processEvents()
        worker.wait(50)
        waited += 50
    for _ in range(40):
        _app.processEvents()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    outdir = argv[0] if argv else os.path.join('docs', 'gui_screenshots')
    os.makedirs(outdir, exist_ok=True)

    # 1) 2D bowtie, deformation-grid view with the magenta fold overlay.
    win = LiveSolverWindow(_bowtie_fixture())
    win.resize(1500, 900)
    win.start()
    win.show()
    _set_view(win, VIEW_GRID)
    _grab(win, outdir, 'gui_2d_grid_folds.png')

    # 2) Same field, 2-tri heatmap + colorbar (auto-levelled).
    _set_view(win, VIEW_2TRI)
    win._autolevel_check.setChecked(True)
    _grab(win, outdir, 'gui_2d_2tri_heatmap.png')

    # 3) 2D M14 solve -> populated convergence chart + history scrubber.
    win2 = LiveSolverWindow(_dense_folded_2d())
    win2.resize(1500, 900)
    win2.start()
    win2.show()
    win2._select_combo_data(win2._constraint_combo, '2tri')
    win2._select_combo_data(win2._method_combo, 'm14')
    win2._budget_spin.setValue(20.0)
    _set_view(win2, VIEW_JDET)
    win2._autolevel_check.setChecked(True)
    win2._on_run(use_roi=False)
    _run_to_completion(win2)
    _grab(win2, outdir, 'gui_2d_full_window_solved.png')

    # 4) 3D mode: synthetic folded volume, 6-tet (3D) heatmap slice.
    win3 = LiveSolverWindow(_synthetic_3d_volume())
    win3.resize(1500, 900)
    win3.start()
    win3.show()
    win3._select_combo_data(win3._constraint_combo, 'tet3d')
    _set_view(win3, VIEW_JDET)  # 3D mode renders the 6-tet min-volume slice
    _grab(win3, outdir, 'gui_3d_sixtet_view.png')

    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
