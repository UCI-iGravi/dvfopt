"""Per-slice fold overview strip for (3, D>1, H, W) volumes.

A thin clickable bar chart under the plot: x = slice index, y = per-slice
2-tri fold count. Instantly answers "which of my 528 slices are bad" and
doubles as navigation (click → jump z). Counts are computed off the GUI
thread by :class:`OverviewWorker`, streamed in chunks so the strip fills
progressively on big volumes.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from dvfopt_gui.worker import _metric_counts

_CHUNK = 32


class OverviewWorker(QtCore.QThread):
    """Compute per-slice 2-tri fold counts; emit ``(start, counts)`` chunks."""

    chunkReady = QtCore.Signal(int, object)

    def __init__(self, volume: np.ndarray, parent=None):
        super().__init__(parent)
        # Copy: the window's volume gets spliced by finishing runs while we
        # read it from this thread.
        self._volume = np.asarray(volume, dtype=np.float64).copy()
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self):
        D = self._volume.shape[1]
        for start in range(0, D, _CHUNK):
            if self._cancel:
                return
            end = min(D, start + _CHUNK)
            counts = np.empty(end - start, dtype=np.int64)
            for i, z in enumerate(range(start, end)):
                if self._cancel:
                    return
                counts[i] = _metric_counts(self._volume[1:, z], '2tri')[0]
            self.chunkReady.emit(start, counts)


class SliceOverviewStrip(pg.PlotWidget):
    """Fixed-height clickable bar chart of per-slice fold counts."""

    sliceClicked = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent, background='w')
        self.setFixedHeight(44)
        pi = self.getPlotItem()
        pi.hideAxis('left')
        pi.setMenuEnabled(False)
        pi.setMouseEnabled(x=False, y=False)
        pi.hideButtons()
        self._bars = pg.BarGraphItem(x=[], height=[], width=0.9, brush='#e67e22')
        pi.addItem(self._bars)
        self._marker = pg.InfiniteLine(angle=90, pen=pg.mkPen('#000', width=2))
        pi.addItem(self._marker)
        self._marker.hide()
        self._n = 0

    def set_counts(self, counts) -> None:
        if counts is None:
            self._bars.setOpts(x=[], height=[])
            self._n = 0
            return
        counts = np.asarray(counts)
        self._n = len(counts)
        self._bars.setOpts(x=np.arange(self._n), height=counts)
        self.getPlotItem().setXRange(-0.5, max(0.5, self._n - 0.5), padding=0)

    def set_current(self, z: int) -> None:
        self._marker.setValue(int(z))
        self._marker.show()

    def _emit_click_at(self, x: float) -> None:
        """Map a view x-coordinate to a slice index and emit (test hook)."""
        if self._n == 0:
            return
        z = round(x)
        if 0 <= z < self._n:
            self.sliceClicked.emit(z)

    def mousePressEvent(self, ev):
        vb = self.getPlotItem().vb
        if self._n and self.sceneBoundingRect().contains(ev.pos()):
            point = vb.mapSceneToView(ev.pos())
            self._emit_click_at(point.x())
        super().mousePressEvent(ev)
