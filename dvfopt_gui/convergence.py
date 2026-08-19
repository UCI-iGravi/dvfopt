"""Live convergence plot for the solver run.

A small dual-axis chart of the two trajectories the worker already
records per step — the fold count ``n_neg`` (left axis, red) and the
worst signed area / Jdet ``min_T`` (right axis, blue, with a dashed zero
line). A vertical cursor marks the step the history slider is parked on,
so scrubbing the slider and reading the curve stay in sync.

Kept as its own widget (like :class:`dvfopt_gui.history.HistoryController`)
so the plumbing stays out of :class:`dvfopt_gui.app.LiveSolverWindow`.
The window feeds it plain arrays via :meth:`set_data` / :meth:`set_cursor`
and clears it with :meth:`clear_data`; it owns no solver state.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

# Deliberately NOT red: the central Jdet heatmap uses red = positive =
# feasible (per the project's preferred reading), so a red fold-count
# curve here would mean the opposite of red over there. Orange keeps the
# two panels from contradicting each other at a glance.
_N_NEG_COLOR = '#e67e22'  # orange — fold count (n_neg)
_MIN_T_COLOR = '#2471a3'  # blue — worst signed area / Jdet


class ConvergencePlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent, background='w')
        pi = self.getPlotItem()
        pi.setLabel('bottom', 'step')
        pi.setLabel('left', 'folds (n_neg)', color=_N_NEG_COLOR)
        pi.showGrid(x=True, y=True, alpha=0.25)
        pi.setMenuEnabled(False)

        self._n_neg_curve = pg.PlotDataItem(pen=pg.mkPen(_N_NEG_COLOR, width=2))
        pi.addItem(self._n_neg_curve)

        # Second ViewBox sharing the x-axis for ``min_T`` on the right.
        self._vb2 = pg.ViewBox()
        pi.showAxis('right')
        pi.scene().addItem(self._vb2)
        pi.getAxis('right').linkToView(self._vb2)
        pi.getAxis('right').setLabel('min area / Jdet', color=_MIN_T_COLOR)
        self._vb2.setXLink(pi)

        self._min_T_curve = pg.PlotDataItem(pen=pg.mkPen(_MIN_T_COLOR, width=2))
        self._vb2.addItem(self._min_T_curve)
        # Dashed zero line on the right axis — the feasibility boundary.
        self._zero = pg.InfiniteLine(angle=0, pen=pg.mkPen('#999', style=QtCore.Qt.DashLine))
        self._vb2.addItem(self._zero)
        self._zero.setValue(0.0)
        # Dashed threshold line (the solver's actual target, thr > 0).
        self._thr_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(_MIN_T_COLOR, width=1, style=QtCore.Qt.DotLine)
        )
        self._vb2.addItem(self._thr_line)
        self._thr_line.hide()
        # Phase-boundary markers (wallbreaker / SLP stage names).
        self._stage_lines: list = []

        # Vertical cursor marking the current history step.
        self._cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen('#000', width=1, style=QtCore.Qt.DashLine)
        )
        pi.addItem(self._cursor)
        self._cursor.hide()

        pi.vb.sigResized.connect(self._sync_views)
        self._sync_views()

    def _sync_views(self):
        """Keep the right-axis ViewBox geometry locked to the main one."""
        pi = self.getPlotItem()
        self._vb2.setGeometry(pi.vb.sceneBoundingRect())
        self._vb2.linkedViewChanged(pi.vb, self._vb2.XAxis)

    def set_data(self, steps, n_neg, min_T) -> None:
        """Plot the ``n_neg`` and ``min_T`` trajectories against ``steps``
        (all 1-D arrays of equal length)."""
        steps = np.asarray(steps)
        self._n_neg_curve.setData(steps, np.asarray(n_neg, dtype=float))
        self._min_T_curve.setData(steps, np.asarray(min_T, dtype=float))
        self._sync_views()

    def set_threshold(self, thr) -> None:
        """Show the feasibility-threshold line on the ``min_T`` axis."""
        self._thr_line.setValue(float(thr))
        self._thr_line.show()

    def set_stage_markers(self, steps, labels) -> None:
        """Mark pipeline-stage boundaries with labeled vertical lines.

        ``steps``/``labels`` are parallel sequences: the history-step
        index where each named stage's snapshot landed. Existing markers
        are replaced (call with empty sequences to clear).
        """
        key = (tuple(steps), tuple(labels))
        if key == getattr(self, '_last_marks', None):
            return  # unchanged — skip the Qt item churn
        self._last_marks = key
        pi = self.getPlotItem()
        for item in self._stage_lines:
            pi.removeItem(item)
        self._stage_lines.clear()
        for step, label in zip(steps, labels):
            # pyqtgraph renders the label as a str.format template —
            # escape braces so stage names like 'bulk:{m14}' can't raise
            # inside InfLineLabel on the GUI thread.
            safe = str(label).replace('{', '{{').replace('}', '}}')
            line = pg.InfiniteLine(
                pos=float(step),
                angle=90,
                pen=pg.mkPen('#888', width=1, style=QtCore.Qt.DotLine),
                label=safe,
                labelOpts={
                    'position': 0.92,
                    'rotateAxis': (1, 0),
                    'color': '#555',
                    'anchors': [(0.0, 0.5), (0.0, 0.5)],
                },
            )
            pi.addItem(line)
            self._stage_lines.append(line)

    def set_cursor(self, step) -> None:
        """Move the vertical step cursor and show it."""
        self._cursor.setValue(float(step))
        self._cursor.show()

    def clear_data(self) -> None:
        """Remove all plotted data and hide the cursor, markers, and
        threshold line (a stale line would otherwise survive a load)."""
        self._n_neg_curve.setData([], [])
        self._min_T_curve.setData([], [])
        self._cursor.hide()
        self._last_marks = None  # force the next marker set to redraw
        self.set_stage_markers([], [])
        self._thr_line.hide()
