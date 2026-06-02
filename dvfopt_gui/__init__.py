"""``dvfopt_gui`` — live, interactive visualisation of dvfopt solvers.

Standalone PyQtGraph app that runs a dvfopt solver in a worker thread
and renders its per-step state (current Jdet heatmap, active SLSQP
window + padded ring, current pixel being optimised) in real time.
Click any pixel to see its current Jdet value in the inspector panel.

v1 inspector limitation: Jdet only. The triangle-area (T1/T2) readout
is planned for v2 — it needs a ``phi`` snapshot which the worker
deliberately doesn't carry to keep memory bounded.

Currently instruments :func:`dvfopt.core.slsqp.iterative.iterative_serial`
(the windowed Jdet SLSQP path). Other solvers can be wired in by
adding a ``step_callback`` kwarg following the same pattern — see the
module docstring of :mod:`dvfopt_gui.worker` for the callback contract.

Entry points
------------

* ``python -m dvfopt_gui.demo`` — load the B0039 z=12 slice (or fall
  back to a 20x20 canonical synthetic), open the live-viz window, and
  start the solver.
* :func:`dvfopt_gui.launch` — programmatic entry; pass a ``(3, 1, H, W)``
  deformation field (channels ``[dz, dy, dx]``; dz row ignored) and it
  builds the window and starts solving.

Optional dependencies
---------------------

Requires the ``[gui]`` extra::

    pip install -e '.[gui]'

which pulls in ``pyqt5`` + ``pyqtgraph``.
"""

from __future__ import annotations

from dvfopt_gui.app import LiveSolverWindow, launch

__all__ = ['LiveSolverWindow', 'launch']
