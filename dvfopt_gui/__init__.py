"""``dvfopt_gui`` — live, interactive visualisation of dvfopt solvers.

Standalone PyQtGraph app that runs a dvfopt solver in a worker thread
and renders its per-step state (current Jdet / 2-tri heatmap, active
SLSQP window, current pixel being optimised) in real time. Click any
pixel to see its T1/T2/Jdet values in the inspector panel.

Currently instruments :func:`dvfopt.core.slsqp.iterative.iterative_serial`
(the windowed Jdet/2-tri SLSQP path). Other solvers can be wired in by
adding a ``step_callback`` kwarg following the same pattern — see the
docstring of :class:`SolverWorker` for the callback contract.

Entry points
------------

* ``python -m dvfopt_gui.demo`` — load the B0039 z=12 slice (or fall
  back to a 20x20 canonical synthetic), open the live-viz window, and
  start the solver.
* :func:`dvfopt_gui.launch` — programmatic entry; pass any
  ``(2, H, W)`` field and it'll build the window and start solving.

Optional dependencies
---------------------

Requires the ``[gui]`` extra::

    pip install -e '.[gui]'

which pulls in ``pyqt5`` + ``pyqtgraph``.
"""

from __future__ import annotations

from dvfopt_gui.app import LiveSolverWindow, launch

__all__ = ['LiveSolverWindow', 'launch']
