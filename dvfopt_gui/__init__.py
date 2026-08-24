"""``dvfopt_gui`` — live, interactive visualisation of dvfopt solvers.

Standalone PyQtGraph app that runs a dvfopt solver in a worker thread
and renders its per-step state (current Jdet / simplex (2D) heatmap or warped
deformation grid, active SLSQP window + padded ring, current pixel
being optimised) in real time. Hover or click any pixel to see its
current Jdet and triangle areas (T1/T2) in the inspector panel.

Multiple solver families are selectable from the toolbar (simplex (2D)
wallbreakers M10/M14/M14-Schwarz, penalty→barrier L-BFGS-B,
windowed SLSQP, NMVF). The windowed-SLSQP path fires a per-sub-window
``step_callback`` for live progress; the wallbreaker family fires it at
each pipeline-stage boundary. See the module docstring of
:mod:`dvfopt_gui.worker` for the callback contract.

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

which pulls in ``pyside6`` + ``pyqtgraph``.
"""

from __future__ import annotations

from dvfopt_gui.app import LiveSolverWindow, launch

__all__ = ['LiveSolverWindow', 'launch']
