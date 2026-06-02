"""Solver worker thread + state-snapshot pipeline.

The solver runs in a :class:`QThread` so the Qt event loop stays
responsive. Per-step state from the solver's ``step_callback`` hook
gets snapshotted into plain ``numpy.ndarray`` copies (the live
``jacobian_matrix`` / ``quality_matrix`` are mutated in-place by the
solver — reading them un-copied from the GUI thread would race), then
pushed onto a bounded ``queue.Queue``. The GUI's ``QTimer`` drains
the queue at ~30 Hz; older snapshots are dropped if the GUI can't keep
up. This bounds memory use and prevents render lag from slowing the
solver.

Callback contract
-----------------
``step_callback(state: dict)`` — fired by the solver after every inner
sub-iteration. ``state`` carries:

* ``phi`` — current ``(2, H, W)`` field (mutates! copy if you need it)
* ``phi_init`` — original input field, same shape (immutable)
* ``jacobian`` — current Jdet ``(1, H, W)`` (mutates! copy if you need it)
* ``quality`` — current 2-tri quality ``(1, H, W)`` (when enforce_*= True;
  for plain Jdet runs this is the same buffer as ``jacobian``)
* ``neg_index`` — ``(y, x)`` tuple of the current worst-Jdet target pixel
* ``window_center`` — ``(cy, cx)`` SLSQP active-window centre
* ``window_size`` — ``(sy, sx)`` SLSQP active-window dimensions
* ``opt_size`` — ``(sy', sx')`` padded optimisation window size
* ``is_padded`` — bool, True iff opt window is padded
* ``per_index_iter`` — sub-iteration index within the current pixel
* ``outer_iter`` — outer-loop iteration (one per worst-pixel target)
* ``n_neg`` — running fold count
* ``min_T`` — running minimum Jdet/triangle area
"""

from __future__ import annotations

import queue
import traceback

import numpy as np
from PyQt5 import QtCore


class StateSnapshot:
    """Plain-numpy snapshot of one solver step, decoupled from solver memory."""

    __slots__ = (
        'is_padded',
        'jacobian',
        'min_T',
        'n_neg',
        'neg_x',
        'neg_y',
        'opt_x0',
        'opt_x1',
        'opt_y0',
        'opt_y1',
        'outer_iter',
        'per_index_iter',
        'window_x0',
        'window_x1',
        'window_y0',
        'window_y1',
    )

    def __init__(
        self,
        *,
        jacobian,
        window_y0,
        window_y1,
        window_x0,
        window_x1,
        opt_y0,
        opt_y1,
        opt_x0,
        opt_x1,
        is_padded,
        neg_y,
        neg_x,
        per_index_iter,
        outer_iter,
        n_neg,
        min_T,
    ):
        self.jacobian = jacobian
        self.window_y0 = window_y0
        self.window_y1 = window_y1
        self.window_x0 = window_x0
        self.window_x1 = window_x1
        self.opt_y0 = opt_y0
        self.opt_y1 = opt_y1
        self.opt_x0 = opt_x0
        self.opt_x1 = opt_x1
        self.is_padded = is_padded
        self.neg_y = neg_y
        self.neg_x = neg_x
        self.per_index_iter = per_index_iter
        self.outer_iter = outer_iter
        self.n_neg = n_neg
        self.min_T = min_T


def _state_to_snapshot(state: dict) -> StateSnapshot:
    """Convert the solver's callback dict into a thread-safe snapshot."""
    jac = np.asarray(state['jacobian'][0]).copy()  # (H, W)
    cy, cx = state['window_center']
    sy, sx = state['window_size']
    osy, osx = state['opt_size']
    hy, hx = sy // 2, sx // 2
    ohy, ohx = osy // 2, osx // 2
    H, W = jac.shape
    return StateSnapshot(
        jacobian=jac,
        window_y0=max(0, cy - hy),
        window_y1=min(H, cy + (sy - hy)),
        window_x0=max(0, cx - hx),
        window_x1=min(W, cx + (sx - hx)),
        opt_y0=max(0, cy - ohy),
        opt_y1=min(H, cy + (osy - ohy)),
        opt_x0=max(0, cx - ohx),
        opt_x1=min(W, cx + (osx - ohx)),
        is_padded=bool(state['is_padded']),
        neg_y=int(state['neg_index'][0]),
        neg_x=int(state['neg_index'][1]),
        per_index_iter=int(state['per_index_iter']),
        outer_iter=int(state['outer_iter'] or 0),
        n_neg=int(state['n_neg']),
        min_T=float(state['min_T']),
    )


class SolverWorker(QtCore.QThread):
    """Run the solver in a worker thread.

    Per-step state is delivered to the GUI thread via a bounded
    ``queue.Queue(maxsize=1)`` only — we deliberately do **not** emit
    a per-callback Qt signal because queued cross-thread signals
    accumulate in the GUI event loop, defeating the bounded-queue
    design. Solver pace is the bottleneck; the GUI drains the queue
    on its own render timer at ~30 Hz.
    """

    finishedWithResult = QtCore.pyqtSignal(object, object)  # corrected_phi, info
    errored = QtCore.pyqtSignal(str)

    def __init__(self, *, deformation_i, solver_kwargs=None, parent=None):
        super().__init__(parent)
        self._deformation_i = deformation_i
        self._solver_kwargs = dict(solver_kwargs or {})
        self._stop_requested = False
        # Bounded queue so a slow GUI doesn't stall the solver. The
        # producer pre-drains the old item before pushing — we always
        # surface only the most recent snapshot.
        self._latest: queue.Queue = queue.Queue(maxsize=1)
        # Atomic counter for callback fires; the GUI reads this for its
        # FPS / callbacks-per-second display.
        self._callback_count = 0

    def request_stop(self):
        self._stop_requested = True

    def _callback(self, state: dict) -> None:
        if self._stop_requested:
            # ``step_callback`` fires AFTER ``scipy.optimize.minimize``
            # returns from each sub-window, not during the inner
            # optimiser. Raising here aborts the run between
            # sub-optimisations, so stop latency is bounded by one
            # window's solve time — not zero.
            raise KeyboardInterrupt('user requested stop')
        snap = _state_to_snapshot(state)
        self._callback_count += 1
        # Drop the previous snapshot if the GUI hasn't picked it up yet.
        try:
            self._latest.get_nowait()
        except queue.Empty:
            pass
        try:
            self._latest.put_nowait(snap)
        except queue.Full:
            pass

    def take_latest(self):
        """GUI thread calls this on its render timer. Returns the most
        recent snapshot (or ``None`` if no new state since the last poll)."""
        try:
            return self._latest.get_nowait()
        except queue.Empty:
            return None

    @property
    def callback_count(self) -> int:
        """Total ``step_callback`` fires since the worker started.

        Read-only; safe to call from the GUI thread without locking
        because we only read a single ``int`` whose update is atomic in
        CPython (single 64-bit store).
        """
        return self._callback_count

    def run(self):
        try:
            from dvfopt.core.slsqp.iterative import iterative_serial

            kwargs = dict(self._solver_kwargs)
            kwargs.setdefault('verbose', 0)
            phi_out = iterative_serial(
                self._deformation_i.copy(),
                step_callback=self._callback,
                **kwargs,
            )
            self.finishedWithResult.emit(phi_out, None)
        except KeyboardInterrupt:
            # Clean stop requested via request_stop().
            self.finishedWithResult.emit(None, 'stopped')
        except Exception as exc:
            self.errored.emit(f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
