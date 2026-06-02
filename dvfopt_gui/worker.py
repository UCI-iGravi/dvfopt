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
    """Plain-numpy snapshot of one solver step, decoupled from solver memory.

    Carries the corrected ``phi`` (a copy — solver mutates the live
    buffer in place) so the GUI can render alternate views (2-tri
    min, deformation-grid wireframe) and the pixel inspector can
    compute T1/T2 alongside Jdet.
    """

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
        'phi',
        'window_x0',
        'window_x1',
        'window_y0',
        'window_y1',
    )

    def __init__(
        self,
        *,
        phi,
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
        self.phi = phi
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
    """Convert the solver's callback dict into a thread-safe snapshot.

    Both ``phi`` and ``jacobian`` are *copied* — the solver mutates
    those buffers in place between callback fires, so a stale view
    from the GUI thread would otherwise race.
    """
    # ``phi`` from the iterative_serial path is ``(2, H, W)`` channels
    # ``[dy, dx]``. Copy so the GUI thread can read it safely.
    phi_arr = np.asarray(state['phi']).copy()
    jac = np.asarray(state['jacobian'][0]).copy()  # (H, W)
    cy, cx = state['window_center']
    sy, sx = state['window_size']
    osy, osx = state['opt_size']
    hy, hx = sy // 2, sx // 2
    ohy, ohx = osy // 2, osx // 2
    H, W = jac.shape
    return StateSnapshot(
        phi=phi_arr,
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

    def __init__(
        self,
        *,
        deformation_i,
        method_id: str = 'slsqp_windowed_2tri',
        params=None,
        parent=None,
    ):
        super().__init__(parent)
        self._deformation_i = deformation_i
        self._method_id = method_id
        self._params = dict(params or {})
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

    # -- one-shot helpers -----------------------------------------------------

    def _emit_synthetic_snapshot(self, phi_2hw, n_neg: int, min_T: float) -> None:
        """Push one synthetic snapshot through the queue so the GUI can
        render the final state of a one-shot (no-step_callback) solver.

        The window / target overlays are collapsed to a zero-sized rect
        since there's no live window to show.
        """
        from dvfopt.jacobian.numpy_jdet import jacobian_det2D

        jac = jacobian_det2D(phi_2hw)[0]
        H, W = jac.shape
        snap = StateSnapshot(
            phi=phi_2hw.copy(),
            jacobian=jac.copy(),
            window_y0=0,
            window_y1=0,
            window_x0=0,
            window_x1=0,
            opt_y0=0,
            opt_y1=0,
            opt_x0=0,
            opt_x1=0,
            is_padded=False,
            neg_y=0,
            neg_x=0,
            per_index_iter=0,
            outer_iter=0,
            n_neg=int(n_neg),
            min_T=float(min_T),
        )
        self._callback_count += 1
        try:
            self._latest.get_nowait()
        except queue.Empty:
            pass
        try:
            self._latest.put_nowait(snap)
        except queue.Full:
            pass

    def _run_windowed_slsqp(self, enforce_triangles: bool):
        """Live-progress path: ``iterative_serial`` with our
        ``step_callback`` hook so the GUI sees every sub-window solve."""
        from dvfopt.core.slsqp.iterative import iterative_serial

        kwargs = {
            'verbose': 0,
            'enforce_triangles': enforce_triangles,
        }
        if 'max_iterations' in self._params:
            kwargs['max_iterations'] = int(self._params['max_iterations'])
        phi_out = iterative_serial(
            self._deformation_i.copy(),
            step_callback=self._callback,
            **kwargs,
        )
        return phi_out

    def _run_via_solver(self, strategy):
        """One-shot path through ``dvfopt.Solver``. Emits one synthetic
        snapshot when finished so the GUI's render loop sees the final
        state. No live progress for these methods (they don't expose
        ``step_callback`` hooks yet)."""
        from dvfopt import L1Objective, Solver, TriConstraint2DFullCoverage

        # iterative_serial gets the (3, 1, H, W) layout; Solver takes
        # the (2, H, W) one. Strip the singleton dz row.
        phi_2hw = np.stack(
            [
                self._deformation_i[1, 0].astype(np.float64),
                self._deformation_i[2, 0].astype(np.float64),
            ]
        )
        H, W = phi_2hw.shape[1:]
        constraint = TriConstraint2DFullCoverage(shape=(H, W))
        objective = L1Objective(eps=1e-4)
        solver = Solver(constraint=constraint, objective=objective, strategy=strategy)
        if self._stop_requested:
            raise KeyboardInterrupt()
        result = solver.fit(phi_2hw)
        self._emit_synthetic_snapshot(
            result.corrected, n_neg=result.final_n_neg, min_T=result.final_min_T
        )
        return result.corrected

    def _build_strategy(self):
        """Build a configured Strategy instance for the chosen method."""
        from dvfopt import (
            BarrierStrategy,
            HarmonicALMBarrierStrategy,
            HarmonicALMRefineRepairStrategy,
            SchwarzHarmonicALMRefineRepairStrategy,
        )

        time_budget = float(self._params.get('time_budget_s', 60.0))
        if self._method_id == 'barrier_2tri':
            return BarrierStrategy()
        if self._method_id == 'm10_2tri':
            return HarmonicALMBarrierStrategy(time_budget_s=time_budget)
        if self._method_id == 'm14_2tri':
            return HarmonicALMRefineRepairStrategy(time_budget_s=time_budget)
        if self._method_id == 'm14_schwarz_2tri':
            return SchwarzHarmonicALMRefineRepairStrategy(time_budget_s=time_budget)
        raise ValueError(f'unknown method_id={self._method_id!r}')

    # -- main entrypoint ------------------------------------------------------

    def run(self):
        try:
            if self._method_id == 'slsqp_windowed_jdet':
                phi_out = self._run_windowed_slsqp(enforce_triangles=False)
            elif self._method_id == 'slsqp_windowed_2tri':
                phi_out = self._run_windowed_slsqp(enforce_triangles=True)
            else:
                phi_out = self._run_via_solver(self._build_strategy())
            self.finishedWithResult.emit(phi_out, None)
        except KeyboardInterrupt:
            # Clean stop requested via request_stop().
            self.finishedWithResult.emit(None, 'stopped')
        except Exception as exc:
            self.errored.emit(f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
