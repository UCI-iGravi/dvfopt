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
from collections import deque

import numpy as np
from PyQt5 import QtCore

# Default cap on snapshots kept in the per-run history buffer. Each
# ``StateSnapshot`` carries only ``phi`` (2*H*W floats) — the jacobian
# is recomputed from phi at render time (it's a pure function of phi,
# costs ~50 µs at 128×128, and removing the cache saves 33% of
# per-snapshot memory). At H=W=256 that's ~1 MB/snapshot, so 500 caps
# memory at ~500 MB worst case. Realistic runs use far less: one-shot
# wallbreakers emit 2 snapshots; SLSQP-windowed emits ~one per fold
# (typically <200). Oldest snapshots are dropped if the cap is
# exceeded.
#
# Overridable per-worker via :class:`SolverWorker`'s ``history_max_size``
# kwarg (the GUI's Params dialog edits the window-level default).
DEFAULT_HISTORY_MAX = 500
# Back-compat alias for callers that referenced the old constant name.
HISTORY_MAX = DEFAULT_HISTORY_MAX


class StateSnapshot:
    """Plain-numpy snapshot of one solver step, decoupled from solver memory.

    Carries ``phi`` (a copy — solver mutates the live buffer in place)
    plus the per-step scalar bookkeeping. The Jdet heatmap is **not**
    cached here: it's a pure function of ``phi`` and gets recomputed
    in the GUI's render path. Dropping the cache saves 33% of
    per-snapshot memory at ~50 µs/scrub of recompute cost — invisible
    to the user.
    """

    __slots__ = (
        'is_padded',
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

    ``phi`` is *copied* — the solver mutates the live buffer in place
    between callback fires, so a stale view from the GUI thread would
    otherwise race. The jacobian is dropped (the GUI recomputes it from
    phi on demand — see ``StateSnapshot`` docstring).
    """
    # ``phi`` from the iterative_serial path is ``(2, H, W)`` channels
    # ``[dy, dx]``. Copy so the GUI thread can read it safely.
    phi_arr = np.asarray(state['phi']).copy()
    cy, cx = state['window_center']
    sy, sx = state['window_size']
    osy, osx = state['opt_size']
    hy, hx = sy // 2, sx // 2
    ohy, ohx = osy // 2, osx // 2
    H, W = phi_arr.shape[1:]
    return StateSnapshot(
        phi=phi_arr,
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
        history_max_size: int = DEFAULT_HISTORY_MAX,
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
        # Per-run history buffer. Every snapshot the solver emits gets
        # appended here as well (independent of the bounded ``_latest``
        # queue) so the GUI's scrub slider can replay the run later.
        # CPython's ``deque.append`` + indexing are atomic under the GIL,
        # so cross-thread access is safe without an explicit lock as
        # long as ``StateSnapshot`` instances are treated as immutable
        # after construction (which they are — ``phi``/``jacobian`` are
        # copies of the solver's live buffers).
        #
        # ``history_max_size`` caps the deque. Bumping it via the GUI's
        # Params dialog only takes effect for the NEXT worker — deques
        # can't be resized in place.
        self._history: deque = deque(maxlen=max(2, int(history_max_size)))
        # Track total snapshots ever emitted (independent of the deque's
        # capped length) so the GUI can show e.g. "step 3000 of 5421"
        # when older entries have aged out.
        self._history_total = 0
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
        self._record(snap)

    def _record(self, snap) -> None:
        """Single funnel for emitting a snapshot — pushes to both the
        bounded live queue (latest-only) and the full history deque."""
        self._callback_count += 1
        self._history.append(snap)
        self._history_total += 1
        # Drop the previous snapshot if the GUI hasn't picked it up yet.
        try:
            self._latest.get_nowait()
        except queue.Empty:
            pass
        try:
            self._latest.put_nowait(snap)
        except queue.Full:
            pass

    def history_len(self) -> int:
        """Number of snapshots currently retained (≤ ``HISTORY_MAX``)."""
        return len(self._history)

    def history_get(self, idx: int):
        """Return the snapshot at index ``idx`` (0-based), or ``None``
        if out of range. Thread-safe: relies on CPython's atomic
        ``deque`` indexing + immutable ``StateSnapshot`` instances."""
        n = len(self._history)
        if n == 0 or idx < 0 or idx >= n:
            return None
        return self._history[idx]

    @property
    def history_total(self) -> int:
        """Total snapshots ever emitted by this worker (may exceed
        ``history_len()`` if old entries have aged out of the deque)."""
        return self._history_total

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
        snap = StateSnapshot(
            phi=phi_2hw.copy(),
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
        self._record(snap)

    def _emit_initial_snapshot(self) -> None:
        """Snapshot the *input* phi as history[0] so the scrub slider
        can always go back to the unsolved field. Called once at the
        start of ``run()`` before any solver work. We still compute the
        jacobian once here to populate ``n_neg`` / ``min_T`` on the
        snapshot, but discard it after — the GUI will recompute it from
        ``phi`` at render time."""
        from dvfopt.jacobian.numpy_jdet import jacobian_det2D

        phi_2hw = np.stack(
            [
                self._deformation_i[1, 0].astype(np.float64),
                self._deformation_i[2, 0].astype(np.float64),
            ]
        )
        jac = jacobian_det2D(phi_2hw)[0]
        n_neg = int((jac < 0).sum())
        min_T = float(jac.min())
        snap = StateSnapshot(
            phi=phi_2hw,
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
            n_neg=n_neg,
            min_T=min_T,
        )
        self._record(snap)

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

    def _build_objective(self):
        """Build an Objective instance for ``self._params['objective_id']``.

        Defaults to L1 if the param is missing (so callers that didn't
        thread the choice through get the same behaviour as before).
        """
        from dvfopt import L1Objective, L2Objective, NoneObjective

        oid = self._params.get('objective_id', 'l1')
        if oid == 'l1':
            return L1Objective(eps=1e-4)
        if oid == 'l2':
            return L2Objective()
        if oid == 'none':
            return NoneObjective()
        raise ValueError(f'unknown objective_id={oid!r}')

    def _run_via_solver(self, strategy, constraint_kind: str):
        """One-shot path through ``dvfopt.Solver``. Strategies that
        support a ``step_callback`` kwarg (currently:
        :class:`HarmonicALMRefineRepairStrategy` / M14) will fire it at
        each pipeline-stage boundary — those snapshots flow into the
        worker's history deque and become scrub-able in the GUI. For
        other strategies the callback is ignored and only the final
        synthetic snapshot lands in history.

        ``constraint_kind`` is ``'2tri'`` or ``'jdet'`` — picks
        :class:`TriConstraint2DFullCoverage` vs :class:`JdetConstraint2D`.
        The objective comes from ``self._params['objective_id']``.
        """
        from dvfopt import (
            JdetConstraint2D,
            Solver,
            TriConstraint2DFullCoverage,
        )
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

        # iterative_serial gets the (3, 1, H, W) layout; Solver takes
        # the (2, H, W) one. Strip the singleton dz row.
        phi_2hw = np.stack(
            [
                self._deformation_i[1, 0].astype(np.float64),
                self._deformation_i[2, 0].astype(np.float64),
            ]
        )
        H, W = phi_2hw.shape[1:]
        if constraint_kind == '2tri':
            constraint = TriConstraint2DFullCoverage(shape=(H, W))
        elif constraint_kind == 'jdet':
            constraint = JdetConstraint2D(shape=(H, W))
        else:
            raise ValueError(f'unknown constraint_kind={constraint_kind!r}')
        objective = self._build_objective()
        solver = Solver(constraint=constraint, objective=objective, strategy=strategy)

        # Per-stage callback adapter: convert {'phi', 'stage'} dicts from
        # the strategy into ``StateSnapshot`` records on the worker's
        # history deque. ``n_neg`` / ``min_T`` are 2-tri stats here
        # (which is what the wallbreakers optimise for) — that keeps the
        # ``stats_label`` numbers meaningful at each stage.
        outer = [0]

        def _stage_callback(state):
            if self._stop_requested:
                raise KeyboardInterrupt('user requested stop')
            phi = np.asarray(state['phi'])
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            min_T = float(min(T1.min(), T2.min()))
            n_neg = int((np.minimum(T1, T2) <= 0).sum())
            outer[0] += 1
            snap = StateSnapshot(
                phi=phi.copy(),
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
                outer_iter=outer[0],
                n_neg=n_neg,
                min_T=min_T,
            )
            self._record(snap)

        if self._stop_requested:
            raise KeyboardInterrupt()
        result = solver.fit(phi_2hw, step_callback=_stage_callback)
        # ``_emit_synthetic_snapshot`` still fires once at the end so
        # there's an authoritative "final" entry — strategies that
        # didn't emit any per-stage snapshots fall back to just this.
        self._emit_synthetic_snapshot(
            result.corrected, n_neg=result.final_n_neg, min_T=result.final_min_T
        )
        return result.corrected

    def _build_strategy(self):
        """Build a configured Strategy instance for the chosen method.

        ``self._method_id`` is always ``<algo>_<constraint>``. The
        wallbreaker family (m10/m14/m14_schwarz) is 2-tri-only by design;
        Barrier and NMVF work with either constraint family.
        """
        from dvfopt import (
            BarrierStrategy,
            HarmonicALMBarrierStrategy,
            HarmonicALMRefineRepairStrategy,
            NMVFStrategy,
            SchwarzHarmonicALMRefineRepairStrategy,
        )

        time_budget = float(self._params.get('time_budget_s', 60.0))
        mid = self._method_id
        if mid in ('barrier_2tri', 'barrier_jdet'):
            return BarrierStrategy()
        if mid == 'm10_2tri':
            return HarmonicALMBarrierStrategy(time_budget_s=time_budget)
        if mid == 'm14_2tri':
            return HarmonicALMRefineRepairStrategy(time_budget_s=time_budget)
        if mid == 'm14_schwarz_2tri':
            return SchwarzHarmonicALMRefineRepairStrategy(time_budget_s=time_budget)
        if mid == 'nmvf_jdet':
            return NMVFStrategy()
        raise ValueError(f'unknown method_id={mid!r}')

    # -- main entrypoint ------------------------------------------------------

    def run(self):
        try:
            # Always seed history[0] with the input field. This makes the
            # scrub slider's leftmost position meaningful even for the
            # one-shot wallbreaker methods (which would otherwise only
            # emit a single final-state snapshot).
            self._emit_initial_snapshot()
            mid = self._method_id
            if mid == 'slsqp_windowed_jdet':
                phi_out = self._run_windowed_slsqp(enforce_triangles=False)
            elif mid == 'slsqp_windowed_2tri':
                phi_out = self._run_windowed_slsqp(enforce_triangles=True)
            else:
                # Method-id always ends in either ``_2tri`` or ``_jdet``;
                # split on the LAST underscore so algo names that contain
                # underscores (e.g. ``m14_schwarz``) round-trip correctly.
                algo, _, kind = mid.rpartition('_')
                if kind not in ('2tri', 'jdet'):
                    raise ValueError(f'unknown method_id={mid!r}')
                phi_out = self._run_via_solver(self._build_strategy(), kind)
            self.finishedWithResult.emit(phi_out, None)
        except KeyboardInterrupt:
            # Clean stop requested via request_stop().
            self.finishedWithResult.emit(None, 'stopped')
        except Exception as exc:
            self.errored.emit(f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
