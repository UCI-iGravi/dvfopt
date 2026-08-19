"""Solver worker thread + state-snapshot pipeline.

The solver runs in a :class:`QThread` so the Qt event loop stays
responsive. Per-step state from the solver's ``step_callback`` hook
gets snapshotted into plain ``numpy.ndarray`` copies (the live
``jacobian_matrix`` / ``quality_matrix`` are mutated in-place by the
solver — reading them un-copied from the GUI thread would race), then
pushed onto a bounded ``queue.Queue``. The GUI's ``QTimer`` drains
the queue at ~10 Hz; older snapshots are dropped if the GUI can't keep
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

from dvfopt._defaults import DEFAULT_PARAMS

# Solver feasibility threshold. Every dvfopt constraint is ``C(phi) >=
# threshold`` (see ``dvfopt/constraints.py``), so a cell is only
# *feasible to the solver* once its Jdet / triangle area clears this
# margin — not merely once it's non-negative. The GUI surfaces this so a
# field reading "0 folds" (no inversions) but ``min < threshold`` is not
# mistaken for "already solved". Sourced from the package default (0.01)
# so the GUI tracks the solver if that default ever changes.
FEASIBILITY_THRESHOLD = float(DEFAULT_PARAMS['threshold'])

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

# 3D runs emit few (phase-level) snapshots, but each is a full
# (3, D, H, W) volume. Cap the deque small and guard total bytes: past
# the budget we keep only the input + final snapshots.
DEFAULT_HISTORY_MAX_3D = 8
MAX_3D_HISTORY_BYTES = 2 * 1024**3  # ~2 GB


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
        'stage',
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
        stage=None,
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
        # Pipeline-stage name for phase-boundary snapshots (wallbreaker /
        # SLP stages); None for windowed per-step snapshots. The GUI's
        # convergence plot uses it to draw phase markers.
        self.stage = stage


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


def _stage_of(state: dict) -> str | None:
    """Stage name from a strategy step_callback payload ('' / missing -> None)."""
    stage = state.get('stage')
    return str(stage) if stage else None


def _volume_snapshot(
    phi3d, *, n_neg: int, min_T: float, outer_iter: int, stage: str | None = None
) -> StateSnapshot:
    """Build a 3D StateSnapshot: phi is the full (3, D, H, W) volume;
    window/opt rects collapse to zero (no active-window overlay in 3D)."""
    return StateSnapshot(
        phi=np.asarray(phi3d, dtype=np.float64).copy(),
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
        outer_iter=int(outer_iter),
        n_neg=int(n_neg),
        min_T=float(min_T),
        stage=stage,
    )


def _metric_field(phi_2hw, kind: str) -> np.ndarray:
    """Return the per-cell metric field for ``phi_2hw`` under one metric.

    ``kind='2tri'`` → per-cell ``min(T1, T2)`` (signed triangle area,
    catches sub-pixel folds the central-difference stencil misses);
    ``kind='jdet'`` → per-pixel central-difference Jacobian determinant.
    """
    if kind == '2tri':
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

        T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
        return np.minimum(T1, T2)
    from dvfopt.jacobian.numpy_jdet import jacobian_det2D

    return jacobian_det2D(phi_2hw)[0]


def _metric_counts(phi_2hw, kind: str) -> tuple[int, float]:
    """Return ``(n_neg, min_T)`` for ``phi_2hw`` under one metric.

    ``n_neg`` counts *folded* cells — those whose metric is ``<= 0`` (an
    inverted or degenerate-zero-area cell) — and ``min_T`` is the worst
    (smallest) signed value. The ``<= 0`` convention is shared by **both**
    metrics and matches the live windowed-SLSQP callback, which reports
    ``(jac <= 0).sum()`` (see ``dvfopt/core/_internal/metrics.py``), so a
    run's step-0 count lines up with its live tail instead of differing by
    the number of exactly-zero cells. For the solver's stricter
    feasibility margin (``< threshold``), see :func:`_infeasible_count`.

    A run uses ONE metric for *every* snapshot it emits — init,
    per-stage, and final — so its convergence trajectory is internally
    consistent (no Jdet-counted step 0 spliced onto a 2-tri-counted
    tail). See :meth:`SolverWorker._trajectory_metric_kind`.
    """
    field = _metric_field(phi_2hw, kind)
    return int((field <= 0).sum()), float(field.min())


def _infeasible_count(phi_2hw, kind: str, threshold: float = FEASIBILITY_THRESHOLD) -> int:
    """Count cells the *solver* still considers infeasible: metric
    ``< threshold`` (default 0.01), not merely ``<= 0``.

    A field can have zero folds yet a nonzero infeasible count when its
    minimum sits in ``(0, threshold)`` — positive, so not inverted, but
    inside the solver's safety margin. Surfacing this keeps the GUI's
    "is it done?" reading honest about what the solver is actually
    chasing.
    """
    return int((_metric_field(phi_2hw, kind) < threshold).sum())


def _metric_field_3d(phi3d, kind: str) -> np.ndarray:
    """Per-cell 3D metric field for ``phi3d`` ``(3, D, H, W)`` ``[dz,dy,dx]``.

    ``kind='tet3d'`` → per-cell min 6-tet signed volume
    ``(D-1, H-1, W-1)``; ``kind='jdet3d'`` → per-voxel 3D Jacobian
    determinant ``(D, H, W)``.
    """
    if kind == 'tet3d':
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        return six_tet_min_volume_3d(phi3d)
    if kind == 'jdet3d':
        from dvfopt.jacobian.numpy_jdet import jacobian_det3D

        return jacobian_det3D(phi3d)
    if kind == 'inj3d':
        # Per-voxel min axial monotonicity gap (the injectivity-gap view).
        from dvfopt.jacobian.monotonicity import injectivity_quality_3d

        return injectivity_quality_3d(phi3d)
    raise ValueError(f'unknown 3D metric kind={kind!r}')


def _metric_counts_3d(phi3d, kind: str) -> tuple[int, float]:
    """``(n_neg, min_T)`` over the whole volume under one 3D metric.
    Folds counted ``<= 0`` (matching the 2D convention)."""
    field = _metric_field_3d(phi3d, kind)
    return int((field <= 0).sum()), float(field.min())


def _infeasible_count_3d(phi3d, kind: str, threshold: float = FEASIBILITY_THRESHOLD) -> int:
    """Voxels/cells the solver still considers infeasible: metric ``< threshold``."""
    return int((_metric_field_3d(phi3d, kind) < threshold).sum())


class ReplayHistory:
    """Read-only stand-in for :class:`SolverWorker` that holds a finished
    run's snapshots loaded from disk.

    The GUI's history-scrub widgets and render tick only need a handful
    of read accessors (``history_len``/``history_get``/``history_total``/
    ``take_latest``/``isRunning``/``callback_count``). A loaded ``.npz``
    run has no live thread, so this exposes exactly that surface — letting
    a saved run be scrubbed through the same code path as a live one.
    """

    def __init__(self, snapshots, history_total: int | None = None):
        self._history = list(snapshots)
        self._history_total = (
            int(history_total) if history_total is not None else len(self._history)
        )

    def history_len(self) -> int:
        return len(self._history)

    def history_get(self, idx: int):
        if 0 <= idx < len(self._history):
            return self._history[idx]
        return None

    @property
    def history_total(self) -> int:
        return self._history_total

    def take_latest(self):
        return None

    @property
    def callback_count(self) -> int:
        return 0

    def isRunning(self) -> bool:  # mirror QThread.isRunning
        return False


class LoadWorker(QtCore.QThread):
    """Load a DVF file off the GUI thread.

    Dispatches by extension: ``.npy``/``.npz`` through
    :func:`dvfopt_gui.persistence.parse_loaded` (full saved-run support),
    SimpleITK formats through :func:`dvfopt_gui.io_formats.load_dvf_sitk`.
    Emits ``loadedRun`` with a ``LoadedRun`` on success, else ``failed``
    with a message. GB-scale ``np.load`` + float64 conversion no longer
    freeze the window.
    """

    loadedRun = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = str(path)

    def run(self):
        try:
            from dvfopt_gui.io_formats import is_sitk_path, load_dvf_sitk
            from dvfopt_gui.persistence import LoadedRun, parse_loaded

            if is_sitk_path(self._path):
                run = LoadedRun(volume=load_dvf_sitk(self._path))
            else:
                loaded = np.load(self._path, allow_pickle=False)
                try:
                    run = parse_loaded(loaded)
                finally:
                    if isinstance(loaded, np.lib.npyio.NpzFile):
                        loaded.close()
            self.loadedRun.emit(run)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class SolverWorker(QtCore.QThread):
    """Run the solver in a worker thread.

    Per-step state is delivered to the GUI thread via a bounded
    ``queue.Queue(maxsize=1)`` only — we deliberately do **not** emit
    a per-callback Qt signal because queued cross-thread signals
    accumulate in the GUI event loop, defeating the bounded-queue
    design. Solver pace is the bottleneck; the GUI drains the queue
    on its own render timer at ~10 Hz.
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
        # Solver verbosity for this run (log-dock level), resolved once.
        self._verbose = int(self._params.get('verbose', 0))
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
        # Set by the 'auto' dispatch: the registry label auto_strategy
        # resolved to (e.g. 'm14_schwarz'); None for explicit methods.
        self.resolved_strategy_label: str | None = None
        # Set by the pipeline runners (_run_marching_25d / _run_pipeline_3d):
        # the Correct25DReport / Correct3DReport for status display, and the
        # last 2.5D progress event for the progress bar.
        self.pipeline_report = None
        self.marching_progress: tuple | None = None
        # SolveInfo of the last Solver-path run (record_history=True) —
        # the window uses it for phase display + the convergence report.
        self.solve_info = None

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

    def _emit_initial_snapshot(self, metric_kind: str) -> None:
        """Snapshot the *input* phi as history[0] so the scrub slider can
        always go back to the unsolved field. Called once at the start of
        ``run()`` before any solver work.

        ``n_neg`` / ``min_T`` are computed under ``metric_kind`` (the same
        metric every later snapshot of this run uses), so step 0 lines up
        with the rest of the convergence trajectory rather than being a
        Jdet count grafted onto a 2-tri tail."""
        phi_2hw = np.stack(
            [
                self._deformation_i[1, 0].astype(np.float64),
                self._deformation_i[2, 0].astype(np.float64),
            ]
        )
        n_neg, min_T = _metric_counts(phi_2hw, metric_kind)
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
            stage='input',
        )
        self._record(snap)

    def _run_windowed_slsqp(self, enforce_triangles: bool):
        """Live-progress path: ``iterative_serial`` with our
        ``step_callback`` hook so the GUI sees every sub-window solve."""
        from dvfopt.core.slsqp.iterative import iterative_serial

        kwargs = {
            'verbose': self._verbose,
            'enforce_triangles': enforce_triangles,
        }
        # Constraint-mode toggles come from the Params -> Strategy overrides
        # (the 2D windowed path drives iterative_serial directly, so the
        # SLSQPWindowedStrategy dataclass isn't constructed here).
        overrides = dict(self._params.get('strategy_overrides') or {})
        for k in ('enforce_shoelace', 'enforce_injectivity', 'injectivity_threshold'):
            if k in overrides:
                kwargs[k] = overrides[k]
        if 'max_iterations' in self._params:
            kwargs['max_iterations'] = int(self._params['max_iterations'])
        if self._params.get('max_per_index_iter') is not None:
            kwargs['max_per_index_iter'] = int(self._params['max_per_index_iter'])
        if self._params.get('method_name'):
            kwargs['method_name'] = str(self._params['method_name'])
        if self._params.get('threshold') is not None:
            kwargs['threshold'] = float(self._params['threshold'])
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

    def _run_via_solver(self, strategy, constraint_kind: str, *, metric_kind: str):
        """One-shot path through ``dvfopt.Solver``. Strategies that
        support a ``step_callback`` kwarg (currently:
        :class:`HarmonicALMRefineRepairStrategy` / M14) will fire it at
        each pipeline-stage boundary — those snapshots flow into the
        worker's history deque and become scrub-able in the GUI. For
        other strategies the callback is ignored and only the final
        synthetic snapshot lands in history.

        ``constraint_kind`` is ``'2tri'`` or ``'jdet'`` — picks
        :class:`TriConstraint2DFullCoverage` vs :class:`JdetConstraint2D`.
        ``metric_kind`` is the metric used to count folds in every
        snapshot (init/stage/final) — normally equal to
        ``constraint_kind``. The objective comes from
        ``self._params['objective_id']``.
        """
        from dvfopt import (
            JdetConstraint2D,
            Solver,
            TriConstraint2DFullCoverage,
        )

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
        solver = Solver(
            constraint=constraint,
            objective=objective,
            strategy=strategy,
            threshold=self._params.get('threshold'),
        )

        # Per-stage callback adapter: convert {'phi', 'stage'} dicts from
        # the strategy into ``StateSnapshot`` records on the worker's
        # history deque. ``n_neg`` / ``min_T`` use ``metric_kind`` — the
        # same metric as the init + final snapshots — so the convergence
        # trajectory is internally consistent.
        outer = [0]

        def _stage_callback(state):
            if self._stop_requested:
                raise KeyboardInterrupt('user requested stop')
            phi = np.asarray(state['phi'])
            n_neg, min_T = _metric_counts(phi, metric_kind)
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
                stage=_stage_of(state),
            )
            self._record(snap)

        if self._stop_requested:
            raise KeyboardInterrupt()
        result = solver.fit(
            phi_2hw,
            step_callback=_stage_callback,
            record_history=True,
            verbose=self._verbose,
        )
        # getattr: test doubles stub Solver.fit with minimal results.
        self.solve_info = getattr(result, 'info', None)
        # ``_emit_synthetic_snapshot`` still fires once at the end so
        # there's an authoritative "final" entry — strategies that
        # didn't emit any per-stage snapshots fall back to just this.
        # Recount under ``metric_kind`` (rather than the SolveResult's own
        # final stats) so it matches the rest of this run's trajectory.
        final_n_neg, final_min_T = _metric_counts(np.asarray(result.corrected), metric_kind)
        self._emit_synthetic_snapshot(result.corrected, n_neg=final_n_neg, min_T=final_min_T)
        return result.corrected

    def _run_via_solver_3d(self, strategy, constraint_kind: str, *, metric_kind: str):
        """Whole-volume 3D path. ``self._deformation_i`` is the full
        ``(3, D, H, W)`` ``[dz, dy, dx]`` volume. The 3D constraints accept
        that layout directly (no reorder); ``result.corrected`` returns it.

        Phased wallbreakers (m10/m14/m14_schwarz) fire ``step_callback`` at
        phase boundaries — we record full-volume stages (and check the stop
        flag). Non-phased methods (slsqp_fullgrid, barrier, slsqp_windowed)
        run to completion and only the init + final snapshots land.
        """
        from dvfopt import JdetConstraint3D, Solver, Tet6Constraint3D

        vol = np.asarray(self._deformation_i, dtype=np.float64)
        if vol.ndim != 4 or vol.shape[0] != 3:
            raise ValueError(f'3D run needs (3, D, H, W); got {vol.shape}')
        _, D, H, W = vol.shape
        if constraint_kind == 'tet3d':
            constraint = Tet6Constraint3D(shape=(D, H, W))
        elif constraint_kind == 'jdet3d':
            constraint = JdetConstraint3D(shape=(D, H, W))
        else:
            raise ValueError(f'unknown 3D constraint_kind={constraint_kind!r}')
        objective = self._build_objective()
        solver = Solver(
            constraint=constraint,
            objective=objective,
            strategy=strategy,
            threshold=self._params.get('threshold'),
        )

        # Memory guard: keep mid stages only if the full deque fits the budget.
        est = DEFAULT_HISTORY_MAX_3D * 3 * D * H * W * 8
        keep_stages = est <= MAX_3D_HISTORY_BYTES

        # Initial snapshot (input volume), under the run metric. The auto
        # dispatch may have just computed these on the identical volume.
        cached = getattr(self, '_init_counts_3d', None)
        if cached is not None and cached[0] == metric_kind:
            n0, m0 = cached[1], cached[2]
        else:
            n0, m0 = _metric_counts_3d(vol, metric_kind)
        self._record(_volume_snapshot(vol, n_neg=n0, min_T=m0, outer_iter=0, stage='input'))

        outer = [0]

        def _stage_callback(state):
            if self._stop_requested:
                raise KeyboardInterrupt('user requested stop')
            phi = np.asarray(state['phi'])
            # Schwarz emits per-cluster crops — use them only for the stop
            # check above; snapshot only the full-volume phases.
            if phi.shape != vol.shape:
                return
            if not keep_stages:
                return
            n, m = _metric_counts_3d(phi, metric_kind)
            outer[0] += 1
            self._record(
                _volume_snapshot(
                    phi,
                    n_neg=n,
                    min_T=m,
                    outer_iter=outer[0],
                    stage=_stage_of(state),
                )
            )

        if self._stop_requested:
            raise KeyboardInterrupt()
        result = solver.fit(
            vol,
            step_callback=_stage_callback,
            record_history=True,
            verbose=self._verbose,
        )
        # getattr: test doubles stub Solver.fit with minimal results.
        self.solve_info = getattr(result, 'info', None)
        corrected = np.asarray(result.corrected, dtype=np.float64)
        nf, mf = _metric_counts_3d(corrected, metric_kind)
        self._record(
            _volume_snapshot(corrected, n_neg=nf, min_T=mf, outer_iter=outer[0] + 1, stage='final')
        )
        return corrected

    def _run_marching_25d(self):
        """Whole-volume 2.5D marching (fold PREVENTION): sweep + mop via
        ``correct_dvf_25d``. Input is the CURRENT (per-slice-corrected)
        volume the window handed us — the pipeline's precondition is
        dz == 0, which per-slice 2D correction guarantees."""
        from dvfopt import correct_dvf_25d

        vol = np.asarray(self._deformation_i, dtype=np.float64)
        if vol.ndim != 4 or vol.shape[0] != 3:
            raise ValueError(f'2.5D marching needs (3, D, H, W); got {vol.shape}')
        _, D, H, W = vol.shape
        thr = self._params.get('threshold')
        thr = float(thr) if thr is not None else 0.01

        n0, m0 = _metric_counts_3d(vol, 'tet3d')
        self._record(_volume_snapshot(vol, n_neg=n0, min_T=m0, outer_iter=0))

        est = DEFAULT_HISTORY_MAX_3D * 3 * D * H * W * 8
        keep_stages = est <= MAX_3D_HISTORY_BYTES
        stride = max(1, D // 6)
        outer = [0]

        def _cb(event):
            if self._stop_requested:
                raise KeyboardInterrupt('user requested stop')
            self.marching_progress = (
                event['phase'],
                event['index'],
                event['total'],
                event['n_neg'],
            )
            if not keep_stages:
                return
            if event['phase'] == 'sweep' and event['index'] % stride != 0:
                return
            outer[0] += 1
            n, m = _metric_counts_3d(event['phi'], 'tet3d')
            self._record(_volume_snapshot(event['phi'], n_neg=n, min_T=m, outer_iter=outer[0]))

        if self._stop_requested:
            raise KeyboardInterrupt()
        # callback_copies=False: zero-copy events are safe here — _cb never
        # retains event['phi']; _volume_snapshot() copies before recording.
        phi_out, report = correct_dvf_25d(
            vol,
            threshold=thr,
            verbose=self._verbose,
            progress_callback=_cb,
            callback_copies=False,
        )
        self.pipeline_report = report
        nf, mf = _metric_counts_3d(phi_out, 'tet3d')
        self._record(_volume_snapshot(phi_out, n_neg=nf, min_T=mf, outer_iter=outer[0] + 1))
        return np.asarray(phi_out, dtype=np.float64)

    def _run_pipeline_3d(self):
        """One-shot end-to-end 3D orchestrator (``correct_dvf_3d``): bulk
        recovery + k-ring escape. No progress hook exists — init + final
        snapshots only; Stop is best-effort (checked before launch)."""
        import dvfopt

        vol = np.asarray(self._deformation_i, dtype=np.float64)
        if vol.ndim != 4 or vol.shape[0] != 3:
            raise ValueError(f'3D pipeline needs (3, D, H, W); got {vol.shape}')
        thr = self._params.get('threshold')
        thr = float(thr) if thr is not None else 0.01

        n0, m0 = _metric_counts_3d(vol, 'tet3d')
        self._record(_volume_snapshot(vol, n_neg=n0, min_T=m0, outer_iter=0))
        if self._stop_requested:
            raise KeyboardInterrupt()
        phi_out, report = dvfopt.correct_dvf_3d(vol, threshold=thr, verbose=self._verbose)
        self.pipeline_report = report
        phi_out = np.asarray(phi_out, dtype=np.float64)
        nf, mf = _metric_counts_3d(phi_out, 'tet3d')
        self._record(_volume_snapshot(phi_out, n_neg=nf, min_T=mf, outer_iter=1))
        return phi_out

    def _resolve_auto(self, constraint, n_neg: int, min_T: float, fallback_label: str):
        """Shared auto-dispatch: route via ``auto_strategy``, fall back to
        the family default if the resolved registry label is unavailable,
        and record ``resolved_strategy_label`` for the status display."""
        from dvfopt import make_strategy
        from dvfopt.solver import auto_strategy

        label = auto_strategy(constraint, n_neg, min_T, str(self._params.get('objective_id', 'l1')))
        try:
            strategy = make_strategy(label)
        except (KeyError, ValueError):
            from dvfopt._logging import log_warning

            log_warning(
                f'auto-strategy resolved to unavailable label {label!r}; '
                f'falling back to {fallback_label!r}'
            )
            label = fallback_label
            strategy = make_strategy(label)
        self.resolved_strategy_label = label
        return strategy

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
        overrides = dict(self._params.get('strategy_overrides') or {})

        def _make(cls, **base):
            try:
                return cls(**{**base, **overrides})
            except TypeError as exc:
                raise ValueError(
                    f'invalid strategy parameter(s) for {cls.__name__}: {exc}'
                ) from exc

        if mid in ('barrier_2tri', 'barrier_jdet'):
            return _make(BarrierStrategy)
        if mid == 'm10_2tri':
            return _make(HarmonicALMBarrierStrategy, time_budget_s=time_budget)
        if mid == 'm14_2tri':
            return _make(HarmonicALMRefineRepairStrategy, time_budget_s=time_budget)
        if mid == 'm14_schwarz_2tri':
            return _make(SchwarzHarmonicALMRefineRepairStrategy, time_budget_s=time_budget)
        if mid == 'slsqp_fullgrid_2tri':
            from dvfopt import SLSQPFullGridStrategy

            return _make(SLSQPFullGridStrategy)
        if mid == 'schwarz_2tri':
            from dvfopt import SchwarzStrategy

            return _make(SchwarzStrategy)
        if mid == 'nmvf_jdet':
            return _make(NMVFStrategy)
        if mid == 'm10_tet3d':
            from dvfopt import HarmonicALMBarrier3DStrategy

            return _make(HarmonicALMBarrier3DStrategy, time_budget_s=time_budget)
        if mid == 'm14_tet3d':
            from dvfopt import HarmonicALMRefineRepair3DStrategy

            return _make(HarmonicALMRefineRepair3DStrategy, time_budget_s=time_budget)
        if mid == 'm14_schwarz_tet3d':
            from dvfopt import SchwarzHarmonicALMRefineRepair3DStrategy

            return _make(SchwarzHarmonicALMRefineRepair3DStrategy, time_budget_s=time_budget)
        if mid == 'slsqp_fullgrid_tet3d':
            from dvfopt import SLSQPFullGrid3DStrategy

            return _make(SLSQPFullGrid3DStrategy)
        if mid == 'active_band_tet3d':
            from dvfopt import ActiveBandALM3DStrategy

            return _make(ActiveBandALM3DStrategy)
        if mid == 'coupled_kring_tet3d':
            from dvfopt import CoupledKRing3DStrategy

            return _make(CoupledKRing3DStrategy)
        if mid in ('barrier_jdet3d',):
            return _make(BarrierStrategy)
        if mid == 'slsqp_windowed_jdet3d':
            from dvfopt import SLSQPWindowedStrategy

            return _make(SLSQPWindowedStrategy)
        if mid in ('auto_2tri', 'auto_jdet'):
            from dvfopt import JdetConstraint2D, TriConstraint2DFullCoverage

            phi_2hw = np.stack(
                [
                    self._deformation_i[1, 0].astype(np.float64),
                    self._deformation_i[2, 0].astype(np.float64),
                ]
            )
            H, W = phi_2hw.shape[1:]
            kind = '2tri' if mid.endswith('_2tri') else 'jdet'
            n_neg, min_T = _metric_counts(phi_2hw, kind)
            constraint = (
                TriConstraint2DFullCoverage(shape=(H, W))
                if kind == '2tri'
                else JdetConstraint2D(shape=(H, W))
            )
            return self._resolve_auto(
                constraint, n_neg, min_T, 'm14' if kind == '2tri' else 'barrier'
            )
        if mid == 'slp_2tri':
            from dvfopt import SLPStrategy

            return _make(SLPStrategy)
        if mid == 'slp_tet3d':
            from dvfopt import SLPStrategy

            return _make(SLPStrategy)
        if mid in ('auto_tet3d', 'auto_jdet3d'):
            from dvfopt import JdetConstraint3D, Tet6Constraint3D

            vol = np.asarray(self._deformation_i, dtype=np.float64)
            _, D, H, W = vol.shape
            kind = 'tet3d' if mid.endswith('_tet3d') else 'jdet3d'
            n_neg, min_T = _metric_counts_3d(vol, kind)
            # The whole-volume metric kernel is the priciest step in the
            # GUI — stash the counts so _run_via_solver_3d's initial
            # snapshot (same kind) doesn't recompute them.
            self._init_counts_3d = (kind, n_neg, min_T)
            constraint = (
                Tet6Constraint3D(shape=(D, H, W))
                if kind == 'tet3d'
                else JdetConstraint3D(shape=(D, H, W))
            )
            return self._resolve_auto(constraint, n_neg, min_T, 'barrier')
        if mid == 'barrier_torch_tet3d':
            from dvfopt import BarrierTet3DTorchStrategy

            return _make(BarrierTet3DTorchStrategy)
        raise ValueError(f'unknown method_id={mid!r}')

    def _trajectory_metric_kind(self) -> str:
        """Pick the single metric used for this run's whole convergence
        trajectory (init + every stage + final).

        The windowed-SLSQP path reports Jdet stats straight from the
        solver's own bookkeeping — regardless of its 2-tri *constraint*
        flag — so its trajectory is Jdet-based. 3D methods (``_tet3d``
        / ``_jdet3d`` suffix) go through ``_run_via_solver_3d`` and use
        the 3D metric. Every other path goes through ``_run_via_solver``
        and counts folds with the constraint's own metric (``_2tri`` →
        2-tri, ``_jdet`` → Jdet).
        """
        mid = self._method_id
        if mid.endswith('_tet3d'):
            return 'tet3d'
        if mid.endswith('_jdet3d'):
            return 'jdet3d'
        if mid.startswith('slsqp_windowed'):
            return 'jdet'
        return '2tri' if mid.endswith('_2tri') else 'jdet'

    # -- main entrypoint ------------------------------------------------------

    def run(self):
        try:
            # Always seed history[0] with the input field. This makes the
            # scrub slider's leftmost position meaningful even for the
            # one-shot wallbreaker methods (which would otherwise only
            # emit a single final-state snapshot). It uses the same metric
            # as the rest of the run so the convergence curve is coherent.
            metric_kind = self._trajectory_metric_kind()
            if metric_kind not in ('tet3d', 'jdet3d'):
                self._emit_initial_snapshot(metric_kind)
            mid = self._method_id
            if mid == 'slsqp_windowed_jdet':
                phi_out = self._run_windowed_slsqp(enforce_triangles=False)
            elif mid == 'slsqp_windowed_2tri':
                phi_out = self._run_windowed_slsqp(enforce_triangles=True)
            elif mid == 'marching25d_tet3d':
                phi_out = self._run_marching_25d()
            elif mid == 'pipeline3d_tet3d':
                phi_out = self._run_pipeline_3d()
            else:
                # Method-id always ends in either ``_2tri``, ``_jdet``,
                # ``_tet3d``, or ``_jdet3d``; split on the LAST underscore
                # so algo names that contain underscores (e.g.
                # ``m14_schwarz``) round-trip correctly.
                algo, _, kind = mid.rpartition('_')
                if kind in ('tet3d', 'jdet3d'):
                    phi_out = self._run_via_solver_3d(
                        self._build_strategy(), kind, metric_kind=metric_kind
                    )
                elif kind in ('2tri', 'jdet'):
                    phi_out = self._run_via_solver(
                        self._build_strategy(), kind, metric_kind=metric_kind
                    )
                else:
                    raise ValueError(f'unknown method_id={mid!r}')
            self.finishedWithResult.emit(phi_out, None)
        except KeyboardInterrupt:
            # Clean stop requested via request_stop().
            self.finishedWithResult.emit(None, 'stopped')
        except Exception as exc:
            self.errored.emit(f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
