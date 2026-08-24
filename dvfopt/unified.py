"""DVFopt -- high-level facade over the parameterized Solver.

A per-slice orchestration layer on top of
:class:`dvfopt.solver.Solver`. Adds:

* automatic 2D/3D detection (`(3, D, H, W)`, `(3, H, W)`, `(2, H, W)`)
* a per-slice loop with tabular + dataframe + plot helpers
* string-config-style API for users who prefer ``DVFoptConfig`` over
  building Constraint/Objective/Strategy instances manually

The actual optimization runs through ``Solver``; DVFopt converts the
config strings to constraint/objective/strategy instances and forwards
each slice into ``solver.fit(phi)``.

Configuration axes (every combination valid; sensible defaults):

    constraint : 'simplex' (default, full-coverage), 'simplex_standard'
        (TR-BL only — for benchmark reproducibility), 'jdet'
        (== 'jdet_2d'); legacy '2tri'/'2tri_standard' labels are still
        accepted. The facade is per-slice 2D; for true-3D
        constraints ('simplex_3d', 'jdet_3d') use correct_dvf_3d /
        correct_dvf_25d or the Solver API.
    solver     : 'nmvf', 'barrier', 'slsqp', 'slsqp_windowed', 'schwarz',
                 'harmonic_alm_barrier' (alias 'm10'),
                 'harmonic_alm_refine_repair' (alias 'm14'),
                 'schwarz_harmonic_alm_refine_repair' (alias 'm14_schwarz'),
                 'auto'
    objective  : 'l1' | 'l2' | 'none'

Example::

    from dvfopt import DVFopt, DVFoptConfig
    opt = DVFopt(DVFoptConfig(constraint='simplex', solver='m14_schwarz',
                                objective='l1', threshold=0.01))
    result = opt.fit(deformation)             # (3, D, H, W), (3, H, W), or (2, H, W)
    print(result.summary())
    print(result.to_dataframe())
    result.plot_convergence(z=0)
    result.plot_feasibility(z=0)
    result.plot_gradient_region(z=0)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from dvfopt._logging import log_info
from dvfopt.constraints import _CONSTRAINT_REGISTRY, Constraint, make_constraint
from dvfopt.objectives import make_objective
from dvfopt.solver import SolveInfo, Solver, SolveResult, auto_strategy
from dvfopt.strategies import Strategy, make_strategy

# ============================================================
# Config
# ============================================================


@dataclass
class DVFoptConfig:
    """High-level config for :class:`DVFopt`.

    Slim by design: the constraint/objective/solver axes are strings
    (or instances), the rest are the knobs you actually tune per run.
    For strategy-specific kwargs (``lam_schedule``, ``mu_schedule``,
    ``pad``, etc.) pass a pre-built :class:`Strategy` instance instead
    of a string label::

        cfg = DVFoptConfig(
            constraint='simplex',
            solver=BarrierStrategy(lam_schedule=(1, 100, 1e4), max_iter=500),
            objective='l1',
        )

    The string ``solver=...`` form uses each strategy's dataclass
    defaults, which are the values validated by the canonical suite.
    """

    # ---- problem ----
    # Any registered 2D constraint label: 'simplex' (=full-coverage, default) |
    # 'simplex_standard' | 'bilinear' | 'finite' | 'jdet' | 'jdet_2d'
    # (legacy '2tri' / '2tri_standard' still accepted).
    # 3D constraints ('simplex_3d', 'jdet_3d') are rejected — the facade is
    # per-slice 2D; use correct_dvf_3d / correct_dvf_25d or Solver for 3D.
    constraint: str = 'simplex'
    threshold: float = 0.01
    err_tol: float = 1e-5

    # ---- strategy / objective ----
    # 'nmvf', 'barrier', 'slsqp', 'slsqp_windowed', 'schwarz',
    # 'harmonic_alm_barrier' (alias 'm10'),
    # 'harmonic_alm_refine_repair' (alias 'm14'),
    # 'schwarz_harmonic_alm_refine_repair' (alias 'm14_schwarz'),
    # or 'auto' (defer to auto_strategy).
    # A Strategy instance is also accepted — use that when you need
    # non-default knobs.
    solver: object = 'auto'
    # NOTE: defaults to 'l1', matching dvfopt.solver.correct_dvf — the two
    # APIs now share the same default objective. (Before v0.2.x this facade
    # historically defaulted to 'l2'.) With solver='auto', simplex (2D) + l1 routes
    # to the SLP champion strategy.
    objective: str = 'l1'  # 'l1', 'l2', 'none'
    eps_l1: float = 1e-4

    # SLP accuracy mode. 'fast' (default) runs SLP directly; 'max' prepends
    # the whole-slice GPU untangler as a low-L1 seed. Only affects the 'slp'
    # solver path (ignored for other strategies); requires PyTorch.
    accuracy: str = 'fast'  # 'fast' | 'max'

    # ---- output ----
    verbose: int = 1
    debug: bool = False
    record_history: bool = True
    record_snapshots: bool = False  # for plot_feasibility

    # Optional strategy-specific overrides for the string-config path.
    # Anyone wanting deeper control should pass a Strategy instance.
    strategy_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        # Fail fast on a bad accuracy mode regardless of which solver /
        # dispatch path would eventually consume it.
        if self.accuracy not in ('fast', 'max'):
            raise ValueError(f"accuracy must be 'fast' or 'max', got {self.accuracy!r}")


# ============================================================
# Result
# ============================================================


@dataclass
class SliceResult(SolveResult):
    """Per-slice DVFopt result — extends :class:`SolveResult` with the
    facade-level metadata (slice index, chosen strategy, history,
    snapshots) the per-volume orchestrator tracks.

    ``corrected`` is a read-only ``(2, H, W)`` [dy, dx] *view* into the
    assembled ``Result.corrected`` volume (not an independent copy) —
    copy it before mutating.

    Aliases ``init_min``/``final_min`` are provided as properties so the
    existing dataframe + plot code (which reads ``s.init_min``) continues
    to work without churn.
    """

    z: int = 0
    solver_used: str = ''
    n_outer_iters: int = 0
    notes: str = ''
    history: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    # snapshots[k] = {'tag': str, 'T': ndarray, 'n_neg': int,
    #                 'min_tri': float, 'phi': ndarray (optional)}

    # ---- Aliases for backwards compat with the older field names ----
    @property
    def init_min(self) -> float:
        return self.init_min_T

    @property
    def final_min(self) -> float:
        return self.final_min_T


@dataclass
class Result:
    """Outcome of a DVFopt.fit call.

    Attributes
    ----------
    corrected : ndarray
        Corrected DVF, same shape and dtype as the input.
    config : DVFoptConfig
        The config used.
    slice_results : list of SliceResult
        One entry per slice processed (length = D for 3D, 1 for 2D).
    total_wall_time : float
        Total wall-clock time across all slices.
    """

    corrected: np.ndarray
    config: DVFoptConfig
    slice_results: list[SliceResult]
    total_wall_time: float

    # ----- summary helpers -----
    @property
    def feasible(self) -> bool:
        return all(s.feasible for s in self.slice_results)

    @property
    def summary_dict(self) -> dict[str, Any]:
        n = len(self.slice_results)
        feas = sum(1 for s in self.slice_results if s.feasible)
        return dict(
            slices=n,
            feasible=feas,
            feasibility_pct=100.0 * feas / max(1, n),
            init_n_neg=sum(s.init_n_neg for s in self.slice_results),
            final_n_neg=sum(s.final_n_neg for s in self.slice_results),
            init_min_tri=min((s.init_min for s in self.slice_results), default=float('nan')),
            final_min_tri=min((s.final_min for s in self.slice_results), default=float('nan')),
            total_wall_time_s=self.total_wall_time,
        )

    def summary(self) -> str:
        d = self.summary_dict
        cfg = self.config
        return (
            f'DVFopt result  ({d["slices"]} slice(s))\n'
            f'  solver         : {cfg.solver}   constraint: {cfg.constraint}   '
            f'objective: {cfg.objective}\n'
            f'  threshold      : {cfg.threshold}\n'
            f'  feasible       : {d["feasible"]}/{d["slices"]}  '
            f'({d["feasibility_pct"]:.1f}%)\n'
            f'  folds          : init {d["init_n_neg"]} -> final '
            f'{d["final_n_neg"]}\n'
            f'  min_tri / jdet : init {d["init_min_tri"]:+.4f} -> final '
            f'{d["final_min_tri"]:+.4f}\n'
            f'  wall time      : {d["total_wall_time_s"]:.1f}s'
        )

    def to_dataframe(self):
        """Per-slice tabular summary."""
        import pandas as pd

        return pd.DataFrame(
            [
                dict(
                    z=s.z,
                    init_n_neg=s.init_n_neg,
                    init_min=s.init_min,
                    final_n_neg=s.final_n_neg,
                    final_min=s.final_min,
                    feasible=s.feasible,
                    solver=s.solver_used,
                    outer_iters=s.n_outer_iters,
                    wall_s=s.wall_time,
                    notes=s.notes,
                )
                for s in self.slice_results
            ]
        )

    def history_df(self):
        """Concatenated per-iteration history across all slices (long form)."""
        import pandas as pd

        rows = []
        for s in self.slice_results:
            for h in s.history:
                rows.append({'z': s.z, **h})
        return pd.DataFrame(rows)

    # ----- visualization -----
    # The plot implementations live in :mod:`dvfopt._plots` to keep
    # matplotlib (a heavy import) out of the unified.py import path
    # until a plot is actually called. The methods here are thin
    # delegating shims so ``result.plot_convergence(z=0)`` still works.
    def plot_convergence(self, z=None, ax=None):
        from dvfopt._plots import plot_convergence as _plot_convergence

        return _plot_convergence(self, z=z, ax=ax)

    def plot_feasibility(self, z=0, snapshot=-1, ax=None):
        from dvfopt._plots import plot_feasibility as _plot_feasibility

        return _plot_feasibility(self, z=z, snapshot=snapshot, ax=ax)

    def plot_gradient_region(self, z=0, ax=None):
        from dvfopt._plots import plot_gradient_region as _plot_gradient_region

        return _plot_gradient_region(self, z=z, ax=ax)


# ============================================================
# Helpers
# ============================================================


def _extract_2d_slice(deformation, z):
    """Return a (2, H, W) [dy, dx] slice from any of the supported shapes."""
    if deformation.ndim == 2:
        raise ValueError('input must be at least 3D (channels + spatial)')
    if deformation.ndim == 3:
        if deformation.shape[0] == 2:  # (2, H, W)
            return deformation
        if deformation.shape[0] == 3:  # (3, H, W)
            return np.stack([deformation[1], deformation[2]])
    if deformation.ndim == 4:  # (3, D, H, W)
        if deformation.shape[0] == 3:
            return np.stack([deformation[1, z], deformation[2, z]])
    raise ValueError(f'unsupported deformation shape {deformation.shape}')


def _compute_constraint_2d(phi2, kind):
    """Constraint values as a (n_constraints,) ndarray — exactly what the
    solver enforces (corner patches included under ``'simplex'``)."""
    c = make_constraint(kind, phi2.shape[-2:])
    return c.values(c.flatten(phi2))


def _stats_2d(phi2, kind):
    """Return (n_neg, min_value) for the 2D constraint of `kind`."""
    T = _compute_constraint_2d(phi2, kind)
    return int((T <= 0).sum()), float(T.min())


# ============================================================
# DVFopt
# ============================================================


class DVFopt:
    """Unified deformation-field optimizer.

    All configuration is held in a :class:`DVFoptConfig`. Either pass a
    pre-built config or override individual keyword arguments at
    construction time::

        opt = DVFopt(solver='barrier', threshold=0.01)
        result = opt.fit(deformation)

    The ``fit`` method auto-detects 2D-vs-3D input and dispatches to the
    appropriate constraint / solver backend.
    """

    def __init__(self, config: Optional[DVFoptConfig] = None, **kwargs):
        if config is None:
            config = DVFoptConfig(**kwargs)
        elif kwargs:
            config = replace(config, **kwargs)
        self.config = config
        self._validate()

    def _validate(self):
        from dvfopt.exceptions import SolverConfigError

        c = self.config
        cls = _CONSTRAINT_REGISTRY.get(c.constraint)
        if cls is None:
            valid = sorted(k for k, v in _CONSTRAINT_REGISTRY.items() if v.dim == 2)
            raise SolverConfigError(f'bad constraint: {c.constraint!r}; valid: {valid}')
        if cls.dim != 2:
            raise SolverConfigError(
                f'constraint {c.constraint!r} is not supported: the DVFopt '
                f'facade is per-slice 2D; for true-3D correction use '
                f'correct_dvf_3d / correct_dvf_25d or the Solver API '
                f"(e.g. Solver.from_spec(constraint='simplex_3d', ...))."
            )
        # solver: 'auto', a registered label, or a Strategy instance.
        if isinstance(c.solver, Strategy) or c.solver == 'auto':
            pass
        elif isinstance(c.solver, str):
            from dvfopt.strategies import _STRATEGY_REGISTRY

            if c.solver not in _STRATEGY_REGISTRY:
                raise SolverConfigError(
                    f'bad solver: {c.solver!r}; valid: {sorted(_STRATEGY_REGISTRY)}'
                )
        else:
            raise SolverConfigError(
                f'solver must be str or Strategy, got {type(c.solver).__name__}'
            )
        if c.objective not in ('l2', 'l1', 'none'):
            raise SolverConfigError(f'bad objective: {c.objective!r}')

    # ---- main entry ----
    def fit(self, deformation) -> Result:
        """Run the optimizer on ``deformation`` and return a :class:`Result`.

        Accepted input layouts (all coerced to canonical ``(3, D, H, W)``
        for per-slice dispatch, then restored to the original shape on
        return):

        * ``(2, H, W)`` — a single 2D slice. Output: ``(2, H, W)``.
        * ``(3, H, W)`` — single 2D slice with a dz channel. Output: ``(3, H, W)``.
        * ``(3, D, H, W)`` — full 3D volume. Output: ``(3, D, H, W)``.

        Anything :func:`numpy.asarray` can interpret (lists, masked
        arrays, torch tensors via ``__array__``) is accepted. NaN/Inf
        values are rejected at the boundary with an actionable error.
        """
        from dvfopt.exceptions import SolverConfigError
        from dvfopt.validation import coerce_to_ndarray, validate_finite

        t0 = time.time()
        deformation = coerce_to_ndarray(deformation, name='deformation')
        validate_finite(deformation, name='deformation')

        # Detect format → canonical (3, D, H, W) buffer for per-slice
        # dispatch. We remember the original layout so we can restore
        # it on return.
        original_layout = (deformation.ndim, deformation.shape)
        if deformation.ndim == 3 and deformation.shape[0] == 2:
            # (2, H, W) — synthesise a (3, 1, H, W) with dz=0, drop dz on return.
            H, W = deformation.shape[1:]
            corrected = np.zeros((3, 1, H, W), dtype=np.float64)
            corrected[1, 0] = deformation[0]
            corrected[2, 0] = deformation[1]
            slices = [0]
        elif deformation.ndim == 3 and deformation.shape[0] == 3:
            # (3, H, W) — promote to (3, 1, H, W).
            corrected = deformation[:, None].copy()
            slices = [0]
        elif deformation.ndim == 4 and deformation.shape[0] == 3:
            # (3, D, H, W) — already canonical.
            corrected = deformation.copy()
            slices = list(range(deformation.shape[1]))
        else:
            raise SolverConfigError(
                f'unsupported deformation layout {deformation.shape}. '
                f'Accepted: (2, H, W), (3, H, W), or (3, D, H, W).'
            )

        slice_results = []
        for z in slices:
            phi2 = _extract_2d_slice(corrected, z)
            sr = self._run_slice(phi2, z)
            self._put_2d_slice(corrected, z, phi2)
            # Store a read-only (2, H, W) [dy, dx] VIEW into the assembled
            # volume rather than a per-slice copy — Result.corrected already
            # holds the assembled data, and duplicating every slice costs
            # ~1.2 GB on a 528-slice volume.
            view = corrected[1:3, z]
            view.flags.writeable = False
            sr.corrected = view
            slice_results.append(sr)

        # Restore the original layout on the return value.
        ndim, shape = original_layout
        if ndim == 3 and shape[0] == 2:
            corrected = np.stack([corrected[1, 0], corrected[2, 0]])
        elif ndim == 3 and shape[0] == 3:
            corrected = corrected[:, 0]
        # else: already canonical (3, D, H, W).

        return Result(
            corrected=corrected,
            config=self.config,
            slice_results=slice_results,
            total_wall_time=time.time() - t0,
        )

    def _put_2d_slice(self, corrected, z, phi2):
        """Write a (2, H, W) slice back into the (3, D, H, W) corrected
        buffer, leaving channel 0 (dz) untouched."""
        if corrected.ndim == 4:
            corrected[1, z] = phi2[0]
            corrected[2, z] = phi2[1]
        elif corrected.ndim == 3 and corrected.shape[0] == 2:
            corrected[0] = phi2[0]
            corrected[1] = phi2[1]
        elif corrected.ndim == 3 and corrected.shape[0] == 3:
            corrected[1] = phi2[0]
            corrected[2] = phi2[1]

    # ---- per-slice dispatcher (delegates to Solver) ----
    def _run_slice(self, phi2, z) -> SliceResult:
        c = self.config
        # Build the Constraint up front so we can read init stats and
        # auto-resolve the strategy without re-deriving anything.
        H, W = phi2.shape[1], phi2.shape[2]
        constraint = make_constraint(c.constraint, (H, W))
        init_n_neg, init_min = _constraint_stats(constraint, phi2)
        if c.verbose >= 1:
            log_info(f'[z={z}] init n_neg={init_n_neg}  min={init_min:+.4f}')
        if init_n_neg == 0 and init_min >= c.threshold - c.err_tol:
            # ``corrected`` is replaced by fit() with a read-only view
            # into the assembled volume; no per-slice copy is retained.
            return SliceResult(
                corrected=phi2,
                init_n_neg=0,
                init_min_T=init_min,
                final_n_neg=0,
                final_min_T=init_min,
                feasible=True,
                wall_time=0.0,
                info={},
                z=z,
                solver_used='none',
                n_outer_iters=0,
                notes='already feasible',
            )

        # Resolve strategy. Accepted forms:
        #   - 'auto' → auto_strategy heuristic
        #   - any other str → make_strategy(label, **c.strategy_kwargs)
        #   - Strategy instance → used as-is (c.strategy_kwargs ignored)
        if isinstance(c.solver, Strategy):
            strategy = c.solver
            strategy_label = type(strategy).__name__
            if c.accuracy != 'fast':
                import warnings

                warnings.warn(
                    f"DVFoptConfig.accuracy={c.accuracy!r} is ignored when "
                    f"solver is a Strategy instance; set it on the instance "
                    f"instead (e.g. SLPStrategy(accuracy={c.accuracy!r}))."
                )
        elif c.solver == 'auto':
            strategy_label = auto_strategy(
                constraint, init_n_neg, init_min, objective_label=c.objective
            )
            kw = dict(c.strategy_kwargs)
            if c.accuracy != 'fast':
                if strategy_label == 'slp':
                    # User-supplied strategy_kwargs['accuracy'] wins over
                    # the config-level shorthand.
                    kw.setdefault('accuracy', c.accuracy)
                else:
                    # auto resolved to a non-SLP label (l2/none objectives,
                    # Jdet constraints), where accuracy would silently do
                    # nothing — say so. (simplex (2D) + l1 auto-resolves to 'slp'
                    # and takes the branch above.)
                    import warnings

                    warnings.warn(
                        f"accuracy={c.accuracy!r} currently applies only to "
                        f"solver='slp'; auto selected {strategy_label!r} — "
                        f"set solver='slp' to use the GPU-seeded mode."
                    )
            strategy = make_strategy(strategy_label, **kw)
        else:
            strategy_label = c.solver
            kw = dict(c.strategy_kwargs)
            if strategy_label == 'slp' and c.accuracy != 'fast':
                # User-supplied strategy_kwargs['accuracy'] wins over
                # the config-level shorthand.
                kw.setdefault('accuracy', c.accuracy)
            strategy = make_strategy(strategy_label, **kw)
        objective = make_objective(c.objective, eps_l1=c.eps_l1)

        # Snapshot init if requested.
        snapshots: list[dict[str, Any]] = []
        if c.record_snapshots:
            T = constraint.values(constraint.flatten(phi2))
            snapshots.append(dict(tag='init', T=T.copy(), n_neg=init_n_neg, min_tri=init_min))

        # Run.
        solver = Solver(
            constraint=constraint,
            objective=objective,
            strategy=strategy,
            threshold=c.threshold,
            err_tol=c.err_tol,
        )
        t0 = time.time()
        res = solver.fit(phi2, verbose=c.verbose, record_history=c.record_history)
        phi2[:] = res.corrected

        if c.record_snapshots:
            T = constraint.values(constraint.flatten(phi2))
            snapshots.append(
                dict(tag='final', T=T.copy(), n_neg=res.final_n_neg, min_tri=res.final_min_T)
            )

        # Solver.fit now returns SolveInfo on res.info. Flatten the
        # SolveInfo.phases to the legacy list-of-dicts shape that
        # ``Result.plot_convergence`` and ``Result.history_df`` expect.
        info = res.info
        if isinstance(info, SolveInfo):
            # Build the legacy list-of-dicts shape from SolveInfo.phases.
            # Extras can include the canonical keys (n_neg / min_T) when
            # they came from a strategy that already populated them; the
            # PhaseInfo fields win.
            history = []
            for p in info.phases:
                row = dict(p.extras)
                row.update(
                    phase=p.name,
                    nit=p.n_iter,
                    wall_s=p.wall_s,
                    n_neg=p.n_neg,
                    min_T=p.min_T,
                )
                history.append(row)
        elif isinstance(info, list):
            history = info
        elif isinstance(info, dict) and isinstance(info.get('history'), list):
            history = info['history']
        elif isinstance(info, dict):
            history = [info] if info else []
        else:
            history = []

        if c.verbose >= 1:
            log_info(
                f'[z={z}] final n_neg={res.final_n_neg}  '
                f'min={res.final_min_T:+.5f}  '
                f'strategy={strategy_label}  '
                f'({time.time() - t0:.1f}s)',
            )
        # ``corrected`` is replaced by fit() with a read-only view into
        # the assembled volume; no per-slice copy is retained.
        return SliceResult(
            corrected=phi2,
            init_n_neg=init_n_neg,
            init_min_T=init_min,
            final_n_neg=res.final_n_neg,
            final_min_T=res.final_min_T,
            feasible=res.feasible,
            wall_time=res.wall_time,
            info=res.info or {},
            z=z,
            solver_used=strategy_label,
            n_outer_iters=1,
            history=history if c.record_history else [],
            snapshots=snapshots,
            notes=('feasible' if res.feasible else 'still folded'),
        )


# ============================================================
# Helpers (constraint + strategy plumbing)
# ============================================================


def _constraint_stats(constraint: Constraint, phi2: np.ndarray) -> tuple[int, float]:
    flat = constraint.flatten(phi2)
    T = constraint.values(flat)
    return int((T <= 0).sum()), float(T.min())


__all__ = [
    'DVFopt',
    'DVFoptConfig',
    'Result',
    'SliceResult',
]
