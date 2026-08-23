"""Solver facade — composes a constraint, an objective, and a strategy.

The :class:`Solver` is the canonical user-facing API for the
parameterized package. It validates compatibility (e.g. m10/m14 require
a 2-triangle constraint, the SLSQP full-grid strategy doesn't support
3D, etc.), runs the strategy, and returns a structured result.

Usage
-----

Direct construction::

    from dvfopt import Solver, TriConstraint2D, L1Objective, BarrierStrategy
    solver = Solver(
        constraint=TriConstraint2D(shape=(320, 456)),
        objective=L1Objective(eps=1e-4),
        strategy=BarrierStrategy(),
    )
    result = solver.fit(phi_in)

String shorthand (constructs the parts from labels)::

    from dvfopt import Solver
    result = Solver.from_spec(
        constraint='2tri', objective='l1',
        strategy='schwarz_harmonic_alm_refine_repair',
        shape=(320, 456),
    ).fit(phi_in)

The legacy :class:`dvfopt.DVFopt` / :class:`dvfopt.DVFoptConfig` API is
a higher-level facade layered on top of this — see
:mod:`dvfopt.unified`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.constraints import (
    Constraint,
    TriConstraint2D,
    TriConstraint2DFullCoverage,
    make_constraint,
)
from dvfopt.objectives import Objective, make_objective
from dvfopt.strategies import Strategy, make_strategy


@dataclass
class PhaseInfo:
    """One phase of a strategy run — e.g. a barrier λ-step, an m14 stage.

    Strategies log one of these per discrete pass in their pipeline.
    Used by visualization code to render convergence curves uniformly
    across strategies (no more bespoke per-strategy info dicts).
    """

    name: str
    n_iter: int = 0
    wall_s: float = 0.0
    n_neg: int = -1
    min_T: float = float('nan')
    extras: dict = field(default_factory=dict)


@dataclass
class SolveInfo:
    """Standardized history container produced by every Strategy.

    Strategies may *additionally* attach strategy-specific state in
    ``extras``, but anything common across strategies (per-phase
    feasibility traces, total wall clock, the strategy name)
    lives here so visualization / dataframes can rely on it.
    """

    strategy_name: str = ''
    phases: list = field(default_factory=list)  # list[PhaseInfo]
    total_iter: int = 0
    feasible_after_phase: int = -1  # index where n_neg first hit 0
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_legacy_history(
        cls, strategy_name: str, history: list, threshold: float = 0.01
    ) -> SolveInfo:
        """Wrap a legacy free-form ``history`` list as a :class:`SolveInfo`.

        ``history`` items typically have ``phase`` (str), ``nit`` or
        ``step``, ``n_neg`` / ``min_T``, ``wall_s``, and free-form
        strategy-specific keys. This adapter pulls the common fields
        into :class:`PhaseInfo` and stashes the rest in ``extras``.
        """
        phases: list[PhaseInfo] = []
        feasible_after = -1
        for i, h in enumerate(history or []):
            if not isinstance(h, dict):
                continue
            n_iter = int(h.get('nit', h.get('step', h.get('n_iter', 0)) or 0))
            n_neg = int(h.get('n_neg', -1))
            phases.append(
                PhaseInfo(
                    name=str(h.get('phase', f'phase_{i}')),
                    n_iter=n_iter,
                    wall_s=float(h.get('wall_s', 0.0)),
                    n_neg=n_neg,
                    min_T=float(h.get('min_T', h.get('min_tri', float('nan')))),
                    extras={
                        k: v
                        for k, v in h.items()
                        if k
                        not in {
                            'phase',
                            'nit',
                            'step',
                            'n_iter',
                            'n_neg',
                            'min_T',
                            'min_tri',
                            'wall_s',
                        }
                    },
                )
            )
            if feasible_after < 0 and n_neg == 0 and phases[-1].min_T >= threshold - 1e-5:
                feasible_after = i
        # Only stash the raw history when phase extraction FAILED (e.g.
        # history items weren't dicts) — otherwise the same data would be
        # retained twice (once as PhaseInfo, once verbatim), which adds up
        # on long per-slice runs.
        extras = {'_legacy_history': history} if (history and not phases) else {}
        return cls(
            strategy_name=strategy_name,
            phases=phases,
            total_iter=sum(p.n_iter for p in phases),
            feasible_after_phase=feasible_after,
            extras=extras,
        )


@dataclass
class SolveResult:
    """Output of :meth:`Solver.fit`.

    Attributes
    ----------
    corrected : ndarray
        Corrected DVF, same shape as input.
    init_n_neg, init_min_T : (int, float)
        Initial constraint stats.
    final_n_neg, final_min_T : (int, float)
        Final constraint stats.
    feasible : bool
        ``True`` when ``final_n_neg == 0`` and
        ``final_min_T >= threshold - err_tol``.
    wall_time : float
    info : SolveInfo
        Standardized run history (phases, totals); strategy-specific
        extras live under ``info.extras``.
    """

    corrected: np.ndarray
    init_n_neg: int
    init_min_T: float
    final_n_neg: int
    final_min_T: float
    feasible: bool
    wall_time: float
    info: SolveInfo = field(default_factory=SolveInfo)


class Solver:
    """Constraint + Objective + Strategy composition.

    Parameters
    ----------
    constraint : :class:`Constraint`
    objective : :class:`Objective`
    strategy : :class:`Strategy`
    threshold : float, optional
        Lower bound for ``constraint.values(...)``. Defaults to
        ``DEFAULT_PARAMS['threshold']``.
    err_tol : float, optional
        Slack used for the feasibility classification in
        :class:`SolveResult`.
    """

    def __init__(
        self,
        constraint: Constraint,
        objective: Objective,
        strategy: Strategy,
        *,
        threshold: Optional[float] = None,
        err_tol: float = 1e-5,
    ):
        if threshold is None:
            threshold = DEFAULT_PARAMS['threshold']
        self.constraint = constraint
        self.objective = objective
        self.strategy = strategy
        self.threshold = float(threshold)
        self.err_tol = float(err_tol)
        # Eagerly check compatibility so failures surface at construction.
        self.strategy._check_constraint(self.constraint)
        self.strategy._check_objective(self.objective)

    @classmethod
    def from_spec(
        cls,
        *,
        constraint: Union[str, Constraint],
        objective: Union[str, Objective] = 'l2',
        strategy: Union[str, Strategy] = 'barrier',
        shape: Optional[tuple[int, ...]] = None,
        threshold: Optional[float] = None,
        err_tol: float = 1e-5,
        eps_l1: float = 1e-4,
        strategy_kwargs: Optional[dict[str, Any]] = None,
    ) -> Solver:
        """Construct a :class:`Solver` from string labels.

        Examples
        --------
        >>> Solver.from_spec(
        ...     constraint='2tri', objective='l1',
        ...     strategy='schwarz_harmonic_alm_refine_repair',
        ...     shape=(320, 456),
        ... )
        """
        if isinstance(constraint, str):
            if shape is None:
                raise ValueError('shape= required when constraint is a string')
            constraint = make_constraint(constraint, shape)
        if isinstance(objective, str):
            objective = make_objective(objective, eps_l1=eps_l1)
        if isinstance(strategy, str):
            kw = dict(strategy_kwargs or {})
            strategy = make_strategy(strategy, **kw)
        return cls(
            constraint=constraint,
            objective=objective,
            strategy=strategy,
            threshold=threshold,
            err_tol=err_tol,
        )

    # ----------------------------- fit -----------------------------
    def fit(
        self,
        phi_in: np.ndarray,
        *,
        verbose: int = 0,
        record_history: bool = False,
        **strategy_kwargs,
    ) -> SolveResult:
        """Run the strategy and return a :class:`SolveResult`.

        The input is coerced ONCE to the constraint's canonical
        ``(C, *shape)`` layout before it reaches the strategy —
        strategies are written against the canonical form and must
        never see loose layouts (e.g. ``(3, H, W)`` with a dz channel,
        or ``(3, 1, H, W)`` singleton-D). The corrected field is then
        restored to the *original* input layout, so ``corrected`` has
        the same shape as ``phi_in``; for a 2D constraint fed a
        3-channel input the dz channel passes through unchanged.

        Extra kwargs are forwarded to the underlying
        :meth:`Strategy.solve` call.
        """
        t0 = time.time()
        phi_canonical = self.constraint.coerce(phi_in)
        init_n_neg, init_min = self._stats(phi_canonical)
        phi_out, info = self.strategy.solve(
            phi_canonical,
            constraint=self.constraint,
            objective=self.objective,
            threshold=self.threshold,
            verbose=verbose,
            record_history=record_history,
            **strategy_kwargs,
        )
        wall = time.time() - t0
        final_n_neg, final_min = self._stats(phi_out)
        phi_out = self._restore_layout(phi_in, phi_out)
        feasible = final_n_neg == 0 and final_min >= self.threshold - self.err_tol

        # Strategies now build SolveInfo directly via
        # :func:`dvfopt.strategies._build_solve_info`. The fall-through
        # normalization here is kept only for back-compat with external
        # strategies that may still return free-form info dicts.
        if isinstance(info, SolveInfo):
            info_obj = info
        else:
            from dvfopt.strategies import _build_solve_info

            info_obj = _build_solve_info(type(self.strategy).__name__, info, self.threshold)
        return SolveResult(
            corrected=phi_out,
            init_n_neg=init_n_neg,
            init_min_T=init_min,
            final_n_neg=final_n_neg,
            final_min_T=final_min,
            feasible=feasible,
            wall_time=wall,
            info=info_obj,
        )

    @staticmethod
    def _normalize_info(strategy_name: str, info, threshold: float) -> SolveInfo:
        """Deprecated. Use :func:`dvfopt.strategies._build_solve_info`.

        Kept for back-compat with code that called this directly.
        Strategies now wrap their own return values via the strategies
        module helper.
        """
        if not info:
            return SolveInfo(strategy_name=strategy_name)
        if isinstance(info, list):
            return SolveInfo.from_legacy_history(strategy_name, info, threshold)
        if isinstance(info, dict):
            history = info.get('history')
            if isinstance(history, list):
                out = SolveInfo.from_legacy_history(strategy_name, history, threshold)
                out.extras.update({k: v for k, v in info.items() if k != 'history'})
                return out
            phases = [
                PhaseInfo(
                    name=k,
                    wall_s=float(v.get('wall', 0.0)) if isinstance(v, dict) else 0.0,
                    n_neg=int(v.get('n_neg', -1)) if isinstance(v, dict) else -1,
                    min_T=(
                        float(v.get('min_T', float('nan'))) if isinstance(v, dict) else float('nan')
                    ),
                    extras=v if isinstance(v, dict) else {'value': v},
                )
                for k, v in info.items()
                if k != 'extras'
            ]
            return SolveInfo(
                strategy_name=strategy_name,
                phases=phases,
                total_iter=sum(p.n_iter for p in phases),
                extras=info.get('extras', {}),
            )
        return SolveInfo(strategy_name=strategy_name, extras={'raw': info})

    # ----------------------------- helpers -----------------------------
    def _restore_layout(self, phi_in, corrected: np.ndarray) -> np.ndarray:
        """Restore the corrected canonical array to the original input layout.

        ``corrected`` is in the constraint's canonical ``(C, *shape)``
        form (``(2, H, W)`` for the 2D families, ``(3, D, H, W)`` for
        3D). When the caller passed a looser layout — ``(3, H, W)``,
        ``(2, 1, H, W)``, ``(3, 1, H, W)`` — the corrected channels are
        written back into a float64 copy shaped like the input, so the
        ``SolveResult.corrected`` "same shape as input" contract holds.
        For 3-channel 2D inputs the dz channel (channel 0) passes
        through unchanged — the 2D constraint families never touch it.
        """
        orig = np.asarray(phi_in, dtype=np.float64)
        if orig.shape == corrected.shape:
            return corrected
        if self.constraint.dim == 2 and corrected.ndim == 3:
            out = orig.copy()
            if orig.ndim == 3 and orig.shape[0] in (2, 3):
                # (2|3, H, W): dy/dx are always the last two channels.
                out[-2] = corrected[0]
                out[-1] = corrected[1]
                return out
            if orig.ndim == 4 and orig.shape[0] in (2, 3) and orig.shape[1] == 1:
                # (2|3, 1, H, W) singleton-D layout.
                out[-2, 0] = corrected[0]
                out[-1, 0] = corrected[1]
                return out
        # Unknown mismatch (coerce() would have rejected it) — return
        # the canonical result rather than guess.
        return corrected

    def _stats(self, phi: np.ndarray) -> tuple[int, float]:
        """Constraint-aware (n_neg, min_T) for the input field."""
        flat = self.constraint.flatten(phi)
        T = self.constraint.values(flat)
        return int((T <= 0).sum()), float(T.min())

    def __repr__(self) -> str:
        return (
            f'Solver(constraint={self.constraint!r}, '
            f'objective={self.objective!r}, '
            f'strategy={self.strategy!r}, threshold={self.threshold})'
        )


# Convenience top-level function for one-shot use ---------------------------


def correct_dvf(
    phi_in: np.ndarray,
    *,
    constraint: Union[str, Constraint] = '2tri',
    objective: Union[str, Objective] = 'l1',
    strategy: Union[str, Strategy] = 'auto',
    shape: Optional[tuple[int, ...]] = None,
    threshold: Optional[float] = None,
    verbose: int = 0,
    record_history: bool = False,
    **strategy_kwargs,
) -> SolveResult:
    """One-shot DVF correction.

    Equivalent to::

        Solver.from_spec(constraint=constraint, objective=objective,
                          strategy=strategy, shape=shape,
                          threshold=threshold).fit(phi_in)

    With ``strategy='auto'``, picks a strategy based on the constraint
    family, objective, and initial fold density (see
    :func:`auto_strategy`; 2-tri + L1 always routes to ``'slp'``).

    .. note::
        The default objective here is ``'l1'``, matching
        :class:`dvfopt.DVFoptConfig` — the two APIs share the same
        default, so results are comparable without passing
        ``objective=`` explicitly.
    """
    if shape is None:
        shape = phi_in.shape[1:]  # infer from input
    if strategy == 'auto':
        # Need the constraint built first to read init stats; do it lazily.
        c = make_constraint(constraint, shape) if isinstance(constraint, str) else constraint
        T = c.values(c.flatten(phi_in))
        init_n_neg = int((T <= 0).sum())
        init_min = float(T.min())
        strategy = auto_strategy(
            c,
            init_n_neg,
            init_min,
            objective_label=(objective if isinstance(objective, str) else objective.label),
        )
    return Solver.from_spec(
        constraint=constraint,
        objective=objective,
        strategy=strategy,
        shape=shape,
        threshold=threshold,
        strategy_kwargs=strategy_kwargs,
    ).fit(phi_in, verbose=verbose, record_history=record_history)


def auto_strategy(
    constraint: Constraint, init_n_neg: int, init_min: float, objective_label: str = 'l1'
) -> str:
    """Pick a strategy label given initial fold stats and constraint family.

    For the 2-triangle constraint:

    * **L1 objective** — ``slp`` at every fold tier. The SLP champion
      (per-cluster trust-region SLP + m14 seed + HiGHS L1 step) reaches
      strict feasibility on every benchmarked slice and Pareto-dominates
      the m14/m10 wallbreakers on wall time at equal-or-better L1; it
      auto-routes small vs large slices internally via
      ``cluster_pixel_threshold``, so no fold-density tiering is needed.
    * **Other objectives (l2, none, …)** — legacy tiering:

      * **Extreme** (``n_neg > 5000`` or ``init_min < -10``) —
        wallbreakers. ``m10`` for L2 (its ALM phase is L2-optimal);
        ``m14_schwarz`` on large slices (>20K corners); ``m14`` on
        smaller.
      * **Moderate-to-dense** (``n_neg > 100`` or ``init_min < -0.25``)
        — ``barrier`` (dominates SLSQP by 100x at this density).
      * **Mild** — ``slsqp`` (active-set machinery is fine, gives KKT
        certs).

    For the 6-tet 3D constraint: extremes (``n_neg > 5000`` or
    ``init_min < -10``) route to the 3D wallbreakers (``m10_3d`` for L2,
    ``m14_schwarz_3d`` on volumes >200K voxels, ``m14_3d`` otherwise) —
    the plain barrier stalls on dense 3D folds. Everything else keeps
    ``barrier``.

    For the Jdet family (no wallbreakers, no SLP): barrier above
    ``n_neg > 500`` or ``init_min < -1``; the mild tier below that
    prefers ``isqp_windowed`` (the no-damage windowed elastic-QP
    engine) when ``osqp`` is installed and the constraint is 2D, else
    ``slsqp_windowed``.
    """
    from dvfopt.constraints import Tet6Constraint3D

    is_tri = isinstance(constraint, (TriConstraint2D, TriConstraint2DFullCoverage))
    if is_tri:
        if objective_label == 'l1':
            # The SLP champion is the validated L1 regime at every fold
            # tier; it handles small/large routing itself.
            return 'slp'
        if init_n_neg > 5000 or init_min < -10.0:
            if objective_label == 'l2':
                return 'm10'
            n_corners = np.prod(constraint.shape)
            if n_corners > 20000:
                return 'm14_schwarz'
            return 'm14'
        if init_n_neg > 100 or init_min < -0.25:
            return 'barrier'
        return 'slsqp'
    # 6-tet 3D: mirror the 2D tiering. Dense 3D folds are exactly where
    # the plain barrier stalls (its penalty phase can't find a feasible
    # step when many tets crowd zero simultaneously) — route extremes to
    # the 3D wallbreakers, whose harmonic seed guarantees a feasible
    # start. Mild-to-moderate folds keep the barrier (fast, and the
    # full-grid tet SLSQP does not scale).
    if isinstance(constraint, Tet6Constraint3D):
        if init_n_neg > 5000 or init_min < -10.0:
            if objective_label == 'l2':
                return 'm10_3d'  # ALM phase is L2-optimal
            if int(np.prod(constraint.shape)) > 200_000:
                return 'm14_schwarz_3d'  # cluster-localized on big volumes
            return 'm14_3d'
        return 'barrier'
    # Jdet 2D/3D
    if init_n_neg > 500 or init_min < -1.0:
        return 'barrier'
    import importlib.util

    # Mild tier: the windowed isqp engine (no-damage, 3-5x faster than
    # scipy-SLSQP) when osqp is available; it is 2D-only, so 3D Jdet
    # keeps the legacy windowed SLSQP.
    if constraint.dim == 2 and importlib.util.find_spec('osqp') is not None:
        return 'isqp_windowed'
    return 'slsqp_windowed'


__all__ = [
    'SolveResult',
    'Solver',
    'auto_strategy',
    'correct_dvf',
]
