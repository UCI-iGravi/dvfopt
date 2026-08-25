"""Wrapper strategies over the windowed no-damage engine.

:class:`WindowedWrapperStrategy` runs
:func:`dvfopt.core.windowed.windowed_correct` — one small frozen-ring
window per fold cluster, no-damage by construction — with an inner
window solver selected by *label*. :class:`ISQPWindowedStrategy` pins
the label to ``'isqp'`` (the tuned elastic-QP SQP), the promoted
PR #61-64 benchmark configuration.

Example::

    from dvfopt import Solver, JdetConstraint2D, L2Objective, ISQPWindowedStrategy
    result = Solver(
        constraint=JdetConstraint2D(shape=(320, 456)),
        objective=L2Objective(),
        strategy=ISQPWindowedStrategy(),
    ).fit(phi)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional

from dvfopt.core.windowed import LOCALITY, windowed_correct
from dvfopt.core.windowed._inners import _ISQP_LABELS, INNER_LABELS
from dvfopt.objectives import L1Objective, L2Objective, NoneObjective
from dvfopt.strategies.base import Strategy, register_strategy


@register_strategy('windowed_wrapper')
@dataclass
class WindowedWrapperStrategy(Strategy):
    """Cluster-windowed no-damage decomposition around an inner window solver.

    Detects fold clusters, solves each inside a small window with a
    hard-frozen context ring (only the free pixels are pasted back —
    the rest of the slice is untouched *by construction*), grows
    windows on failure, tiles giant regions, and finishes with a
    large-margin mop pass. The engine lives in
    :mod:`dvfopt.core.windowed`.

    ``inner`` is a window-inner **label**
    (``'isqp'`` / ``'slsqp'`` / ``'slsqp+trust-constr'``, see
    :data:`dvfopt.core.windowed._inners.INNER_LABELS`), NOT a
    :class:`Strategy`. Each window is a frozen-ring REDUCED problem
    (:class:`~dvfopt.core.windowed._inners.WindowSub`): ring variables
    are hard-frozen inside the solve and every constraint row a free
    pixel influences is enforced with global-matching evaluation. A
    ``Strategy.fit`` on a crop cannot express frozen variables or row
    restriction — shoehorning one in reproduces the
    optimize-the-ring-then-discard-it seam gap ``core/slsqp_windowed``
    documents in its own FOLLOW-UP comment (see the 2026-08-23
    windowed-engine-promotion design spec, Decision 1). A
    ``StrategyInnerAdapter`` for masked-solve-capable strategies is
    stage 2.

    Parameters
    ----------
    inner : str
        Window-inner label (required; see above).
    margin : int
        Free-box margin around each fold cluster (clamped to at least
        the constraint family's ring width).
    maxiter : int
        Inner-solver iteration cap per window.
    max_rounds : int
        Outer find-windows/solve rounds before the mop.
    margin_delta : float
        Constraints are driven to ``threshold + margin_delta`` so the
        QP tolerance cannot land a hair below the strict fold check.
    max_window_area : int
        Free-box area above which a merged cluster is cleared by
        overlapping-tile Schwarz decomposition instead of one QP.
    mop_margin : int
        Margin for the terminal large-window mop pass (0 disables).
    time_budget_s : float, optional
        Wall-clock budget, checked at round/window boundaries.
    no_tr_fallback : bool
        Retry a failed window once with the trust region OFF (legacy
        backtracking line search) before growing it. On by default: the
        TR ratio test freezes on sliver-scale violations (~1e-4, inside
        OSQP's noise) that the line search still clears.
    fallback_maxiter : int
        SQP iteration budget for that fallback retry (the line search
        otherwise runs far past convergence).
    qp_max_iter, qp_max_iter_fallback : int
        OSQP ADMM iteration cap per subproblem, normal / fallback solves.
    """

    inner: Optional[str] = None
    margin: int = 3
    maxiter: int = 400
    max_rounds: int = 8
    margin_delta: float = 1e-3
    max_window_area: int = 3000
    mop_margin: int = 25
    time_budget_s: Optional[float] = None
    no_tr_fallback: bool = True
    fallback_maxiter: int = 200
    qp_max_iter: int = 2000
    qp_max_iter_fallback: int = 500

    accepts_constraints = tuple(LOCALITY)
    accepts_objectives = (L1Objective, L2Objective, NoneObjective)
    supports_3d = False

    def __post_init__(self):
        if self.inner is None:
            raise ValueError(
                'WindowedWrapperStrategy requires inner=<label>; '
                'use ISQPWindowedStrategy for the pinned default.'
            )
        if self.inner not in INNER_LABELS:
            raise ValueError(f'unknown inner {self.inner!r}; valid labels: {list(INNER_LABELS)}')

    def solve(
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        step_callback=None,
        **_,
    ):
        self._check_constraint(constraint)
        if self.inner in _ISQP_LABELS and importlib.util.find_spec('osqp') is None:
            raise ImportError("inner='isqp' requires osqp — pip install dvfopt[solvers]")
        phi, report = windowed_correct(
            phi_in,
            self.inner,
            constraint=constraint,
            objective=objective,
            threshold=threshold,
            margin=self.margin,
            maxiter=self.maxiter,
            max_rounds=self.max_rounds,
            margin_delta=self.margin_delta,
            max_window_area=self.max_window_area,
            mop_margin=self.mop_margin,
            no_tr_fallback=self.no_tr_fallback,
            fallback_maxiter=self.fallback_maxiter,
            qp_max_iter=self.qp_max_iter,
            qp_max_iter_fallback=self.qp_max_iter_fallback,
            time_budget_s=self.time_budget_s,
            verbose=verbose,
            record_history=record_history,
            step_callback=step_callback,
        )
        if not record_history:
            return self._finish(phi, record_history, threshold)
        # Marshal the engine's history entries ('name' key, final entry
        # carrying an 'extras' dict) into the legacy-history shape
        # SolveInfo.from_legacy_history maps: 'phase' names the phase, and
        # the final damage/move stats surface at SolveInfo.extras top-level.
        hist = [
            {'phase': h['name'], **{k: v for k, v in h.items() if k != 'name'}}
            for h in report.history
        ]
        info = {'history': hist}
        if hist and isinstance(hist[-1].get('extras'), dict):
            info.update(hist[-1].pop('extras'))
        return self._finish((phi, info), record_history, threshold)


@register_strategy('isqp_windowed')
@dataclass
class ISQPWindowedStrategy(WindowedWrapperStrategy):
    """Windowed engine with the ``'isqp'`` elastic-QP inner pinned.

    Zero-arg constructible — the promoted PR #61-64 configuration
    (528/528 B0039 slices cleared on jdet/finite, damage = 0 on all
    2178 benchmark tasks). Requires the optional ``osqp`` dependency
    (``pip install dvfopt[solvers]``).
    """

    inner: Optional[str] = 'isqp'


__all__ = ['ISQPWindowedStrategy', 'WindowedWrapperStrategy']
