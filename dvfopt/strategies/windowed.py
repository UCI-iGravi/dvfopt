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
    qp_backend : str
        QP solver behind each subproblem: ``'hybrid'`` (default —
        interior-point Clarabel on a window's cold first solve and after
        any ADMM solve that hit ``ip_after_admm_iters``, warm-started
        OSQP otherwise) or ``'osqp'`` (the pre-hybrid path, byte for
        byte). Hybrid measured 262 s vs 300 s on raw B0039 z16 at zero
        simplex folds and damage 0.
    ip_cold : bool
        Use interior point for a window's cold first solve (where the
        ADMM warm start is worth nothing). ``qp_backend='hybrid'`` only.
    ip_after_admm_iters : int
        Use interior point for the solve after any ADMM solve that ran
        at least this many iterations — the tail signal that the warm
        start has gone stale. 800 measured best (cold-only 296 s,
        400 -> 289 s, 800 -> 262 s, 1500 -> 269 s).
    giant_tile : int
        Tile size for the overlapping-tile Schwarz decomposition of an
        over-``max_window_area`` region. Bigger tiles mean fewer Schwarz
        seams and fewer sweeps: 64 (the default) measured 1.9x faster
        than 32 on a full raw B0039 slice at equal feasibility.
    giant_max_sweeps : int
        Sweep cap for that decomposition (it stops early once the region
        is clear or stops improving).
    giant_tile_fit : bool
        Treat ``giant_tile`` as a target and fit the effective tile to
        each region's geometry, so an integer number of near-equal tiles
        covers its longest side. Tile size matters through grid
        alignment (sweep-round count), not size: on the raw B0039 z16
        giant the fitted 51 and the lucky 64 both take 1 round, while
        56 and 80 take 2 (~600 s vs ~350 s). ``False`` = literal tile.
    tr_delta, tr_max : float
        Initial radius / cap of the ``isqp`` inner's trust region, in grid
        units. 2.0 is what every measured windowed number was taken at;
        ``tr_delta=1.0`` trades fidelity for speed (raw B0039 z16: 267 s /
        1022 SQP iterations / L2 move 344 vs 300 s / 1320 / L2 325).
        ``tr_max`` never binds on the measured B0039 windows.
    step_rule : str
        How the ``isqp`` inner turns a QP step into an iterate:
        ``'exact_ls'`` (default) minimises the merit EXACTLY along the
        step — free, since the 2D rows are exactly quadratic along a
        line and the model reuses an evaluation the ratio test already
        makes — or ``'tr'``, the trust-region ratio test that was the
        engine's only rule before this knob (byte for byte). Raw B0039
        z16: 200 s / 563 SQP iterations vs 244 s / 780 (-18% / -28%) at
        0 folds, damage 0 and a smaller move (L2 268 vs 280); 9/9 wall
        and iteration wins over a 9-real-slice sample (-19% / -27%).
        2D only (a 6-tet row is cubic along a line).
    exact_ls_fallback_steps : int
        Consecutive ``'exact_ls'`` steps with ``a* < 0.25`` after which a
        window stops and hands itself to the escalation ladder (0 =
        off). The exact minimiser always finds SOME decrease, so on a
        window it cannot solve it never fires the ratio test's fast
        bail-out — this is the same stop on the signal it CAN see. 3 is
        measured: it never fires on the window ``'exact_ls'`` turns from
        a failure into a solve, and takes the ``z0_sliver`` crop from
        1684 SQP iterations to 212 (``'tr'``: 540) while also cutting
        raw B0039 z16 from 563 to 396 and every case's L2 move.
    coarse_to_fine : bool
        Prepend a coarse-grid warm start: solve the same problem on a
        ``coarse_factor`` x coarsened field and seed the fine solve with
        the prolongated correction, masked to the window free boxes the
        fine engine would open anyway (so no-damage is unchanged). Raw
        B0039 z16: 205 s / 909 SQP iterations (841 fine + a 16 s,
        68-iteration coarse solve) vs 283 s / 1320 cold, at a slightly
        smaller L2 move (320.6 vs 325.1). Skipped — byte-identical to ``False`` — on a fold-free
        field or one with ``min(H, W) < 4 * giant_tile``.
    coarse_factor : int
        Coarsening factor for that stage (box-average blocks).
    reanchor : str
        Optional post-feasibility **re-anchor stage**: ``'none'``
        (default — off), ``'l2'`` or ``'l1'``. The robust recipe
        solves with ``objective='none'`` (pure feasibility, which keeps
        the inner out of the objective-basin traps a distance anchor
        pins it in) and is therefore close to the input only by
        construction. When the field comes out fold-free this stage
        tiles the MOVED region and re-solves each tile minimising the
        chosen distance to the INPUT, keeping a tile only if every
        enforced row stays at or above ``threshold``. It frees only
        pixels the main solve already moved, so no-damage accounting is
        unchanged. Measured on already-feasible fields: B0039 z16 L2
        move 76.7 -> 59.9, z0 194 -> 170, 0 folds throughout.
    reanchor_maxiter : int
        Inner-solver iteration budget per re-anchor tile.
    reanchor_sweeps : int
        Maximum re-anchor sweeps (stops early once a sweep buys < 1% of
        the L2 move).
    reanchor_tile : int
        Re-anchor tile size in px (tiles overlap by 8).
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
    giant_tile: int = 64
    giant_max_sweeps: int = 8
    giant_tile_fit: bool = True
    qp_backend: str = 'hybrid'
    ip_cold: bool = True
    ip_after_admm_iters: int = 800
    tr_delta: float = 2.0
    tr_max: float = 16.0
    step_rule: str = 'exact_ls'
    exact_ls_fallback_steps: int = 3
    patience_retry: bool = True
    orientation_delta: Optional[float] = None
    orientation_scope: str = 'all'
    coarse_to_fine: bool = True
    coarse_factor: int = 4
    reanchor: str = 'none'
    reanchor_maxiter: int = 60
    reanchor_sweeps: int = 3
    reanchor_tile: int = 48
    reseed_rounds: int = 3
    reseed_radius: int = 2
    reseed_before_mop: bool = False
    untangle_delta: Optional[float] = None

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
            giant_tile=self.giant_tile,
            giant_max_sweeps=self.giant_max_sweeps,
            giant_tile_fit=self.giant_tile_fit,
            qp_backend=self.qp_backend,
            ip_cold=self.ip_cold,
            ip_after_admm_iters=self.ip_after_admm_iters,
            tr_delta=self.tr_delta,
            tr_max=self.tr_max,
            step_rule=self.step_rule,
            exact_ls_fallback_steps=self.exact_ls_fallback_steps,
            patience_retry=self.patience_retry,
            orientation_delta=self.orientation_delta,
            orientation_scope=self.orientation_scope,
            coarse_to_fine=self.coarse_to_fine,
            coarse_factor=self.coarse_factor,
            reanchor=self.reanchor,
            reanchor_maxiter=self.reanchor_maxiter,
            reanchor_sweeps=self.reanchor_sweeps,
            reanchor_tile=self.reanchor_tile,
            reseed_rounds=self.reseed_rounds,
            reseed_radius=self.reseed_radius,
            reseed_before_mop=self.reseed_before_mop,
            untangle_delta=self.untangle_delta,
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
