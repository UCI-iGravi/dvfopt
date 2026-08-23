"""Generic Schwarz domain-decomposition wrapper around any inner Strategy.

:class:`SchwarzWrapperStrategy` composes an inner :class:`Strategy` with
cluster-based Schwarz: detect connected fold components, run
``inner.solve()`` on each padded crop independently, splice back, and
(optionally) finish with a barrier polish.

The schwarz core lives in
:mod:`dvfopt.core.schwarz._common` — this Strategy is just
the composable façade. The legacy
:class:`SchwarzHarmonicALMRefineRepairStrategy` (alias
:class:`M14SchwarzStrategy`) remains exported for back-compat; it pins
the inner to refine-repair. New code should prefer
``SchwarzWrapperStrategy(inner=HarmonicALMRefineRepairStrategy(...))``
for the same behaviour with the flexibility to swap the inner.

Example::

    from dvfopt import (
        Solver, TriConstraint2D, L1Objective,
        SchwarzWrapperStrategy, HarmonicALMRefineRepairStrategy,
    )
    result = Solver(
        constraint=TriConstraint2D(shape=(320, 456)),
        objective=L1Objective(eps=1e-4),
        strategy=SchwarzWrapperStrategy(
            inner=HarmonicALMRefineRepairStrategy(),
            pad=4,
        ),
    ).fit(phi)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Optional

import numpy as np

from dvfopt.constraints import (
    Constraint,
    Tet6Constraint3D,
    TriConstraint2D,
    TriConstraint2DFullCoverage,
)
from dvfopt.exceptions import IncompatibleConstraintError
from dvfopt.strategies.base import Strategy, _build_solve_info, register_strategy

# Constraint families the schwarz wrapper knows how to decompose.
# Both must support a single-argument ``type(c)(shape=new_shape)`` clone
# so per-cluster crops get a fresh constraint of matching shape.
_SCHWARZ_2D = (TriConstraint2D, TriConstraint2DFullCoverage)
_SCHWARZ_3D = (Tet6Constraint3D,)


@register_strategy('schwarz_wrapper')
@dataclass
class SchwarzWrapperStrategy(Strategy):
    """Schwarz domain decomposition around an arbitrary inner Strategy.

    Detects connected fold components, runs ``inner.solve()`` on each
    padded crop, splices results back. Falls back to running ``inner``
    globally when a single cluster spans more than
    ``fallback_size_ratio`` of any axis or when outer rounds stall.

    Compatible inner strategies are the ones whose
    ``accepts_constraints`` covers either the 2D 2-triangle family
    (:class:`TriConstraint2D`, :class:`TriConstraint2DFullCoverage`) or
    the 3D 6-tet family (:class:`Tet6Constraint3D`). The wrapper
    auto-detects 2D vs 3D from the outer constraint and dispatches
    accordingly.

    Parameters
    ----------
    inner : Strategy
        Strategy to invoke on each cluster crop. The wrapper constructs
        a fresh constraint of matching crop shape (cloned via
        ``type(constraint)(shape=...)``) and forwards the outer
        ``objective`` and ``threshold`` unchanged.
    pad : int
        Cells/voxels of context around each cluster's bounding box.
    merge_dilation : int
        Dilation applied to the fold mask before CCL (merges
        near-touching clusters).
    max_outer_iters : int
        Outer-loop budget if splicing introduces new folds at crop
        boundaries.
    fallback_size_ratio : float
        Single cluster covering > this fraction of any axis →
        fall back to running ``inner.solve`` globally.
    time_budget_s : float
        Total wall-clock budget for the sweep.
    final_polish : bool
        If True, run a final global :class:`BarrierStrategy` polish
        once the sweep finishes (only if ``min_T < threshold + 1e-5``
        or ``n_neg > 0``). The polish anchors to the input via the
        outer ``objective``.
    final_polish_max_iter : int
        ``max_minimize_iter`` for the polish.
    """

    inner: Optional[Strategy] = None
    pad: int = 4
    merge_dilation: int = 2
    max_outer_iters: int = 3
    fallback_size_ratio: float = 0.7
    time_budget_s: float = 600.0
    final_polish: bool = True
    final_polish_max_iter: int = 200
    # Forwarded to the final polish's BarrierStrategy.barrier_grad_rtol
    # (near-feasible start → sparsified gradient up to ~5-9x faster).
    final_polish_grad_rtol: float = 0.0

    # 2D + 3D both go through this single class. The dispatch happens at
    # solve() time based on the outer constraint.
    accepts_constraints = _SCHWARZ_2D + _SCHWARZ_3D
    supports_3d: bool = field(init=False, default=True)

    def __post_init__(self):
        if self.inner is None:
            raise ValueError(
                'SchwarzWrapperStrategy requires inner=<Strategy>; '
                "use e.g. inner=HarmonicALMRefineRepairStrategy()."
            )
        if not isinstance(self.inner, Strategy):
            raise TypeError(f'inner must be a Strategy; got {type(self.inner).__name__}')

    def _check_inner_compatible(self, constraint: Constraint) -> None:
        accepted = getattr(self.inner, 'accepts_constraints', None)
        if accepted is None:
            return
        if not isinstance(constraint, tuple(accepted)):
            raise IncompatibleConstraintError(
                f'inner {type(self.inner).__name__} does not accept '
                f'{type(constraint).__name__}; '
                f'inner.accepts_constraints={accepted!r}'
            )

    def solve(
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        **_,
    ):
        from dvfopt.core.schwarz._common import (
            cluster_schwarz_2d_tri,
            cluster_schwarz_3d_tet,
        )

        self._check_constraint(constraint)
        self._check_inner_compatible(constraint)

        is_2d = isinstance(constraint, _SCHWARZ_2D)
        is_3d = isinstance(constraint, _SCHWARZ_3D)
        if not (is_2d or is_3d):  # defence-in-depth — _check_constraint should catch this
            raise IncompatibleConstraintError(
                f'SchwarzWrapperStrategy does not support {type(constraint).__name__}'
            )

        # Mutable container so the inner_solve closure can write back the
        # last cluster's info dict for record_history.
        last_inner_info: dict = {}

        def _clone_constraint_for(crop_shape: tuple[int, ...]) -> Constraint:
            return type(constraint)(shape=crop_shape)

        # If the inner is a dataclass with a ``time_budget_s`` field,
        # we use :func:`dataclasses.replace` to make a per-call copy
        # with the Schwarz-computed per-cluster budget. Without this the
        # inner would use its own static ``time_budget_s`` field and a
        # single large cluster could blow the wrapper's total wall-clock.
        inner_has_budget_field = is_dataclass(self.inner) and any(
            f.name == 'time_budget_s' for f in fields(self.inner)
        )

        def inner_solve(phi_crop: np.ndarray, time_budget_s: Optional[float] = None):
            crop_constraint = _clone_constraint_for(phi_crop.shape[1:])
            inner = self.inner
            if time_budget_s is not None and inner_has_budget_field:
                inner = replace(inner, time_budget_s=time_budget_s)
            phi_out, info = inner.solve(
                phi_crop,
                constraint=crop_constraint,
                objective=objective,
                threshold=threshold,
                verbose=max(0, verbose - 1),
                record_history=record_history,
            )
            if record_history:
                last_inner_info.clear()
                last_inner_info.update(info=info, time_budget_s=time_budget_s)
            return phi_out

        final_polish_fn = None
        if self.final_polish:
            from dvfopt.strategies.barrier import BarrierStrategy

            polisher = BarrierStrategy(
                max_iter=self.final_polish_max_iter,
                barrier_grad_rtol=self.final_polish_grad_rtol,
            )

            def final_polish_fn(phi: np.ndarray) -> np.ndarray:
                phi_out, _info = polisher.solve(
                    phi,
                    constraint=constraint,
                    objective=objective,
                    threshold=threshold,
                    verbose=0,
                    record_history=False,
                )
                return phi_out

        kw = dict(
            threshold=threshold,
            pad=self.pad,
            merge_dilation=self.merge_dilation,
            max_outer_iters=self.max_outer_iters,
            fallback_size_ratio=self.fallback_size_ratio,
            time_budget_s=self.time_budget_s,
            final_polish_fn=final_polish_fn,
            verbose=verbose,
            record_history=record_history,
        )
        if is_2d:
            out = cluster_schwarz_2d_tri(phi_in, inner_solve, **kw)
        else:
            out = cluster_schwarz_3d_tet(phi_in, inner_solve, **kw)

        if record_history:
            phi_out, info = out
            info['inner_strategy'] = type(self.inner).__name__
            return phi_out, _build_solve_info('SchwarzWrapperStrategy', info, threshold)
        return out, _build_solve_info('SchwarzWrapperStrategy', {}, threshold)


__all__ = ['SchwarzWrapperStrategy']
