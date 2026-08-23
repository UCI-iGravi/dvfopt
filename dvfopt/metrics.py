"""Canonical fold-statistics helpers.

Reporting layers (pipelines, CLI, benchmarks) derive their fold numbers
from :func:`fold_stats` so the definitions of "folded" (``<= 0``),
"below threshold" (``< threshold - err_tol``), and "fold severity"
(summed depth below threshold) live in exactly one place. Solver inner
loops keep their local 2-line stats — those are hot paths and their
tuple returns are deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dvfopt._defaults import DEFAULT_PARAMS


@dataclass(frozen=True)
class FoldStats:
    """Fold statistics of one constraint-values array (areas / volumes / Jdets)."""

    n_neg: int  # values <= 0 — true folds
    n_below: int  # values < threshold - err_tol — strict-feasibility misses
    min_val: float
    neg_volume: float  # sum(threshold - v) over v < threshold — fold severity

    @property
    def feasible(self) -> bool:
        return self.n_below == 0


def fold_stats(values, threshold: Optional[float] = None, err_tol: float = 1e-5) -> FoldStats:
    """Compute :class:`FoldStats` for an array of constraint values.

    ``threshold=None`` uses ``DEFAULT_PARAMS['threshold']`` (0.01).
    """
    v = np.asarray(values, dtype=np.float64)
    thr = DEFAULT_PARAMS['threshold'] if threshold is None else float(threshold)
    return FoldStats(
        n_neg=int((v <= 0).sum()),
        n_below=int((v < thr - err_tol).sum()),
        min_val=float(v.min()),
        neg_volume=float(np.clip(thr - v, 0.0, None).sum()),
    )


def constraint_fold_stats(
    phi,
    constraint: str = 'auto',
    threshold: Optional[float] = None,
    err_tol: float = 1e-5,
) -> tuple[str, FoldStats]:
    """:class:`FoldStats` of a DVF under a named constraint.

    ``constraint`` is a registry name ('2tri', '2tri_standard', 'jdet',
    'jdet_2d', 'finite', 'jdet_3d', '6tet'); ``'auto'`` picks '2tri' for 2D layouts
    and '6tet' for true-3D ``(3, D>1, H, W)`` volumes. Returns the
    resolved name plus the stats. Mirrors ``Solver._stats``
    (coerce -> flatten -> values), so the numbers agree with
    ``SolveResult.init_n_neg``/``init_min_T``.
    """
    from dvfopt.constraints import make_constraint

    phi = np.asarray(phi, dtype=np.float64)
    if constraint == 'auto':
        constraint = '6tet' if phi.ndim == 4 and phi.shape[1] > 1 else '2tri'
    shape = phi.shape[-3:] if constraint in ('6tet', 'jdet_3d') else phi.shape[-2:]
    c = make_constraint(constraint, shape)
    vals = c.values(c.flatten(c.coerce(phi)))
    return constraint, fold_stats(vals, threshold, err_tol)
