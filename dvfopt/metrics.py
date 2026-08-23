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
    'jdet_2d', 'jdet_3d', '6tet'); ``'auto'`` picks '2tri' for 2D layouts
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


@dataclass(frozen=True)
class InjectivityStats:
    """Sub-pixel injectivity diagnostics of a DVF.

    From the quantitative-IFT radius map (and, in 2D, the bilinear cell
    certificate) in :mod:`dvfopt.jacobian.injectivity_radius` — see that
    module's docstring for the math and references.
    """

    min_radius: float  # smallest certified IFT injectivity radius (px/voxels)
    frac_subpixel: float  # fraction of samples with certified radius < 1
    cell_min_jdet: Optional[float]  # 2D only: min bilinear cell Jdet (None in 3D)
    n_cells_nonpos: Optional[int]  # 2D only: folded cells under the bilinear model


def injectivity_stats(phi, max_window: int = 8) -> InjectivityStats:
    """Neighbourhood-injectivity diagnostics for a 2D field or 3D volume.

    Accepts ``(2, H, W)`` ``[dy, dx]`` layouts as well as the canonical
    ``(3, 1, H, W)`` / ``(3, D, H, W)`` ``[dz, dy, dx]``. 2D fields also get
    the bilinear cell certificate; 3D volumes report the radius map only —
    the trilinear Jdet is not multi-affine, so sub-voxel folds are the
    6-tet constraint family's job (:func:`constraint_fold_stats`).
    """
    from dvfopt.jacobian.injectivity_radius import (
        cell_min_jdet_2d,
        ift_radius_2d,
        ift_radius_3d,
    )

    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 4 and phi.shape[0] == 3 and phi.shape[1] > 1:
        r = ift_radius_3d(phi, max_window=max_window)
        cell_min = None
    else:
        phi2 = phi[1:] if phi.shape[0] == 3 else phi  # drop the dz channel
        r = ift_radius_2d(phi2, max_window=max_window)
        cell_min = cell_min_jdet_2d(phi2)
    return InjectivityStats(
        min_radius=float(r.min()),
        frac_subpixel=float((r < 1.0).mean()),
        cell_min_jdet=None if cell_min is None else float(cell_min.min()),
        n_cells_nonpos=None if cell_min is None else int((cell_min <= 0).sum()),
    )
