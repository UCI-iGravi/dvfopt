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


def _is_volume(phi: np.ndarray) -> bool:
    """True for a true-3D ``(C, D>1, H, W)`` layout, False for 2D layouts."""
    return phi.ndim == 4 and phi.shape[1] > 1


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

    ``constraint`` is a registry name ('simplex', 'simplex_standard',
    'bilinear', 'jdet', 'jdet_2d', 'finite', 'jdet_3d', 'simplex_3d' —
    legacy '2tri'/'2tri_standard'/'6tet'/'6tet_3d' still accepted);
    ``'auto'`` picks 'simplex' for 2D layouts and 'simplex_3d' for
    true-3D ``(3, D>1, H, W)`` volumes. Returns the
    resolved name plus the stats. Mirrors ``Solver._stats``
    (coerce -> flatten -> values), so the numbers agree with
    ``SolveResult.init_n_neg``/``init_min_T``.
    """
    from dvfopt.constraints import infer_shape, make_constraint

    phi = np.asarray(phi, dtype=np.float64)
    if constraint == 'auto':
        constraint = 'simplex_3d' if _is_volume(phi) else 'simplex'
    c = make_constraint(constraint, infer_shape(constraint, phi))
    vals = c.values(c.flatten(phi))
    return constraint, fold_stats(vals, threshold, err_tol)


@dataclass(frozen=True)
class InjectivityStats:
    """Sub-pixel injectivity diagnostics of a DVF.

    From the quantitative-IFT radius *estimate* (and, in 2D, the exact
    bilinear cell certificate) in :mod:`dvfopt.jacobian.injectivity_radius`
    — see that module's docstring for the math, references, and caveats:
    the radius is an estimate, not a certificate, and it is
    orientation-blind (a uniformly reflected region scores clean), so read
    these numbers alongside :func:`fold_stats`, never instead of it.
    """

    min_radius: float  # smallest IFT radius estimate; saturates at max_window
    frac_subpixel: float  # fraction of samples with radius estimate < 1
    cell_min_jdet: Optional[float]  # 2D only: min bilinear cell Jdet (None in 3D)
    n_cells_nonpos: Optional[int]  # 2D only: folded cells under the bilinear model
    max_window: int  # ladder cap used — min_radius == max_window means "at least this"


def injectivity_stats(phi, max_window: int = 8) -> InjectivityStats:
    """Neighbourhood-injectivity diagnostics for a 2D field or 3D volume.

    Accepts the layouts the constraint paths accept: ``(2, H, W)``
    ``[dy, dx]``, ``(3, H, W)`` / ``(3, 1, H, W)`` (dz dropped), and
    true-3D ``(3, D>1, H, W)`` ``[dz, dy, dx]``. 2D fields also get the
    exact bilinear cell certificate; 3D volumes report the radius map
    only — the trilinear Jdet is not multi-affine, so sub-voxel 3D folds
    are the simplex (3D) constraint family's job (:func:`constraint_fold_stats`).
    See :class:`InjectivityStats` for how (not) to read the numbers.
    """
    from dvfopt.jacobian.injectivity_radius import (
        cell_min_jdet_2d,
        ift_radius_2d,
        ift_radius_3d,
    )
    from dvfopt.validation import validate_dvf

    phi = np.asarray(phi)
    if _is_volume(phi):
        vol = validate_dvf(phi, dim=3)
        r = ift_radius_3d(vol, max_window=max_window)
        cell_min = None
    else:
        phi2 = validate_dvf(phi, dim=2)
        r = ift_radius_2d(phi2, max_window=max_window)
        cell_min = cell_min_jdet_2d(phi2)
    return InjectivityStats(
        min_radius=float(r.min()),
        frac_subpixel=float((r < 1.0).mean()),
        cell_min_jdet=None if cell_min is None else float(cell_min.min()),
        n_cells_nonpos=None if cell_min is None else int((cell_min <= 0).sum()),
        max_window=int(max_window),
    )


def _change_jdet(phi: np.ndarray) -> np.ndarray:
    """Central-difference Jdet of ``phi``, picking the 2D/3D/per-slice path by shape."""
    from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D

    if phi.ndim == 4 and phi.shape[0] == 3 and phi.shape[1] > 1:
        return np.asarray(jacobian_det3D(phi))
    if phi.ndim == 4:  # (3, 1, H, W) or (2, 1, H, W): per-slice 2D
        return np.stack(
            [
                np.asarray(jacobian_det2D(np.stack([phi[-2, z], phi[-1, z]])))
                for z in range(phi.shape[1])
            ]
        )
    return np.asarray(jacobian_det2D(np.stack([phi[-2], phi[-1]])))


def field_change_stats(phi_in, phi_out, *, moved_eps: float = 1e-6, moved_px: float = 0.5) -> dict:
    """Grid-size-independent measures of how much a correction changed a field.

    Raw L1/L2 move sums scale with voxel count and have no physical reading;
    these are in px (or dimensionless for the Jdet change) and answer "how
    much did the correction touch the field, typically and at worst" —
    independent of grid resolution.

    Keys
    ----
    move_med_px, move_p95_px, move_max_px
        Per-voxel displacement change ``||phi_out - phi_in||``, over ALL voxels.
    move_med_moved_px, move_p95_moved_px
        The same, restricted to voxels with change ``> moved_eps`` (``0.0``
        when none moved).
    moved_frac, moved_frac_0p5px
        Fraction of voxels with change ``> moved_eps`` / ``> moved_px``.
    jdet_change_med, jdet_change_p95, jdet_change_max
        ``|J_out - J_in|`` per cell, central-difference Jdet
        (:func:`dvfopt.jacobian.numpy_jdet.jacobian_det2D`/``jacobian_det3D``,
        picked by shape: true-3D ``(3, D>1, H, W)`` uses ``jacobian_det3D``;
        ``(3|2, 1, H, W)`` runs ``jacobian_det2D`` per slice on the last two
        channels; ``(2, H, W)`` runs it directly).

    ``phi_in``/``phi_out`` must have the same shape; channels are the leading
    axis (``dz``/``dy``/``dx`` or ``dy``/``dx``).
    """
    a = np.asarray(phi_in, dtype=np.float64)
    b = np.asarray(phi_out, dtype=np.float64)
    d = np.sqrt(((b - a) ** 2).sum(axis=0))
    moved = d > moved_eps
    dm = d[moved]
    dj = np.abs(_change_jdet(b) - _change_jdet(a))
    return dict(
        move_med_px=float(np.median(d)),
        move_p95_px=float(np.percentile(d, 95)),
        move_max_px=float(d.max()),
        move_med_moved_px=float(np.median(dm)) if dm.size else 0.0,
        move_p95_moved_px=float(np.percentile(dm, 95)) if dm.size else 0.0,
        moved_frac=float(moved.mean()),
        moved_frac_0p5px=float((d > moved_px).mean()),
        jdet_change_med=float(np.median(dj)),
        jdet_change_p95=float(np.percentile(dj, 95)),
        jdet_change_max=float(dj.max()),
    )
