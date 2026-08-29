"""Fold-morphology row for one 2D field — the columns of the paper's §4.2 table.

Three metrics on the same field, per cell where possible:

* ``jdet``     central-difference Jacobian determinant, per pixel;
* ``simplex``  exact Jacobian of the piecewise-linear interpolant on the fixed
               BL-TR diagonal (2 triangles / cell) — a certificate for that
               interpolant; reported per cell (min of the 2) and per triangle;
* ``bilinear`` exact sub-pixel certificate for the bilinear interpolant
               (``cell_min_jdet_2d``); ``bilinear_only`` = cells it folds that
               the simplex metric passes — the column nobody reports.

Clusters are 8-connected components of the simplex per-cell fold mask.

Scales: ``jdet`` and ``bilinear`` are Jacobians (1.0 on the identity);
``simplex`` values are raw triangle areas exactly as ``dvfopt``'s constraints
report them (0.5 on the identity — the library's ``threshold`` applies to
that scale), so the counts here agree with ``dvfopt.metrics``.
"""

import numpy as np
from scipy import ndimage

from dvfopt.core.primitives.tri import tri_areas_flat
from dvfopt.jacobian.injectivity_radius import cell_min_jdet_2d
from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.metrics import fold_stats

COLUMNS = [
    'H',
    'W',
    'n_cells',
    'disp_max',
    'disp_mean',
    'jdet_neg_px',
    'jdet_below_px',
    'jdet_min',
    'simplex_neg_tris',
    'simplex_below_tris',
    'simplex_neg_cells',
    'simplex_below_cells',
    'simplex_min',
    'simplex_severity',
    'bilinear_neg_cells',
    'bilinear_only_cells',
    'bilinear_min',
    'fold_frac',
    'n_clusters',
    'cluster_area_med',
    'cluster_area_max',
]


def morphology(phi, threshold=0.01):
    """Return the :data:`COLUMNS` dict for a ``(3, 1, H, W)`` field."""
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 4 or phi.shape[0] != 3 or phi.shape[1] != 1 or min(phi.shape[-2:]) < 2:
        raise ValueError(f'expected (3, 1, H, W) with H, W >= 2, got {phi.shape}')
    H, W = phi.shape[-2:]
    dy, dx = phi[1, 0], phi[2, 0]
    n_cells = (H - 1) * (W - 1)

    jdet = jacobian_det2D(phi[1:])[0]
    tri = tri_areas_flat(np.concatenate([dy.ravel(), dx.ravel()]), H, W)  # DY_FIRST pack
    simplex_cell = np.minimum(tri[:n_cells], tri[n_cells:]).reshape(H - 1, W - 1)
    bil = cell_min_jdet_2d(phi[1:])

    js, ts = fold_stats(jdet, threshold), fold_stats(tri, threshold)
    ss, bs = fold_stats(simplex_cell, threshold), fold_stats(bil, threshold)

    lab, n = ndimage.label(simplex_cell <= 0, structure=np.ones((3, 3), dtype=int))
    areas = np.bincount(lab.ravel())[1:] if n else np.zeros(0, dtype=int)
    disp = np.hypot(dy, dx)
    return dict(
        H=H,
        W=W,
        n_cells=n_cells,
        disp_max=float(disp.max()),
        disp_mean=float(disp.mean()),
        jdet_neg_px=js.n_neg,
        jdet_below_px=js.n_below,
        jdet_min=js.min_val,
        simplex_neg_tris=ts.n_neg,
        simplex_below_tris=ts.n_below,  # cohort_benchmark's "simplex folds" (< threshold)
        simplex_neg_cells=ss.n_neg,
        simplex_below_cells=ss.n_below,
        simplex_min=ss.min_val,
        simplex_severity=ss.neg_volume,
        bilinear_neg_cells=bs.n_neg,
        bilinear_only_cells=int(((bil <= 0) & (simplex_cell > 0)).sum()),
        bilinear_min=bs.min_val,
        fold_frac=ss.n_neg / n_cells,
        n_clusters=int(n),
        cluster_area_med=float(np.median(areas)) if n else 0.0,
        cluster_area_max=int(areas.max()) if n else 0,
    )
