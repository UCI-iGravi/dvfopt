"""Hybrid overlapping-Schwarz + cluster solver for 2-triangle correction.

Promoted from the experiment in
``notebooks/experiments/overlapping_tiles_schwarz.ipynb``, which
demonstrated 100% feasibility on the 7 dense-fold B0039 slices the main
cluster runner couldn't crack.

Algorithm — each outer iteration:

1. Detect connected components of the cell-fold mask (after a small
   ``merge_dilation`` so near-touching folds merge).
2. For each component, route by bbox size:
     * **large** (span > ``large_span`` or area > ``large_area``) →
       overlapping-tile Schwarz: tile the component's bbox into
       ``tile``-cell tiles with ``overlap`` cells overlap. Multiplicative
       (Gauss-Seidel) sweeps; each tile solved with frozen edges. The
       overlap propagates corrections between tiles across sweeps.
     * **small** → normal frozen-edge crop with a per-cell pad boost
       (the pad grows on next iteration when the crop stalls).
3. Re-detect components, repeat until ``n_neg = 0`` or ``max_outer``.

Once Schwarz reduces a big component to sparse residuals, the outer
loop sees those residuals as small components and routes them to the
normal solver. Schwarz tiling is therefore a *fallback for the
large-component case only*, not a replacement for normal per-cluster
SLSQP.

Both branches call :func:`dvfopt.core._cluster_2tri.solve_cluster_2tri_2d`
for the actual SLSQP work.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt._logging import log_info
from dvfopt.core._cluster_2tri import solve_cluster_2tri_2d
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _fold_components(phi, feas_lb, merge_dilation=1):
    """Connected components of the cell-fold mask.

    A cell is "folded" when ``min(T1, T2) < feas_lb`` — the same
    threshold-consistent predicate :func:`solve_cluster_2tri_2d` gates
    feasibility on (``feas_lb = threshold - err_tol``), so sub-threshold
    cells cannot survive as "converged".

    Returns a list of cell-coord bboxes ``(cy0, cy1, cx0, cx1)`` in the
    ``find_objects`` half-open convention (``cy1`` is exclusive).
    """
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    fold = np.minimum(T1, T2) < feas_lb
    if not fold.any():
        return []
    mask = binary_dilation(fold, iterations=merge_dilation) if merge_dilation > 0 else fold
    labels, _ = cc_label(mask)
    comps = []
    for sl in find_objects(labels):
        if sl is None:
            continue
        comps.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return comps


def _make_tiles(bbox, H, W, tile, overlap):
    """Overlapping cell-coord tiles covering ``bbox=(cy0, cy1, cx0, cx1)``."""
    cy0, cy1, cx0, cx1 = bbox
    stride = max(1, tile - overlap)
    out = set()
    for y0 in range(cy0, max(cy0 + 1, cy1), stride):
        for x0 in range(cx0, max(cx0 + 1, cx1), stride):
            y1 = min(y0 + tile, H - 1)
            x1 = min(x0 + tile, W - 1)
            y0c = max(0, y1 - tile)
            x0c = max(0, x1 - tile)
            if (y1 - y0c) >= 4 and (x1 - x0c) >= 4:
                out.add((y0c, y1, x0c, x1))
    return sorted(out)


def _solve_crop(phi, phi_anchor, y0, y1, x0, x1, *, threshold, eps_l1, l2_passes, l2_iter, l1_iter):
    """Solve one crop with a rectangular frozen-edge interior mask.

    Splices the interior-corner updates back into ``phi`` in place.
    Returns the per-cluster info dict from :func:`solve_cluster_2tri_2d`.
    """
    sy, sx = y1 - y0, x1 - x0
    if sy < 4 or sx < 4:
        return {'feasible': False}
    im = np.zeros((sy + 1, sx + 1), dtype=bool)
    im[1:-1, 1:-1] = True
    phi_win = phi[:, y0 : y1 + 1, x0 : x1 + 1].copy()
    anc_win = phi_anchor[:, y0 : y1 + 1, x0 : x1 + 1].copy()
    phi_out, info = solve_cluster_2tri_2d(
        phi_win,
        anc_win,
        im,
        threshold=threshold,
        eps_l1=eps_l1,
        l2_max_passes=l2_passes,
        l2_max_iter=l2_iter,
        l1_max_iter=l1_iter,
    )
    if info.get('feasible'):
        yy, xx = np.where(im)
        phi[:, y0 + yy, x0 + xx] = phi_out[:, yy, xx]
    return info


def _solve_region_schwarz(
    phi, phi_anchor, bbox, *, threshold, feas_lb, eps_l1, tile, overlap, max_sweeps
):
    """Overlapping-tile multiplicative Schwarz on one large component's bbox.

    Light per-tile budget — Schwarz relies on repeated sweeps to
    propagate corrections through the overlap regions.
    """
    H, W = phi.shape[1], phi.shape[2]
    cy0, cy1, cx0, cx1 = bbox
    for sweep in range(max_sweeps):
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        fold = np.minimum(T1, T2) < feas_lb
        sub = np.zeros_like(fold)
        sub[cy0:cy1, cx0:cx1] = fold[cy0:cy1, cx0:cx1]
        if not sub.any():
            return
        ys, xs = np.where(sub)
        rbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        tiles = _make_tiles(rbox, H, W, tile, overlap)
        if sweep % 2 == 1:
            tiles = tiles[::-1]
        for y0, y1, x0, x1 in tiles:
            phi_win = phi[:, y0 : y1 + 1, x0 : x1 + 1]
            t1w, t2w = _triangle_areas_2d(phi_win[0], phi_win[1])
            if not (np.minimum(t1w, t2w) < feas_lb).any():
                continue
            _solve_crop(
                phi,
                phi_anchor,
                y0,
                y1,
                x0,
                x1,
                threshold=threshold,
                eps_l1=eps_l1,
                l2_passes=4,
                l2_iter=30,
                l1_iter=40,
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def iterative_2d_tri_schwarz(
    deformation_2hw: np.ndarray,
    *,
    threshold: Optional[float] = None,
    eps_l1: float = 1e-4,
    max_outer: int = 30,
    large_span: int = 40,
    large_area: int = 1500,
    tile: int = 16,
    overlap: int = 4,
    schwarz_max_sweeps: int = 6,
    l2_passes: int = 12,
    l2_iter: int = 80,
    l1_iter: int = 120,
    merge_dilation: int = 1,
    verbose: int = 1,
    record_history: bool = False,
):
    """Hybrid Schwarz + per-cluster SLSQP for 2-triangle 2D correction.

    Designed for the dense-fold slices the plain per-cluster runner
    can't crack (the 7 stuck B0039 slices, e.g. ``z=12``, ``z=30``).

    Parameters
    ----------
    deformation_2hw : ndarray
        ``(2, H, W)`` ``[dy, dx]`` field, or ``(3, 1, H, W)``.
    threshold : float, optional
        Lower bound for both triangle areas. Defaults to
        ``DEFAULT_PARAMS['threshold']`` (0.01).
    eps_l1 : float
        L1 polish smoothing constant.
    max_outer : int
        Max outer iterations. Each outer iteration redetects components.
    large_span : int
        Component bbox longer-axis (cells) above which routing switches
        from per-cluster SLSQP to Schwarz tiling.
    large_area : int
        Component bbox area (cells) above which routing switches.
    tile : int
        Schwarz tile size (cells).
    overlap : int
        Schwarz tile overlap (cells).
    schwarz_max_sweeps : int
        Max sweeps per Schwarz call.
    l2_passes : int
        Max L2-SLSQP passes per cluster (small-component branch).
    l2_iter, l1_iter : int
        SLSQP ``maxiter`` per pass.
    merge_dilation : int
        Cell dilation used when detecting components (near-touching
        folds merge into one component).
    verbose : int
    record_history : bool
        If True, returns ``(phi, history)``.

    Returns
    -------
    phi_corrected : ndarray, shape ``(2, H, W)``
    history : list of dict, only if ``record_history=True``
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    # Fold-detection predicate consistent with the per-cluster solver's
    # feasibility gate (and the SLSQP constraint lb=threshold): a cell
    # with min area in (0, threshold) is still a fold. Without this, the
    # splice-on-feasible logic and the ``<= 0`` re-detection disagreed and
    # sub-threshold cells survived as "converged".
    feas_lb = threshold - DEFAULT_PARAMS['err_tol']

    if deformation_2hw.ndim == 4:
        if deformation_2hw.shape[0] == 3:
            deformation_2hw = np.stack([deformation_2hw[1, 0], deformation_2hw[2, 0]])
        else:
            deformation_2hw = deformation_2hw[:, 0]

    # SLSQP requires float64; many real-data DVFs are stored as float32.
    if deformation_2hw.dtype != np.float64:
        deformation_2hw = deformation_2hw.astype(np.float64)

    H, W = deformation_2hw.shape[1], deformation_2hw.shape[2]
    phi = deformation_2hw.copy()
    phi_anchor = deformation_2hw.copy()
    pad_boost = np.zeros((H - 1, W - 1), dtype=int)
    history = []

    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    init_n_neg = int((feas_lb > T1).sum() + (feas_lb > T2).sum())
    init_min = float(min(T1.min(), T2.min()))
    if verbose >= 1:
        log_info(
            f'[2d-tri-schwarz init] grid {H}x{W}  threshold={threshold}  '
            f'n_neg={init_n_neg}  min_tri={init_min:+.4f}'
        )

    t_start = time.time()
    converged = False
    final_n_neg = init_n_neg
    final_min = init_min

    for outer in range(1, max_outer + 1):
        comps = _fold_components(phi, feas_lb, merge_dilation=merge_dilation)
        if not comps:
            converged = True
            break
        t_outer = time.time()
        n_large = n_small = 0
        for cy0, cy1, cx0, cx1 in comps:
            span = max(cy1 - cy0, cx1 - cx0)
            area = (cy1 - cy0) * (cx1 - cx0)
            if span > large_span or area > large_area:
                _solve_region_schwarz(
                    phi,
                    phi_anchor,
                    (cy0, cy1, cx0, cx1),
                    threshold=threshold,
                    feas_lb=feas_lb,
                    eps_l1=eps_l1,
                    tile=tile,
                    overlap=overlap,
                    max_sweeps=schwarz_max_sweeps,
                )
                n_large += 1
            else:
                boost = int(pad_boost[cy0:cy1, cx0:cx1].max())
                pad = 1 + boost
                y0 = max(0, cy0 - pad)
                y1 = min(H - 1, cy1 + pad)
                x0 = max(0, cx0 - pad)
                x1 = min(W - 1, cx1 + pad)
                info = _solve_crop(
                    phi,
                    phi_anchor,
                    y0,
                    y1,
                    x0,
                    x1,
                    threshold=threshold,
                    eps_l1=eps_l1,
                    l2_passes=l2_passes,
                    l2_iter=l2_iter,
                    l1_iter=l1_iter,
                )
                if info.get('feasible'):
                    pad_boost[cy0:cy1, cx0:cx1] = 0
                else:
                    pad_boost[cy0:cy1, cx0:cx1] += 1
                n_small += 1

        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        final_n_neg = int((feas_lb > T1).sum() + (feas_lb > T2).sum())
        final_min = float(min(T1.min(), T2.min()))
        elapsed = time.time() - t_outer
        if record_history:
            history.append(
                dict(
                    outer=outer,
                    n_neg=final_n_neg,
                    min_tri=final_min,
                    n_components=len(comps),
                    n_large=n_large,
                    n_small=n_small,
                    wall_s=elapsed,
                )
            )
        if verbose >= 1:
            log_info(
                f'  outer {outer:2d}: n_neg={final_n_neg:5d}  '
                f'min_tri={final_min:+.4f}  '
                f'comps={len(comps):3d} (large={n_large} small={n_small})  '
                f'({elapsed:.1f}s)'
            )
        if final_n_neg == 0:
            converged = True
            break

    if verbose >= 1:
        status = 'converged' if converged else f'max_outer={max_outer} reached'
        log_info(
            f'[2d-tri-schwarz done] {status}  '
            f'n_neg {init_n_neg} -> {final_n_neg}  '
            f'min_tri {init_min:+.4f} -> {final_min:+.4f}  '
            f'total_t={time.time() - t_start:.1f}s'
        )

    if record_history:
        return phi, history
    return phi
