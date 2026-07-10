"""Harmonic / Laplacian extension of the warped grid over fold cores.

Algorithm
---------
1.  Find every connected component of cells with ``min(T1, T2) < threshold``.
2.  Dilate each by ``ring_pad`` cells so the boundary lies in a fold-free
    annulus. The interior corners are unknowns; the ring corners are
    Dirichlet data (kept exactly equal to ``phi_in``).
3.  Solve a Laplacian on the warped coordinate fields ``(def_y, def_x)``
    over the union of interior corners. This yields a harmonic map.
4.  Radó-Kneser-Choquet: a harmonic map onto a CONVEX boundary curve
    is a diffeomorphism. The dilated ring is rarely convex, so step 5.
5.  If any cell in the patch still has ``min(T1, T2) < threshold``, dilate
    the patch by one cell and retry, up to ``max_grow_iters``.
6.  Stop when feasibility is reached or growth fails. Either way, return
    the best-effort patch substituted into ``phi_in``.

Why it works
------------
The wall in SLSQP-land is a degeneracy of the constraint ACTIVE-SET when
many ``T_i`` are crowded at zero. A harmonic extension never even
considers the constraint; it just produces the unique smooth field with
the given boundary values, and on a sufficiently nice boundary that
field is automatically diffeomorphic.

The cost of getting around the wall is L2 deviation from the input
*inside the fold core only* — the harmonic patch is the smoothest
possible reconstruction, but it does not respect any feature in the
interior, only the boundary ring.

Promoted from ``notebooks/experiments/wall_breakers/methods/m02_harmonic_extension.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from scipy.ndimage import label as cc_label

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _fold_cell_mask(phi: np.ndarray, threshold: float) -> np.ndarray:
    """``(H-1, W-1)`` boolean mask of cells with ``min(T1, T2) < threshold``."""
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return np.minimum(T1, T2) < threshold


def _cells_to_corner_mask(cell_mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Promote a ``(H-1, W-1)`` cell mask to a ``(H, W)`` corner mask."""
    corner = np.zeros((H, W), dtype=bool)
    cy, cx = np.where(cell_mask)
    corner[cy, cx] = True
    corner[cy, cx + 1] = True
    corner[cy + 1, cx] = True
    corner[cy + 1, cx + 1] = True
    return corner


def _solve_laplace_patch(values: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    """Solve the 5-point Laplacian on ``values`` with ``free_mask`` marking
    interior unknowns; fixed corners are Dirichlet data from ``values``.

    Returns a copy of ``values`` with free corners replaced by the
    harmonic extension. Fixed corners are untouched.

    Vectorised: builds the sparse matrix without a Python loop over the
    free nodes. For a patch of ~300 free corners this is ~10x faster
    than the per-node loop the function used to use.
    """
    H, W = values.shape
    out = values.copy()
    if not free_mask.any():
        return out

    free_idx = np.argwhere(free_mask)
    n = len(free_idx)
    yi, xi = free_idx[:, 0], free_idx[:, 1]
    idx_of = np.full((H, W), -1, dtype=np.int64)
    idx_of[yi, xi] = np.arange(n)

    # Per-node diagonal starts at 4 (full 5-point stencil) and is reduced
    # by 1 for each off-grid neighbour (homogeneous Neumann at the image
    # edge).
    diag = np.full(n, 4.0)
    rhs = np.zeros(n)
    off_rows = []
    off_cols = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy = yi + dy
        xx = xi + dx
        # In-bounds mask
        in_bounds = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        oob = ~in_bounds
        diag[oob] -= 1.0

        if not in_bounds.any():
            continue
        # Among in-bounds neighbours, split by free vs Dirichlet.
        k_in = np.where(in_bounds)[0]
        yy_in = yy[k_in]
        xx_in = xx[k_in]
        is_free = free_mask[yy_in, xx_in]

        # Off-diagonal contribution for free->free edges.
        k_free = k_in[is_free]
        if k_free.size:
            off_rows.append(k_free)
            off_cols.append(idx_of[yy[k_free], xx[k_free]])

        # RHS contribution from Dirichlet neighbours.
        k_dir = k_in[~is_free]
        if k_dir.size:
            np.add.at(rhs, k_dir, values[yy[k_dir], xx[k_dir]])

    if off_rows:
        off_rows_arr = np.concatenate(off_rows)
        off_cols_arr = np.concatenate(off_cols)
        rows = np.concatenate([np.arange(n), off_rows_arr])
        cols = np.concatenate([np.arange(n), off_cols_arr])
        data = np.concatenate([diag, -np.ones(off_rows_arr.size)])
    else:
        rows = np.arange(n)
        cols = np.arange(n)
        data = diag

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    sol = spla.spsolve(A, rhs)
    out[yi, xi] = sol
    return out


def harmonic_extension_2d(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    ring_pad: int = 2,
    max_grow_iters: int = 6,
    merge_dilation: int = 2,
    margin: float = 0.0,
    record_history: bool = False,
):
    """Harmonic extension over each (dilated) fold component.

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)``
        Field with channels ``[dy, dx]``.
    threshold : float
        Triangle-area lower bound used to detect fold cells.
    ring_pad : int
        Initial cell-dilation of each fold component before the boundary
        ring is locked.
    max_grow_iters : int
        If the harmonic patch still has folds, dilate by one more cell
        and retry up to this many times.
    merge_dilation : int
        Cell dilation used only for grouping cells into components (so
        two nearly-touching cores share one patch). Does NOT itself free
        those cells.
    margin : float
        Minimum acceptable ``min(T1, T2)`` above ``threshold``.
    record_history : bool
        If True, returns ``(phi, info)`` instead of just ``phi``.

    Returns
    -------
    dict with keys ``phi_out`` (corrected field) and ``info`` (patch records).
    """
    from dvfopt._defaults import DEFAULT_PARAMS

    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    if merge_dilation < 0:
        raise ValueError(f'merge_dilation must be >= 0, got {merge_dilation}')
    if ring_pad < 0:
        raise ValueError(f'ring_pad must be >= 0, got {ring_pad}')
    H, W = phi_in.shape[1], phi_in.shape[2]
    accept_thr = threshold + margin

    cell_fold = _fold_cell_mask(phi_in, threshold)
    if not cell_fold.any():
        info = {'patches': 0, 'reason': 'already-feasible'}
        phi_out = phi_in.copy()
        return (phi_out, info) if record_history else phi_out

    # scipy treats iterations < 1 as "repeat until convergence" (fills
    # the grid), so only dilate for merge_dilation >= 1.
    grouped = (
        binary_dilation(cell_fold, iterations=merge_dilation)
        if merge_dilation >= 1
        else cell_fold
    )
    labels, n_comp = cc_label(grouped, structure=generate_binary_structure(2, 2))

    phi_out = phi_in.copy()
    patch_records = []

    for comp_id in range(1, n_comp + 1):
        comp_cells = (labels == comp_id) & cell_fold
        if not comp_cells.any():
            continue
        cur_cells = comp_cells.copy()
        last_min = -np.inf
        for grow in range(max_grow_iters + 1):
            # binary_dilation(iterations=0) would "iterate to convergence"
            # (fill the grid); ring_pad=0 with grow=0 must mean "no
            # dilation this round".
            iters = ring_pad + grow
            patch_cells = (
                binary_dilation(cur_cells, iterations=iters) if iters >= 1 else cur_cells
            )
            interior_cells = binary_erosion(patch_cells, iterations=1)
            free_mask = _cells_to_corner_mask(interior_cells, H, W)

            new_dy = _solve_laplace_patch(phi_out[0], free_mask)
            new_dx = _solve_laplace_patch(phi_out[1], free_mask)
            phi_trial = np.stack([new_dy, new_dx])

            T1, T2 = _triangle_areas_2d(phi_trial[0], phi_trial[1])
            patch_T_min = np.minimum(T1, T2)[patch_cells].min() if patch_cells.any() else np.inf

            if patch_T_min >= accept_thr:
                phi_out = phi_trial
                patch_records.append(
                    {
                        'comp_id': comp_id,
                        'grow': grow,
                        'n_cells': int(patch_cells.sum()),
                        'patch_T_min': float(patch_T_min),
                    }
                )
                break

            last_min = patch_T_min
            cur_cells = binary_dilation(cur_cells, iterations=1)
        else:
            phi_out = phi_trial
            patch_records.append(
                {
                    'comp_id': comp_id,
                    'grow': max_grow_iters,
                    'failed': True,
                    'patch_T_min': float(last_min),
                    'n_cells': int(patch_cells.sum()),
                }
            )

    info = {
        'patches': len(patch_records),
        'n_components': int(n_comp),
        'records_first5': patch_records[:5],
    }
    return (phi_out, info) if record_history else phi_out
