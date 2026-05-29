"""Harmonic / Laplacian extension of the warped grid over fold cores.

Algorithm
---------
1.  Find every connected component of cells with min(T1, T2) <= threshold.
2.  Dilate each by ``ring_pad`` cells so the boundary lies in a fold-free
    annulus. The interior corners are unknowns; the ring corners are
    Dirichlet data (kept exactly equal to phi_in).
3.  Solve a Laplacian on the warped *coordinate* fields (def_y, def_x)
    over the union of interior corners. This yields a harmonic map.
4.  Radó-Kneser-Choquet: a harmonic map onto a CONVEX boundary curve
    is a diffeomorphism. The dilated ring is rarely convex, so step 5.
5.  If any cell in the patch still has min(T1, T2) < threshold, dilate
    the patch by one cell and retry, up to ``max_grow_iters``.
6.  Stop when feasibility is reached or growth fails. Either way, return
    the best-effort patch substituted into phi_in.

Why it works
------------
The wall in SLSQP-land is a degeneracy of the constraint ACTIVE-SET when
many T_i are crowded at zero. A harmonic extension never even considers
the constraint; it just produces the unique smooth field with the given
boundary values, and on a sufficiently nice boundary that field is
automatically diffeomorphic.

The cost of getting around the wall is L2 deviation from the input
*inside the fold core only* -- the harmonic patch is the smoothest
possible reconstruction, but it does not respect any feature in the
interior, only the boundary ring.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import (
    label as cc_label, binary_dilation, find_objects, generate_binary_structure)

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

NAME = 'harmonic'
DESCRIPTION = 'Harmonic extension over dilated fold cores (Dirichlet from fold-free ring)'


def _fold_cell_mask(phi: np.ndarray, threshold: float) -> np.ndarray:
    """``(H-1, W-1)`` boolean mask of cells with min(T1, T2) < threshold."""
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return np.minimum(T1, T2) < threshold


def _cells_to_corner_mask(cell_mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Promote a ``(H-1, W-1)`` cell mask to a ``(H, W)`` corner mask: every
    corner of every flagged cell becomes a free unknown."""
    corner = np.zeros((H, W), dtype=bool)
    cy, cx = np.where(cell_mask)
    corner[cy,     cx]     = True
    corner[cy,     cx + 1] = True
    corner[cy + 1, cx]     = True
    corner[cy + 1, cx + 1] = True
    return corner


def _solve_laplace_patch(values: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    """Solve the 5-point Laplacian on ``values`` with ``free_mask`` indicating
    interior unknowns; fixed corners are Dirichlet data taken from ``values``.

    Returns a new array of the same shape with free corners replaced by the
    harmonic extension and fixed corners untouched.
    """
    H, W = values.shape
    out = values.copy()
    if not free_mask.any():
        return out

    free_idx = np.argwhere(free_mask)
    n = len(free_idx)
    idx_of = np.full((H, W), -1, dtype=np.int64)
    idx_of[free_idx[:, 0], free_idx[:, 1]] = np.arange(n)

    rows, cols, data = [], [], []
    rhs = np.zeros(n)
    for k, (y, x) in enumerate(free_idx):
        rows.append(k); cols.append(k); data.append(4.0)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                # treat OOB as Dirichlet = local value (zero-flux-ish)
                rhs[k] += values[y, x]
                continue
            if free_mask[yy, xx]:
                rows.append(k); cols.append(idx_of[yy, xx]); data.append(-1.0)
            else:
                rhs[k] += values[yy, xx]

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    sol = spla.spsolve(A, rhs)
    out[free_idx[:, 0], free_idx[:, 1]] = sol
    return out


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          ring_pad: int = 2,
          max_grow_iters: int = 6,
          merge_dilation: int = 2,
          margin: float = 0.0) -> dict:
    """Harmonic extension over each (dilated) fold component.

    Parameters
    ----------
    ring_pad : initial cell-dilation of each fold component before the
        boundary ring is locked.
    max_grow_iters : if the harmonic patch still has folds, dilate by one
        more cell and retry up to this many times.
    merge_dilation : cell dilation used only for grouping cells into
        components (so two nearly-touching cores share one patch). Does
        NOT itself free those cells.
    margin : minimum acceptable min(T1, T2) above threshold; pushing
        margin > 0 forces extra iterations to bury the constraint deeper
        in the feasible interior.
    """
    H, W = phi_in.shape[1], phi_in.shape[2]
    accept_thr = threshold + margin

    # Detect ALL initial fold cells and group them into components.
    cell_fold = _fold_cell_mask(phi_in, threshold)
    if not cell_fold.any():
        return {'phi_out': phi_in.copy(),
                'info': {'patches': 0, 'reason': 'already-feasible'}}

    grouped = binary_dilation(cell_fold, iterations=merge_dilation)
    labels, n_comp = cc_label(grouped, structure=generate_binary_structure(2, 2))

    phi_out = phi_in.copy()
    patch_records = []

    for comp_id in range(1, n_comp + 1):
        comp_cells = (labels == comp_id) & cell_fold  # restrict to ACTUAL folds
        if not comp_cells.any():
            continue
        cur_cells = comp_cells.copy()
        last_min = -np.inf
        for grow in range(max_grow_iters + 1):
            patch_cells = binary_dilation(cur_cells, iterations=ring_pad + grow)
            # Free-corner mask in (H, W): every corner of every patch cell.
            free_mask = _cells_to_corner_mask(patch_cells, H, W)
            # ... but corners on the patch boundary that are NOT in any
            # patch cell stay fixed. By construction _cells_to_corner_mask
            # already encodes only patch corners; we want the OUTER ring
            # to be Dirichlet, so subtract corners that lie on the ring.
            # Ring = corners of patch cells whose 4-neighbour cells include
            # at least one non-patch cell. Equivalently: erode by 1 cell.
            from scipy.ndimage import binary_erosion
            interior_cells = binary_erosion(patch_cells, iterations=1)
            interior_corner = _cells_to_corner_mask(interior_cells, H, W)
            # Only interior corners become unknowns; ring corners stay fixed.
            free_mask = interior_corner

            # Solve harmonic extension for dy and dx independently.
            new_dy = _solve_laplace_patch(phi_out[0], free_mask)
            new_dx = _solve_laplace_patch(phi_out[1], free_mask)
            phi_trial = np.stack([new_dy, new_dx])

            # Check feasibility over the entire patch (and the ring, which
            # is unchanged and must still be feasible -- it was if the ring
            # was originally fold-free).
            T1, T2 = _triangle_areas_2d(phi_trial[0], phi_trial[1])
            patch_T_min = np.minimum(T1, T2)[patch_cells].min() if patch_cells.any() else np.inf

            if patch_T_min >= accept_thr:
                phi_out = phi_trial
                patch_records.append({
                    'comp_id': comp_id, 'grow': grow,
                    'n_cells': int(patch_cells.sum()),
                    'patch_T_min': float(patch_T_min),
                })
                break

            last_min = patch_T_min
            cur_cells = binary_dilation(cur_cells, iterations=1)
        else:
            # ran out of grow iterations; accept best-effort
            phi_out = phi_trial
            patch_records.append({
                'comp_id': comp_id, 'grow': max_grow_iters, 'failed': True,
                'patch_T_min': float(last_min),
                'n_cells': int(patch_cells.sum()),
            })

    return {'phi_out': phi_out,
            'info': {'patches': len(patch_records),
                     'n_components': int(n_comp),
                     'records_first5': patch_records[:5]}}
