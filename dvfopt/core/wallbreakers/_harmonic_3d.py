"""Harmonic / Laplacian extension over 3D fold cores (m10-3D).

3D analogue of :mod:`dvfopt.core.wallbreakers._harmonic` (the 2D m02
harmonic extension that the 2D m10 wallbreaker builds on).

Algorithm
---------

1. Find every connected component of voxel cells where some tetrahedron
   has signed volume ``< threshold`` (``six_tet_fold_classification > 0``).
2. Dilate each by ``ring_pad`` voxels so the boundary lies in a fold-free
   shell. Interior corners are unknowns; ring corners are Dirichlet data
   (kept exactly equal to ``phi_in``).
3. Solve a 7-point Laplacian on each warped coordinate channel
   ``(def_z, def_y, def_x)`` over the union of interior corners.
4. If any cell in the patch still has a folded tet, dilate the patch by
   one more voxel and retry, up to ``max_grow_iters``.

Why it works
------------

Radó-Kneser-Choquet generalises to higher dimensions for *convex*
boundaries: a harmonic map onto a convex shell is a diffeomorphism. The
dilated patch is rarely strictly convex in 3D either, so step 4 is the
practical escape hatch.

The cost is L2 deviation from the input *inside the fold core only* —
the harmonic patch is the smoothest possible reconstruction, but it
doesn't respect any feature in the interior, only the boundary shell.

This is a focused first cut. Compared to the 2D m10 pipeline (harmonic
→ ALM → polish), only the harmonic step is ported. m14 (refinement /
repair) and m14-Schwarz (cluster decomposition) are still 2D-only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from scipy.ndimage import label as cc_label

from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


def _fold_cell_mask_3d(phi: np.ndarray, threshold: float) -> np.ndarray:
    """``(D-1, H-1, W-1)`` boolean mask of cells with any tet below ``threshold``."""
    # Fused per-cube min kernel — identical to
    # ``six_tet_volumes_3d(phi).min(axis=0)`` without materialising the
    # full (6, Dc, Hc, Wc) array.
    return six_tet_min_volume_3d(phi) < threshold


def _cells_to_corner_mask_3d(cell_mask: np.ndarray, D: int, H: int, W: int) -> np.ndarray:
    """Promote a ``(D-1, H-1, W-1)`` cell mask to a ``(D, H, W)`` corner mask.

    Each cell at ``(cz, cy, cx)`` has 8 corners at the standard offsets
    ``(oz, oy, ox) in {0, 1}^3``.
    """
    corner = np.zeros((D, H, W), dtype=bool)
    cz, cy, cx = np.where(cell_mask)
    for oz in (0, 1):
        for oy in (0, 1):
            for ox in (0, 1):
                corner[cz + oz, cy + oy, cx + ox] = True
    return corner


def _solve_laplace_patch_3d(values: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    """Solve the 7-point 3D Laplacian on ``values`` with ``free_mask`` marking
    interior unknowns; fixed corners are Dirichlet data from ``values``.

    3D analogue of the 2D version in :mod:`._harmonic` — same vectorised
    build pattern (no Python loop over free nodes), just 6 neighbours
    instead of 4, with the per-node diagonal starting at 6 and reduced
    by 1 for each off-grid neighbour (homogeneous Neumann at the volume
    edge).

    Returns a copy of ``values`` with free voxels replaced by the
    harmonic extension. Fixed voxels are untouched.
    """
    D, H, W = values.shape
    out = values.copy()
    if not free_mask.any():
        return out

    free_idx = np.argwhere(free_mask)
    n = len(free_idx)
    zi, yi, xi = free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]
    idx_of = np.full((D, H, W), -1, dtype=np.int64)
    idx_of[zi, yi, xi] = np.arange(n)

    diag = np.full(n, 6.0)
    rhs = np.zeros(n)
    off_rows: list[np.ndarray] = []
    off_cols: list[np.ndarray] = []

    for dz, dy, dx in (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ):
        zz = zi + dz
        yy = yi + dy
        xx = xi + dx
        in_bounds = (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        oob = ~in_bounds
        diag[oob] -= 1.0

        if not in_bounds.any():
            continue
        k_in = np.where(in_bounds)[0]
        zz_in = zz[k_in]
        yy_in = yy[k_in]
        xx_in = xx[k_in]
        is_free = free_mask[zz_in, yy_in, xx_in]

        k_free = k_in[is_free]
        if k_free.size:
            off_rows.append(k_free)
            off_cols.append(idx_of[zz[k_free], yy[k_free], xx[k_free]])

        k_dir = k_in[~is_free]
        if k_dir.size:
            np.add.at(rhs, k_dir, values[zz[k_dir], yy[k_dir], xx[k_dir]])

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
    out[zi, yi, xi] = sol
    return out


def harmonic_extension_3d(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    ring_pad: int = 2,
    max_grow_iters: int = 6,
    merge_dilation: int = 2,
    margin: float = 0.0,
    record_history: bool = False,
):
    """Harmonic extension over each (dilated) 3D fold component.

    Parameters
    ----------
    phi_in : ndarray, shape ``(3, D, H, W)``
        Field with channels ``[dz, dy, dx]``.
    threshold : float
        Per-tet lower bound used to detect fold cells. Default
        ``DEFAULT_PARAMS['threshold']`` (0.01).
    ring_pad : int
        Initial cell-dilation of each fold component before the boundary
        ring is locked.
    max_grow_iters : int
        If the harmonic patch still has folds, dilate by one more cell
        and retry up to this many times.
    merge_dilation : int
        Cell dilation used only for grouping cells into components.
    margin : float
        Minimum acceptable per-tet volume above ``threshold``.
    record_history : bool
        If True, returns ``(phi, info)`` instead of just ``phi``.

    Returns
    -------
    phi_out : ndarray, shape ``(3, D, H, W)`` — corrected field.
    info : dict, only if ``record_history=True`` — per-component records.
    """
    from dvfopt._defaults import DEFAULT_PARAMS

    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    if merge_dilation < 0:
        raise ValueError(f'merge_dilation must be >= 0, got {merge_dilation}')
    if ring_pad < 0:
        raise ValueError(f'ring_pad must be >= 0, got {ring_pad}')
    _, D, H, W = phi_in.shape
    accept_thr = threshold + margin

    cell_fold = _fold_cell_mask_3d(phi_in, threshold)
    if not cell_fold.any():
        info = {'patches': 0, 'reason': 'already-feasible', 'records': []}
        phi_out = phi_in.copy()
        return (phi_out, info) if record_history else phi_out

    # Group nearby fold cells into components (3D 26-connectivity for
    # `merge_dilation` so diagonally-adjacent voxels join the same patch).
    # NOTE: scipy treats iterations < 1 as "repeat until convergence"
    # (fills the grid), so only dilate for merge_dilation >= 1.
    grouped = (
        binary_dilation(cell_fold, iterations=merge_dilation) if merge_dilation >= 1 else cell_fold
    )
    labels, n_comp = cc_label(grouped, structure=generate_binary_structure(3, 3))

    phi_out = phi_in.copy()
    patch_records: list[dict] = []

    for comp_id in range(1, n_comp + 1):
        comp_cells = (labels == comp_id) & cell_fold
        if not comp_cells.any():
            continue
        cur_cells = comp_cells.copy()
        for grow in range(max_grow_iters + 1):
            # scipy's binary_dilation(iterations=0) means "iterate to
            # convergence" (fills the whole volume -> near-full-volume
            # spsolve), so ring_pad=0 with grow=0 must mean "no dilation
            # this round", not "dilate forever".
            iters = ring_pad + grow
            patch_cells = binary_dilation(cur_cells, iterations=iters) if iters >= 1 else cur_cells
            interior_cells = binary_erosion(patch_cells, iterations=1)
            free_mask = _cells_to_corner_mask_3d(interior_cells, D, H, W)

            # Harmonic solve on each of the 3 displacement channels.
            new_dz = _solve_laplace_patch_3d(phi_out[0], free_mask)
            new_dy = _solve_laplace_patch_3d(phi_out[1], free_mask)
            new_dx = _solve_laplace_patch_3d(phi_out[2], free_mask)
            phi_trial = np.stack([new_dz, new_dy, new_dx])

            # Fused per-cube min kernel — avoids materialising the full
            # (6, Dc, Hc, Wc) volume array per grow attempt per component.
            min_V_trial = six_tet_min_volume_3d(phi_trial)
            patch_min = min_V_trial[patch_cells].min() if patch_cells.any() else np.inf

            if patch_min >= accept_thr:
                phi_out = phi_trial
                patch_records.append(
                    {
                        'comp_id': comp_id,
                        'grow': grow,
                        'n_cells': int(patch_cells.sum()),
                        'patch_V_min': float(patch_min),
                    }
                )
                break
        else:
            # Exhausted grow budget — accept the best we have.
            phi_out = phi_trial
            patch_records.append(
                {
                    'comp_id': comp_id,
                    'grow': max_grow_iters,
                    'n_cells': int(patch_cells.sum()),
                    'patch_V_min': float(patch_min),
                    'note': 'grow-budget-exhausted',
                }
            )

    if record_history:
        info = {'patches': len(patch_records), 'records': patch_records}
        return phi_out, info
    return phi_out


__all__ = ['harmonic_extension_3d']
