"""Laplacian inpainting of the fold zone.

For the 94 unfixable cells (z=0..7, y=136..221, x=191..283), dilate
by RING_PAD corners to form an "inpaint mask". Solve the Laplace
equation on each channel of phi inside this mask with Dirichlet
boundary conditions from outside.

Resulting field: smooth inside the fold zone (Laplacian extensions
are harmonic, so no internal extrema in any single component).
Combined with smooth boundary values from outside the mask, the
6-tet feasibility should hold throughout the fold zone — the field
becomes a continuous interpolation from the surrounding region.

Cost: L1 distance from input. We're DISCARDING the registration's
intent in the fold zone and replacing with smooth interpolation.

This is a deliberate trade-off: pay L1 for guaranteed feasibility.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import cg, spsolve

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def laplacian_inpaint(phi, mask_corners, verbose=True):
    """Solve Laplace equation on the masked corners with Dirichlet BCs
    from outside.

    Parameters
    ----------
    phi : (3, D, H, W) — field. The corners flagged in mask will be
        modified; others kept.
    mask_corners : (D, H, W) bool — True where corner displacement
        should be solved for.

    Returns
    -------
    phi_out : (3, D, H, W) — field with masked corners replaced by
        Laplacian extension.
    """
    D, H, W = phi.shape[1:]
    inside_idx = np.where(mask_corners)
    n_inside = len(inside_idx[0])
    if n_inside == 0:
        return phi.copy()

    if verbose:
        print(f'  Building Laplacian system: {n_inside} interior corners', flush=True)

    # Map (z, y, x) → linear index for inside corners.
    inside_set = {(int(z), int(y), int(x)): i for i, (z, y, x) in enumerate(zip(*inside_idx))}

    # Build 7-point Laplacian on inside corners with Dirichlet BCs from outside.
    # A_ii = number of neighbours; A_ij = -1 for inside neighbours.
    # b_i = sum of outside-neighbour boundary values.
    rows, cols, vals = [], [], []
    b = np.zeros((n_inside, 3))
    for i, (z, y, x) in enumerate(zip(*inside_idx)):
        z, y, x = int(z), int(y), int(x)
        n_neigh = 0
        for dz, dy, dx in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            zn, yn, xn = z + dz, y + dy, x + dx
            if not (0 <= zn < D and 0 <= yn < H and 0 <= xn < W):
                continue  # outside the grid — treat as boundary at 0 displacement
            n_neigh += 1
            if (zn, yn, xn) in inside_set:
                j = inside_set[(zn, yn, xn)]
                rows.append(i)
                cols.append(j)
                vals.append(-1.0)
            else:
                # Outside the inpaint mask → boundary value from phi.
                b[i] += phi[:, zn, yn, xn]
        rows.append(i)
        cols.append(i)
        vals.append(float(n_neigh))

    A = csr_matrix((vals, (rows, cols)), shape=(n_inside, n_inside))
    if verbose:
        print(f'  Solving Laplacian system ({n_inside} unknowns)...', flush=True)
    t0 = time.time()
    # Solve per-channel.
    x_sol = np.zeros((n_inside, 3))
    for c in range(3):
        x_sol[:, c] = spsolve(A, b[:, c])
    if verbose:
        print(f'    solve wall: {time.time() - t0:.1f}s', flush=True)

    # Replace masked corners in phi.
    phi_out = phi.copy()
    for i, (z, y, x) in enumerate(zip(*inside_idx)):
        phi_out[:, int(z), int(y), int(x)] = x_sol[i]
    return phi_out


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    print(f'Loaded, shape={phi.shape}', flush=True)
    V0 = six_tet_volumes_3d(phi)
    n_neg0 = int((V0 <= 0).sum())
    n_below0 = int((V0 < THRESHOLD - 1e-5).sum())
    print(f'Start: n_neg={n_neg0}  n<0.01={n_below0}  min_T={float(V0.min()):+.6f}', flush=True)

    # Identify unfixable cells.
    best_min = _best_min_per_cell(phi)
    unfix_mask = best_min <= 0
    print(f'Unfixable cells: {int(unfix_mask.sum())}', flush=True)

    # Convert unfixable cell mask to corner mask.
    # A cube has 8 corners; corner (z, y, x) is shared by cubes at
    # (z-1, y-1, x-1) ... (z, y, x). So a corner is in the mask if
    # ANY of its 8 surrounding cubes is unfixable.
    D, H, W = phi.shape[1:]
    Dc, Hc, Wc = unfix_mask.shape
    # Build corner mask via dilation in cube space then convert.
    cube_mask = unfix_mask.copy()

    # Test different ring_pads.
    for ring_pad in [2, 5, 10]:
        print(f'\n--- ring_pad={ring_pad} ---', flush=True)
        # Dilate cube mask by ring_pad cells.
        dilated = binary_dilation(cube_mask, iterations=ring_pad)
        # Convert to corner mask: corner (z, y, x) is in mask if any
        # neighboring cube (z' ∈ [z-1, z], etc.) is in dilated.
        corner_mask = np.zeros((D, H, W), dtype=bool)
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    sz = slice(dz, dz + Dc)
                    sy = slice(dy, dy + Hc)
                    sx = slice(dx, dx + Wc)
                    corner_mask[sz, sy, sx] |= dilated
        n_corners_in = int(corner_mask.sum())
        print(f'  inpaint corner mask: {n_corners_in} corners', flush=True)

        phi_new = laplacian_inpaint(phi, corner_mask, verbose=True)
        V_new = six_tet_volumes_3d(phi_new)
        n_neg = int((V_new <= 0).sum())
        n_below = int((V_new < THRESHOLD - 1e-5).sum())
        L1 = float(np.abs(phi_new - phi).sum())
        print(
            f'  After inpaint:  n_neg={n_neg}  n<0.01={n_below}  '
            f'min_T={float(V_new.min()):+.6f}  L1_from_input={L1:.1f}',
            flush=True,
        )
        if n_neg == 0 and n_below == 0:
            print(f'  *** STRICT 100% FEASIBLE at ring_pad={ring_pad} ***', flush=True)
            np.save(
                OUTPUT / f'b0039_z0_15_strict_via_laplacian_inpaint_ring{ring_pad}.npy', phi_new
            )


if __name__ == '__main__':
    main()
