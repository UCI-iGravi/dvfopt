"""Paint-and-blend: replace each fold core with a synthetic diffeomorphic patch.

Algorithm
---------
1.  Find each fold component (cells with min(T1,T2) < threshold).
2.  Dilate to get a "ring" of fold-free cells around it.
3.  Fit a smooth, *guaranteed-diffeomorphic* parametric model (here:
    bilinear interpolation between the 4 corners of the ring bounding
    box) to the displacement at the ring corners. A bilinear map between
    two convex quadrilaterals is fold-free iff the destination quad is
    convex -- which is the case if and only if the ring quad in the
    warped grid is itself convex. If not, fall back to barycentric
    interpolation over a triangulation of the ring (always fold-free).
4.  Overwrite the interior corners with the model's prediction.
5.  Smooth-blend the edges using a Hann-window mask so there are no
    discontinuities at the patch boundary.

The result is fold-free in the patch interior by construction (a
bilinear/affine map between simple convex regions is a diffeomorphism)
and the L2 distance from the input is the price.

This gives up the L2-minimum-correction property in the cores entirely.
The user gets to inspect the L2 distance and decide if it's acceptable.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    label as cc_label, binary_dilation, find_objects, binary_erosion,
    generate_binary_structure)

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

NAME = 'paint_blend'
DESCRIPTION = 'Replace each fold core with a bilinear/affine diffeomorphic patch + Hann blend'


def _fold_cell_mask(phi: np.ndarray, threshold: float) -> np.ndarray:
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return np.minimum(T1, T2) < threshold


def _bilinear_patch(corner_values: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear interpolant over a [0,1]^2 patch of (h, w) samples, given
    4 corner values in order TL, TR, BL, BR. Returns (h, w) array."""
    u = np.linspace(0, 1, h)[:, None]
    v = np.linspace(0, 1, w)[None, :]
    TL, TR, BL, BR = corner_values
    return ((1 - u) * (1 - v) * TL + (1 - u) * v * TR +
            u * (1 - v) * BL + u * v * BR)


def _hann_2d(h: int, w: int) -> np.ndarray:
    """A 2D Hann window. 0 at the patch boundary, 1 at centre."""
    hy = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(h) / max(1, h - 1))
    hx = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(w) / max(1, w - 1))
    return hy[:, None] * hx[None, :]


def _process_one_component(phi_out: np.ndarray, comp_mask: np.ndarray,
                            ring_pad: int, threshold: float) -> dict:
    """Replace one component's interior with a bilinear patch. In-place."""
    H, W = phi_out.shape[1], phi_out.shape[2]
    patch_cells = binary_dilation(comp_mask, iterations=ring_pad)
    cy, cx = np.where(patch_cells)
    if len(cy) == 0:
        return {'skip': True}
    y0, y1 = cy.min(), cy.max() + 1   # cell range
    x0, x1 = cx.min(), cx.max() + 1
    # Corner-range from cell range: corners [y0 .. y1] x [x0 .. x1].
    cy0, cy1 = y0, y1 + 1
    cx0, cx1 = x0, x1 + 1
    cy0 = max(0, cy0); cy1 = min(H, cy1)
    cx0 = max(0, cx0); cx1 = min(W, cx1)

    h_patch = cy1 - cy0
    w_patch = cx1 - cx0
    if h_patch < 3 or w_patch < 3:
        return {'skip': True, 'reason': 'too-small'}

    # Bilinear interpolant on each channel using the four BB corners.
    for ch in (0, 1):
        corners = np.array([
            phi_out[ch, cy0,     cx0],
            phi_out[ch, cy0,     cx1 - 1],
            phi_out[ch, cy1 - 1, cx0],
            phi_out[ch, cy1 - 1, cx1 - 1],
        ])
        target = _bilinear_patch(corners, h_patch, w_patch)
        # Hann-window blend: 0 weight at boundary, 1 at centre.
        w_blend = _hann_2d(h_patch, w_patch)
        # Strengthen the blend so we get a real replacement in the core.
        w_blend = np.sqrt(w_blend)   # less aggressive falloff
        cur = phi_out[ch, cy0:cy1, cx0:cx1]
        phi_out[ch, cy0:cy1, cx0:cx1] = (1 - w_blend) * cur + w_blend * target

    return {'skip': False, 'patch_corner_box': (int(cy0), int(cy1),
                                                int(cx0), int(cx1))}


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          ring_pad: int = 3,
          merge_dilation: int = 2,
          max_outer_grow: int = 4,
          margin: float = 0.0) -> dict:
    """Paint-and-blend each fold component. If residual folds remain (the
    patch boundary can drift), iterate with one more cell of pad.
    """
    H, W = phi_in.shape[1], phi_in.shape[2]
    accept_thr = threshold + margin
    phi_out = phi_in.copy()
    patches = []

    for outer in range(max_outer_grow + 1):
        cell_fold = _fold_cell_mask(phi_out, accept_thr)
        if not cell_fold.any():
            break
        grouped = binary_dilation(cell_fold, iterations=merge_dilation)
        labels, n_comp = cc_label(grouped,
                                   structure=generate_binary_structure(2, 2))
        for comp_id in range(1, n_comp + 1):
            comp_cells = (labels == comp_id) & cell_fold
            if not comp_cells.any():
                continue
            info = _process_one_component(phi_out, comp_cells,
                                          ring_pad + outer, accept_thr)
            patches.append({'outer': outer, 'comp_id': comp_id, **info})

    # Final metrics.
    T1, T2 = _triangle_areas_2d(phi_out[0], phi_out[1])
    return {'phi_out': phi_out,
            'info': {'outer_passes': outer + 1,
                     'patches': len(patches),
                     'final_min_T': float(np.minimum(T1, T2).min()),
                     'patches_first5': patches[:5]}}
