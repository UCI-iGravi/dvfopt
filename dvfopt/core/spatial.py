"""Spatial helpers for sub-window positioning, selection, and edge detection."""

import numpy as np
from scipy.ndimage import label

from dvfopt._defaults import _unpack_size


def get_nearest_center(neg_index, slice_shape, submatrix_size, near_cent_dict=None):
    """Return the nearest valid sub-window centre for *neg_index*.

    Computed in O(1) by clamping the requested position to the valid
    range ``[hy, H - hy_hi]`` × ``[hx, W - hx_hi]``.  The *near_cent_dict*
    parameter is accepted but ignored (kept for call-site compatibility).
    """
    sy, sx = _unpack_size(submatrix_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    max_y = slice_shape[1] - hy_hi
    max_x = slice_shape[2] - hx_hi
    y, x = neg_index
    return [0, max(hy, min(int(y), max_y)), max(hx, min(int(x), max_x))]


def argmin_quality(jacobian_matrix):
    """Index of the pixel with the lowest Jacobian determinant."""
    flat_index = np.argmin(jacobian_matrix[0])
    idx = np.unravel_index(flat_index, jacobian_matrix.shape[1:])
    return (int(idx[0]), int(idx[1]))


def neg_jdet_bounding_window(jacobian_matrix, center_yx, threshold, err_tol, labeled=None):
    """Compute the smallest window enclosing the negative-Jdet region around *center_yx*.

    The window is the bounding box of all pixels with Jdet <= *threshold* - *err_tol*
    that are **connected** (8-connectivity) to *center_yx*, expanded by 1 pixel on
    each side so the frozen edges sit on positive-Jdet pixels.

    The window dimensions match the bounding box — the height and width
    can differ, producing a rectangular window.  The bbox centre is returned
    so callers can position the window on the region's centre (via
    ``nearest_center``) rather than on the worst pixel, avoiding an
    oversized window when the worst pixel is off-centre.  Each dimension
    is at least 3.

    Parameters
    ----------
    jacobian_matrix : ndarray, shape ``(1, H, W)``
    center_yx : tuple of int
        ``(y, x)`` of the worst pixel.
    threshold, err_tol : float
    labeled : ndarray or None
        Pre-computed connected-component label array (same shape as
        ``jacobian_matrix[0]``).  When provided, the ``scipy.ndimage.label``
        call is skipped.  Pass this when calling in a loop over many pixels
        to avoid recomputing labels for every pixel.

    Returns
    -------
    size : tuple of int
        ``(height, width)`` — each >= 3.
    bbox_center : tuple of int
        ``(y, x)`` centre of the bounding box.
    """
    if labeled is None:
        neg_mask = jacobian_matrix[0] <= threshold - err_tol
        labeled, _ = label(neg_mask)
    region_label = labeled[center_yx[0], center_yx[1]]

    if region_label == 0:
        # Pixel is not negative (shouldn't happen, but be safe)
        return (3, 3), center_yx

    region_ys, region_xs = np.where(labeled == region_label)
    # Bounding box of the connected negative region + 1 pixel border,
    # clamped to the grid so edge-touching regions don't go out of bounds.
    H, W = jacobian_matrix.shape[1:]
    y_min = max(int(region_ys.min()) - 1, 0)
    y_max = min(int(region_ys.max()) + 1, H - 1)
    x_min = max(int(region_xs.min()) - 1, 0)
    x_max = min(int(region_xs.max()) + 1, W - 1)

    # Rectangular window matching the bounding box (floor 3 per dim).
    height = max(y_max - y_min + 1, 3)
    width = max(x_max - x_min + 1, 3)

    # Round centre UP so that the window [cy - hy, cy + hy_hi) aligns with the
    # bbox.  hy = height // 2 rounds down, so the window extends further in the
    # +y / +x direction.  Rounding the centre up compensates for this.
    bbox_center = ((y_min + y_max + 1) // 2, (x_min + x_max + 1) // 2)
    return (height, width), bbox_center


def _frozen_edges_clean(jacobian_matrix, cy, cx, submatrix_size, threshold, err_tol):
    """Return True if the frozen edges of the window have positive Jdet.

    Checks the outer ring of the window centred at ``(cy, cx)``.
    If any edge pixel has Jdet <= threshold - err_tol, the optimiser's
    frozen-edge constraint would be infeasible, so the caller should
    grow the window instead of running the optimiser.
    """
    sy, sx = _unpack_size(submatrix_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    H, W = jacobian_matrix.shape[1], jacobian_matrix.shape[2]
    y0 = max(cy - hy, 0)
    y1 = min(cy + hy_hi - 1, H - 1)
    x0 = max(cx - hx, 0)
    x1 = min(cx + hx_hi - 1, W - 1)
    edge_vals = np.concatenate([
        jacobian_matrix[0, y0, x0:x1 + 1].ravel(),
        jacobian_matrix[0, y1, x0:x1 + 1].ravel(),
        jacobian_matrix[0, y0:y1 + 1, x0].ravel(),
        jacobian_matrix[0, y0:y1 + 1, x1].ravel(),
    ])
    return edge_vals.min() > threshold - err_tol


def get_phi_sub_flat(phi, cz, cy, cx, shape, submatrix_size):
    """Extract and flatten a rectangular sub-window of *phi* around (cy, cx)."""
    sy, sx = _unpack_size(submatrix_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    phix = phi[1, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi]
    phiy = phi[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi]
    return np.concatenate([phix.flatten(), phiy.flatten()])


def get_phi_sub_flat_padded(phi, cz, cy, cx, shape, submatrix_size):
    """Extract a sub-window with 1-pixel padding on all sides when possible.

    When the window fits 1 pixel inside the grid boundary on all sides, the
    extraction is expanded to ``(sy+2) x (sx+2)``.  The extra ring is used as
    the frozen boundary in the SLSQP constraints, so the full original
    ``sy x sx`` window (including its boundary ring) is optimised and its
    Jacobian is constrained with proper central-difference context.

    When padding is not possible (window too close to the grid edge), falls
    back to the standard ``sy x sx`` extraction.

    Returns
    -------
    flat : ndarray
        Flattened ``[dx, dy]`` values of the (possibly padded) sub-window.
    actual_size : tuple of int
        ``(sy+2, sx+2)`` if padded, ``(sy, sx)`` otherwise.
    """
    sy, sx = _unpack_size(submatrix_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    H, W = shape[1], shape[2]

    can_pad = (cy - hy - 1 >= 0 and cy + hy_hi + 1 <= H and
               cx - hx - 1 >= 0 and cx + hx_hi + 1 <= W)

    if can_pad:
        phix = phi[1, cy - hy - 1:cy + hy_hi + 1, cx - hx - 1:cx + hx_hi + 1]
        phiy = phi[0, cy - hy - 1:cy + hy_hi + 1, cx - hx - 1:cx + hx_hi + 1]
        actual_size = (sy + 2, sx + 2)
    else:
        phix = phi[1, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi]
        phiy = phi[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi]
        actual_size = (sy, sx)

    return np.concatenate([phix.flatten(), phiy.flatten()]), actual_size


def _window_bounds(cy, cx, submatrix_size):
    sy, sx = _unpack_size(submatrix_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    return (cy - hy, cy + hy_hi - 1, cx - hx, cx + hx_hi - 1)


def _windows_overlap(b1, b2):
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])


def _select_non_overlapping(neg_pixels, pixel_window_sizes, slice_shape,
                             near_cent_dict, pixel_bbox_centers=None):
    """Greedily select non-overlapping windows using an occupancy grid.

    Uses a 2D boolean grid for O(window_area) overlap checks instead of
    O(k) pairwise comparisons per candidate.

    The occupancy footprint is expanded by 1 pixel on each side beyond the
    window bounds.  This enforces a 1-pixel gap between parallel windows so
    that the padded frozen boundary of each window is never modified by
    another window in the same batch.
    """
    H, W = slice_shape[1], slice_shape[2]
    occupied = np.zeros((H, W), dtype=bool)
    selected = []

    for neg_idx in neg_pixels:
        ws = pixel_window_sizes[neg_idx]
        center_key = (pixel_bbox_centers or {}).get(neg_idx, neg_idx)
        cz, cy, cx = get_nearest_center(center_key, slice_shape, ws, near_cent_dict)
        y0, y1, x0, x1 = _window_bounds(cy, cx, ws)

        # Padded footprint: 1 extra pixel on each side for gap enforcement
        py0 = max(y0 - 1, 0)
        py1 = min(y1 + 1, H - 1)
        px0 = max(x0 - 1, 0)
        px1 = min(x1 + 1, W - 1)

        if occupied[py0:py1 + 1, px0:px1 + 1].any():
            continue

        occupied[py0:py1 + 1, px0:px1 + 1] = True
        selected.append((neg_idx, (cz, cy, cx), ws))

    return selected


def _edge_flags(cy, cx, submatrix_size, slice_shape, max_window):
    """Return (is_at_edge, window_reached_max) for a window."""
    sy, sx = _unpack_size(submatrix_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    start_y = cy - hy
    end_y = cy + hy_hi - 1          # inclusive last pixel
    start_x = cx - hx
    end_x = cx + hx_hi - 1
    max_y, max_x = slice_shape[1:]
    is_at_edge = (start_y == 0 or end_y >= max_y - 1
                  or start_x == 0 or end_x >= max_x - 1)
    max_sy, max_sx = _unpack_size(max_window)
    window_reached_max = sy >= max_sy and sx >= max_sx
    return is_at_edge, window_reached_max
