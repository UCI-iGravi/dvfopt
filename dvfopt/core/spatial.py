"""Spatial helpers for sub-window positioning, selection, and edge detection."""

import numpy as np
from scipy.ndimage import label

from dvfopt._defaults import _unpack_size


def nearest_center(shape, submatrix_size):
    """Build a dict mapping every (z,y,x) to the nearest valid sub-window centre."""
    sy, sx = _unpack_size(submatrix_size)
    hy, hx = sy // 2, sx // 2
    max_y = shape[1] - sy + hy
    max_x = shape[2] - sx + hx
    near_cent = {}
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                near_cent[(z, y, x)] = [z, max(hy, min(y, max_y)),
                                           max(hx, min(x, max_x))]
    return near_cent


def get_nearest_center(neg_index, slice_shape, submatrix_size, near_cent_dict):
    """Look up (or compute) the nearest valid centre for *neg_index*."""
    key = _unpack_size(submatrix_size)  # normalise to tuple for caching
    if key in near_cent_dict:
        return near_cent_dict[key][(0, *neg_index)]
    else:
        near_cent = nearest_center(slice_shape, key)
        near_cent_dict[key] = near_cent
        return near_cent[(0, *neg_index)]


def argmin_quality(jacobian_matrix):
    """Index of the pixel with the lowest Jacobian determinant."""
    flat_index = np.argmin(jacobian_matrix[0])
    idx = np.unravel_index(flat_index, jacobian_matrix.shape[1:])
    return (int(idx[0]), int(idx[1]))


def neg_jdet_bounding_window(jacobian_matrix, center_yx, threshold, err_tol):
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

    Returns
    -------
    size : tuple of int
        ``(height, width)`` — each >= 3.
    bbox_center : tuple of int
        ``(y, x)`` centre of the bounding box.
    """
    neg_mask = jacobian_matrix[0] <= threshold - err_tol
    labeled, _ = label(neg_mask)  # 8-connectivity is the scipy default via structure
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
    y0, y1 = cy - hy, cy + hy_hi - 1
    x0, x1 = cx - hx, cx + hx_hi - 1
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
    """
    H, W = slice_shape[1], slice_shape[2]
    occupied = np.zeros((H, W), dtype=bool)
    selected = []

    for neg_idx in neg_pixels:
        ws = pixel_window_sizes[neg_idx]
        center_key = (pixel_bbox_centers or {}).get(neg_idx, neg_idx)
        cz, cy, cx = get_nearest_center(center_key, slice_shape, ws, near_cent_dict)
        y0, y1, x0, x1 = _window_bounds(cy, cx, ws)

        if occupied[y0:y1 + 1, x0:x1 + 1].any():
            continue

        occupied[y0:y1 + 1, x0:x1 + 1] = True
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
