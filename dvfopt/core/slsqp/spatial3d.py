"""3D spatial helpers: sub-volume positioning, bounding windows, boundary masks."""

import numpy as np
from scipy.ndimage import label

from dvfopt._defaults import _unpack_size_3d


def get_nearest_center_3d(neg_index, volume_shape, subvolume_size):
    """Compute the nearest valid sub-volume centre for *neg_index* (O(1) clamp).

    Parameters
    ----------
    neg_index : tuple of int ``(z, y, x)``
    volume_shape : tuple ``(D, H, W)``
    subvolume_size : int or ``(sz, sy, sx)``
    """
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    hz, hy, hx = sz // 2, sy // 2, sx // 2
    D, H, W = volume_shape
    max_z = D - sz + hz
    max_y = H - sy + hy
    max_x = W - sx + hx
    cz = max(hz, min(neg_index[0], max_z))
    cy = max(hy, min(neg_index[1], max_y))
    cx = max(hx, min(neg_index[2], max_x))
    return cz, cy, cx


def argmin_worst_voxel(jacobian_matrix):
    """Index of the voxel with the lowest Jacobian determinant (full grid).

    Returns ``(z, y, x)`` tuple.
    """
    flat_index = np.argmin(jacobian_matrix)
    return np.unravel_index(flat_index, jacobian_matrix.shape)


def neg_jdet_bounding_window_3d(jacobian_matrix, center_zyx, threshold, err_tol,
                                labeled_array=None, pad=2):
    """Compute the smallest sub-volume enclosing the negative-Jdet region.

    The sub-volume is the bounding box of all voxels with
    Jdet <= *threshold* - *err_tol* that are **connected**
    (26-connectivity) to *center_zyx*, expanded by *pad* voxels on each
    side.  Each dimension is at least 3.

    The default ``pad=2`` provides 1 voxel of optimisation room around
    the negative region plus 1 voxel for the frozen boundary ring.

    Returns
    -------
    size : tuple of int ``(sz, sy, sx)``
    bbox_center : tuple of int ``(z, y, x)``
    """
    if labeled_array is None:
        neg_mask = jacobian_matrix <= threshold - err_tol
        structure = np.ones((3, 3, 3))  # 26-connectivity
        labeled_array, _ = label(neg_mask, structure=structure)

    region_label = labeled_array[center_zyx[0], center_zyx[1], center_zyx[2]]

    if region_label == 0:
        return (3, 3, 3), center_zyx

    region_zs, region_ys, region_xs = np.where(labeled_array == region_label)

    D, H, W = jacobian_matrix.shape
    z_min = max(int(region_zs.min()) - pad, 0)
    z_max = min(int(region_zs.max()) + pad, D - 1)
    y_min = max(int(region_ys.min()) - pad, 0)
    y_max = min(int(region_ys.max()) + pad, H - 1)
    x_min = max(int(region_xs.min()) - pad, 0)
    x_max = min(int(region_xs.max()) + pad, W - 1)

    depth  = max(z_max - z_min + 1, 3)
    height = max(y_max - y_min + 1, 3)
    width  = max(x_max - x_min + 1, 3)

    bbox_center = ((z_min + z_max + 1) // 2,
                   (y_min + y_max + 1) // 2,
                   (x_min + x_max + 1) // 2)

    return (depth, height, width), bbox_center


def _frozen_boundary_mask_3d(cz, cy, cx, subvolume_size, volume_shape):
    """Boolean mask of sub-volume boundary voxels that should be frozen.

    Boundary voxels on the *grid* edge are NOT frozen (they have no
    neighbours outside the grid).  All other boundary voxels ARE frozen
    so the optimiser cannot displace negativity outside the window.
    """
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    hz, hy, hx = sz // 2, sy // 2, sx // 2
    hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx
    start_z, end_z = cz - hz, cz + hz_hi - 1
    start_y, end_y = cy - hy, cy + hy_hi - 1
    start_x, end_x = cx - hx, cx + hx_hi - 1
    D, H, W = volume_shape

    mask = np.zeros((sz, sy, sx), dtype=bool)
    if start_z > 0:
        mask[0, :, :] = True
    if end_z < D - 1:
        mask[-1, :, :] = True
    if start_y > 0:
        mask[:, 0, :] = True
    if end_y < H - 1:
        mask[:, -1, :] = True
    if start_x > 0:
        mask[:, :, 0] = True
    if end_x < W - 1:
        mask[:, :, -1] = True
    return mask



def _frozen_edges_clean_3d(jacobian_matrix, cz, cy, cx, subvolume_size,
                           threshold, err_tol, freeze_mask):
    """Return True if frozen boundary voxels have positive Jdet."""
    if not freeze_mask.any():
        return True
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    hz, hy, hx = sz // 2, sy // 2, sx // 2
    hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx
    sub_jdet = jacobian_matrix[cz - hz:cz + hz_hi,
                               cy - hy:cy + hy_hi,
                               cx - hx:cx + hx_hi]
    frozen_vals = sub_jdet[freeze_mask]
    return frozen_vals.min() > threshold - err_tol


def get_phi_sub_flat_3d(phi, cz, cy, cx, subvolume_size):
    """Extract and flatten a sub-volume of *phi* around ``(cz, cy, cx)``.

    Parameters
    ----------
    phi : ndarray, shape ``(3, D, H, W)`` with ``[dz, dy, dx]``

    Returns
    -------
    1-D array packed as ``[dx_flat, dy_flat, dz_flat]``
    """
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    hz, hy, hx = sz // 2, sy // 2, sx // 2
    hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx

    slc = (slice(cz - hz, cz + hz_hi),
           slice(cy - hy, cy + hy_hi),
           slice(cx - hx, cx + hx_hi))

    phi_dx = phi[2][slc]
    phi_dy = phi[1][slc]
    phi_dz = phi[0][slc]
    return np.concatenate([phi_dx.flatten(), phi_dy.flatten(), phi_dz.flatten()])


def _edge_flags_3d(cz, cy, cx, subvolume_size, volume_shape, max_window):
    """Return ``(is_at_edge, window_reached_max)`` for a sub-volume."""
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    hz, hy, hx = sz // 2, sy // 2, sx // 2
    hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx

    D, H, W = volume_shape

    is_at_edge = (cz - hz == 0 or cz + hz_hi - 1 >= D - 1
                  or cy - hy == 0 or cy + hy_hi - 1 >= H - 1
                  or cx - hx == 0 or cx + hx_hi - 1 >= W - 1)

    max_sz, max_sy, max_sx = _unpack_size_3d(max_window)
    window_reached_max = sz >= max_sz and sy >= max_sy and sx >= max_sx

    return is_at_edge, window_reached_max
