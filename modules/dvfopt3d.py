"""
Iterative SLSQP optimisation for correcting negative Jacobian determinants
in 3D deformation (displacement) fields.

3D extension of ``dvfopt.py``.  Operates on ``(3, D, H, W)`` deformation
fields with channels ``[dz, dy, dx]``.

Usage::

    from modules.dvfopt3d import iterative_3d, jacobian_det3D
"""

import os
import time
from collections import defaultdict

import numpy as np
from scipy.ndimage import label, zoom
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


# ---------------------------------------------------------------------------
# DVF generation utilities
# ---------------------------------------------------------------------------
def generate_random_dvf_3d(shape, max_magnitude=5.0, seed=None):
    """Generate a random 3D deformation vector field (DVF).

    Parameters
    ----------
    shape : tuple
        ``(3, D, H, W)`` — standard 3D deformation field shape.
    max_magnitude : float
        Max displacement in voxels (uniform in ``[-mag, +mag]``).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(3, D, H, W)``
    """
    rng = np.random.default_rng(seed)
    C = shape[0]
    assert C == 3, "DVF must have 3 channels (dz, dy, dx)"
    return rng.uniform(-max_magnitude, max_magnitude, size=shape).astype(np.float64)


def scale_dvf_3d(dvf, new_size):
    """Rescale a ``(3, D, H, W)`` deformation field to *new_size* ``(new_D, new_H, new_W)``.

    Spatial interpolation is trilinear (``order=1``) and displacement
    magnitudes are scaled proportionally.
    """
    C, D, H, W = dvf.shape
    new_D, new_H, new_W = new_size
    scale_z = new_D / D
    scale_y = new_H / H
    scale_x = new_W / W

    dvf_resized = np.zeros((C, new_D, new_H, new_W), dtype=dvf.dtype)
    for c in range(C):
        dvf_resized[c] = zoom(dvf[c], (scale_z, scale_y, scale_x), order=1)

    dvf_resized[0] *= scale_z  # dz
    dvf_resized[1] *= scale_y  # dy
    dvf_resized[2] *= scale_x  # dx
    return dvf_resized


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "threshold": 0.01,
    "err_tol": 1e-5,
    "max_iterations": 10000,
    "max_per_index_iter": 50,
    "max_minimize_iter": 1000,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _log(verbose, level, msg):
    if verbose >= level:
        print(msg)


def _unpack_size_3d(subvolume_size):
    """Normalize *subvolume_size* to a ``(sz, sy, sx)`` tuple.

    Accepts a single int (cubic) or a 3-tuple/list (rectangular).
    """
    if isinstance(subvolume_size, (tuple, list)):
        if len(subvolume_size) == 3:
            return int(subvolume_size[0]), int(subvolume_size[1]), int(subvolume_size[2])
    return int(subvolume_size), int(subvolume_size), int(subvolume_size)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def objectiveEuc(phi, phi_init):
    """L2 norm objective function."""
    return np.linalg.norm(phi - phi_init)


# ---------------------------------------------------------------------------
# 3D Jacobian determinant
# ---------------------------------------------------------------------------
def _numpy_jdet_3d(dz, dy, dx):
    """Compute 3D Jacobian determinant from displacement components (numpy).

    Uses central differences via ``np.gradient`` for all 9 partial
    derivatives of the deformation gradient tensor.

    Parameters
    ----------
    dz, dy, dx : ndarray, shape ``(D, H, W)``

    Returns
    -------
    ndarray, shape ``(D, H, W)``
    """
    # Partial derivatives: axis 0=z, 1=y, 2=x
    ddx_dx = np.gradient(dx, axis=2)  # ∂dx/∂x
    ddx_dy = np.gradient(dx, axis=1)  # ∂dx/∂y
    ddx_dz = np.gradient(dx, axis=0)  # ∂dx/∂z

    ddy_dx = np.gradient(dy, axis=2)  # ∂dy/∂x
    ddy_dy = np.gradient(dy, axis=1)  # ∂dy/∂y
    ddy_dz = np.gradient(dy, axis=0)  # ∂dy/∂z

    ddz_dx = np.gradient(dz, axis=2)  # ∂dz/∂x
    ddz_dy = np.gradient(dz, axis=1)  # ∂dz/∂y
    ddz_dz = np.gradient(dz, axis=0)  # ∂dz/∂z

    # Deformation gradient: J = I + nabla(u)
    # det(J) via cofactor expansion along first row
    a11 = 1 + ddx_dx;  a12 = ddx_dy;      a13 = ddx_dz
    a21 = ddy_dx;       a22 = 1 + ddy_dy;  a23 = ddy_dz
    a31 = ddz_dx;       a32 = ddz_dy;      a33 = 1 + ddz_dz

    return (a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31))


def jacobian_det3D(phi):
    """Compute the Jacobian determinant from a ``(3, D, H, W)`` phi array.

    Returns shape ``(D, H, W)``.
    """
    dz = phi[0]
    dy = phi[1]
    dx = phi[2]
    return _numpy_jdet_3d(dz, dy, dx)


def jacobian_constraint_3d(phi_flat, subvolume_size, freeze_mask=None):
    """Return flattened Jacobian determinant values for optimiser constraints.

    phi_flat packing: ``[dx_flat, dy_flat, dz_flat]``.

    When *freeze_mask* is given, only non-frozen voxels are returned.
    """
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    voxels = sz * sy * sx
    dx = phi_flat[:voxels].reshape((sz, sy, sx))
    dy = phi_flat[voxels:2 * voxels].reshape((sz, sy, sx))
    dz = phi_flat[2 * voxels:].reshape((sz, sy, sx))
    jdet = _numpy_jdet_3d(dz, dy, dx)
    if freeze_mask is not None:
        return jdet[~freeze_mask].flatten()
    return jdet.flatten()


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
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
                                labeled_array=None):
    """Compute the smallest sub-volume enclosing the negative-Jdet region.

    The sub-volume is the bounding box of all voxels with
    Jdet <= *threshold* - *err_tol* that are **connected**
    (26-connectivity) to *center_zyx*, expanded by 1 voxel on each
    side.  Each dimension is at least 3.

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
    z_min = max(int(region_zs.min()) - 1, 0)
    z_max = min(int(region_zs.max()) + 1, D - 1)
    y_min = max(int(region_ys.min()) - 1, 0)
    y_max = min(int(region_ys.max()) + 1, H - 1)
    x_min = max(int(region_xs.min()) - 1, 0)
    x_max = min(int(region_xs.max()) + 1, W - 1)

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


# ---------------------------------------------------------------------------
# Constraint builder
# ---------------------------------------------------------------------------
def _build_constraints_3d(phi_sub_flat, subvolume_size, freeze_mask, threshold):
    """Build SLSQP constraints for a 3D sub-volume optimisation.

    The Jacobian constraint excludes only frozen boundary voxels.
    Grid-edge boundary voxels are NOT frozen and ARE constrained.
    """
    fm = freeze_mask
    nlc = NonlinearConstraint(
        lambda phi1: jacobian_constraint_3d(phi1, subvolume_size, fm),
        threshold, np.inf,
    )
    constraints = [nlc]

    if freeze_mask.any():
        sz, sy, sx = _unpack_size_3d(subvolume_size)
        voxels = sz * sy * sx
        edge_indices = np.argwhere(freeze_mask)
        fixed_indices = []
        for z, y, x in edge_indices:
            idx = z * sy * sx + y * sx + x
            fixed_indices.extend([idx, idx + voxels, idx + 2 * voxels])

        fixed_values = phi_sub_flat[fixed_indices]
        A_eq = np.zeros((len(fixed_indices), phi_sub_flat.size))
        for row, col in enumerate(fixed_indices):
            A_eq[row, col] = 1

        constraints.append(LinearConstraint(A_eq, fixed_values, fixed_values))

    return constraints


# ---------------------------------------------------------------------------
# Init / metrics helpers
# ---------------------------------------------------------------------------
def _resolve_params(**overrides):
    p = dict(DEFAULT_PARAMS)
    for name, val in overrides.items():
        if val is not None:
            p[name] = val
    return p


def _init_phi_3d(deformation):
    """Create the initial ``phi`` working array from a ``(3, D, H, W)`` deformation.

    Returns ``(phi, phi_init, D, H, W)``.
    """
    D, H, W = deformation.shape[1:]
    phi = deformation.copy().astype(np.float64)
    phi_init = phi.copy()
    return phi, phi_init, D, H, W


def _update_metrics_3d(phi, phi_init, num_neg_jac, min_jdet_list,
                       error_list=None):
    """Recompute Jacobian and append to accumulator lists.

    Returns ``(jacobian_matrix, cur_neg, cur_min)``.
    """
    jac = jacobian_det3D(phi)  # (D, H, W)
    cur_neg = int((jac <= 0).sum())
    cur_min = float(jac.min())
    num_neg_jac.append(cur_neg)
    min_jdet_list.append(cur_min)
    if error_list is not None:
        error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))
    return jac, cur_neg, cur_min


def _apply_result_3d(phi, result_x, cz, cy, cx, sub_size):
    """Write optimised sub-volume back into *phi*."""
    sz, sy, sx = _unpack_size_3d(sub_size)
    hz, hy, hx = sz // 2, sy // 2, sx // 2
    hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx
    voxels = sz * sy * sx

    slc = (slice(cz - hz, cz + hz_hi),
           slice(cy - hy, cy + hy_hi),
           slice(cx - hx, cx + hx_hi))

    phi[2][slc] = result_x[:voxels].reshape(sz, sy, sx)              # dx
    phi[1][slc] = result_x[voxels:2 * voxels].reshape(sz, sy, sx)    # dy
    phi[0][slc] = result_x[2 * voxels:].reshape(sz, sy, sx)          # dz


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


# ---------------------------------------------------------------------------
# SLSQP worker
# ---------------------------------------------------------------------------
def _optimize_single_window_3d(
    phi_sub_flat,
    phi_init_sub_flat,
    subvolume_size,
    freeze_mask,
    threshold,
    max_minimize_iter,
    method_name,
):
    """Run SLSQP on one 3D sub-volume.  Returns ``(result_x, elapsed)``."""
    constraints = _build_constraints_3d(
        phi_sub_flat, subvolume_size, freeze_mask, threshold,
    )

    t0 = time.time()
    result = minimize(
        lambda phi1: objectiveEuc(phi1, phi_init_sub_flat),
        phi_sub_flat,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": False},
        method=method_name,
    )
    elapsed = time.time() - t0
    return result.x, elapsed


# ---------------------------------------------------------------------------
# Full-grid optimisation fallback (non-cubic grids)
# ---------------------------------------------------------------------------
def _full_grid_step_3d(phi, phi_init, D, H, W, threshold,
                       max_minimize_iter, methodName, verbose):
    """Optimize the entire D x H x W grid at once."""
    voxels = D * H * W
    phi_flat = np.concatenate([phi[2].flatten(),
                               phi[1].flatten(),
                               phi[0].flatten()])
    phi_init_flat = np.concatenate([phi_init[2].flatten(),
                                    phi_init[1].flatten(),
                                    phi_init[0].flatten()])

    def jac_con(pf):
        dx = pf[:voxels].reshape(D, H, W)
        dy = pf[voxels:2 * voxels].reshape(D, H, W)
        dz = pf[2 * voxels:].reshape(D, H, W)
        return _numpy_jdet_3d(dz, dy, dx).flatten()

    constraints = [NonlinearConstraint(jac_con, threshold, np.inf)]

    _log(verbose, 1,
         f"  [full-grid] Optimizing entire {D}x{H}x{W} grid "
         f"({3 * voxels} variables)")

    result = minimize(
        lambda phi1: objectiveEuc(phi1, phi_init_flat),
        phi_flat,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": verbose >= 2},
        method=methodName,
    )

    phi[2] = result.x[:voxels].reshape(D, H, W)
    phi[1] = result.x[voxels:2 * voxels].reshape(D, H, W)
    phi[0] = result.x[2 * voxels:].reshape(D, H, W)


# ---------------------------------------------------------------------------
# Serial inner loop
# ---------------------------------------------------------------------------
def _serial_fix_voxel(
    neg_index, phi, phi_init, jacobian_matrix,
    volume_shape, window_counts,
    max_per_index_iter, max_minimize_iter,
    max_window, threshold, err_tol, methodName, verbose,
    error_list, num_neg_jac, min_jdet_list, iter_times,
):
    """Fix a single voxel using the serial adaptive-window inner loop.

    Mutates *phi* and the accumulator lists in-place.

    Returns
    -------
    jacobian_matrix, subvolume_size, per_index_iter, (cz, cy, cx)
    """
    subvolume_size, bbox_center = neg_jdet_bounding_window_3d(
        jacobian_matrix, neg_index, threshold, err_tol)
    max_sz, max_sy, max_sx = _unpack_size_3d(max_window)
    subvolume_size = (min(subvolume_size[0], max_sz),
                      min(subvolume_size[1], max_sy),
                      min(subvolume_size[2], max_sx))

    per_index_iter = 0
    window_reached_max = False

    while per_index_iter < max_per_index_iter:
        per_index_iter += 1

        cz, cy, cx = get_nearest_center_3d(
            bbox_center, volume_shape, subvolume_size)
        sz, sy, sx = _unpack_size_3d(subvolume_size)
        hz, hy, hx = sz // 2, sy // 2, sx // 2
        hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx

        phi_init_sub_flat = get_phi_sub_flat_3d(
            phi_init, cz, cy, cx, subvolume_size)
        phi_sub_flat = get_phi_sub_flat_3d(
            phi, cz, cy, cx, subvolume_size)

        if per_index_iter > 1:
            _log(verbose, 2,
                 f"  [window] Index {neg_index}: window grew to "
                 f"{sz}x{sy}x{sx} (sub-iter {per_index_iter})")

        is_at_edge, w_max = _edge_flags_3d(
            cz, cy, cx, subvolume_size, volume_shape, max_window)
        window_reached_max = window_reached_max or w_max
        freeze_mask = _frozen_boundary_mask_3d(
            cz, cy, cx, subvolume_size, volume_shape)

        _log(verbose, 2,
             f"  [edge] at_edge={is_at_edge}  "
             f"window_reached_max={window_reached_max}  "
             f"frozen_voxels={int(freeze_mask.sum())}")

        # Skip optimizer if frozen edges have negative Jdet
        if (freeze_mask.any()
                and not _frozen_edges_clean_3d(
                    jacobian_matrix, cz, cy, cx,
                    subvolume_size, threshold, err_tol, freeze_mask)):
            _log(verbose, 2,
                 f"  [skip] Frozen edges have neg Jdet at "
                 f"win {sz}x{sy}x{sx} — growing")
            if sz < max_sz or sy < max_sy or sx < max_sx:
                subvolume_size = (min(sz + 2, max_sz),
                                  min(sy + 2, max_sy),
                                  min(sx + 2, max_sx))
            continue

        window_counts[subvolume_size] += 1

        result_x, elapsed = _optimize_single_window_3d(
            phi_sub_flat, phi_init_sub_flat, subvolume_size,
            freeze_mask,
            threshold, max_minimize_iter, methodName,
        )
        iter_times.append(elapsed)

        _apply_result_3d(phi, result_x, cz, cy, cx, subvolume_size)

        jacobian_matrix, cur_neg, cur_min = _update_metrics_3d(
            phi, phi_init, num_neg_jac, min_jdet_list, error_list)

        _log(verbose, 2,
             f"  [sub-Jdet] centre ({cz},{cy},{cx}) "
             f"window {sz}x{sy}x{sx}")

        if float(jacobian_matrix.min()) > threshold - err_tol:
            break

        # Check local window and grow for next sub-iteration
        if window_reached_max:
            break
        local = jacobian_matrix[cz - hz:cz + hz_hi,
                                cy - hy:cy + hy_hi,
                                cx - hx:cx + hx_hi]
        if not (local < threshold - err_tol).any():
            break
        if sz < max_sz or sy < max_sy or sx < max_sx:
            subvolume_size = (min(sz + 2, max_sz),
                              min(sy + 2, max_sy),
                              min(sx + 2, max_sx))
        else:
            window_reached_max = True

    return jacobian_matrix, subvolume_size, per_index_iter, (cz, cy, cx)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def _save_results_3d(save_path, *, method, threshold, err_tol, max_iterations,
                     max_per_index_iter, max_minimize_iter,
                     D, H, W, elapsed, final_err, init_neg, final_neg,
                     init_min, final_min, iteration, phi, error_list,
                     num_neg_jac, iter_times, min_jdet_list, window_counts):
    os.makedirs(save_path, exist_ok=True)

    output_text = "Settings:\n"
    output_text += f"\tMethod: {method}\n"
    output_text += f"\tThreshold: {threshold}\n"
    output_text += f"\tError tolerance: {err_tol}\n"
    output_text += f"\tMax iterations: {max_iterations}\n"
    output_text += f"\tMax per index iterations: {max_per_index_iter}\n"
    output_text += f"\tMax minimize iterations: {max_minimize_iter}\n"
    output_text += "\nResults:\n"
    output_text += f"\tInput deformation field resolution (D x H x W): {D} x {H} x {W}\n"
    output_text += f"\tTotal run-time: {elapsed} seconds\n"
    output_text += f"\tFinal L2 error: {final_err}\n"
    output_text += f"\tStarting number of non-positive Jacobian determinants: {init_neg}\n"
    output_text += f"\tFinal number of non-positive Jacobian determinants: {final_neg}\n"
    output_text += f"\tStarting Jacobian determinant minimum value: {init_min}\n"
    output_text += f"\tFinal Jacobian determinant minimum value: {final_min}\n"
    output_text += f"\tNumber of index iterations: {iteration}"

    with open(os.path.join(save_path, "results.txt"), "w") as f:
        f.write(output_text)

    np.save(os.path.join(save_path, "phi.npy"), phi)
    np.save(os.path.join(save_path, "error_list.npy"), error_list)
    np.save(os.path.join(save_path, "num_neg_jac.npy"), num_neg_jac)
    np.save(os.path.join(save_path, "iter_times.npy"), iter_times)
    np.save(os.path.join(save_path, "min_jdet_list.npy"), min_jdet_list)

    with open(os.path.join(save_path, "window_counts.csv"), "w") as f:
        f.write("window_depth,window_height,window_width,count\n")
        for ws in sorted(window_counts):
            sz, sy, sx = _unpack_size_3d(ws)
            f.write(f"{sz},{sy},{sx},{window_counts[ws]}\n")


# ---------------------------------------------------------------------------
# Main iterative SLSQP algorithm (3D)
# ---------------------------------------------------------------------------
def iterative_3d(
    deformation,
    methodName="SLSQP",
    verbose=1,
    save_path=None,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
):
    """Iterative SLSQP correction of negative Jacobian determinants in 3D.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``
        Input deformation field with channels ``[dz, dy, dx]``.
    methodName : str
        Optimiser method passed to ``scipy.optimize.minimize``.
    verbose : int
        ``0`` = silent, ``1`` = per-iteration progress, ``2`` = debug.
    save_path : str or None
        Directory to save results.  ``None`` disables saving.
    threshold, err_tol, max_iterations, max_per_index_iter,
    max_minimize_iter :
        Override the corresponding default parameters.

    Returns
    -------
    phi : ndarray, shape ``(3, D, H, W)``
        Corrected displacement field ``[dz, dy, dx]``.
    """
    p = _resolve_params(threshold=threshold, err_tol=err_tol,
                        max_iterations=max_iterations,
                        max_per_index_iter=max_per_index_iter,
                        max_minimize_iter=max_minimize_iter)
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]

    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

    # Accumulators
    error_list = []
    num_neg_jac = []
    iter_times = []
    min_jdet_list = []
    window_counts = defaultdict(int)

    start_time = time.time()
    phi, phi_init, D, H, W = _init_phi_3d(deformation)
    volume_shape = (D, H, W)
    max_window = (D, H, W)

    _log(verbose, 1,
         f"[init] Grid {D}x{H}x{W}  |  threshold={threshold}  "
         f"|  method={methodName}")

    jacobian_matrix, init_neg, init_min = _update_metrics_3d(
        phi, phi_init, num_neg_jac, min_jdet_list)

    _log(verbose, 1,
         f"[init] Neg-Jdet voxels: {init_neg}  |  min Jdet: {init_min:.6f}")

    iteration = 0
    while (iteration < max_iterations
           and (jacobian_matrix <= threshold - err_tol).any()):
        iteration += 1

        neg_index = argmin_worst_voxel(jacobian_matrix)

        jacobian_matrix, subvolume_size, per_index_iter, (cz, cy, cx) = \
            _serial_fix_voxel(
                neg_index, phi, phi_init, jacobian_matrix,
                volume_shape, window_counts,
                max_per_index_iter, max_minimize_iter,
                max_window, threshold, err_tol, methodName, verbose,
                error_list, num_neg_jac, min_jdet_list, iter_times,
            )

        # Full-grid fallback for non-cubic grids
        sz, sy, sx = _unpack_size_3d(subvolume_size)
        if (sz >= D and sy >= H and sx >= W
                and not (D == H == W)
                and (jacobian_matrix <= threshold - err_tol).any()):
            iter_start = time.time()
            _full_grid_step_3d(phi, phi_init, D, H, W, threshold,
                               max_minimize_iter, methodName, verbose)
            iter_times.append(time.time() - iter_start)

            jacobian_matrix, cur_neg, cur_min = _update_metrics_3d(
                phi, phi_init, num_neg_jac, min_jdet_list, error_list)

        cur_neg = int((jacobian_matrix <= 0).sum())
        cur_min = float(jacobian_matrix.min())
        cur_err = error_list[-1] if error_list else 0.0
        _log(verbose, 1,
             f"[iter {iteration:4d}]  fix ({neg_index[0]:3d},"
             f"{neg_index[1]:3d},{neg_index[2]:3d})  "
             f"win {sz}x{sy}x{sx}  neg_jdet {cur_neg:5d}  "
             f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
             f"sub-iters {per_index_iter}")

        if float(jacobian_matrix.min()) > threshold - err_tol:
            _log(verbose, 1,
                 f"[done] All Jdet > threshold after iter {iteration}")
            break

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _log(verbose, 1, "")
    _log(verbose, 1, "=" * 60)
    _log(verbose, 1, f"  SUMMARY  ({methodName} — 3D)")
    _log(verbose, 1, "-" * 60)
    _log(verbose, 1, f"  Grid size        : {D} x {H} x {W}")
    _log(verbose, 1, f"  Iterations       : {iteration}")
    _log(verbose, 1, f"  Neg-Jdet  {init_neg:>5d} -> {final_neg:>5d}")
    _log(verbose, 1, f"  Min Jdet  {init_min:+.6f} -> {final_min:+.6f}")
    _log(verbose, 1, f"  L2 error         : {final_err:.6f}")
    _log(verbose, 1, f"  Time             : {elapsed:.2f}s")
    _log(verbose, 1, "=" * 60)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results_3d(
            save_path, method=methodName, threshold=threshold,
            err_tol=err_tol, max_iterations=max_iterations,
            max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            D=D, H=H, W=W, elapsed=elapsed, final_err=final_err,
            init_neg=init_neg, final_neg=final_neg, init_min=init_min,
            final_min=final_min, iteration=iteration, phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi
