"""3D solver primitives: init/metrics/save, sub-volume SLSQP worker."""

import time

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from dvfopt._defaults import _log, _unpack_size_3d
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_3d, jacobian_det3D
from dvfopt.core.objective import objectiveEuc
from dvfopt.core.spatial3d import (
    get_nearest_center_3d,
    argmin_worst_voxel,
    neg_jdet_bounding_window_3d,
    _frozen_boundary_mask_3d,
    _frozen_edges_clean_3d,
    get_phi_sub_flat_3d,
    _edge_flags_3d,
)
from dvfopt.core.constraints3d import (
    jacobian_constraint_3d,
    _build_constraints_3d,
)
from dvfopt.core.gradients3d import jdet_constraint_jacobian_3d


# ---------------------------------------------------------------------------
# 3D init / metrics helpers
# ---------------------------------------------------------------------------
def _init_phi_3d(deformation):
    """Create the initial ``phi`` working array from a ``(3, D, H, W)`` deformation.

    Returns ``(phi, phi_init, D, H, W)``.
    """
    D, H, W = deformation.shape[1:]
    phi = deformation.copy().astype(np.float64)
    phi_init = phi.copy()
    return phi, phi_init, D, H, W


def _update_metrics_3d(phi, phi_init, num_neg_jac, min_jdet_list,
                       error_list=None, jacobian_matrix=None,
                       patch_center=None, patch_size=None):
    """Recompute Jacobian and append to accumulator lists.

    When *jacobian_matrix*, *patch_center*, and *patch_size* are provided,
    only the affected sub-volume + 1-voxel border is recomputed.

    Returns ``(jacobian_matrix, cur_neg, cur_min)``.
    """
    if jacobian_matrix is not None and patch_center is not None and patch_size is not None:
        jac = _patch_jacobian_3d(jacobian_matrix, phi, patch_center, patch_size)
    elif jacobian_matrix is not None and patch_center is None:
        jac = jacobian_matrix
    else:
        jac = jacobian_det3D(phi)  # (D, H, W)
    cur_neg = int((jac <= 0).sum())
    cur_min = float(jac.min())
    num_neg_jac.append(cur_neg)
    min_jdet_list.append(cur_min)
    if error_list is not None:
        error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))
    return jac, cur_neg, cur_min


def _patch_jacobian_3d(jacobian_matrix, phi, center, sub_size):
    """Recompute Jacobian only in the modified sub-volume + 1-voxel border.

    Mutates *jacobian_matrix* in place and returns it.
    """
    cz, cy, cx = center
    sz, sy, sx = _unpack_size_3d(sub_size)
    hz, hy, hx = sz // 2, sy // 2, sx // 2
    hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx
    D, H, W = phi.shape[1], phi.shape[2], phi.shape[3]

    z0 = max(cz - hz - 1, 0)
    z1 = min(cz + hz_hi + 1, D)
    y0 = max(cy - hy - 1, 0)
    y1 = min(cy + hy_hi + 1, H)
    x0 = max(cx - hx - 1, 0)
    x1 = min(cx + hx_hi + 1, W)

    dz_sub = phi[0, z0:z1, y0:y1, x0:x1]
    dy_sub = phi[1, z0:z1, y0:y1, x0:x1]
    dx_sub = phi[2, z0:z1, y0:y1, x0:x1]
    jdet_sub = _numpy_jdet_3d(dz_sub, dy_sub, dx_sub)
    jacobian_matrix[z0:z1, y0:y1, x0:x1] = jdet_sub
    return jacobian_matrix


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


# ---------------------------------------------------------------------------
# 3D SLSQP worker
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
        jac=True,
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

    grid_size = (D, H, W)
    constraints = [NonlinearConstraint(
        jac_con, threshold, np.inf,
        jac=lambda pf: jdet_constraint_jacobian_3d(pf, grid_size),
    )]

    _log(verbose, 1,
         f"  [full-grid] Optimizing entire {D}x{H}x{W} grid "
         f"({3 * voxels} variables)")

    result = minimize(
        lambda phi1: objectiveEuc(phi1, phi_init_flat),
        phi_flat,
        jac=True,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": verbose >= 2},
        method=methodName,
    )

    phi[2] = result.x[:voxels].reshape(D, H, W)
    phi[1] = result.x[voxels:2 * voxels].reshape(D, H, W)
    phi[0] = result.x[2 * voxels:].reshape(D, H, W)


# ---------------------------------------------------------------------------
# Serial inner loop — fix a single voxel
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
            phi, phi_init, num_neg_jac, min_jdet_list, error_list,
            jacobian_matrix=jacobian_matrix, patch_center=(cz, cy, cx),
            patch_size=subvolume_size)

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


