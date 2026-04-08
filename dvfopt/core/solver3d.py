"""3D solver primitives: init/metrics/save, sub-volume SLSQP worker."""

import time

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from dvfopt._defaults import _log, _unpack_size_3d, _adaptive_maxiter
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_3d, jacobian_det3D
from dvfopt.core.objective import objective_euc
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

    Uses a two-layer design matching `_patch_jacobian_2d`:
    - Write-back region: sub-volume ± 1-voxel border (what gets stored).
    - Computation region: write-back ± 1 extra voxel for central-difference
      context, so ``np.gradient`` uses central differences at the write-back
      boundary rather than one-sided differences.

    Mutates *jacobian_matrix* in place and returns it.
    """
    cz, cy, cx = center
    sz, sy, sx = _unpack_size_3d(sub_size)
    hz, hy, hx = sz // 2, sy // 2, sx // 2
    hz_hi, hy_hi, hx_hi = sz - hz, sy - hy, sx - hx
    D, H, W = phi.shape[1], phi.shape[2], phi.shape[3]

    # Write-back region: sub-volume + 1-voxel border
    wz0 = max(cz - hz - 1, 0);  wz1 = min(cz + hz_hi + 1, D)
    wy0 = max(cy - hy - 1, 0);  wy1 = min(cy + hy_hi + 1, H)
    wx0 = max(cx - hx - 1, 0);  wx1 = min(cx + hx_hi + 1, W)

    # Computation region: 1 extra voxel on each side for gradient context
    cz0 = max(wz0 - 1, 0);  cz1 = min(wz1 + 1, D)
    cy0 = max(wy0 - 1, 0);  cy1 = min(wy1 + 1, H)
    cx0 = max(wx0 - 1, 0);  cx1 = min(wx1 + 1, W)

    jdet_comp = _numpy_jdet_3d(phi[0, cz0:cz1, cy0:cy1, cx0:cx1],
                                phi[1, cz0:cz1, cy0:cy1, cx0:cx1],
                                phi[2, cz0:cz1, cy0:cy1, cx0:cx1])

    # Trim to write-back region
    tz0 = wz0 - cz0;  ty0 = wy0 - cy0;  tx0 = wx0 - cx0
    jacobian_matrix[wz0:wz1, wy0:wy1, wx0:wx1] = \
        jdet_comp[tz0:tz0 + (wz1 - wz0),
                  ty0:ty0 + (wy1 - wy0),
                  tx0:tx0 + (wx1 - wx0)]
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
    """Run SLSQP on one 3D sub-volume.  Returns ``(result_x, elapsed, success)``."""
    constraints = _build_constraints_3d(
        phi_sub_flat, subvolume_size, freeze_mask, threshold,
    )

    t0 = time.time()
    result = minimize(
        lambda phi1: objective_euc(phi1, phi_init_sub_flat),
        phi_sub_flat,
        jac=True,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": False},
        method=method_name,
    )
    elapsed = time.time() - t0
    if not np.all(np.isfinite(result.x)):
        return phi_sub_flat, elapsed, False
    return result.x, elapsed, result.success


# ---------------------------------------------------------------------------
# Full-grid optimisation fallback (non-cubic grids)
# ---------------------------------------------------------------------------
def _full_grid_step_3d(phi, phi_init, D, H, W, threshold,
                       max_minimize_iter, method_name, verbose):
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
        lambda phi1: objective_euc(phi1, phi_init_flat),
        phi_flat,
        jac=True,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": verbose >= 2},
        method=method_name,
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
    max_window, threshold, err_tol, method_name, verbose,
    error_list, num_neg_jac, min_jdet_list, iter_times,
    min_window=(3, 3, 3),
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
    min_sz, min_sy, min_sx = _unpack_size_3d(min_window)
    subvolume_size = (max(min(subvolume_size[0], max_sz), min_sz),
                      max(min(subvolume_size[1], max_sy), min_sy),
                      max(min(subvolume_size[2], max_sx), min_sx))

    per_index_iter = 0
    window_reached_max = False

    while per_index_iter < max_per_index_iter:
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

        # Skip optimizer if frozen edges have negative Jdet (infeasible constraint).
        # Only skip when the window can still grow; at max window the optimizer
        # must run unconditionally (no frozen edges apply when window_reached_max).
        if (freeze_mask.any()
                and not window_reached_max
                and not is_at_edge
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

        per_index_iter += 1
        window_counts[subvolume_size] += 1

        _n_vars = 3 * sz * sy * sx
        _eff_max_iter = _adaptive_maxiter(_n_vars, max_minimize_iter)

        result_x, elapsed, opt_success = _optimize_single_window_3d(
            phi_sub_flat, phi_init_sub_flat, subvolume_size,
            freeze_mask,
            threshold, _eff_max_iter, method_name,
        )
        iter_times.append(elapsed)
        if not opt_success:
            _log(verbose, 2,
                 f"  [warn] SLSQP did not converge at win {sz}x{sy}x{sx} "
                 f"centre ({cz},{cy},{cx})")

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


