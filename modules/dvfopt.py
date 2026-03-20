"""
Iterative SLSQP optimisation for correcting negative Jacobian determinants
in 2D deformation (displacement) fields.

This module contains both the serial and hybrid-parallel algorithms, plus
all shared helpers.  No matplotlib or pandas dependency — visualisation
lives in ``modules.dvfviz``.

Usage::

    from modules.dvfopt import iterative_with_jacobians2, jacobian_det2D
    from modules.dvfopt import iterative_parallel  # hybrid parallel variant

Verbosity levels (``verbose`` parameter):

* ``0`` — silent, no output
* ``1`` — one-line progress per outer iteration + final summary
* ``2`` — full debug output (edge masks, constraints, sub-matrices)
"""

import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


# ---------------------------------------------------------------------------
# DVF generation utilities
# ---------------------------------------------------------------------------
def generate_random_dvf(shape, max_magnitude=5.0, seed=None):
    """Generate a random 2D deformation vector field (DVF).

    Parameters
    ----------
    shape : tuple
        ``(3, 1, H, W)`` — standard deformation field shape.
    max_magnitude : float
        Max displacement in pixels (uniform in ``[-mag, +mag]``).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(3, 1, H, W)``
    """
    if seed is not None:
        np.random.seed(seed)

    C, _, H, W = shape
    assert C == 3, "DVF must have 3 channels (dz, dy, dx)"
    return np.random.uniform(-max_magnitude, max_magnitude, size=shape).astype(np.float32)


def scale_dvf(dvf, new_size):
    """Rescale a ``(3, 1, H, W)`` deformation field to *new_size* ``(new_H, new_W)``.

    Spatial interpolation is bilinear (``order=1``) and displacement
    magnitudes are scaled proportionally.
    """
    C, _, H, W = dvf.shape
    new_H, new_W = new_size
    scale_y = new_H / H
    scale_x = new_W / W

    dvf_resized = np.zeros((C, 1, new_H, new_W), dtype=dvf.dtype)
    for c in range(C):
        dvf_resized[c, 0] = zoom(dvf[c, 0], (scale_y, scale_x), order=1)

    dvf_resized[2, 0] *= scale_x  # dx
    dvf_resized[1, 0] *= scale_y  # dy
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
# Internal logging helpers
# ---------------------------------------------------------------------------
def _log(verbose, level, msg):
    """Print *msg* if *verbose* >= *level*."""
    if verbose >= level:
        print(msg)


# ---------------------------------------------------------------------------
# Window size helper
# ---------------------------------------------------------------------------
def _unpack_size(submatrix_size):
    """Normalize *submatrix_size* to a ``(sy, sx)`` tuple.

    Accepts a single int (square) or a 2-tuple/list (rectangular).
    """
    if isinstance(submatrix_size, (tuple, list)):
        return int(submatrix_size[0]), int(submatrix_size[1])
    return int(submatrix_size), int(submatrix_size)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------
def objectiveEuc(phi, phi_init):
    """L2 norm objective function."""
    return np.linalg.norm(phi - phi_init)


# ---------------------------------------------------------------------------
# Jacobian helpers
# ---------------------------------------------------------------------------
def _numpy_jdet_2d(dy, dx):
    """Compute 2D Jacobian determinant from displacement components (numpy).

    Uses central differences for interior pixels, matching SimpleITK's
    ``DisplacementFieldJacobianDeterminant`` for interior values.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    ndarray, shape ``(H, W)``
    """
    ddx_dx = np.gradient(dx, axis=1)  # ∂dx/∂x
    ddy_dy = np.gradient(dy, axis=0)  # ∂dy/∂y
    ddx_dy = np.gradient(dx, axis=0)  # ∂dx/∂y
    ddy_dx = np.gradient(dy, axis=1)  # ∂dy/∂x
    return (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx


def jacobian_det2D(phi_xy):
    """Compute the Jacobian determinant from a ``(2, H, W)`` phi array.

    Also accepts ``(2, 1, H, W)`` (the extra unit dimension is squeezed).
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    jdet = _numpy_jdet_2d(dy, dx)
    return jdet[np.newaxis, :, :]  # (1, H, W) to match existing API


def jacobian_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened Jacobian determinant values for optimiser constraints."""
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    jdet = _numpy_jdet_2d(dy, dx)
    if exclude_boundaries:
        return jdet[1:-1, 1:-1].flatten()
    else:
        return jdet.flatten()


# ---------------------------------------------------------------------------
# Shoelace (geometric quad-area) helpers
# ---------------------------------------------------------------------------
def _shoelace_areas_2d(dy, dx):
    """Signed area of each deformed quad cell via the shoelace formula.

    Uses vertex order TL → TR → BR → BL, matching ``dvfviz._quad_signed_areas``.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``
        Displacement components.

    Returns
    -------
    ndarray, shape ``(H-1, W-1)``
    """
    H, W = dy.shape
    ref_y, ref_x = np.mgrid[:H, :W]
    def_x = ref_x + dx
    def_y = ref_y + dy

    x0, y0 = def_x[:-1, :-1], def_y[:-1, :-1]   # TL
    x1, y1 = def_x[:-1, 1:],  def_y[:-1, 1:]     # TR
    x2, y2 = def_x[1:, 1:],   def_y[1:, 1:]      # BR
    x3, y3 = def_x[1:, :-1],  def_y[1:, :-1]     # BL
    return 0.5 * ((x0*y1 - x1*y0) + (x1*y2 - x2*y1)
                  + (x2*y3 - x3*y2) + (x3*y0 - x0*y3))


def shoelace_det2D(phi_xy):
    """Compute shoelace quad-cell areas from a ``(2, H, W)`` phi array.

    Returns shape ``(1, H-1, W-1)`` — analogous to ``jacobian_det2D``.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    return _shoelace_areas_2d(dy, dx)[np.newaxis, :, :]


def shoelace_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened shoelace quad areas for optimiser constraints.

    Analogous to ``jacobian_constraint`` but checks geometric cell areas
    instead of gradient-based Jacobian determinants.
    """
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    areas = _shoelace_areas_2d(dy, dx)
    if exclude_boundaries:
        return areas[1:-1, 1:-1].flatten()
    else:
        return areas.flatten()


# ---------------------------------------------------------------------------
# Monotonicity (global injectivity) helpers
# ---------------------------------------------------------------------------
def _monotonicity_diffs_2d(dy, dx):
    """Forward-difference monotonicity metrics for deformed coordinates.

    For a structured grid with unit spacing, deformed positions are
    ``def_x[i,j] = j + dx[i,j]``.  Global injectivity (on the discrete
    structured grid) is guaranteed when deformed coordinates are strictly
    monotonic along both axes:

    * ``h_mono[i,j] = 1 + dx[i,j+1] - dx[i,j] > 0``  (x increases along rows)
    * ``v_mono[i,j] = 1 + dy[i+1,j] - dy[i,j] > 0``  (y increases along cols)

    Returns ``(h_mono, v_mono)`` with shapes ``(H, W-1)`` and ``(H-1, W)``.
    """
    h_mono = 1.0 + np.diff(dx, axis=1)   # (H, W-1)
    v_mono = 1.0 + np.diff(dy, axis=0)   # (H-1, W)
    return h_mono, v_mono


def injectivity_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened monotonicity diffs for the SLSQP injectivity constraint.

    All returned values must be > threshold for montonicity (and hence
    global injectivity on the structured grid) to hold.
    """
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
    if exclude_boundaries:
        h_vals = h_mono[1:-1, 1:-1].flatten()
        v_vals = v_mono[1:-1, 1:-1].flatten()
    else:
        h_vals = h_mono.flatten()
        v_vals = v_mono.flatten()
    return np.concatenate([h_vals, v_vals])


def _quality_map(phi, enforce_shoelace, enforce_injectivity=False):
    """Per-pixel quality metric combining gradient-Jdet and optional extras.

    When both *enforce_shoelace* and *enforce_injectivity* are ``False``,
    returns ``jacobian_det2D(phi)``.  Otherwise, each active metric is
    spread to per-pixel values and the element-wise minimum is returned
    so that the worst violation drives pixel selection and convergence.

    Returns shape ``(1, H, W)`` — same as ``jacobian_det2D``.
    """
    jdet = jacobian_det2D(phi)
    if not enforce_shoelace and not enforce_injectivity:
        return jdet
    result = jdet.copy()
    H, W = jdet.shape[1:]

    if enforce_shoelace:
        areas = shoelace_det2D(phi)           # (1, H-1, W-1)
        shoe = np.full((1, H, W), np.inf)
        a = areas[0]
        shoe[0, :-1, :-1] = np.minimum(shoe[0, :-1, :-1], a)
        shoe[0, :-1, 1:]  = np.minimum(shoe[0, :-1, 1:],  a)
        shoe[0, 1:,  :-1] = np.minimum(shoe[0, 1:,  :-1], a)
        shoe[0, 1:,  1:]  = np.minimum(shoe[0, 1:,  1:],  a)
        result = np.minimum(result, shoe)

    if enforce_injectivity:
        dy = phi[0]
        dx = phi[1]
        h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
        mono = np.full((1, H, W), np.inf)
        # h_mono is (H, W-1): gap between col j and col j+1
        mono[0, :, :-1] = np.minimum(mono[0, :, :-1], h_mono)
        mono[0, :, 1:]  = np.minimum(mono[0, :, 1:],  h_mono)
        # v_mono is (H-1, W): gap between row i and row i+1
        mono[0, :-1, :] = np.minimum(mono[0, :-1, :], v_mono)
        mono[0, 1:,  :] = np.minimum(mono[0, 1:,  :], v_mono)
        result = np.minimum(result, mono)

    return result


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
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


def argmin_excluding_edges(jacobian_matrix):
    """Index of the pixel with the lowest Jacobian determinant, excluding edges."""
    inner = jacobian_matrix[0, 1:-1, 1:-1]
    flat_index = np.argmin(inner)
    inner_idx = np.unravel_index(flat_index, inner.shape)
    return (inner_idx[0] + 1, inner_idx[1] + 1)


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
    from scipy.ndimage import label

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


# ---------------------------------------------------------------------------
# Shared constraint builder
# ---------------------------------------------------------------------------
def _build_constraints(phi_sub_flat, submatrix_size, is_at_edge,
                       window_reached_max, threshold, enforce_shoelace=False,
                       enforce_injectivity=False):
    """Build SLSQP constraints for a sub-window optimisation.

    Returns a list of constraint objects suitable for
    ``scipy.optimize.minimize``.

    When *enforce_shoelace* is ``True``, an additional
    ``NonlinearConstraint`` requires all shoelace quad-cell areas to
    exceed *threshold* as well.

    When *enforce_injectivity* is ``True``, an additional
    ``NonlinearConstraint`` enforces monotonicity of deformed coordinates
    (sufficient condition for global injectivity on structured grids).
    """
    exclude_bounds = not is_at_edge and not window_reached_max

    nlc = NonlinearConstraint(
        lambda phi1: jacobian_constraint(phi1, submatrix_size, exclude_bounds),
        threshold, np.inf,
    )
    constraints = [nlc]

    if enforce_shoelace:
        constraints.append(NonlinearConstraint(
            lambda phi1: shoelace_constraint(phi1, submatrix_size, exclude_bounds),
            threshold, np.inf,
        ))

    if enforce_injectivity:
        constraints.append(NonlinearConstraint(
            lambda phi1: injectivity_constraint(phi1, submatrix_size, exclude_bounds),
            threshold, np.inf,
        ))

    if exclude_bounds:
        sy, sx = _unpack_size(submatrix_size)
        edge_mask = np.zeros((sy, sx), dtype=bool)
        edge_mask[[0, -1], :] = True
        edge_mask[:, [0, -1]] = True

        edge_indices = np.argwhere(edge_mask)
        fixed_indices = []
        y_offset_sub = sy * sx
        for y, x in edge_indices:
            idx = y * sx + x
            fixed_indices.extend([idx, idx + y_offset_sub])

        fixed_values = phi_sub_flat[fixed_indices]
        A_eq = np.zeros((len(fixed_indices), phi_sub_flat.size))
        for row, idx in enumerate(fixed_indices):
            A_eq[row, idx] = 1

        constraints.append(LinearConstraint(A_eq, fixed_values, fixed_values))

    return constraints


# ---------------------------------------------------------------------------
# Shared init / save helpers
# ---------------------------------------------------------------------------
def _resolve_params(**overrides):
    """Merge *overrides* into ``DEFAULT_PARAMS``, returning resolved dict."""
    p = dict(DEFAULT_PARAMS)
    for name, val in overrides.items():
        if val is not None:
            p[name] = val
    return p


def _init_phi(deformation_i):
    """Create the initial ``phi`` working array from a ``(3,1,H,W)`` deformation.

    Returns ``(phi, phi_init, H, W)``.
    """
    H, W = deformation_i.shape[-2:]
    phi = np.zeros((2, H, W))
    phi[1] = deformation_i[-1]
    phi[0] = deformation_i[-2]
    phi_init = phi.copy()
    return phi, phi_init, H, W


def _update_metrics(phi, phi_init, enforce_shoelace, enforce_injectivity,
                    num_neg_jac, min_jdet_list, error_list=None):
    """Recompute Jacobian/quality matrices and append to accumulator lists.

    Parameters
    ----------
    error_list : list or None
        When not ``None``, the L2 error is appended.

    Returns
    -------
    jacobian_matrix, quality_matrix, cur_neg, cur_min
    """
    jac = jacobian_det2D(phi)
    use_q = enforce_shoelace or enforce_injectivity
    qm = _quality_map(phi, enforce_shoelace, enforce_injectivity) if use_q else jac
    cur_neg = int((jac <= 0).sum())
    cur_min = float(jac.min())
    num_neg_jac.append(cur_neg)
    min_jdet_list.append(cur_min)
    if error_list is not None:
        error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))
    return jac, qm, cur_neg, cur_min


def _save_results(save_path, *, method, threshold, err_tol, max_iterations,
                  max_per_index_iter, max_minimize_iter,
                  H, W, elapsed, final_err, init_neg, final_neg, init_min,
                  final_min, iteration, phi, error_list, num_neg_jac,
                  iter_times, min_jdet_list, window_counts,
                  extra_settings="", extra_results=""):
    """Write correction results to *save_path*."""
    os.makedirs(save_path, exist_ok=True)

    output_text = "Settings:\n"
    output_text += f"\tMethod: {method}\n"
    output_text += f"\tThreshold: {threshold}\n"
    output_text += f"\tError tolerance: {err_tol}\n"
    output_text += f"\tMax iterations: {max_iterations}\n"
    output_text += f"\tMax per index iterations: {max_per_index_iter}\n"
    output_text += f"\tMax minimize iterations: {max_minimize_iter}\n"
    if extra_settings:
        output_text += extra_settings
    output_text += "\nResults:\n"
    output_text += f"\tInput deformation field resolution (height x width): {H} x {W}\n"
    output_text += f"\tTotal run-time: {elapsed} seconds\n"
    output_text += f"\tFinal L2 error: {final_err}\n"
    output_text += f"\tStarting number of non-positive Jacobian determinants: {init_neg}\n"
    output_text += f"\tFinal number of non-positive Jacobian determinants: {final_neg}\n"
    output_text += f"\tStarting Jacobian determinant minimum value: {init_min}\n"
    output_text += f"\tFinal Jacobian determinant minimum value: {final_min}\n"
    output_text += f"\tNumber of index iterations: {iteration}"
    if extra_results:
        output_text += "\n" + extra_results

    with open(save_path + "/results.txt", "w") as f:
        f.write(output_text)

    np.save(save_path + "/phi.npy", phi)
    np.save(save_path + "/error_list.npy", error_list)
    np.save(save_path + "/num_neg_jac.npy", num_neg_jac)
    np.save(save_path + "/iter_times.npy", iter_times)
    np.save(save_path + "/min_jdet_list.npy", min_jdet_list)

    with open(save_path + "/window_counts.csv", "w") as f:
        f.write("window_height,window_width,count\n")
        for ws in sorted(window_counts):
            sy, sx = _unpack_size(ws)
            f.write(f"{sy},{sx},{window_counts[ws]}\n")


# ---------------------------------------------------------------------------
# Full-grid optimisation fallback (non-square grids)
# ---------------------------------------------------------------------------
def _full_grid_step(phi, phi_init, H, W, threshold, max_minimize_iter,
                    methodName, verbose, enforce_shoelace, enforce_injectivity):
    """Optimize the entire H×W grid at once.

    Used as a fallback when the square sub-window (capped at
    ``min(H, W)``) cannot cover the full grid.  Constraints are applied
    to **all** pixels (including boundary), matching the behaviour of
    windowed optimisations whose windows touch the grid edge.
    """
    pixels = H * W
    phi_flat = np.concatenate([phi[1].flatten(), phi[0].flatten()])
    phi_init_flat = np.concatenate([phi_init[1].flatten(), phi_init[0].flatten()])

    def jac_con(phi_xy):
        dx = phi_xy[:pixels].reshape(H, W)
        dy = phi_xy[pixels:].reshape(H, W)
        return _numpy_jdet_2d(dy, dx).flatten()

    constraints = [NonlinearConstraint(jac_con, threshold, np.inf)]

    if enforce_shoelace:
        def shoe_con(phi_xy):
            dx = phi_xy[:pixels].reshape(H, W)
            dy = phi_xy[pixels:].reshape(H, W)
            return _shoelace_areas_2d(dy, dx).flatten()
        constraints.append(NonlinearConstraint(shoe_con, threshold, np.inf))

    if enforce_injectivity:
        def inject_con(phi_xy):
            dx = phi_xy[:pixels].reshape(H, W)
            dy = phi_xy[pixels:].reshape(H, W)
            h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
            return np.concatenate([h_mono.flatten(), v_mono.flatten()])
        constraints.append(NonlinearConstraint(inject_con, threshold, np.inf))

    _log(verbose, 1,
         f"  [full-grid] Optimizing entire {H}x{W} grid "
         f"({2 * pixels} variables)")

    result = minimize(
        lambda phi1: objectiveEuc(phi1, phi_init_flat),
        phi_flat,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": verbose >= 2},
        method=methodName,
    )

    phi[1] = result.x[:pixels].reshape(H, W)
    phi[0] = result.x[pixels:].reshape(H, W)


# ---------------------------------------------------------------------------
# Main iterative SLSQP algorithm
# ---------------------------------------------------------------------------
def iterative_with_jacobians2(
    deformation_i,
    methodName="SLSQP",
    verbose=1,
    save_path=None,
    plot_every=0,
    plot_callback=None,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
    enforce_shoelace=False,
    enforce_injectivity=False,
):
    """Iterative SLSQP correction of negative Jacobian determinants.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
        Input deformation field with channels ``[dz, dy, dx]``.
    methodName : str
        Optimiser method passed to ``scipy.optimize.minimize``.
    verbose : int
        Verbosity level. ``0`` = silent, ``1`` = per-iteration progress
        line + final summary, ``2`` = full debug output (edge masks,
        constraints, sub-Jacobian matrices).  Accepts ``True``/``False``
        for backward compatibility (mapped to 1/0).
    save_path : str or None
        Directory to save results. ``None`` disables saving.
    plot_every : int
        Show a Jacobian heatmap snapshot every *plot_every* outer
        iterations.  ``0`` disables (default).
    plot_callback : callable or None
        Optional callback receiving ``(deformation_i, phi)``
        after each sub-optimisation.
    threshold, err_tol, max_iterations, max_per_index_iter,
    max_minimize_iter :
        Override the corresponding default parameters.
    enforce_shoelace : bool
        When ``True``, the optimiser also enforces positive shoelace
        quad-cell areas (geometric fold detection) in addition to the
        gradient-based Jacobian determinant.  Convergence and pixel
        selection use both metrics.
    enforce_injectivity : bool
        When ``True``, the optimiser enforces monotonicity of deformed
        coordinates along grid axes — a sufficient condition for global
        injectivity on structured grids.  This is more restrictive than
        Jacobian-only or shoelace enforcement.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
        Corrected displacement field ``[dy, dx]``.
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
    phi, phi_init, H, W = _init_phi(deformation_i)
    slice_shape = (1, H, W)
    max_window = (H, W)
    near_cent_dict = {}

    _log(verbose, 1, f"[init] Grid {H}x{W}  |  threshold={threshold}  |  method={methodName}")
    _log(verbose, 2, f"[init] deformation_i shape: {deformation_i.shape}, phi shape: {phi.shape}")

    # Initial Jacobian
    jacobian_matrix, quality_matrix, init_neg, init_min = _update_metrics(
        phi, phi_init, enforce_shoelace, enforce_injectivity,
        num_neg_jac, min_jdet_list)

    _log(verbose, 1, f"[init] Neg-Jdet pixels: {init_neg}  |  min Jdet: {init_min:.6f}")

    iteration = 0
    while iteration < max_iterations and (quality_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any():
        iteration += 1

        neg_index_tuple = argmin_excluding_edges(quality_matrix)

        # Snapshot bookkeeping — capture pre-step state and initial window.
        _show_snap = plot_every and iteration % plot_every == 0
        if _show_snap:
            _snap_before = jacobian_matrix.copy()
            _init_size, _init_center = neg_jdet_bounding_window(
                quality_matrix, neg_index_tuple, threshold, err_tol)
            _init_size = (min(_init_size[0], H), min(_init_size[1], W))
            cz0, cy0, cx0 = get_nearest_center(
                _init_center, slice_shape, _init_size, near_cent_dict)
            edge0, _ = _edge_flags(cy0, cx0, _init_size,
                                   slice_shape, max_window)
            _snap_windows = [(cy0, cx0, _init_size, edge0)]

        # Delegate to shared serial inner loop
        jacobian_matrix, quality_matrix, submatrix_size, per_index_iter, (cy, cx) = \
            _serial_fix_pixel(
                neg_index_tuple, phi, phi_init, jacobian_matrix,
                slice_shape, near_cent_dict, window_counts,
                max_per_index_iter, max_minimize_iter,
                max_window, threshold, err_tol, methodName, verbose,
                error_list, num_neg_jac, min_jdet_list, iter_times,
                enforce_shoelace=enforce_shoelace,
                enforce_injectivity=enforce_injectivity,
                plot_callback=plot_callback,
                deformation_i=deformation_i,
            )

        # Side-by-side before/after snapshot for this iteration.
        if _show_snap:
            from modules.dvfviz import plot_step_snapshot
            plot_step_snapshot(jacobian_matrix, iteration,
                               int((jacobian_matrix <= 0).sum()),
                               float(jacobian_matrix.min()),
                               windows=_snap_windows,
                               jacobian_before=_snap_before)

        # Full-grid fallback for non-square grids where the window
        # can't cover the entire grid.
        _sub_sy, _sub_sx = _unpack_size(submatrix_size)
        if (_sub_sy >= H and _sub_sx >= W
                and H != W
                and (quality_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any()):
            iter_start = time.time()
            _full_grid_step(phi, phi_init, H, W, threshold,
                            max_minimize_iter, methodName, verbose,
                            enforce_shoelace, enforce_injectivity)
            iter_times.append(time.time() - iter_start)

            jacobian_matrix, quality_matrix, cur_neg, cur_min = _update_metrics(
                phi, phi_init, enforce_shoelace, enforce_injectivity,
                num_neg_jac, min_jdet_list, error_list)

        # One-line progress per outer iteration
        cur_neg = int((jacobian_matrix <= 0).sum())
        cur_min = float(jacobian_matrix.min())
        cur_err = error_list[-1] if error_list else 0.0
        _sy, _sx = _unpack_size(submatrix_size)
        _log(verbose, 1,
             f"[iter {iteration:4d}]  fix ({neg_index_tuple[0]:3d},{neg_index_tuple[1]:3d})  "
             f"win {_sy}x{_sx}  neg_jdet {cur_neg:5d}  "
             f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
             f"sub-iters {per_index_iter}")

        if float(quality_matrix[0, 1:-1, 1:-1].min()) > threshold - err_tol:
            _log(verbose, 1, f"[done] All Jdet > threshold after iter {iteration}")
            break

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _log(verbose, 1, "")
    _log(verbose, 1, "=" * 60)
    _log(verbose, 1, f"  SUMMARY  ({methodName})")
    _log(verbose, 1, "-" * 60)
    _log(verbose, 1, f"  Grid size        : {H} x {W}")
    _log(verbose, 1, f"  Iterations       : {iteration}")
    _log(verbose, 1, f"  Neg-Jdet  {init_neg:>5d} -> {final_neg:>5d}")
    _log(verbose, 1, f"  Min Jdet  {init_min:+.6f} -> {final_min:+.6f}")
    _log(verbose, 1, f"  L2 error         : {final_err:.6f}")
    _log(verbose, 1, f"  Time             : {elapsed:.2f}s")
    _log(verbose, 1, "=" * 60)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results(
            save_path, method=methodName, threshold=threshold, err_tol=err_tol,
            max_iterations=max_iterations, max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            H=H, W=W, elapsed=elapsed, final_err=final_err,
            init_neg=init_neg, final_neg=final_neg, init_min=init_min,
            final_min=final_min, iteration=iteration, phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi


# ===================================================================
# Hybrid parallel iterative SLSQP
# ===================================================================

# ---------------------------------------------------------------------------
# Standalone worker (must be picklable — top-level function)
# ---------------------------------------------------------------------------
def _optimize_single_window(
    phi_sub_flat,
    phi_init_sub_flat,
    submatrix_size,
    is_at_edge,
    window_reached_max,
    threshold,
    max_minimize_iter,
    method_name,
    enforce_shoelace=False,
    enforce_injectivity=False,
):
    """Run SLSQP on one sub-window.  Returns ``(result_x, elapsed)``."""

    constraints = _build_constraints(
        phi_sub_flat, submatrix_size, is_at_edge, window_reached_max, threshold,
        enforce_shoelace=enforce_shoelace,
        enforce_injectivity=enforce_injectivity,
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
# Parallel helpers
# ---------------------------------------------------------------------------
def _find_negative_pixels(jacobian_matrix, threshold, err_tol):
    """Return list of (y, x) for inner pixels below threshold, worst first."""
    inner = jacobian_matrix[0, 1:-1, 1:-1]
    ys, xs = np.where(inner <= threshold - err_tol)
    vals = inner[ys, xs]
    order = np.argsort(vals)
    return [(int(ys[i]) + 1, int(xs[i]) + 1) for i in order]


def _window_bounds(cy, cx, submatrix_size):
    sy, sx = _unpack_size(submatrix_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    return (cy - hy, cy + hy_hi - 1, cx - hx, cx + hx_hi - 1)


def _windows_overlap(b1, b2):
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])


def _select_non_overlapping(neg_pixels, pixel_window_sizes, slice_shape,
                             near_cent_dict, pixel_bbox_centers=None):
    """Greedily select non-overlapping windows (each pixel has its own size)."""
    selected = []
    used_bounds = []

    for neg_idx in neg_pixels:
        ws = pixel_window_sizes[neg_idx]
        center_key = (pixel_bbox_centers or {}).get(neg_idx, neg_idx)
        cz, cy, cx = get_nearest_center(center_key, slice_shape, ws, near_cent_dict)
        bounds = _window_bounds(cy, cx, ws)

        overlaps = False
        for ub in used_bounds:
            if _windows_overlap(bounds, ub):
                overlaps = True
                break

        if not overlaps:
            selected.append((neg_idx, (cz, cy, cx), ws))
            used_bounds.append(bounds)

    return selected


def _apply_result(phi, result_x, cy, cx, sub_size):
    """Write optimised sub-window back into *phi*."""
    sy, sx = _unpack_size(sub_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    pixels = sy * sx
    phi[1, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi] = \
        result_x[:pixels].reshape(sy, sx)
    phi[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi] = \
        result_x[pixels:].reshape(sy, sx)


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


# ---------------------------------------------------------------------------
# Serial inner loop — runs when batch_size == 1
# ---------------------------------------------------------------------------
def _serial_fix_pixel(
    neg_index_tuple, phi, phi_init, jacobian_matrix,
    slice_shape, near_cent_dict, window_counts,
    max_per_index_iter, max_minimize_iter,
    max_window, threshold, err_tol, methodName, verbose,
    error_list, num_neg_jac, min_jdet_list, iter_times,
    enforce_shoelace=False,
    enforce_injectivity=False,
    plot_callback=None,
    deformation_i=None,
):
    """Fix a single pixel using the serial adaptive-window inner loop.

    Start from the bounding-box-derived window, then grow by 2 each
    sub-iteration until the local region is clean or the window hits the
    grid boundary.

    Mutates *phi* and the accumulator lists in-place.

    Returns
    -------
    jacobian_matrix, quality_matrix, submatrix_size, per_index_iter, (cy, cx)
    """
    _use_quality = enforce_shoelace or enforce_injectivity
    quality_matrix = _quality_map(phi, enforce_shoelace, enforce_injectivity) if _use_quality else jacobian_matrix

    # Adaptive starting size from negative-Jdet bounding box
    submatrix_size, bbox_center = neg_jdet_bounding_window(
        quality_matrix, neg_index_tuple, threshold, err_tol)
    max_sy, max_sx = _unpack_size(max_window)
    submatrix_size = (min(submatrix_size[0], max_sy), min(submatrix_size[1], max_sx))

    per_index_iter = 0
    window_reached_max = False

    while (
        per_index_iter == 0
        or (
            (not window_reached_max)
            and per_index_iter < max_per_index_iter
            and (quality_matrix[0,
                    cy - hy:cy + hy_hi,
                    cx - hx:cx + hx_hi]
                 < threshold - err_tol).any()
        )
    ):
        per_index_iter += 1

        window_counts[_unpack_size(submatrix_size)] += 1

        cz, cy, cx = get_nearest_center(
            bbox_center, slice_shape, submatrix_size, near_cent_dict)
        sy, sx = _unpack_size(submatrix_size)
        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx

        phi_init_sub_flat = get_phi_sub_flat(phi_init, cz, cy, cx, slice_shape, submatrix_size)
        phi_sub_flat = get_phi_sub_flat(phi, cz, cy, cx, slice_shape, submatrix_size)

        if per_index_iter > 1:
            _log(verbose, 2, f"  [window] Index {neg_index_tuple}: window grew to {sy}x{sx} (sub-iter {per_index_iter})")

        is_at_edge, w_max = _edge_flags(cy, cx, submatrix_size, slice_shape, max_window)
        window_reached_max = window_reached_max or w_max

        _log(verbose, 2, f"  [edge] at_edge={is_at_edge}  window_reached_max={window_reached_max}")

        # Skip optimizer if frozen edges have negative Jdet (likely infeasible)
        if (not is_at_edge and not window_reached_max
                and not _frozen_edges_clean(quality_matrix, cy, cx,
                                           submatrix_size, threshold, err_tol)):
            _log(verbose, 2, f"  [skip] Frozen edges have neg Jdet at win {sy}x{sx} — growing")
            sy, sx = _unpack_size(submatrix_size)
            if sy < max_sy or sx < max_sx:
                submatrix_size = (min(sy + 2, max_sy), min(sx + 2, max_sx))
            continue

        # Run optimisation directly — no process pool
        result_x, elapsed = _optimize_single_window(
            phi_sub_flat, phi_init_sub_flat, submatrix_size,
            is_at_edge, window_reached_max,
            threshold, max_minimize_iter, methodName,
            enforce_shoelace=enforce_shoelace,
            enforce_injectivity=enforce_injectivity,
        )
        iter_times.append(elapsed)

        _apply_result(phi, result_x, cy, cx, submatrix_size)

        jacobian_matrix, quality_matrix, cur_neg, cur_min = _update_metrics(
            phi, phi_init, enforce_shoelace, enforce_injectivity,
            num_neg_jac, min_jdet_list, error_list)

        _log(verbose, 2, f"  [sub-Jdet] centre ({cy},{cx}) window {sy}x{sx}:\n"
             + np.array2string(
                 jacobian_matrix[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi],
                 precision=4, suppress_small=True))

        if plot_callback is not None:
            plot_callback(deformation_i, phi)

        if float(quality_matrix[0, 1:-1, 1:-1].min()) > threshold - err_tol:
            break

        # Grow window for next sub-iteration
        sy, sx = _unpack_size(submatrix_size)
        if sy < max_sy or sx < max_sx:
            submatrix_size = (min(sy + 2, max_sy), min(sx + 2, max_sx))
        else:
            window_reached_max = True

    return jacobian_matrix, quality_matrix, submatrix_size, per_index_iter, (cy, cx)


# ---------------------------------------------------------------------------
# Main hybrid parallel algorithm
# ---------------------------------------------------------------------------
def iterative_parallel(
    deformation_i,
    methodName="SLSQP",
    verbose=1,
    save_path=None,
    plot_every=0,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
    max_workers=None,
    enforce_shoelace=False,
    enforce_injectivity=False,
):
    """Hybrid serial/parallel iterative SLSQP correction.

    * batch_size > 1 → ``ProcessPoolExecutor`` (true parallelism)
    * batch_size == 1 → direct in-process with full adaptive window
      growth (avoids Windows ``spawn`` overhead)

    Parameters
    ----------
    max_workers : int or None
        Number of worker processes.  ``None`` → ``os.cpu_count()``.
    enforce_shoelace : bool
        When ``True``, also enforce positive shoelace quad-cell areas
        (geometric fold detection) alongside gradient-based Jacobian.
    enforce_injectivity : bool
        When ``True``, enforce monotonicity of deformed coordinates
        (sufficient condition for global injectivity on structured grids).

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    """
    # Resolve parameters
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

    if max_workers is None:
        max_workers = os.cpu_count()

    # Accumulators
    error_list = []
    num_neg_jac = []
    iter_times = []
    min_jdet_list = []
    window_counts = defaultdict(int)

    start_time = time.time()
    phi, phi_init, H, W = _init_phi(deformation_i)
    slice_shape = (1, H, W)
    max_window = (H, W)
    near_cent_dict = {}

    _log(verbose, 1,
         f"[init] Grid {H}x{W}  |  threshold={threshold}  "
         f"|  method={methodName}  |  workers={max_workers}")

    jacobian_matrix, quality_matrix, init_neg, init_min = _update_metrics(
        phi, phi_init, enforce_shoelace, enforce_injectivity,
        num_neg_jac, min_jdet_list)

    _log(verbose, 1,
         f"[init] Neg-Jdet pixels: {init_neg}  |  min Jdet: {init_min:.6f}")

    # Per-pixel window size tracker for parallel batches
    pixel_window_sizes = {}
    pixel_bbox_centers = {}

    iteration = 0
    serial_iters = 0
    parallel_iters = 0
    global_min_window = (3, 3)
    prev_neg = init_neg
    attempted_pixels = set()  # pixels attempted in the last batch
    executor = None  # lazy — only created if we actually need parallelism

    try:
        while (iteration < max_iterations
               and (quality_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any()):
            iteration += 1

            neg_pixels = _find_negative_pixels(quality_matrix, threshold, err_tol)
            if not neg_pixels:
                break

            # Assign / grow window sizes for batching decision
            current_neg_set = set(neg_pixels)
            new_window_sizes = {}
            new_bbox_centers = {}
            for px in neg_pixels:
                # Always recompute bounding box for current Jacobian state
                bbox_size, bbox_center = neg_jdet_bounding_window(
                    quality_matrix, px, threshold, err_tol
                )
                bsy, bsx = _unpack_size(bbox_size)
                max_sy, max_sx = _unpack_size(max_window)
                bsy = min(max(bsy, 3), max_sy)
                bsx = min(max(bsx, 3), max_sx)
                bbox_size = (bsy, bsx)
                new_bbox_centers[px] = bbox_center

                if px in attempted_pixels and px in pixel_window_sizes:
                    # Was attempted last iteration but still negative → grow
                    old_sy, old_sx = _unpack_size(pixel_window_sizes[px])
                    new_sy = min(max(bsy, old_sy + 2), max_sy)
                    new_sx = min(max(bsx, old_sx + 2), max_sx)
                    new_window_sizes[px] = (new_sy, new_sx)
                else:
                    new_window_sizes[px] = bbox_size

                gsy, gsx = _unpack_size(global_min_window)
                ws_sy, ws_sx = _unpack_size(new_window_sizes[px])
                new_window_sizes[px] = (max(ws_sy, gsy), max(ws_sx, gsx))
            pixel_window_sizes = new_window_sizes
            pixel_bbox_centers = new_bbox_centers

            # Select non-overlapping batch
            batch = _select_non_overlapping(
                neg_pixels, pixel_window_sizes, slice_shape, near_cent_dict,
                pixel_bbox_centers=pixel_bbox_centers,
            )

            # Snapshot bookkeeping — save pre-step state for side-by-side.
            _show_snap = plot_every and iteration % plot_every == 0
            if _show_snap:
                _snap_before = jacobian_matrix.copy()
                _snap_windows = []
                for _, (_, b_cy, b_cx), b_sz in batch:
                    b_edge, _ = _edge_flags(b_cy, b_cx, b_sz,
                                            slice_shape, max_window)
                    _snap_windows.append((b_cy, b_cx, b_sz, b_edge))

            if len(batch) <= 1:
                # ──────────────────────────────────────────────────────
                # SERIAL PATH — run directly, no process pool overhead
                # ──────────────────────────────────────────────────────
                serial_iters += 1
                neg_idx = neg_pixels[0]

                _log(verbose, 1,
                     f"[iter {iteration:4d}]  serial  "
                     f"fix ({neg_idx[0]:3d},{neg_idx[1]:3d})  "
                     f"neg_pixels={len(neg_pixels)}")

                jacobian_matrix, quality_matrix, sub_size, sub_iters, _final_center = \
                    _serial_fix_pixel(
                        neg_idx, phi, phi_init, jacobian_matrix,
                        slice_shape, near_cent_dict, window_counts,
                        max_per_index_iter,
                        max_minimize_iter, max_window,
                        threshold, err_tol, methodName, verbose,
                        error_list, num_neg_jac, min_jdet_list, iter_times,
                        enforce_shoelace=enforce_shoelace,
                        enforce_injectivity=enforce_injectivity,
                    )

                cur_neg = int((jacobian_matrix <= 0).sum())
                cur_min = float(jacobian_matrix.min())
                cur_err = error_list[-1] if error_list else 0.0
                _ssy, _ssx = _unpack_size(sub_size)
                _log(verbose, 1,
                     f"         -> neg_jdet {cur_neg:5d}  "
                     f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
                     f"win {_ssy}x{_ssx}  sub-iters {sub_iters}")

                # Full-grid fallback for non-square grids
                _sub_sy, _sub_sx = _unpack_size(sub_size)
                if (_sub_sy >= H and _sub_sx >= W
                        and H != W
                        and (quality_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any()):
                    iter_start = time.time()
                    _full_grid_step(phi, phi_init, H, W, threshold,
                                    max_minimize_iter, methodName, verbose,
                                    enforce_shoelace, enforce_injectivity)
                    iter_times.append(time.time() - iter_start)

                    jacobian_matrix, quality_matrix, cur_neg, cur_min = _update_metrics(
                        phi, phi_init, enforce_shoelace, enforce_injectivity,
                        num_neg_jac, min_jdet_list, error_list)
                    _log(verbose, 1,
                         f"         -> full-grid  neg_jdet {cur_neg:5d}  "
                         f"min_jdet {cur_min:+.6f}  L2 {error_list[-1]:.4f}")

                prev_neg = cur_neg
                attempted_pixels = {neg_idx}

            else:
                # ──────────────────────────────────────────────────────
                # PARALLEL PATH — submit batch to process pool
                # ──────────────────────────────────────────────────────
                parallel_iters += 1

                # Lazy-create executor on first parallel batch
                if executor is None:
                    executor = ProcessPoolExecutor(max_workers=max_workers)

                batch_sizes = [_unpack_size(ws) for _, _, ws in batch]
                batch_strs = [f"{sy}x{sx}" for sy, sx in batch_sizes]
                _log(verbose, 1,
                     f"[iter {iteration:4d}]  parallel  batch={len(batch)}  "
                     f"neg_pixels={len(neg_pixels)}  "
                     f"windows={','.join(batch_strs)}")

                futures = {}
                for neg_idx, (cz, cy, cx), sub_size in batch:
                    window_counts[_unpack_size(sub_size)] += 1

                    phi_init_sub_flat = get_phi_sub_flat(
                        phi_init, cz, cy, cx, slice_shape, sub_size)
                    phi_sub_flat = get_phi_sub_flat(
                        phi, cz, cy, cx, slice_shape, sub_size)

                    is_at_edge, window_reached_max = _edge_flags(
                        cy, cx, sub_size, slice_shape, max_window)

                    fut = executor.submit(
                        _optimize_single_window,
                        phi_sub_flat, phi_init_sub_flat, sub_size,
                        is_at_edge, window_reached_max,
                        threshold, max_minimize_iter, methodName,
                        enforce_shoelace,
                        enforce_injectivity,
                    )
                    futures[fut] = (neg_idx, cz, cy, cx, sub_size)

                batch_time = 0.0
                for fut in as_completed(futures):
                    neg_idx, cz, cy, cx, sub_size = futures[fut]
                    result_x, elapsed = fut.result()
                    batch_time = max(batch_time, elapsed)
                    _apply_result(phi, result_x, cy, cx, sub_size)

                iter_times.append(batch_time)

                jacobian_matrix, quality_matrix, cur_neg, cur_min = _update_metrics(
                    phi, phi_init, enforce_shoelace, enforce_injectivity,
                    num_neg_jac, min_jdet_list, error_list)

                cur_err = error_list[-1]
                _log(verbose, 1,
                     f"         -> neg_jdet {cur_neg:5d}  "
                     f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}")

                # Escalate global min window when parallel doesn't
                # improve — forces larger windows and eventually serial.
                gsy, gsx = _unpack_size(global_min_window)
                if cur_neg >= prev_neg and (gsy < H or gsx < W):
                    global_min_window = (min(gsy + 2, H), min(gsx + 2, W))
                    _log(verbose, 1,
                         f"  [escalate] parallel didn't improve "
                         f"({prev_neg}->{cur_neg}), "
                         f"global min window -> {global_min_window[0]}x{global_min_window[1]}")
                prev_neg = cur_neg
                attempted_pixels = {item[0] for item in batch}

            # Side-by-side before/after snapshot for this iteration.
            if _show_snap:
                from modules.dvfviz import plot_step_snapshot
                plot_step_snapshot(jacobian_matrix, iteration,
                                   int((jacobian_matrix <= 0).sum()),
                                   float(jacobian_matrix.min()),
                                   windows=_snap_windows,
                                   jacobian_before=_snap_before)

            if float(quality_matrix[0, 1:-1, 1:-1].min()) > threshold - err_tol:
                _log(verbose, 1,
                     f"[done] All Jdet > threshold after iter {iteration}")
                break

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _log(verbose, 1, "")
    _log(verbose, 1, "=" * 60)
    _log(verbose, 1, f"  SUMMARY  ({methodName} — hybrid parallel)")
    _log(verbose, 1, "-" * 60)
    _log(verbose, 1, f"  Grid size        : {H} x {W}")
    _log(verbose, 1, f"  Iterations       : {iteration}  "
         f"(serial={serial_iters}, parallel={parallel_iters})")
    _log(verbose, 1, f"  Neg-Jdet  {init_neg:>5d} -> {final_neg:>5d}")
    _log(verbose, 1, f"  Min Jdet  {init_min:+.6f} -> {final_min:+.6f}")
    _log(verbose, 1, f"  L2 error         : {final_err:.6f}")
    _log(verbose, 1, f"  Time             : {elapsed:.2f}s")
    _log(verbose, 1, "=" * 60)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results(
            save_path, method=f"{methodName} (hybrid parallel)",
            threshold=threshold, err_tol=err_tol,
            max_iterations=max_iterations, max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            H=H, W=W, elapsed=elapsed, final_err=final_err,
            init_neg=init_neg, final_neg=final_neg, init_min=init_min,
            final_min=final_min, iteration=iteration, phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
            extra_results=(f"\tSerial iterations: {serial_iters}\n"
                           f"\tParallel iterations: {parallel_iters}"),
        )

    return phi
