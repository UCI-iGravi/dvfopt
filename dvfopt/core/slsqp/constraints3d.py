"""3D constraint builders for SLSQP optimisation."""

import numpy as np
import scipy.sparse
from scipy.optimize import LinearConstraint, NonlinearConstraint

from dvfopt._defaults import _unpack_size_3d
from dvfopt.core.slsqp.gradients3d import jdet_constraint_jacobian_3d
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_3d


def jacobian_constraint_3d(phi_flat, subvolume_size, freeze_mask=None):
    """Return flattened Jacobian determinant values for optimiser constraints.

    phi_flat packing: ``[dx_flat, dy_flat, dz_flat]``.

    When *freeze_mask* is given, only non-frozen voxels are returned.
    """
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    voxels = sz * sy * sx
    dx = phi_flat[:voxels].reshape((sz, sy, sx))
    dy = phi_flat[voxels : 2 * voxels].reshape((sz, sy, sx))
    dz = phi_flat[2 * voxels :].reshape((sz, sy, sx))
    jdet = _numpy_jdet_3d(dz, dy, dx)
    if freeze_mask is not None:
        return jdet[~freeze_mask].flatten()
    return jdet.flatten()


def _injectivity_linear_constraint_3d(subvolume_size, inj_lb, freeze_mask=None):
    """Axial monotonicity of deformed coordinates as ONE sparse LinearConstraint.

    Each axial gap is linear in phi: ``gap = 1 + d_next - d_prev`` (the
    deformed coordinate is ``index + displacement``), so unlike the 2D
    path no NonlinearConstraint is needed — SLSQP gets exact linear rows.
    One row per neighbouring voxel pair along each axis (x-gaps read the
    dx block, y-gaps dy, z-gaps dz), bounded ``gap >= inj_lb`` i.e.
    ``A @ phi >= inj_lb - 1``.

    When *freeze_mask* is given, only rows whose BOTH endpoints are
    non-frozen are kept (the 2D ``exclude_boundaries`` spirit) — frozen
    ring pairs the input may already violate would otherwise make the
    sub-problem structurally infeasible.
    """
    sz, sy, sx = _unpack_size_3d(subvolume_size)
    voxels = sz * sy * sx
    lin = np.arange(voxels).reshape(sz, sy, sx)
    free = None if freeze_mask is None else ~freeze_mask

    rows_prev, rows_next, block = [], [], []
    # (prev-slice, next-slice, channel block index) per axis:
    # x-gaps -> dx block 0, y-gaps -> dy block 1, z-gaps -> dz block 2.
    specs = [
        (
            lin[:, :, :-1],
            lin[:, :, 1:],
            0,
            None if free is None else free[:, :, :-1] & free[:, :, 1:],
        ),
        (
            lin[:, :-1, :],
            lin[:, 1:, :],
            1,
            None if free is None else free[:, :-1, :] & free[:, 1:, :],
        ),
        (
            lin[:-1, :, :],
            lin[1:, :, :],
            2,
            None if free is None else free[:-1, :, :] & free[1:, :, :],
        ),
    ]
    for prev, nxt, blk, keep in specs:
        p = prev.ravel()
        n = nxt.ravel()
        if keep is not None:
            k = keep.ravel()
            p, n = p[k], n[k]
        rows_prev.append(p + blk * voxels)
        rows_next.append(n + blk * voxels)
        block.append(len(p))
    prev_cols = np.concatenate(rows_prev)
    next_cols = np.concatenate(rows_next)
    n_rows = prev_cols.size
    if n_rows == 0:
        return None
    row_idx = np.repeat(np.arange(n_rows), 2)
    col_idx = np.stack([prev_cols, next_cols], axis=1).ravel()
    data = np.tile(np.array([-1.0, 1.0]), n_rows)
    A = scipy.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, 3 * voxels))
    return LinearConstraint(A, inj_lb - 1.0, np.inf)


def _build_constraints_3d(
    phi_sub_flat,
    subvolume_size,
    freeze_mask,
    threshold,
    window_reached_max=False,
    enforce_injectivity=False,
    injectivity_threshold=None,
):
    """Build SLSQP constraints for a 3D sub-volume optimisation.

    The Jacobian constraint excludes only frozen boundary voxels.
    Grid-edge boundary voxels are NOT frozen and ARE constrained.

    When *window_reached_max* is ``True`` the window cannot grow any
    further, so no frozen edges apply (mirroring the 2D
    ``exclude_bounds = not is_at_edge and not window_reached_max``
    semantics in :func:`dvfopt.core.slsqp.constraints._build_constraints`):
    the boundary equality constraints are dropped and the Jacobian
    constraint covers **all** voxels, including the rim.  Without this
    release, a fold component larger than the maximum window keeps its
    negative rim pinned by equality constraints — an infeasible SLSQP
    problem that can never make progress.

    The serial solver no longer uses the ``window_reached_max=True`` branch
    of this function: max-window solves go through
    :func:`_build_constraints_3d_maxwindow` (patch-based halo no-damage
    constraints). The flag remains for API compatibility.
    """
    fm = None if window_reached_max else freeze_mask
    nlc = NonlinearConstraint(
        lambda phi1: jacobian_constraint_3d(phi1, subvolume_size, fm),
        threshold,
        np.inf,
        jac=lambda phi1: jdet_constraint_jacobian_3d(phi1, subvolume_size, fm),
    )
    constraints = [nlc]

    if enforce_injectivity:
        inj_lb = threshold if injectivity_threshold is None else injectivity_threshold
        inj = _injectivity_linear_constraint_3d(subvolume_size, inj_lb, freeze_mask=fm)
        if inj is not None:
            constraints.append(inj)

    if fm is not None and fm.any():
        sz, sy, sx = _unpack_size_3d(subvolume_size)
        voxels = sz * sy * sx
        edge_indices = np.argwhere(freeze_mask)
        fixed_indices = []
        for z, y, x in edge_indices:
            idx = z * sy * sx + y * sx + x
            fixed_indices.extend([idx, idx + voxels, idx + 2 * voxels])

        fixed_indices = np.array(fixed_indices)
        fixed_values = phi_sub_flat[fixed_indices]
        n_fixed = len(fixed_indices)
        A_eq = scipy.sparse.csr_matrix(
            (np.ones(n_fixed), (np.arange(n_fixed), fixed_indices)),
            shape=(n_fixed, phi_sub_flat.size),
        )

        constraints.append(LinearConstraint(A_eq, fixed_values, fixed_values))

    return constraints


def _build_constraints_3d_maxwindow(
    patch_flat,
    patch_size,
    win_start,
    win_size,
    threshold,
    enforce_injectivity=False,
    injectivity_threshold=None,
):
    """Constraints for a max-window solve: Jdet over window ∪ halo on a
    context patch, with per-row lower bounds.

    The decision vector stays window-only (``[dx, dy, dz]`` packing over
    ``win_size``); constraint evaluation embeds it into the frozen
    *patch_flat* context (window + 2 voxels per side, clamped to the
    volume by the caller). Rows cover the window dilated by 1 — exactly
    the region the outer accept/rollback check measures — and, because
    every constrained voxel sits ≥ 1 voxel inside the patch (or on a
    patch edge that coincides with a volume edge), the ``np.gradient``
    stencils here equal the full-field ones: feasible ⇒ paste-back
    acceptable, for any SLSQP implementation's choice of optimum.

    Lower bounds: ``threshold`` on window rows; ``min(threshold, current
    Jdet)`` on halo rows (healthy border voxels must stay healthy,
    already-bad ones must not get worse — x0 is halo-feasible by
    construction).
    """
    pz, py, px = (int(s) for s in patch_size)
    n_patch = pz * py * px
    oz, oy, ox = (int(s) for s in win_start)
    sz, sy, sx = _unpack_size_3d(win_size)

    # Window-voxel linear indices in patch C-order; variable columns in
    # the [dx, dy, dz] channel-block layout.
    win_lin = (
        np.arange(oz, oz + sz)[:, None, None] * (py * px)
        + np.arange(oy, oy + sy)[None, :, None] * px
        + np.arange(ox, ox + sx)[None, None, :]
    ).ravel()
    cols = np.concatenate([win_lin, win_lin + n_patch, win_lin + 2 * n_patch])

    # Constrained rows: window dilated by 1, clamped to the patch. The
    # patch is clamped to the volume by the caller, so clamping to the
    # patch equals clamping to the volume (= the accept-check region).
    window = np.zeros((pz, py, px), dtype=bool)
    window[oz : oz + sz, oy : oy + sy, ox : ox + sx] = True
    region = np.zeros((pz, py, px), dtype=bool)
    region[
        max(oz - 1, 0) : min(oz + sz + 1, pz),
        max(oy - 1, 0) : min(oy + sy + 1, py),
        max(ox - 1, 0) : min(ox + sx + 1, px),
    ] = True
    rows = np.flatnonzero(region.ravel())
    window_rows = window.ravel()[rows]

    patch_base = np.asarray(patch_flat, dtype=np.float64).copy()

    def _patch_jdet(vec):
        dx = vec[:n_patch].reshape(pz, py, px)
        dy = vec[n_patch : 2 * n_patch].reshape(pz, py, px)
        dz = vec[2 * n_patch :].reshape(pz, py, px)
        return _numpy_jdet_3d(dz, dy, dx).ravel()

    def _embed(x):
        vec = patch_base.copy()
        vec[cols] = x
        return vec

    jdet0 = _patch_jdet(patch_base)[rows]
    lb = np.where(window_rows, threshold, np.minimum(threshold, jdet0))

    nlc = NonlinearConstraint(
        lambda x: _patch_jdet(_embed(x))[rows],
        lb,
        np.inf,
        jac=lambda x: jdet_constraint_jacobian_3d(_embed(x), (pz, py, px))[rows][:, cols].tocsr(),
    )
    constraints = [nlc]

    if enforce_injectivity:
        # Window-interior gaps only (both endpoints are decision vars).
        # Halo separation is not enforced here — the outer loop's quality
        # gate re-selects any remaining boundary violation.
        inj_lb = threshold if injectivity_threshold is None else injectivity_threshold
        inj = _injectivity_linear_constraint_3d((sz, sy, sx), inj_lb, freeze_mask=None)
        if inj is not None:
            constraints.append(inj)
    return constraints
