"""Metric tracking helpers shared by the 2D solver.

* :func:`_update_metrics` — recompute Jdet / quality and append to accumulator lists.
* :func:`_patch_jacobian_2d` — recompute Jdet only in the modified sub-region.
* :func:`_patch_quality_2d` — recompute the combined quality map only in the
  modified sub-region.

Originally bundled in ``coordinator.py`` — kept re-exported there
for backward compatibility.
"""

import numpy as np

from dvfopt._defaults import _unpack_size
from dvfopt.core.slsqp_windowed.constraints import _quality_map
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d, jacobian_det2D


def _update_metrics(
    phi,
    phi_init,
    enforce_shoelace,
    enforce_injectivity,
    num_neg_jac,
    min_jdet_list,
    error_list=None,
    jacobian_matrix=None,
    patch_center=None,
    patch_size=None,
    enforce_triangles=False,
    quality_matrix=None,
):
    """Recompute Jacobian/quality matrices and append to accumulator lists.

    Parameters
    ----------
    error_list : list or None
        When not ``None``, the L2 error is appended.
    jacobian_matrix : ndarray or None
        When provided along with *patch_center* and *patch_size*, only the
        affected sub-region (+ 1px gradient border) is recomputed, avoiding
        a full-grid Jacobian computation.
    patch_center : tuple or None
        ``(cy, cx)`` center of the optimised sub-window.
    patch_size : tuple or None
        ``(sy, sx)`` size of the optimised sub-window.
    quality_matrix : ndarray or None
        The current quality map (only meaningful when any ``enforce_*``
        flag is set).  When provided together with *jacobian_matrix*:

        * with *patch_center*/*patch_size* — the quality map is patched
          window-locally (exact: every quality metric has the same
          bounded footprint as the Jacobian determinant), avoiding the
          full-grid recomputation;
        * with ``patch_center=None`` — the quality map is trusted as
          already patched externally (e.g. per-window in a parallel
          batch), mirroring the Jacobian convention.

        When ``None`` (legacy behaviour), the quality map is recomputed
        over the full grid.

    Returns
    -------
    jacobian_matrix, quality_matrix, cur_neg, cur_min
    """
    if jacobian_matrix is not None and patch_center is not None and patch_size is not None:
        jac = _patch_jacobian_2d(jacobian_matrix, phi, patch_center, patch_size)
    elif jacobian_matrix is not None and patch_center is None:
        # Jacobian already patched externally (e.g., parallel batch)
        jac = jacobian_matrix
    else:
        jac = jacobian_det2D(phi)
    use_q = enforce_shoelace or enforce_injectivity or enforce_triangles
    if not use_q:
        qm = jac
    elif (
        quality_matrix is not None
        and jacobian_matrix is not None
        and patch_center is not None
        and patch_size is not None
    ):
        qm = _patch_quality_2d(
            quality_matrix,
            phi,
            jac,
            patch_center,
            patch_size,
            enforce_shoelace,
            enforce_injectivity,
            enforce_triangles=enforce_triangles,
        )
    elif quality_matrix is not None and jacobian_matrix is not None and patch_center is None:
        # Quality already patched externally alongside the Jacobian.
        qm = quality_matrix
    else:
        qm = _quality_map(
            phi,
            enforce_shoelace,
            enforce_injectivity,
            enforce_triangles=enforce_triangles,
            jacobian_matrix=jac,
        )
    cur_neg = int((jac <= 0).sum())
    cur_min = float(jac.min())
    num_neg_jac.append(cur_neg)
    min_jdet_list.append(cur_min)
    if error_list is not None:
        error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))
    return jac, qm, cur_neg, cur_min


def _patch_regions_2d(center, sub_size, H, W):
    """Shared window-patch geometry for Jacobian and quality maps.

    Returns ``(wy0, wy1, wx0, wx1, cy0, cy1, cx0, cx1)`` where
    ``[wy0:wy1, wx0:wx1]`` is the write-back region (sub-window + 1px
    border, clamped to the grid) and ``[cy0:cy1, cx0:cx1]`` is the
    computation region (write-back + 1 extra pixel of context).
    """
    cy, cx = center
    sy, sx = _unpack_size(sub_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx

    wy0 = max(cy - hy - 1, 0)
    wy1 = min(cy + hy_hi + 1, H)
    wx0 = max(cx - hx - 1, 0)
    wx1 = min(cx + hx_hi + 1, W)

    cy0 = max(wy0 - 1, 0)
    cy1 = min(wy1 + 1, H)
    cx0 = max(wx0 - 1, 0)
    cx1 = min(wx1 + 1, W)
    return wy0, wy1, wx0, wx1, cy0, cy1, cx0, cx1


def _patch_jacobian_2d(jacobian_matrix, phi, center, sub_size):
    """Recompute Jacobian only in the modified sub-region + 1px border.

    The computation region is expanded by an extra pixel beyond the
    write-back region so that ``np.gradient`` uses central differences
    at the write-back boundary (matching full-grid computation).

    Mutates *jacobian_matrix* in place and returns it.
    """
    H, W = phi.shape[1], phi.shape[2]
    wy0, wy1, wx0, wx1, cy0, cy1, cx0, cx1 = _patch_regions_2d(center, sub_size, H, W)

    jdet_comp = _numpy_jdet_2d(phi[0, cy0:cy1, cx0:cx1], phi[1, cy0:cy1, cx0:cx1])

    # Trim to write-back region
    ty0 = wy0 - cy0
    tx0 = wx0 - cx0
    jacobian_matrix[0, wy0:wy1, wx0:wx1] = jdet_comp[ty0 : ty0 + wy1 - wy0, tx0 : tx0 + wx1 - wx0]
    return jacobian_matrix


def _patch_quality_2d(
    quality_matrix,
    phi,
    jacobian_matrix,
    center,
    sub_size,
    enforce_shoelace,
    enforce_injectivity,
    enforce_triangles=False,
):
    """Recompute the combined quality map only in the modified sub-region.

    Exactness: every metric folded into :func:`_quality_map` has the same
    bounded footprint as the Jacobian determinant — the value at pixel
    ``p`` depends only on ``phi`` within the 3x3 neighbourhood of ``p``:

    * *shoelace* / *triangles* — per-cell areas from the cell's 4 corner
      pixels, spread (via min) to the incident pixels;
    * *injectivity* — monotonicity gaps between horizontally / vertically
      / diagonally adjacent pixel pairs, spread to the pair members;
    * *Jdet* — taken from the (already patched) *jacobian_matrix* slice.

    So recomputing :func:`_quality_map` on the computation region
    (write-back + 1px of context) and writing back the write-back region
    reproduces the full-grid values exactly: every cell / pixel pair
    incident to a write-back pixel lies inside the computation region
    (or does not exist globally either, at true grid edges).

    Mutates *quality_matrix* in place and returns it.
    """
    H, W = phi.shape[1], phi.shape[2]
    wy0, wy1, wx0, wx1, cy0, cy1, cx0, cx1 = _patch_regions_2d(center, sub_size, H, W)

    q_comp = _quality_map(
        phi[:, cy0:cy1, cx0:cx1],
        enforce_shoelace,
        enforce_injectivity,
        enforce_triangles=enforce_triangles,
        jacobian_matrix=jacobian_matrix[:, cy0:cy1, cx0:cx1],
    )

    ty0 = wy0 - cy0
    tx0 = wx0 - cx0
    quality_matrix[0, wy0:wy1, wx0:wx1] = q_comp[0, ty0 : ty0 + wy1 - wy0, tx0 : tx0 + wx1 - wx0]
    return quality_matrix
