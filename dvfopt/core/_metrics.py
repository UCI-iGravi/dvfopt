"""Metric tracking helpers shared by the 2D solver.

* :func:`_update_metrics` — recompute Jdet / quality and append to accumulator lists.
* :func:`_patch_jacobian_2d` — recompute Jdet only in the modified sub-region.

Originally bundled in ``dvfopt/core/solver.py`` — kept re-exported there
for backward compatibility.
"""

import numpy as np

from dvfopt._defaults import _unpack_size
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d, jacobian_det2D
from dvfopt.core.slsqp.constraints import _quality_map


def _update_metrics(phi, phi_init, enforce_shoelace, enforce_injectivity,
                    num_neg_jac, min_jdet_list, error_list=None,
                    jacobian_matrix=None, patch_center=None, patch_size=None,
                    enforce_triangles=False):
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
    qm = _quality_map(phi, enforce_shoelace, enforce_injectivity,
                      enforce_triangles=enforce_triangles,
                      jacobian_matrix=jac) if use_q else jac
    cur_neg = int((jac <= 0).sum())
    cur_min = float(jac.min())
    num_neg_jac.append(cur_neg)
    min_jdet_list.append(cur_min)
    if error_list is not None:
        error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))
    return jac, qm, cur_neg, cur_min


def _patch_jacobian_2d(jacobian_matrix, phi, center, sub_size):
    """Recompute Jacobian only in the modified sub-region + 1px border.

    The computation region is expanded by an extra pixel beyond the
    write-back region so that ``np.gradient`` uses central differences
    at the write-back boundary (matching full-grid computation).

    Mutates *jacobian_matrix* in place and returns it.
    """
    cy, cx = center
    sy, sx = _unpack_size(sub_size)
    hy, hx = sy // 2, sx // 2
    hy_hi, hx_hi = sy - hy, sx - hx
    H, W = phi.shape[1], phi.shape[2]

    # Write-back region: sub-window + 1px border, clamped to grid
    wy0 = max(cy - hy - 1, 0)
    wy1 = min(cy + hy_hi + 1, H)
    wx0 = max(cx - hx - 1, 0)
    wx1 = min(cx + hx_hi + 1, W)

    # Computation region: 1 extra pixel for central-difference context
    cy0 = max(wy0 - 1, 0)
    cy1 = min(wy1 + 1, H)
    cx0 = max(wx0 - 1, 0)
    cx1 = min(wx1 + 1, W)

    jdet_comp = _numpy_jdet_2d(phi[0, cy0:cy1, cx0:cx1],
                                phi[1, cy0:cy1, cx0:cx1])

    # Trim to write-back region
    ty0 = wy0 - cy0
    tx0 = wx0 - cx0
    jacobian_matrix[0, wy0:wy1, wx0:wx1] = \
        jdet_comp[ty0:ty0 + wy1 - wy0, tx0:tx0 + wx1 - wx0]
    return jacobian_matrix
