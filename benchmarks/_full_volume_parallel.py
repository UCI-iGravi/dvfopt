"""Worker-side helpers for parallel full-volume 3D correction.

Module-level functions are required for ProcessPoolExecutor on Windows,
which uses spawn and serialises the target callable across processes.
"""

from dvfopt import iterative_3d, jacobian_det3D


def solve_group(args):
    """Sequentially correct a list of patches within a single region.

    Parameters
    ----------
    args : tuple
        ``(region_bbox, phi_region, patches, max_iter, max_pi, max_mi)``
        where ``patches`` is a list of ``(lpz0, lpz1, lpy0, lpy1, lpx0, lpx1)``
        tuples in coordinates *local* to ``phi_region``.

    Returns
    -------
    region_bbox : tuple
    phi_region : ndarray
    pre_group_neg : int
    post_group_neg : int
    """
    (region_bbox, phi_region, patches,
     max_iter, max_pi, max_mi) = args

    pre_group_neg = int((jacobian_det3D(phi_region) <= 0).sum())

    for (lpz0, lpz1, lpy0, lpy1, lpx0, lpx1) in patches:
        sub = phi_region[:, lpz0:lpz1, lpy0:lpy1, lpx0:lpx1].copy()
        jac_sub = jacobian_det3D(sub)
        if not (jac_sub <= 0).any():
            continue
        corrected = iterative_3d(
            sub,
            verbose=0,
            max_iterations=max_iter,
            max_per_index_iter=max_pi,
            max_minimize_iter=max_mi,
        )
        phi_region[:, lpz0:lpz1, lpy0:lpy1, lpx0:lpx1] = corrected

    post_group_neg = int((jacobian_det3D(phi_region) <= 0).sum())
    return region_bbox, phi_region, pre_group_neg, post_group_neg
