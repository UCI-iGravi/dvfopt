"""Per-pixel two-triangle sign-only Jacobian check (2D).

Geometric alternative to central-difference Jacobians: for every interior
pixel (x, y) form two triangles in the warped grid and check only their
signed-area sign (negative = flip, 0 = collapse, positive = valid).

Triangle vertices (image coords, origin top-left, +x right, +y down):
    T1 at (x, y):  (x, y), (x-1, y+1), (x, y+1)
    T2 at (x, y):  (x, y), (x,   y+1), (x+1, y)

With +y pointing down the raw 2D cross product is negated w.r.t. a
math-origin convention, so we negate to keep "positive = valid".

The output is indexed by cell (y, x) under a TR-BL triangulation so the
shape ``(2, H-1, W-1)`` mirrors ``triangulated_shoelace_det2D`` (which uses
the TL-BR diagonal) for drop-in comparability in notebooks.
"""

import numpy as np

from dvfopt.jacobian.shoelace import _ref_grid

# Optional Numba JIT — paired with tri_grad_T_v JIT in
# `dvfopt.core.tri_primitives`. Forward T-area computation is the
# other ~497k calls/run inside L-BFGS-B objective evaluations; JIT
# folds the 4 corner-slice broadcasts into a single fused loop.
try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore


def _triangle_areas_2d_numpy(dy, dx):
    H, W = dy.shape
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy

    x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
    x_tr, y_tr = def_x[:-1, 1:], def_y[:-1, 1:]
    x_bl, y_bl = def_x[1:, :-1], def_y[1:, :-1]
    x_br, y_br = def_x[1:, 1:], def_y[1:, 1:]

    AB_x = x_bl - x_tr
    AB_y = y_bl - y_tr
    AC_x = x_br - x_tr
    AC_y = y_br - y_tr
    T1 = -0.5 * (AB_x * AC_y - AB_y * AC_x)

    AB_x = x_bl - x_tl
    AB_y = y_bl - y_tl
    AC_x = x_tr - x_tl
    AC_y = y_tr - y_tl
    T2 = -0.5 * (AB_x * AC_y - AB_y * AC_x)

    return T1, T2


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False)
    def _triangle_areas_2d_numba_kernel(dy, dx, H, W):
        T1 = np.empty((H - 1, W - 1))
        T2 = np.empty((H - 1, W - 1))
        for i in range(H - 1):
            for j in range(W - 1):
                # Deformed corner positions (ref_y = i, ref_x = j).
                x_tl = j + dx[i, j]
                y_tl = i + dy[i, j]
                x_tr = (j + 1) + dx[i, j + 1]
                y_tr = i + dy[i, j + 1]
                x_bl = j + dx[i + 1, j]
                y_bl = (i + 1) + dy[i + 1, j]
                x_br = (j + 1) + dx[i + 1, j + 1]
                y_br = (i + 1) + dy[i + 1, j + 1]
                # T1 at (i, j+1): A=TR, B=BL, C=BR
                ABx = x_bl - x_tr
                ABy = y_bl - y_tr
                ACx = x_br - x_tr
                ACy = y_br - y_tr
                T1[i, j] = -0.5 * (ABx * ACy - ABy * ACx)
                # T2 at (i, j): A=TL, B=BL, C=TR
                ABx = x_bl - x_tl
                ABy = y_bl - y_tl
                ACx = x_tr - x_tl
                ACy = y_tr - y_tl
                T2[i, j] = -0.5 * (ABx * ACy - ABy * ACx)
        return T1, T2


def _triangle_areas_2d(dy, dx):
    """Signed areas of the per-pixel two-triangle pair, TR-BL diagonal.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    T1, T2 : tuple of ndarray, shape ``(H-1, W-1)`` each
        ``T1[y, x]`` = signed area of triangle at pixel ``(x+1, y)``
        (lower-right triangle of cell (y, x) under TR-BL split).
        ``T2[y, x]`` = signed area of triangle at pixel ``(x, y)``
        (upper-left triangle of cell (y, x) under TR-BL split).
        Positive = valid under the +y-down convention.

    Uses a Numba @njit kernel when available; falls back to the
    pure-numpy implementation otherwise. The two paths are
    numerically equivalent to 1e-14 absolute on representative
    inputs.
    """
    if not _HAVE_NUMBA:
        return _triangle_areas_2d_numpy(dy, dx)
    H, W = dy.shape
    dy_c = np.ascontiguousarray(dy)
    dx_c = np.ascontiguousarray(dx)
    return _triangle_areas_2d_numba_kernel(dy_c, dx_c, H, W)


def _corner_patch_areas_2d(dy, dx):
    """Two extra triangles needed for full vertex coverage of the TR-BL split.

    The per-cell TR-BL scheme in :func:`_triangle_areas_2d` leaves
    vertex ``(0, 0)`` (top-left grid corner) and vertex ``(H-1, W-1)``
    (bottom-right grid corner) each covered by only one triangle —
    a coverage gap. This helper returns two patch triangles using the
    *opposite* (TL-BR) diagonal at cells ``(0, 0)`` and ``(H-2, W-2)``,
    one per gap corner. Combining the standard areas with these patches
    gives every vertex of the grid coverage of at least two triangles.

    Sign convention matches :func:`_triangle_areas_2d`: positive = valid
    under the +y-down convention.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    ndarray of shape ``(2,)`` — ``[patch_TL, patch_BR]``.
    """
    H, W = dy.shape
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy

    # Patch triangle at cell (0, 0), vertices (TL, BR, TR) — winds so that
    # an identity field gives +0.5 area. Vertex (0, 0) is A here.
    Ax, Ay = def_x[0, 0], def_y[0, 0]
    Bx, By = def_x[1, 1], def_y[1, 1]
    Cx, Cy = def_x[0, 1], def_y[0, 1]
    patch_tl = -0.5 * ((Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax))

    # Patch triangle at cell (H-2, W-2), vertices (TL, BL, BR). Vertex
    # (H-1, W-1) is C here.
    Ax, Ay = def_x[H - 2, W - 2], def_y[H - 2, W - 2]
    Bx, By = def_x[H - 1, W - 2], def_y[H - 1, W - 2]
    Cx, Cy = def_x[H - 1, W - 1], def_y[H - 1, W - 1]
    patch_br = -0.5 * ((Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax))

    return np.array([patch_tl, patch_br])


def _triangle_areas_2d_full_coverage(dy, dx):
    """Standard per-cell triangles plus the two corner-patch triangles.

    Returns ``(T1, T2, patches)`` where ``T1, T2`` are exactly
    :func:`_triangle_areas_2d`'s output and ``patches`` is
    :func:`_corner_patch_areas_2d`'s output. Together they form the
    full-coverage 2-triangle check: every vertex of the ``(H, W)`` grid,
    including the two diagonally-opposite corner vertices that the
    standard scheme under-covers, is touched by at least two triangles.
    """
    T1, T2 = _triangle_areas_2d(dy, dx)
    patches = _corner_patch_areas_2d(dy, dx)
    return T1, T2, patches


def _triangle_signs_2d(dy, dx):
    """Sign of each per-pixel triangle area in {-1, 0, +1}.

    Returns ndarray of shape ``(2, H-1, W-1)``, dtype ``int8``.
    Channel 0 = sign(T1), channel 1 = sign(T2).
    """
    T1, T2 = _triangle_areas_2d(dy, dx)
    return np.stack([np.sign(T1), np.sign(T2)]).astype(np.int8)


def triangle_sign_det2D(phi_xy):
    """Per-pixel two-triangle sign check from a ``(2, H, W)`` phi array.

    Parameters
    ----------
    phi_xy : ndarray, shape ``(2, H, W)``
        Channels ``[dy, dx]`` (same convention as ``shoelace_det2D``).

    Returns
    -------
    ndarray, shape ``(2, H-1, W-1)``, dtype ``int8``
        Signs in {-1, 0, +1}. Positive = valid, zero = collapsed,
        negative = flipped.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    return _triangle_signs_2d(dy, dx)


def triangle_sign_count_negatives(phi_xy):
    """Count of per-pixel triangles with sign <= 0 (flips + collapses).

    Convenience scalar for comparing against
    ``(jacobian_det2D(phi) <= 0).sum()``.
    """
    return int((triangle_sign_det2D(phi_xy) <= 0).sum())


def triangle_sign_areas2D(phi_xy):
    """Signed areas (not just signs) of the per-pixel two-triangle check.

    Smooth in the deformation — suitable as a constraint value for SLSQP
    or L-BFGS-B penalty methods. For a sign-only test use
    :func:`triangle_sign_det2D`.

    Parameters
    ----------
    phi_xy : ndarray, shape ``(2, H, W)``
        Channels ``[dy, dx]``.

    Returns
    -------
    ndarray, shape ``(2, H-1, W-1)``
        Signed areas; positive = valid, negative = flip.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    T1, T2 = _triangle_areas_2d(dy, dx)
    return np.stack([T1, T2])


def triangle_sign_constraint(phi_xy, submatrix_size, exclude_boundaries=False):
    """Flattened 2-triangle signed areas for an SLSQP ``NonlinearConstraint``.

    Takes a flattened phi vector packed as ``[dx_flat, dy_flat]`` (same
    convention as ``shoelace_constraint``) and the submatrix ``(H, W)``.
    Returns a 1-D array of length ``2 * (H-1) * (W-1)`` (or fewer if
    ``exclude_boundaries=True``).
    """
    from dvfopt._defaults import _unpack_size

    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    T1, T2 = _triangle_areas_2d(dy, dx)
    if exclude_boundaries:
        T1 = T1[1:-1, 1:-1]
        T2 = T2[1:-1, 1:-1]
    return np.concatenate([T1.flatten(), T2.flatten()])
