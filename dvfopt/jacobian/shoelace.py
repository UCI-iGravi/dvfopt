"""Shoelace (geometric quad-area) helpers for deformation fields."""

import functools

import numpy as np

from dvfopt._defaults import _unpack_size


@functools.lru_cache(maxsize=8)
def _ref_grid(H, W):
    """Cached reference coordinate grids for a given (H, W).

    Returned arrays are marked read-only so a caller cannot poison the
    LRU cache via in-place assignment. ``ref_y + dy`` / ``ref_x + dx``
    (the only operations the callers perform) allocate fresh arrays
    and remain unaffected.
    """
    grids = np.mgrid[:H, :W]
    grids.setflags(write=False)
    return grids


def _shoelace_areas_from_positions(def_x, def_y):
    """Signed area of each quad cell from deformed-position grids.

    Uses vertex order TL -> TR -> BR -> BL.

    Parameters
    ----------
    def_x, def_y : ndarray, shape ``(H, W)``
        Per-vertex deformed positions.

    Returns
    -------
    ndarray, shape ``(H-1, W-1)``
    """
    x0, y0 = def_x[:-1, :-1], def_y[:-1, :-1]   # TL
    x1, y1 = def_x[:-1, 1:],  def_y[:-1, 1:]     # TR
    x2, y2 = def_x[1:, 1:],   def_y[1:, 1:]      # BR
    x3, y3 = def_x[1:, :-1],  def_y[1:, :-1]     # BL
    return 0.5 * ((x0*y1 - x1*y0) + (x1*y2 - x2*y1)
                  + (x2*y3 - x3*y2) + (x3*y0 - x0*y3))


def _shoelace_areas_2d(dy, dx):
    """Signed area of each deformed quad cell via the shoelace formula.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    ndarray, shape ``(H-1, W-1)``
    """
    H, W = dy.shape
    ref_y, ref_x = _ref_grid(H, W)
    return _shoelace_areas_from_positions(ref_x + dx, ref_y + dy)


def shoelace_det2D(phi_xy):
    """Compute shoelace quad-cell areas from a ``(2, H, W)`` phi array.

    Returns shape ``(1, H-1, W-1)``.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    return _shoelace_areas_2d(dy, dx)[np.newaxis, :, :]


def shoelace_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened shoelace quad areas for optimiser constraints."""
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
# Triangulated shoelace — TL-BR diagonal split (per-triangle signed areas)
# ---------------------------------------------------------------------------

def _triangulated_shoelace_areas_2d(dy, dx):
    """Signed areas of the two triangles formed by splitting each quad
    along its TL-BR diagonal.

    A single quad's shoelace area equals ``T1 + T2``; requiring both
    triangles to be positive separately closes the loophole where a
    self-intersecting bowtie has ``|upper lobe| > |lower lobe|`` and so
    fools the plain shoelace constraint into accepting a folded quad.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    T1, T2 : tuple of ndarray, shape ``(H-1, W-1)`` each
        ``T1`` is the upper triangle (TL, TR, BR); ``T2`` is the lower
        triangle (TL, BR, BL).
    """
    H, W = dy.shape
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy

    x0, y0 = def_x[:-1, :-1], def_y[:-1, :-1]    # TL
    x1, y1 = def_x[:-1, 1:],  def_y[:-1, 1:]     # TR
    x2, y2 = def_x[1:, 1:],   def_y[1:, 1:]      # BR
    x3, y3 = def_x[1:, :-1],  def_y[1:, :-1]     # BL

    # T1 = signed area of (TL, TR, BR)
    T1 = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    # T2 = signed area of (TL, BR, BL)
    T2 = 0.5 * ((x2 - x0) * (y3 - y0) - (x3 - x0) * (y2 - y0))
    return T1, T2


def triangulated_shoelace_det2D(phi_xy):
    """Stack both triangle areas from a ``(2, H, W)`` phi array.

    Returns shape ``(2, H-1, W-1)`` — channel 0 is ``T1`` (upper), channel
    1 is ``T2`` (lower).  The original (signed) shoelace area equals the
    sum of the two channels.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    T1, T2 = _triangulated_shoelace_areas_2d(dy, dx)
    return np.stack([T1, T2])


def triangulated_shoelace_constraint(phi_xy, submatrix_size,
                                     exclude_boundaries=True):
    """Flatten both triangle areas for the SLSQP constraint vector.

    Layout: ``[T1(r, c) for (r, c) in cells] + [T2(r, c) for (r, c) in cells]``,
    i.e. all T1 entries first, then all T2 entries.
    """
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    T1, T2 = _triangulated_shoelace_areas_2d(dy, dx)
    if exclude_boundaries:
        T1 = T1[1:-1, 1:-1]
        T2 = T2[1:-1, 1:-1]
    return np.concatenate([T1.flatten(), T2.flatten()])


# ---------------------------------------------------------------------------
# Strict both-diagonal triangle criterion (4 triangles per cell)
# ---------------------------------------------------------------------------

def _all_triangle_areas_2d(dy, dx):
    """Signed areas of all 4 triangles per cell, both diagonal splits.

    For each cell (TL, TR, BR, BL), returns four signed areas::

        T1 = area(TL, TR, BR)   — diagonal TL-BR, upper
        T2 = area(TL, BR, BL)   — diagonal TL-BR, lower
        T3 = area(TL, TR, BL)   — diagonal TR-BL, upper
        T4 = area(TR, BR, BL)   — diagonal TR-BL, lower

    Each signed area is the exact Jacobian determinant of the piecewise-
    linear map on that triangle (not a finite-difference approximation).
    Requiring all four to be positive is the strict PL-bijectivity
    condition — stricter than ``_shoelace_areas_2d`` (which can pass a
    non-convex deformed quad where one diagonal split has a flipped
    triangle) and stricter than ``_triangulated_shoelace_areas_2d``
    (which only checks the TL-BR split).

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    ndarray, shape ``(4, H-1, W-1)``
        Channels correspond to ``[T1, T2, T3, T4]`` above.
    """
    H, W = dy.shape
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy

    x0, y0 = def_x[:-1, :-1], def_y[:-1, :-1]   # TL
    x1, y1 = def_x[:-1, 1:],  def_y[:-1, 1:]    # TR
    x2, y2 = def_x[1:, 1:],   def_y[1:, 1:]     # BR
    x3, y3 = def_x[1:, :-1],  def_y[1:, :-1]    # BL

    T1 = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    T2 = 0.5 * ((x2 - x0) * (y3 - y0) - (x3 - x0) * (y2 - y0))
    T3 = 0.5 * ((x1 - x0) * (y3 - y0) - (x3 - x0) * (y1 - y0))
    T4 = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return np.stack([T1, T2, T3, T4])


def triangle_det2D(phi_xy):
    """Stack all 4 triangle signed areas from a ``(2, H, W)`` phi array.

    Returns shape ``(4, H-1, W-1)`` — one channel per triangle as in
    :func:`_all_triangle_areas_2d`.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    return _all_triangle_areas_2d(dy, dx)


def triangle_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Flatten all 4 triangle areas for the SLSQP constraint vector.

    Row layout: ``[T1_cells, T2_cells, T3_cells, T4_cells]``, each block in
    row-major cell order.  This matches ``triangle_constraint_jacobian_2d``.
    """
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    A = _all_triangle_areas_2d(dy, dx)  # (4, sy-1, sx-1)
    if exclude_boundaries:
        A = A[:, 1:-1, 1:-1]
    return A.reshape(A.shape[0], -1).ravel()
