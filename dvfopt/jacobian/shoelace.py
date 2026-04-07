"""Shoelace (geometric quad-area) helpers for deformation fields."""

import functools

import numpy as np

from dvfopt._defaults import _unpack_size


@functools.lru_cache(maxsize=8)
def _ref_grid(H, W):
    """Cached reference coordinate grids for a given (H, W)."""
    return np.mgrid[:H, :W]


def _shoelace_areas_2d(dy, dx):
    """Signed area of each deformed quad cell via the shoelace formula.

    Uses vertex order TL -> TR -> BR -> BL.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    ndarray, shape ``(H-1, W-1)``
    """
    H, W = dy.shape
    ref_y, ref_x = _ref_grid(H, W)
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
