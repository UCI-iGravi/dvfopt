"""Neighbourhood-size diagnostics for local injectivity of 2D DVFs.

Two companion maps that complement the standard pixel Jacobian determinant:

1. ``ift_radius_2d`` — quantitative-IFT lower bound on the size of the
   neighbourhood in which the continuous deformation is injective.
   From the quantitative inverse-function theorem
   (Krantz & Parks 2002, §3.2):

       r ≳ σ_min(I + ∇u) / (2 · ‖∇²u‖)

   Larger r ⇒ larger certified-invertible region around the sample.

2. ``cell_min_jdet_2d`` — sub-pixel injectivity certificate on each quad.
   The Jacobian determinant of the bilinear interpolant restricted to a
   unit cell is biaffine in (α, β) ∈ [0,1]², so its minimum over the cell
   is attained at one of the four corners and has a closed form:

       min over cell = min over 4 corners of the Jdet built from
                       forward differences *local to the cell*.

   Positivity of this minimum guarantees the bilinear interpolant is
   injective over the entire cell — a statement the central-difference
   pixel Jdet cannot make.

Neither uses scipy; both are pure numpy and vectorised.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Quantitative-IFT neighbourhood radius (per pixel)
# ---------------------------------------------------------------------------

def _sigma_min_2d(a, b, c, d):
    """Smallest singular value of each 2×2 matrix [[a,b],[c,d]], vectorised.

    Uses the identity σ_min · σ_max = |det|, σ_min² + σ_max² = ‖·‖_F².
    """
    det = a * d - b * c
    frob_sq = a * a + b * b + c * c + d * d
    disc = np.clip(frob_sq * frob_sq - 4.0 * det * det, 0.0, None)
    sigma_min_sq = 0.5 * (frob_sq - np.sqrt(disc))
    return np.sqrt(np.clip(sigma_min_sq, 0.0, None))


def _hessian_frob_norm_2d(dy, dx):
    """Pointwise Frobenius norm of the second-derivative tensor of u.

    For a 2-component field (u_x, u_y) there are six independent second
    partials: {xx, xy, yy} for each component. Returns their sqrt-sum-of-
    squares at each pixel.
    """
    dxx = np.gradient(np.gradient(dx, axis=1), axis=1)
    dxy = np.gradient(np.gradient(dx, axis=1), axis=0)
    dyy = np.gradient(np.gradient(dx, axis=0), axis=0)
    exx = np.gradient(np.gradient(dy, axis=1), axis=1)
    exy = np.gradient(np.gradient(dy, axis=1), axis=0)
    eyy = np.gradient(np.gradient(dy, axis=0), axis=0)
    return np.sqrt(dxx * dxx + dxy * dxy + dyy * dyy
                   + exx * exx + exy * exy + eyy * eyy)


def ift_radius_2d(phi_xy, eps=1e-8):
    """Per-pixel lower bound on the IFT neighbourhood radius.

    Parameters
    ----------
    phi_xy : ndarray, shape ``(2, H, W)`` or ``(2, 1, H, W)``
        Displacement field with channels ``[dy, dx]``.
    eps : float
        Regulariser added to ``‖∇²u‖`` to avoid division-by-zero in the
        (rare) constant-Jacobian regime.

    Returns
    -------
    ndarray, shape ``(H, W)``
        ``r(x) = σ_min(I + ∇u) / (2 · ‖∇²u‖ + eps)``.  Large values mean
        the continuous deformation is provably injective over a large
        neighbourhood of ``x``; tiny values mean the IFT guarantee shrinks
        toward a point.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)

    a = 1.0 + np.gradient(dx, axis=1)   # 1 + ∂dx/∂x
    b = np.gradient(dx, axis=0)         # ∂dx/∂y
    c = np.gradient(dy, axis=1)         # ∂dy/∂x
    d = 1.0 + np.gradient(dy, axis=0)   # 1 + ∂dy/∂y

    sigma_min = _sigma_min_2d(a, b, c, d)
    hess = _hessian_frob_norm_2d(dy, dx)
    return sigma_min / (2.0 * hess + eps)


# ---------------------------------------------------------------------------
# Bilinear cell-minimum Jacobian (closed form, biaffine extremum)
# ---------------------------------------------------------------------------

def cell_min_jdet_2d(phi_xy):
    """Minimum Jdet over each quad for the bilinear interpolant.

    For cell ``(r, c)`` with grid corners ``(r,c), (r,c+1), (r+1,c),
    (r+1,c+1)``, the Jacobian determinant of the bilinear map in local
    coordinates ``(α, β) ∈ [0, 1]²`` is biaffine and attains its extrema
    at the four corners.  This function evaluates the Jdet at each corner
    using forward differences *local to the cell* and returns the
    elementwise minimum.

    ``cell_min_jdet > 0`` ⇒ the bilinear interpolant has positive Jdet
    throughout the whole cell (true sub-pixel injectivity certificate).

    Parameters
    ----------
    phi_xy : ndarray, shape ``(2, H, W)`` or ``(2, 1, H, W)``
        Displacement field with channels ``[dy, dx]``.

    Returns
    -------
    ndarray, shape ``(H-1, W-1)``
        One value per quad, indexed by its top-left corner.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)

    # β-direction (column / x) forward diffs, one value per cell-row
    dbx_top = dx[:-1, 1:] - dx[:-1, :-1]    # at α=0 (top row of cell)
    dbx_bot = dx[1:,  1:] - dx[1:,  :-1]    # at α=1 (bottom row)
    dby_top = dy[:-1, 1:] - dy[:-1, :-1]
    dby_bot = dy[1:,  1:] - dy[1:,  :-1]

    # α-direction (row / y) forward diffs, one value per cell-col
    dax_left  = dx[1:, :-1] - dx[:-1, :-1]  # at β=0 (left col of cell)
    dax_right = dx[1:, 1:]  - dx[:-1, 1:]   # at β=1 (right col)
    day_left  = dy[1:, :-1] - dy[:-1, :-1]
    day_right = dy[1:, 1:]  - dy[:-1, 1:]

    def corner_jdet(dbx, dby, dax, day):
        return (1.0 + dbx) * (1.0 + day) - dax * dby

    j00 = corner_jdet(dbx_top, dby_top, dax_left,  day_left)
    j01 = corner_jdet(dbx_top, dby_top, dax_right, day_right)
    j10 = corner_jdet(dbx_bot, dby_bot, dax_left,  day_left)
    j11 = corner_jdet(dbx_bot, dby_bot, dax_right, day_right)
    return np.minimum(np.minimum(j00, j01), np.minimum(j10, j11))


def cell_to_pixel_min(cell_map, H, W):
    """Project a ``(H-1, W-1)`` per-cell scalar to a ``(H, W)`` per-pixel map.

    Each pixel is assigned the minimum of the (up to four) cells meeting
    at that corner.  Boundary pixels participate in fewer cells; they take
    the min of whatever cells they touch.  Useful for overlaying cell
    diagnostics on pixel-aligned heatmaps.
    """
    out = np.full((H, W), np.inf, dtype=cell_map.dtype)
    out[:-1, :-1] = np.minimum(out[:-1, :-1], cell_map)  # cell is TL of pixel
    out[:-1, 1:]  = np.minimum(out[:-1, 1:],  cell_map)  # cell is TR
    out[1:,  :-1] = np.minimum(out[1:,  :-1], cell_map)  # cell is BL
    out[1:,  1:]  = np.minimum(out[1:,  1:],  cell_map)  # cell is BR
    return out
