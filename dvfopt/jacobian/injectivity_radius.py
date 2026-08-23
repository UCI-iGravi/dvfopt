"""Neighbourhood-size diagnostics for local injectivity of 2D/3D DVFs.

Companion maps that complement the standard pixel Jacobian determinant.
The classical inverse function theorem [1] guarantees that a C¹ map with
invertible ``DF(x₀)`` is injective on *some* neighbourhood of ``x₀`` — but
says nothing about its size, which is why a positive central-difference
Jdet can coexist with a folded cell.  The *quantitative* IFT [2] does bound
the size: if ``‖DF(x) − DF(x₀)‖ ≤ σ_min(DF(x₀))/2`` for all ``x`` in the
ball ``B_r(x₀)``, then ``F`` is injective on ``B_r``.  With ``DF``
``L``-Lipschitz on the ball this yields the radius

    r = σ_min(I + ∇u) / (2 L),        L = sup over B_r of ‖∇²u‖.

1. ``ift_radius_2d`` / ``ift_radius_3d`` — per-sample *certified* lower
   bound on that radius.  The Lipschitz constant is taken as the max of the
   Hessian Frobenius norm over growing ``(2w+1)``-windows and the claim at
   window ``w`` is capped at ``w`` (see the functions for the ladder).
   Larger r ⇒ larger certified-injective region around the sample; r → 0
   flags samples whose IFT guarantee shrinks below the cell — exactly
   where sub-pixel folds live even when the pixel Jdet stays positive
   (the sample-point-Jdet gap discussed in [4], [5], [6]).

2. ``cell_min_jdet_2d`` — sub-pixel injectivity certificate on each quad.
   The Jacobian determinant of the bilinear interpolant restricted to a
   unit cell is biaffine in (α, β) ∈ [0,1]², so its minimum over the cell
   is attained at one of the four corners [3] and has a closed form:

       min over cell = min over 4 corners of the Jdet built from
                       forward differences *local to the cell*.

   Positivity of this minimum guarantees the bilinear interpolant is
   injective over the entire cell — a statement the central-difference
   pixel Jdet cannot make.  (No trilinear analogue exists: the trilinear
   Jdet is not multi-affine, so 3D sub-voxel certification goes through
   the 6-tet constraint family instead.)

Model caveat: the radius maps evaluate the continuous bound on a smooth
(C²) model field whose derivatives match the difference stencils.  The
bilinear interpolant itself is only piecewise-smooth (its ``DF`` jumps
across cell edges), so the radius is a principled diagnostic for the
underlying continuous deformation, while ``cell_min_jdet_2d`` is the exact
statement for the bilinear model.

Neither uses scipy; both are pure numpy and vectorised.

References
----------
[1] W. Rudin, *Principles of Mathematical Analysis*, 3rd ed., McGraw-Hill,
    1976.  Theorem 9.24 (inverse function theorem).
[2] S. G. Krantz and H. R. Parks, *The Implicit Function Theorem: History,
    Theory, and Applications*, Birkhäuser, 2002.  §3.2 (quantitative
    inverse function theorem; the factor 1/2 is the contraction-mapping
    convention, which also gives quantitative control on the inverse).
[3] P. M. Knupp, "On the invertibility of the isoparametric map,"
    *Computer Methods in Applied Mechanics and Engineering* 78(3):313–329,
    1990.  Bilinear quad element invertible iff the Jacobian is positive
    at the four corners — the basis of ``cell_min_jdet_2d`` (the standard
    FEM element-validity check).
[4] B. Karaçalı and C. Davatzikos, "Estimating topology preserving and
    smooth displacement fields," *IEEE Transactions on Medical Imaging*
    23(7):868–880, 2004.  Sample-point Jdet is insufficient for discrete
    deformation fields.
[5] Y. Choi and S. Lee, "Injectivity conditions of 2D and 3D uniform cubic
    B-spline functions," *Graphical Models* 62(6):411–427, 2000.
[6] S. Y. Chun and J. A. Fessler, "A simple regularizer for B-spline
    nonrigid image registration that encourages local invertibility,"
    *IEEE Journal of Selected Topics in Signal Processing* 3(1):159–169,
    2009.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Quantitative-IFT neighbourhood radius (per sample)
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

    For a 2-component field (u_x, u_y) the Hessian of each component is a
    symmetric 2x2 tensor with three independent entries {xx, xy, yy}. The
    Frobenius norm squared of a symmetric 2x2 matrix is
    H_xx^2 + 2*H_xy^2 + H_yy^2 (off-diagonal counted twice). Summing over
    both components gives the field's Hessian Frobenius norm at each pixel.
    Frobenius bounds the operator norm from above, so using it keeps the
    radius bound conservative.
    """
    dxx = np.gradient(np.gradient(dx, axis=1), axis=1)
    dxy = np.gradient(np.gradient(dx, axis=1), axis=0)
    dyy = np.gradient(np.gradient(dx, axis=0), axis=0)
    exx = np.gradient(np.gradient(dy, axis=1), axis=1)
    exy = np.gradient(np.gradient(dy, axis=1), axis=0)
    eyy = np.gradient(np.gradient(dy, axis=0), axis=0)
    return np.sqrt(
        dxx * dxx + 2.0 * dxy * dxy + dyy * dyy + exx * exx + 2.0 * exy * exy + eyy * eyy
    )


def _dilate_max(a):
    """One step of a 3^ndim moving maximum (w steps ⇒ (2w+1)^ndim window)."""
    p = np.pad(a, 1, mode='edge')
    out = a
    for idx in np.ndindex(*(3,) * a.ndim):
        out = np.maximum(out, p[tuple(slice(i, i + n) for i, n in zip(idx, a.shape))])
    return out


def _certified_ladder(sigma_min, hess, eps, max_window):
    """Best certified radius over windowed Lipschitz estimates.

    The quantitative IFT [2] certifies injectivity on B_r whenever
    r ≤ σ_min / (2 L) with L = sup of ‖∇²u‖ over B_r.  Taking L_w as the
    max of the pointwise Hessian norm over the (2w+1)-window makes the
    claim self-consistent only up to radius w, so the certificate at
    window w is min(σ_min / (2 L_w), w); the ladder returns the largest
    certificate over w = 1..max_window (and therefore saturates at
    max_window — a returned value of max_window means "at least this").

    ``max_window=0`` reproduces the legacy pointwise *estimate*
    σ_min / (2‖∇²u(x)‖ + eps), which evaluates the Lipschitz constant only
    at the sample itself — optimistic, not a certificate.
    """
    if max_window == 0:
        return sigma_min / (2.0 * hess + eps)
    best = np.zeros_like(sigma_min)
    L = hess
    for w in range(1, int(max_window) + 1):
        L = _dilate_max(L)
        best = np.maximum(best, np.minimum(sigma_min / (2.0 * L + eps), float(w)))
    return best


def ift_radius_2d(phi_xy, eps=1e-8, max_window=8):
    """Per-pixel certified lower bound on the IFT injectivity radius.

    Parameters
    ----------
    phi_xy : ndarray, shape ``(2, H, W)`` or ``(2, 1, H, W)``
        Displacement field with channels ``[dy, dx]``.
    eps : float
        Regulariser added to ``2·‖∇²u‖`` to avoid division-by-zero in the
        (locally affine) zero-Hessian regime.
    max_window : int
        Certification cap in pixels (see :func:`_certified_ladder`); the
        returned map saturates at this value.  ``0`` selects the legacy
        pointwise estimate (optimistic, not certified).

    Returns
    -------
    ndarray, shape ``(H, W)``
        Certified radius map.  Large values mean the (smooth-model)
        deformation is injective over a large neighbourhood of the pixel;
        values collapsing toward 0 flag pixels whose IFT guarantee is
        sub-pixel — where cells can fold even under positive pixel Jdet.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)

    a = 1.0 + np.gradient(dx, axis=1)  # 1 + ∂dx/∂x
    b = np.gradient(dx, axis=0)  # ∂dx/∂y
    c = np.gradient(dy, axis=1)  # ∂dy/∂x
    d = 1.0 + np.gradient(dy, axis=0)  # 1 + ∂dy/∂y

    sigma_min = _sigma_min_2d(a, b, c, d)
    hess = _hessian_frob_norm_2d(dy, dx)
    return _certified_ladder(sigma_min, hess, eps, max_window)


def ift_radius_3d(phi_zyx, eps=1e-8, max_window=8):
    """Per-voxel certified lower bound on the IFT injectivity radius (3D).

    Same bound and window ladder as :func:`ift_radius_2d`, on a true-3D
    volume.  Needs ``D >= 2`` (single-slice volumes are 2D — use
    :func:`ift_radius_2d`).

    Parameters
    ----------
    phi_zyx : ndarray, shape ``(3, D, H, W)``
        Displacement field with channels ``[dz, dy, dx]``.
    eps, max_window
        As in :func:`ift_radius_2d`.

    Returns
    -------
    ndarray, shape ``(D, H, W)``
        Certified radius map in voxel units.
    """
    if phi_zyx.ndim != 4 or phi_zyx.shape[0] != 3:
        raise ValueError(f'expected a (3, D, H, W) volume, got {phi_zyx.shape}')
    D, H, W = phi_zyx.shape[-3:]
    grads = [np.gradient(phi_zyx[i]) for i in range(3)]  # [dz, dy, dx] × (z, y, x)

    DF = np.empty((D, H, W, 3, 3))
    for i in range(3):
        for j in range(3):
            DF[..., i, j] = grads[i][j] + (1.0 if i == j else 0.0)
    # ponytail: full-volume batched 3x3 SVD (~9 float64 copies of the volume);
    # chunk over z if memory ever matters on big cohort volumes.
    sigma_min = np.linalg.svd(DF.reshape(-1, 3, 3), compute_uv=False)[:, -1].reshape(D, H, W)

    hess_sq = np.zeros((D, H, W))
    for g in grads:
        for j in range(3):
            gj = np.gradient(g[j])
            for k in range(j, 3):
                hess_sq += (1.0 if j == k else 2.0) * gj[k] ** 2
    return _certified_ladder(sigma_min, np.sqrt(hess_sq), eps, max_window)


# ---------------------------------------------------------------------------
# Bilinear cell-minimum Jacobian (closed form, biaffine extremum)
# ---------------------------------------------------------------------------


def cell_min_jdet_2d(phi_xy):
    """Minimum Jdet over each quad for the bilinear interpolant.

    For cell ``(r, c)`` with grid corners ``(r,c), (r,c+1), (r+1,c),
    (r+1,c+1)``, the Jacobian determinant of the bilinear map in local
    coordinates ``(α, β) ∈ [0, 1]²`` is biaffine and attains its extrema
    at the four corners (Knupp [3]; the standard FEM quad-validity check).
    This function evaluates the Jdet at each corner using forward
    differences *local to the cell* and returns the elementwise minimum.

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
    dbx_top = dx[:-1, 1:] - dx[:-1, :-1]  # at α=0 (top row of cell)
    dbx_bot = dx[1:, 1:] - dx[1:, :-1]  # at α=1 (bottom row)
    dby_top = dy[:-1, 1:] - dy[:-1, :-1]
    dby_bot = dy[1:, 1:] - dy[1:, :-1]

    # α-direction (row / y) forward diffs, one value per cell-col
    dax_left = dx[1:, :-1] - dx[:-1, :-1]  # at β=0 (left col of cell)
    dax_right = dx[1:, 1:] - dx[:-1, 1:]  # at β=1 (right col)
    day_left = dy[1:, :-1] - dy[:-1, :-1]
    day_right = dy[1:, 1:] - dy[:-1, 1:]

    def corner_jdet(dbx, dby, dax, day):
        return (1.0 + dbx) * (1.0 + day) - dax * dby

    j00 = corner_jdet(dbx_top, dby_top, dax_left, day_left)
    j01 = corner_jdet(dbx_top, dby_top, dax_right, day_right)
    j10 = corner_jdet(dbx_bot, dby_bot, dax_left, day_left)
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
    out[:-1, 1:] = np.minimum(out[:-1, 1:], cell_map)  # cell is TR
    out[1:, :-1] = np.minimum(out[1:, :-1], cell_map)  # cell is BL
    out[1:, 1:] = np.minimum(out[1:, 1:], cell_map)  # cell is BR
    return out
