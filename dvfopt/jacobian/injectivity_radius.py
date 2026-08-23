"""Neighbourhood-size diagnostics for local injectivity of 2D/3D DVFs.

Companion maps to the standard pixel Jacobian determinant.  The classical
inverse function theorem [1] guarantees that a C¹ map with invertible
``DF(x₀)`` is injective on *some* neighbourhood of ``x₀`` — but says
nothing about its size, which is how a positive central-difference Jdet
can coexist with a folded cell.  The *quantitative* IFT [2] bounds the
size: ``F`` is injective on ``B_r(x₀)`` whenever
``sup over B_r of ‖DF − DF(x₀)‖ ≤ σ_min(DF(x₀))/2``, which for ``DF``
``L``-Lipschitz on the ball gives ``r = σ_min(I + ∇u) / (2 L)``.

1. ``ift_radius_2d`` / ``ift_radius_3d`` — per-sample lower-bound
   *estimate* of that radius from finite differences: σ_min of the
   central-difference ``DF``, ``L`` from windowed maxima of tight
   second differences (:func:`_window_ladder` has the exact semantics).
   ``r`` collapsing toward 0 flags samples whose injective neighbourhood
   is sub-pixel — where cells can fold without the pixel Jdet noticing
   (the sample-point-Jdet gap discussed in [4], [5], [6]).

   Two limits to respect when reading the map:

   * **Estimate, not certificate.**  No finite set of samples can
     upper-bound the derivatives of an arbitrary interpolant, so no
     finite-difference radius can *prove* injectivity of the underlying
     continuous deformation.  The exact sub-grid certificates are
     ``cell_min_jdet_2d`` below (bilinear model, 2D) and the 6-tet
     volume family (3D).
   * **Injectivity is orientation-blind.**  σ_min sees a reflection as
     perfectly invertible: a uniformly ``Jdet < 0`` region reports a
     *large* radius.  Read this map alongside the fold statistics
     (:mod:`dvfopt.metrics`), never instead of them.

2. ``cell_min_jdet_2d`` — exact sub-pixel injectivity certificate on
   each quad.  The Jacobian determinant of the bilinear interpolant
   restricted to a unit cell is biaffine in (α, β) ∈ [0,1]², so its
   minimum over the cell is attained at one of the four corners [3] and
   has a closed form.  Positivity guarantees the bilinear interpolant is
   injective over the entire cell — a statement the central-difference
   pixel Jdet cannot make.  (No trilinear analogue exists: the trilinear
   Jdet is not multi-affine, so 3D sub-voxel certification goes through
   the 6-tet constraint family instead.)

Pure numpy + scipy.ndimage; vectorised.

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
from scipy.ndimage import maximum_filter

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


def _shift(p, shifts, shape):
    """View of the edge-padded array ``p`` offset by ``shifts`` (each ±1/0)."""
    return p[tuple(slice(1 + s, 1 + s + n) for s, n in zip(shifts, shape))]


def _hessian_frob_norm(comps):
    """Frobenius norm of the second-difference tensor of u, per sample.

    One implementation for any dimensionality: for each displacement
    component in ``comps``, pure second derivatives use the tight
    spacing-1 stencil ``f[i-1] − 2 f[i] + f[i+1]`` and mixed ones the
    standard /4 cross stencil, with edge replication at the boundary.
    Off-diagonal entries of each symmetric per-component Hessian count
    twice in the Frobenius sum; Frobenius bounds the operator norm from
    above, keeping the radius estimate conservative.

    The tight stencils matter: a double ``np.gradient`` chain has
    effective spacing 2, which smears single-pixel spikes (about a 4×
    curvature underestimate) and is entirely blind to period-2 (Nyquist)
    fields — the adversarial regimes this diagnostic exists to flag.
    """
    shape = comps[0].shape
    nd = len(shape)
    total = np.zeros(shape)
    for f in comps:
        p = np.pad(f, 1, mode='edge')
        for j in range(nd):
            ej = [int(j == ax) for ax in range(nd)]
            total += (_shift(p, ej, shape) - 2.0 * f + _shift(p, [-e for e in ej], shape)) ** 2
            for k in range(j + 1, nd):
                ek = [int(k == ax) for ax in range(nd)]
                cross = 0.25 * (
                    _shift(p, [a + b for a, b in zip(ej, ek)], shape)
                    - _shift(p, [a - b for a, b in zip(ej, ek)], shape)
                    - _shift(p, [b - a for a, b in zip(ej, ek)], shape)
                    + _shift(p, [-a - b for a, b in zip(ej, ek)], shape)
                )
                total += 2.0 * cross**2
    return np.sqrt(total)


def _window_ladder(sigma_min, hess, eps, max_window):
    """Best radius estimate over windowed Lipschitz constants.

    The quantitative IFT [2] gives injectivity on ``B_r`` when
    ``r ≤ σ_min / (2 L)`` with ``L = sup over B_r of ‖∇²u‖``.  Taking
    ``L_w`` as the max of the pointwise second-difference norm over the
    ``(2w+1)``-window is self-consistent only up to radius ``w``, so the
    claim at window ``w`` is ``min(σ_min / (2 L_w), w)``; the ladder
    returns the largest claim over ``w = 1..max_window`` and therefore
    saturates at ``max_window`` (a returned value of ``max_window`` means
    "at least this").  ``max_window=0`` selects the pointwise variant
    ``σ_min / (2 ‖∇²u(x)‖ + eps)``, which evaluates ``L`` only at the
    sample itself (more optimistic still).

    Estimate, not certificate — see the module docstring.
    """
    mw = int(max_window)
    if mw != max_window or mw < 0:
        raise ValueError(f'max_window must be a non-negative integer, got {max_window!r}')
    if mw == 0:
        return sigma_min / (2.0 * hess + eps)
    best = np.zeros_like(sigma_min)
    L = hess
    for w in range(1, mw + 1):
        L = maximum_filter(L, size=3, mode='nearest')  # (2w+1)-window max after w steps
        best = np.maximum(best, np.minimum(sigma_min / (2.0 * L + eps), float(w)))
    return best


def ift_radius_2d(phi_xy, eps=1e-8, max_window=8):
    """Per-pixel lower-bound estimate of the IFT injectivity radius.

    Parameters
    ----------
    phi_xy : ndarray, shape ``(2, H, W)`` or ``(2, 1, H, W)``
        Displacement field with channels ``[dy, dx]``.
    eps : float
        Regulariser added to ``2·‖∇²u‖`` to avoid division-by-zero in the
        (locally affine) zero-Hessian regime.
    max_window : int
        Window-ladder cap in pixels — the returned map saturates at this
        value; ``0`` selects the pointwise variant.  Semantics in
        :func:`_window_ladder`.

    Returns
    -------
    ndarray, shape ``(H, W)``
        Radius-estimate map.  Values collapsing toward 0 flag pixels
        whose injective neighbourhood is sub-pixel — where cells can fold
        even under positive pixel Jdet.  An estimate, orientation-blind;
        see the module docstring for both caveats.
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)

    a = 1.0 + np.gradient(dx, axis=1)  # 1 + ∂dx/∂x
    b = np.gradient(dx, axis=0)  # ∂dx/∂y
    c = np.gradient(dy, axis=1)  # ∂dy/∂x
    d = 1.0 + np.gradient(dy, axis=0)  # 1 + ∂dy/∂y

    sigma_min = _sigma_min_2d(a, b, c, d)
    return _window_ladder(sigma_min, _hessian_frob_norm([dy, dx]), eps, max_window)


def ift_radius_3d(phi_zyx, eps=1e-8, max_window=8):
    """Per-voxel lower-bound estimate of the IFT injectivity radius (3D).

    Same estimate and window ladder as :func:`ift_radius_2d`, on a
    true-3D volume.

    Parameters
    ----------
    phi_zyx : ndarray, shape ``(3, D, H, W)`` with ``D >= 2``
        Displacement field with channels ``[dz, dy, dx]``.  Single-slice
        volumes are 2D — use :func:`ift_radius_2d`.
    eps, max_window
        As in :func:`ift_radius_2d`.

    Returns
    -------
    ndarray, shape ``(D, H, W)``
        Radius-estimate map in voxel units.
    """
    if phi_zyx.ndim != 4 or phi_zyx.shape[0] != 3 or phi_zyx.shape[1] < 2:
        raise ValueError(
            f'expected a true-3D (3, D>=2, H, W) volume, got {phi_zyx.shape}; '
            'single-slice fields are 2D — use ift_radius_2d'
        )
    D, H, W = phi_zyx.shape[-3:]
    comps = [phi_zyx[0], phi_zyx[1], phi_zyx[2]]  # [dz, dy, dx]

    DF = np.empty((D, H, W, 3, 3))
    for i in range(3):
        for j in range(3):
            DF[..., i, j] = np.gradient(comps[i], axis=j) + (1.0 if i == j else 0.0)
    # ponytail: full-volume batched 3x3 SVD (~9 float64 copies of the volume);
    # chunk over z if memory ever matters on big cohort volumes.
    sigma_min = np.linalg.svd(DF.reshape(-1, 3, 3), compute_uv=False)[:, -1].reshape(D, H, W)

    return _window_ladder(sigma_min, _hessian_frob_norm(comps), eps, max_window)


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
