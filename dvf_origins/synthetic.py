"""Synthetic, seeded generators — one per fold-origin mechanism.

Each returns ``(phi (3,1,H,W), meta)``; see the package docstring for the
field convention. Sizes are in pixels, so parameters scale with ``shape``.
"""

import numpy as np
from scipy import ndimage


def _rng(seed):
    return np.random.default_rng(seed)


def _pack(dy, dx):
    return np.stack([np.zeros_like(dy), dy, dx])[:, None].astype(np.float64)


def smooth_field(shape, sigma, max_disp, rng):
    """Random smooth ``(dy, dx)``: gaussian-filtered white noise, scaled so the
    largest displacement magnitude is ``max_disp`` pixels."""
    H, W = shape
    f = np.stack([ndimage.gaussian_filter(rng.standard_normal((H, W)), sigma) for _ in range(2)])
    f *= max_disp / np.hypot(f[0], f[1]).max()
    return f


def _sample_at(field, y, x):
    """Bilinear sample of a ``(2, H, W)`` field at float ``(y, x)`` points."""
    return np.stack([ndimage.map_coordinates(c, [y, x], order=1, mode='nearest') for c in field])


def _warp_image(img, field):
    """``out(x) = img(x + field(x))`` — pull-back resampling."""
    H, W = img.shape
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    return ndimage.map_coordinates(img, [Y + field[0], X + field[1]], order=1, mode='nearest')


def _texture(shape, rng, blob_n=12):
    """Textured test image in [0, 1]: smoothed noise plus a few blobs."""
    H, W = shape
    img = ndimage.gaussian_filter(rng.standard_normal((H, W)), 2.0)
    Y, X = np.mgrid[0:H, 0:W]
    for _ in range(blob_n):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        r = rng.uniform(0.04, 0.12) * min(H, W)
        img += rng.uniform(0.5, 1.5) * np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2 * r * r))
    img -= img.min()
    return img / img.max()


# --------------------------------------------------------------------------
# 1. interpolation of sparse correspondences
# --------------------------------------------------------------------------
def interp_sparse(
    shape=(192, 192),
    seed=0,
    n_contour=160,
    n_interior=40,
    warp_sigma=20.0,
    warp_max=8.0,
    outlier_frac=0.0,
    outlier_mag=30.0,
    n_collapse=0,
    collapse_size=8,
    jitter=0.0,
    rtol=1e-4,
    maxiter=2000,
):
    """Laplacian interpolation of correspondences sampled from a smooth
    ground-truth warp, corrupted the way real contour matching is:

    * ``outlier_frac`` of the points get an independent ``outlier_mag``-px
      offset (mismatched contour points),
    * ``n_collapse`` groups of ``collapse_size`` ADJACENT contour points are
      all sent to one moving point (many-to-one collapse),
    * ``jitter`` px of gaussian noise on every moving point (incoherence).

    With all three at zero the field is the interpolant of a smooth warp and
    should be (nearly) fold-free — the control row.
    """
    from dvfopt.laplacian import solveLaplacianFromCorrespondences

    rng = _rng(seed)
    H, W = shape
    gt = smooth_field(shape, warp_sigma, warp_max, rng)

    t = np.linspace(0, 2 * np.pi, n_contour, endpoint=False)
    cy = np.clip(np.round(H / 2 + 0.35 * H * np.sin(t)), 1, H - 2)
    cx = np.clip(np.round(W / 2 + 0.40 * W * np.cos(t)), 1, W - 2)
    iy = rng.integers(1, H - 1, n_interior)
    ix = rng.integers(1, W - 1, n_interior)
    fy, fx = (
        np.concatenate([cy, iy]).astype(np.float64),
        np.concatenate([cx, ix]).astype(np.float64),
    )
    N = len(fy)

    m = np.stack([fy, fx], 1) + _sample_at(gt, fy, fx).T  # moving = fixed + gt

    n_out = round(outlier_frac * N)
    if n_out:
        idx = rng.choice(N, n_out, replace=False)
        ang = rng.uniform(0, 2 * np.pi, n_out)
        m[idx] += outlier_mag * np.stack([np.sin(ang), np.cos(ang)], 1)
    for _ in range(n_collapse):
        s = rng.integers(0, n_contour - collapse_size)
        m[s : s + collapse_size] = m[s : s + collapse_size].mean(0)
    if jitter:
        m += rng.normal(0, jitter, m.shape)

    zeros = np.zeros((N, 1))
    fixed = np.hstack([zeros, fy[:, None], fx[:, None]])
    moving = np.hstack([zeros, m])
    phi = solveLaplacianFromCorrespondences(
        (1, H, W), moving, fixed, axes=(1, 2), rtol=rtol, maxiter=maxiter, log_fn=lambda *_: None
    )
    meta = dict(
        source='synthetic',
        tool='Laplacian (dvfopt.laplacian)',
        seed=seed,
        n_pts=N,
        outlier_frac=outlier_frac,
        outlier_mag=outlier_mag,
        n_collapse=n_collapse,
        collapse_size=collapse_size,
        jitter=jitter,
        gt_max_disp=warp_max,
    )
    return np.asarray(phi, dtype=np.float64), meta


# --------------------------------------------------------------------------
# 2. dense optimization with weak regularization
# --------------------------------------------------------------------------
def dense_weak_reg(
    shape=(192, 192),
    seed=0,
    warp_sigma=12.0,
    warp_max=20.0,
    method='tvl1',
    attachment=60.0,
    tightness=0.3,
    num_warp=5,
    num_iter=10,
    radius=3,
):
    """Dense optical flow (skimage TV-L1 or iterative Lucas-Kanade) between a
    textured image and its warp by a smooth ground truth.

    The regularization dial: TV-L1 ``attachment`` (data-term weight — HIGHER
    = weaker regularization), ILK ``radius`` (SMALLER = weaker). The recovered
    flow is in pull-back convention (``moving(x + flow(x)) ≈ reference(x)``),
    the same as ``dvfopt``'s, so it is returned unchanged.

    Measured (192², σ 12): at 10 px of motion TV-L1 does not fold at any
    attachment; at 20 px it folds 54 / 126 / 355 simplex cells for attachment
    15 / 60 / 200, ILK radius 3 folds ~2000. Halving ``warp_sigma`` to 6
    multiplies the counts by ~20.
    """
    from skimage.registration import optical_flow_ilk, optical_flow_tvl1

    rng = _rng(seed)
    gt = smooth_field(shape, warp_sigma, warp_max, rng)
    moving = _texture(shape, rng)
    reference = _warp_image(moving, gt)  # reference(x) = moving(x + gt(x)) -> true flow is gt
    if method == 'tvl1':
        flow = optical_flow_tvl1(
            reference,
            moving,
            attachment=attachment,
            tightness=tightness,
            num_warp=num_warp,
            num_iter=num_iter,
            dtype=np.float64,
        )
        reg = dict(attachment=attachment, tightness=tightness)
    elif method == 'ilk':
        flow = optical_flow_ilk(
            reference, moving, radius=radius, num_warp=num_warp, dtype=np.float64
        )
        reg = dict(radius=radius)
    else:
        raise ValueError(f'method must be tvl1 or ilk, got {method!r}')
    flow = np.asarray(flow, dtype=np.float64)
    err = float(np.sqrt(((flow - gt) ** 2).sum(0).mean()))
    meta = dict(
        source='synthetic',
        tool=f'skimage optical_flow_{method}',
        seed=seed,
        gt_max_disp=warp_max,
        flow_rmse_vs_gt=err,
        **reg,
    )
    return _pack(flow[0], flow[1]), meta


# --------------------------------------------------------------------------
# 3. learned displacement field — PROXY
# --------------------------------------------------------------------------
def learned_proxy(
    shape=(192, 192),
    seed=0,
    warp_sigma=12.0,
    warp_max=10.0,
    noise_sigma=1.0,
    noise_amp=1.0,
):
    """Smooth warp plus band-limited grid-scale noise (gaussian σ =
    ``noise_sigma`` px, RMS amplitude ``noise_amp`` px): the signature of an
    unregularized network output (many shallow scattered folds).

    This is a PROXY — it reproduces the morphology, not the mechanism. Real
    learned fields come from the VoxelMorph / TransMorph notebooks
    (``benchmarks/registration/``, needs torch) via ``real.saved_field``.
    """
    rng = _rng(seed)
    gt = smooth_field(shape, warp_sigma, warp_max, rng)
    noise = np.stack(
        [ndimage.gaussian_filter(rng.standard_normal(shape), noise_sigma) for _ in range(2)]
    )
    noise *= noise_amp / noise.std()
    f = gt + noise
    meta = dict(
        source='synthetic',
        tool='proxy: smooth warp + grid-scale noise',
        proxy=True,
        seed=seed,
        gt_max_disp=warp_max,
        noise_sigma=noise_sigma,
        noise_amp=noise_amp,
    )
    return _pack(f[0], f[1]), meta


# --------------------------------------------------------------------------
# 4. discretized diffeomorphic warp
# --------------------------------------------------------------------------
def diffeo_discretized(
    shape=(256, 256),
    seed=0,
    svf_sigma=6.0,
    svf_max=24.0,
    n_steps=6,
    decimate=2,
):
    """Exponential of a smooth stationary velocity field by scaling and
    squaring (``n_steps`` squarings, linear interpolation), then decimated by
    ``decimate``. Diffeomorphic in the continuum by construction; whatever
    folds remain are discretization artifacts — few squaring steps (a big
    first step), decimation of a strongly compressing region, or sub-pixel
    folds of the interpolant that the central-difference Jdet cannot see.

    ``meta['fine_*']`` records the central-difference fold count BEFORE
    decimation so the sources can be told apart. Measured (256²): the
    defaults give a fine field with Jdet min 0.058 (clean) and 40 folded
    simplex cells after decimation; ``svf_sigma=10, svf_max=60, decimate=1``
    gives Jdet min 0.035 everywhere yet 421 simplex-folded cells, 297 of them
    bilinear-only; ``n_steps=1`` (one big step) folds ~800 cells outright.
    """
    from dvfopt.jacobian.numpy_jdet import jacobian_det2D

    rng = _rng(seed)
    H, W = shape
    v = smooth_field(shape, svf_sigma, svf_max, rng)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    u = v / 2.0**n_steps
    for _ in range(n_steps):
        u = u + _sample_at(u, Y + u[0], X + u[1])  # phi∘phi: u + u(x + u)
    fine_jdet = jacobian_det2D(u)
    fine_neg = int((fine_jdet <= 0).sum())
    if decimate > 1:
        u = u[:, ::decimate, ::decimate] / decimate
    meta = dict(
        source='synthetic',
        tool='SVF scaling-and-squaring (numpy)',
        seed=seed,
        svf_max=svf_max,
        n_steps=n_steps,
        decimate=decimate,
        fine_shape=(H, W),
        fine_jdet_neg=fine_neg,
        fine_jdet_min=float(fine_jdet.min()),
    )
    return _pack(np.ascontiguousarray(u[0]), np.ascontiguousarray(u[1])), meta
