"""Real registrations of a real image pair (mechanism 2 with actual tools).

Pair: ``data/mouse_brain/average_template_25.nii.gz`` (fixed) vs
``B0039_brain_25.nii.gz`` (moving), one coronal slice each at the same
fractional depth, moving center-cropped/padded to the fixed grid. The pair is
NOT pre-aligned, so the registrations work hard — that is the point.

Needs the (gitignored) data; raises ``FileNotFoundError`` cleanly otherwise.
Fields come back in voxel units, pull-back convention (SimpleITK's
displacement-field transform maps fixed points to moving points, the same as
``dvfopt``'s).
"""

from functools import cache

import numpy as np

from dvf_origins import ROOT, pack2d

FIXED = ROOT / 'data' / 'mouse_brain' / 'average_template_25.nii.gz'
MOVING = ROOT / 'data' / 'mouse_brain' / 'B0039_brain_25.nii.gz'


def _fit(a, shape):
    """Center-crop / zero-pad ``a`` to ``shape``."""
    out = np.zeros(shape, dtype=np.float64)
    sl_a, sl_o = [], []
    for n_a, n_o in zip(a.shape, shape):
        n = min(n_a, n_o)
        sl_a.append(slice((n_a - n) // 2, (n_a - n) // 2 + n))
        sl_o.append(slice((n_o - n) // 2, (n_o - n) // 2 + n))
    out[tuple(sl_o)] = a[tuple(sl_a)]
    return out


def _norm(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)


@cache  # every registered case starts from the same pair
def load_pair(z_frac=0.5, downsample=2):
    """``(fixed, moving)`` 2D float64 slices in [0, 1] on the same grid."""
    import SimpleITK as sitk

    for p in (FIXED, MOVING):
        if not p.is_file():
            raise FileNotFoundError(f'mouse-brain image not found (data is gitignored): {p}')
    fx = sitk.GetArrayFromImage(sitk.ReadImage(str(FIXED)))
    mv = sitk.GetArrayFromImage(sitk.ReadImage(str(MOVING)))
    f = fx[int(z_frac * fx.shape[0])].astype(np.float64)
    m = _fit(mv[int(z_frac * mv.shape[0])].astype(np.float64), f.shape)
    f, m = _norm(f), _norm(m)
    if downsample > 1:
        from skimage.transform import downscale_local_mean

        f = downscale_local_mean(f, (downsample, downsample))
        m = downscale_local_mean(m, (downsample, downsample))
    return f, m


def _pair_meta(tool, z_frac, downsample, **kw):
    return dict(
        source='registered',
        tool=tool,
        fixed=FIXED.name,
        moving=MOVING.name,
        z_frac=z_frac,
        downsample=downsample,
        **kw,
    )


def _sitk_field_to_phi(sitk, field):
    """SimpleITK 2-component displacement image -> ``(3,1,H,W)`` ``[0, dy, dx]``."""
    arr = sitk.GetArrayFromImage(field)  # (H, W, 2) = [dx, dy]
    return pack2d(arr[..., 1], arr[..., 0])


def demons(sigma=1.0, iterations=200, z_frac=0.5, downsample=2):
    """SimpleITK fast-symmetric-forces demons. ``sigma`` = gaussian smoothing
    (px) of the DISPLACEMENT field after each update (``SetStandardDeviations``
    — elastic-like regularization; small = weak)."""
    import SimpleITK as sitk

    f, m = load_pair(z_frac, downsample)
    fi, mi = sitk.GetImageFromArray(f), sitk.GetImageFromArray(m)
    mi = sitk.HistogramMatching(mi, fi, 64, 7, True)
    filt = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    filt.SetNumberOfIterations(iterations)
    filt.SetStandardDeviations(sigma)
    phi = _sitk_field_to_phi(sitk, filt.Execute(fi, mi))
    meta = _pair_meta(
        'SimpleITK fast-symmetric-forces demons',
        z_frac,
        downsample,
        sigma=sigma,
        iterations=iterations,
    )
    return phi, meta


def bspline_ffd(mesh=16, iterations=100, z_frac=0.5, downsample=2):
    """SimpleITK free-form B-spline registration, mean-squares metric, L-BFGS-B,
    NO bending penalty; ``mesh`` = control points per axis (finer = more folds)."""
    import SimpleITK as sitk

    f, m = load_pair(z_frac, downsample)
    fi, mi = sitk.GetImageFromArray(f), sitk.GetImageFromArray(m)
    tx = sitk.BSplineTransformInitializer(fi, [mesh, mesh], 3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=2000,
    )
    R.SetInitialTransform(tx, inPlace=True)
    out = R.Execute(fi, mi)
    field = sitk.TransformToDisplacementField(
        out,
        sitk.sitkVectorFloat64,
        fi.GetSize(),
        fi.GetOrigin(),
        fi.GetSpacing(),
        fi.GetDirection(),
    )
    phi = _sitk_field_to_phi(sitk, field)
    meta = _pair_meta(
        'SimpleITK B-spline FFD (no bending penalty)',
        z_frac,
        downsample,
        mesh=mesh,
        iterations=iterations,
    )
    return phi, meta


def tvl1(attachment=40.0, tightness=0.3, num_warp=5, num_iter=10, z_frac=0.5, downsample=2):
    """skimage TV-L1 optical flow on the real pair (``attachment`` higher =
    weaker regularization)."""
    from skimage.registration import optical_flow_tvl1

    f, m = load_pair(z_frac, downsample)
    flow = optical_flow_tvl1(
        f,
        m,
        attachment=attachment,
        tightness=tightness,
        num_warp=num_warp,
        num_iter=num_iter,
        dtype=np.float64,
    )
    meta = _pair_meta(
        'skimage optical_flow_tvl1', z_frac, downsample, attachment=attachment, tightness=tightness
    )
    return pack2d(flow[0], flow[1]), meta
