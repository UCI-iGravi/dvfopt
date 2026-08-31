"""Mechanism 3 for real: small learned registration networks, trained here.

The setups are the ``benchmarks/registration/{voxelmorph,transmorph}-registration``
notebooks' — 200 epochs x 50 steps, MSE + λ·smoothness — but seeded, and
returning the inference field on a held-out pair instead of correcting it in
place. ``integration_steps=0`` is a DIRECT displacement regressor (the
unregularized-network signature the paper's mechanism 3 is about); ``7`` puts
a scaling-and-squaring layer on top (a learned diffeomorphism — folds, if any,
are then mechanism 4's).

``data=None`` trains on the notebooks' synthetic ellipse images (unrelated
random pairs); ``data=cohort_data`` (any callable returning the same dict
shape) trains on REAL brains — the cohort affinely aligned onto the template,
coronal planes paired with the template's — with ``image_size`` / ``n_train``
/ ``n_test_pairs`` / ``pair`` then unused (they describe the synthetic set).

Needs torch (+ voxelmorph / timm), which the main venv does not carry — the
separate-venv recipe is in ``dvf_origins/README.md``. Without them the builders
raise ``ModuleNotFoundError`` and ``generate`` skips the rows. Every builder
records ``warp_rmse`` — the RMSE between the network's own warped source and a
pull-back resampling of the source by the RETURNED field, using the network's
own off-image padding — next to ``warp_rmse_swapped`` (same with the channels
swapped), so the field convention (``[dy, dx]``, ``moving(x + u(x))``) is
checked on what is shipped, not assumed; ``off_image_frac`` is the collapse
detector (see :func:`transmorph`).
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from dvf_origins._common import ORIGINS, ROOT, pack2d
from dvf_origins.registered import FIXED as TEMPLATE
from dvf_origins.registered import _fit, _norm
from dvf_origins.synthetic import _warp_image

# RegTools pipeline outputs (external, not in this repo): per brain the axis-aligned
# volume and the ANTs affine that puts it on the template grid. Default = the
# sibling RegTools checkout the cohort copy came from; override with the env var.
REGTOOLS_COHORT = Path(
    os.environ.get(
        'DVF_ORIGINS_REGTOOLS',
        ROOT.parent.parent / 'UCI-XuLab' / 'UCI-XuLab-RegTools' / 'output' / 'brain25_cohort',
    )
)
_ALIGNED = Path('01_axis_alignment') / 'axisAlignedData.nii.gz'
_AFFINE = Path('02_nonlinear') / 'parameters' / 'fwd_transforms' / 'ants_affine_1.mat'
CACHE = ORIGINS / 'cache'


def make_random_image(size, rng):
    """The notebooks' synthetic image: 1-4 random ellipses on black."""
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for _ in range(rng.integers(1, 5)):
        cy = rng.uniform(size * 0.15, size * 0.85)
        cx = rng.uniform(size * 0.15, size * 0.85)
        ry = rng.uniform(size * 0.06, size * 0.25)
        rx = rng.uniform(size * 0.06, size * 0.25)
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        img[mask] = rng.uniform(0.3, 1.0)
    return img


def _dataset(image_size, n_train, n_test_pairs, data_seed):
    rng = np.random.default_rng(data_seed)
    images = np.stack(
        [make_random_image(image_size, rng) for _ in range(n_train + 2 * n_test_pairs)]
    )
    return images[:n_train], images[n_train:]


def _synthetic_data(image_size, n_train, n_test_pairs, pair, data_seed):
    """The notebooks' data: unrelated random images, any two make a training pair."""
    if not 0 <= pair < n_test_pairs:
        raise ValueError(f'pair={pair} out of range for n_test_pairs={n_test_pairs}')
    train, test = _dataset(image_size, n_train, n_test_pairs, data_seed)
    return dict(
        train=(train, train),
        test=(test[2 * pair], test[2 * pair + 1]),
        paired=False,
        info=dict(data='synthetic', image_size=image_size, n_train=n_train, pair=pair),
    )


def _prep_plane(a, downsample, crop):
    """Block-mean downsample, [1, 99]-percentile normalise, centre-crop to ``crop``."""
    from skimage.transform import downscale_local_mean

    a = downscale_local_mean(a.astype(np.float64), (downsample, downsample))
    if any(n < c for n, c in zip(a.shape, crop)):
        raise ValueError(f'plane {a.shape} after /{downsample} is smaller than crop {tuple(crop)}')
    if any(c % 32 for c in crop):
        raise ValueError(f'crop {tuple(crop)} must be multiples of 32 (five-level UNet)')
    return _fit(_norm(a), tuple(crop)).astype(np.float32)


def _cohort_planes(vol_arr, zs, downsample, crop):
    """Coronal planes ``i = z`` of a SimpleITK ``(k, j, i)`` array as ``(H, W) = (j, k)``."""
    return np.stack([_prep_plane(vol_arr[:, :, z].T, downsample, crop) for z in zs])


def cohort_brains():
    """Brains under ``REGTOOLS_COHORT`` that have the aligned volume and the affine."""
    if not REGTOOLS_COHORT.is_dir():
        raise FileNotFoundError(
            f'RegTools cohort outputs not found at {REGTOOLS_COHORT} (set DVF_ORIGINS_REGTOOLS); '
            f'expects <brain>/{_ALIGNED.as_posix()} and <brain>/{_AFFINE.as_posix()}'
        )
    return sorted(
        p.name
        for p in REGTOOLS_COHORT.iterdir()
        if (p / _ALIGNED).is_file() and (p / _AFFINE).is_file()
    )


def cohort_data(
    test_brain='B0039',
    test_z=264,
    zs=tuple(range(60, 480, 12)),
    downsample=3,
    crop=(96, 128),
    cache=True,
):
    """REAL training data: cohort brains affinely aligned onto the template grid
    (the ANTs fwd affine of the RegTools pipeline — exactly the input SyN then
    deformed nonlinearly; verified on B0039 z=264: slice correlation with the
    template 0.18 identity -> 0.87 affine -> 0.94 SyN), coronal planes ``i = z``,
    paired with the template's plane at the same ``z``. Every brain except
    ``test_brain`` trains; the test pair is ``test_brain`` at ``test_z`` (default:
    the plane the real m1 / m4 rows use — but on THIS grid: block-mean downsampled
    by ``downsample`` and centre-cropped to ``crop``, multiples of 32 for the
    five-level UNet; 3 / 96x128 keeps ~85 % of the 320x456 field of view, so
    compare fold fractions, not counts, against the native-resolution rows).
    Cached under ``data/dvfs/origins/cache/`` (gitignored), keyed by a hash of every
    input and verified on load. Raises ``FileNotFoundError`` when the RegTools
    outputs (external) or the template are absent.
    """
    zs = [int(z) for z in zs]
    brains = cohort_brains()
    if test_brain not in brains:
        raise FileNotFoundError(f'{test_brain} not among the cohort brains on disk: {brains}')
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f'template not found (data is gitignored): {TEMPLATE}')
    train_brains = [b for b in brains if b != test_brain]
    info = dict(
        data='cohort',
        root=str(REGTOOLS_COHORT),
        train_brains=train_brains,
        test_brain=test_brain,
        test_z=int(test_z),
        zs=zs,
        downsample=int(downsample),
        crop=[int(c) for c in crop],
    )
    key = hashlib.sha1(json.dumps(info, sort_keys=True).encode()).hexdigest()[:10]
    cache_file = CACHE / f'cohort_slices_{test_brain}_z{test_z}_{key}.npz'

    def bundle(train_src, tgt, test_src, test_tgt):
        return dict(
            train=(train_src, tgt),  # brain-major: row i pairs with tgt[i % len(tgt)]
            test=(test_src, test_tgt),
            paired=True,
            info=dict(info, n_train=len(train_src)),
        )

    if cache and cache_file.is_file():
        with np.load(cache_file) as z:
            if json.loads(str(z['info'])) == info:
                return bundle(z['train_src'], z['train_tgt'], z['test_src'], z['test_tgt'])
    import SimpleITK as sitk

    tpl = sitk.ReadImage(str(TEMPLATE))
    tpl_arr = sitk.GetArrayFromImage(tpl)
    n_i = tpl.GetSize()[0]
    for z in [*zs, test_z]:
        if not 0 <= z < n_i:
            raise ValueError(f'z={z} out of range for the template depth {n_i}')

    def aligned(brain):
        d = REGTOOLS_COHORT / brain
        mov = sitk.ReadImage(str(d / _ALIGNED))
        aff = sitk.ReadTransform(str(d / _AFFINE))
        return sitk.GetArrayFromImage(
            sitk.Resample(mov, tpl, aff, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        )

    tgt = _cohort_planes(tpl_arr, zs, downsample, crop)
    train_src = np.concatenate(
        [_cohort_planes(aligned(b), zs, downsample, crop) for b in train_brains]
    )
    test_src = _cohort_planes(aligned(test_brain), [test_z], downsample, crop)[0]
    if test_z in zs:
        test_tgt = tgt[zs.index(test_z)]
    else:
        test_tgt = _cohort_planes(tpl_arr, [test_z], downsample, crop)[0]
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            train_src=train_src,
            train_tgt=tgt,
            test_src=test_src,
            test_tgt=test_tgt,
            info=json.dumps(info),
        )
    return bundle(train_src, tgt, test_src, test_tgt)


def _resolve_data(data, image_size, n_train, n_test_pairs, pair, seed):
    if data is None:
        return _synthetic_data(image_size, n_train, n_test_pairs, pair, data_seed=42 + seed)
    if callable(data):
        return data()
    raise TypeError('data must be None (the synthetic set) or a callable returning the data dict')


def _batches(data_d, device, torch):
    """Training tensors and a pair sampler honouring ``paired``, plus the test pair."""
    A_np, B_np = data_d['train']
    paired = bool(data_d['paired'])
    if paired and len(A_np) % len(B_np):
        raise ValueError(f'paired data: {len(A_np)} sources vs {len(B_np)} targets')
    if not paired and len(A_np) != len(B_np):
        raise ValueError('unpaired data needs the same number of sources and targets')
    A, B = (
        torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(1).to(device)
        for x in (A_np, B_np)
    )

    def sample(gen):
        i, j = torch.randint(0, len(A), (2,), generator=gen).tolist()
        if paired:
            j = i % len(B)
        return A[i : i + 1], B[j : j + 1]

    test = tuple(
        torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(device)
        for x in data_d['test']
    )
    return sample, test


def _finish(tool, source, disp, warped, losses, t0, pad, **params):
    phi = pack2d(disp[0], disp[1])
    field = phi[1:, 0]  # the RETURNED channels, so a swap in the packing would show here
    if not np.isfinite(field).all():
        raise FloatingPointError(f'{tool}: non-finite displacement (diverged)')
    # the network's own off-image padding, so the comparison is exact over the whole image
    # (grid_sample 'zeros' blends out-of-bounds taps with 0 = scipy 'grid-constant', NOT
    # 'constant', which replaces the whole out-of-bounds sample; 'border' = 'nearest')
    mode = {'zeros': 'grid-constant', 'border': 'nearest'}[pad]

    def _rmse(f):
        return float(np.sqrt(((_warp_image(source, f, mode=mode) - warped) ** 2).mean()))

    rmse, rmse_swapped = _rmse(field), _rmse(field[::-1])
    H, W = source.shape
    Y, X = np.mgrid[0:H, 0:W]
    sy, sx = Y + field[0], X + field[1]
    off_image = float(((sy < 0) | (sy > H - 1) | (sx < 0) | (sx > W - 1)).mean())
    meta = dict(
        source='learned',
        tool=tool,
        final_loss=float(losses[-1]) if losses else float('nan'),
        train_s=round(time.perf_counter() - t0, 1),
        warp_rmse=rmse,
        warp_rmse_swapped=rmse_swapped,
        # collapse detector: a network that shifts everything off-image samples only its
        # padding, gets a low MSE for free, and both RMSEs above read 0 — see transmorph
        off_image_frac=off_image,
        **params,
    )
    return phi, meta


def voxelmorph(
    seed=0,
    image_size=64,
    n_train=200,
    n_test_pairs=4,
    pair=0,
    nb_features=(16, 32, 32, 32, 16),
    integration_steps=0,
    epochs=200,
    steps_per_epoch=50,
    lr=1e-3,
    lambda_smooth=0.05,
    data=None,
    device=None,
):
    """VoxelMorph (``vxm.nn.models.VxmPairwise``, pytorch backend) as in the
    notebook; ``integration_steps=0`` direct, ``7`` diffeomorphic. ``data``:
    ``None`` = the notebook's synthetic images, or a callable such as
    :func:`cohort_data` (real brains)."""
    os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
    os.environ.setdefault('VXM_BACKEND', 'pytorch')
    import neurite as ne
    import torch
    import voxelmorph as vxm

    data_d = _resolve_data(data, image_size, n_train, n_test_pairs, pair, seed)
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=list(nb_features),
        integration_steps=integration_steps,
        device=device,
    ).to(device)
    image_loss, grad_loss = ne.nn.modules.MSE(), ne.nn.modules.SpatialGradient('l2')
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sample, (te_src, te_tgt) = _batches(data_d, device, torch)
    gen = torch.Generator().manual_seed(seed)
    kw = dict(return_warped_source=True, return_field_type='displacement')

    model.train()
    losses = []
    for _ in range(epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            src, tgt = sample(gen)
            opt.zero_grad()
            disp, warped = model(src, tgt, **kw)
            loss = image_loss(tgt, warped) + lambda_smooth * grad_loss(disp)
            loss.backward()
            opt.step()
            tot += loss.item()
        losses.append(tot / steps_per_epoch)

    model.eval()
    with torch.no_grad():
        disp, warped = model(te_src, te_tgt, **kw)  # disp (1, 2, H, W) = [dy, dx]
    disp = disp.cpu().numpy()[0].astype(np.float64)
    return _finish(
        f'VoxelMorph VxmPairwise ({"direct" if integration_steps == 0 else "diffeo"})',
        data_d['test'][0].astype(np.float64),
        disp,
        warped.cpu().numpy()[0, 0].astype(np.float64),
        losses,
        t0,
        pad='zeros',  # vxm.nn.functional.spatial_transform pads with zeros
        seed=seed,
        **data_d['info'],
        integration_steps=integration_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lambda_smooth=lambda_smooth,
        device=device,
    )


def transmorph(
    seed=0,
    image_size=64,
    n_train=200,
    n_test_pairs=4,
    pair=0,
    integration_steps=0,
    epochs=200,
    steps_per_epoch=50,
    lr=1e-3,
    lambda_smooth=0.05,
    feature_stage=0,
    data=None,
    device=None,
):
    """The notebook's ``SwinRegNet`` — a Swin-Tiny encoder (timm, 2-channel
    input, no pretraining) + small ConvNet decoder regressing a displacement;
    ``integration_steps=0`` direct, ``7`` scaling-and-squaring. ``data`` as in
    :func:`voxelmorph`.

    One deliberate deviation from the notebook: the decoder reads the encoder's
    ``feature_stage`` feature map (stage 0 = 16x16 tokens at 64 px) instead of
    the final 2x2 bottleneck. Measured on the notebook's design (10k steps, CPU):
    the bottleneck can only emit near-global fields and settles on a constant
    -58 px translation that shifts the whole source off-image — the border
    padding then returns black, the MSE equals mean(target²) ≈ 0.08, and the
    "field" is a fold-free translation (``off_image_frac`` 1.0). At 1000 steps
    stage 0 reaches loss 0.074 with 3 % off-image and a genuinely local field
    (446 folded cells) vs the bottleneck's 0.103 / 60 % off-image, and trains
    5.7x faster. ``feature_stage=None`` restores the notebook's bottleneck.
    """
    import timm
    import torch
    import torch.nn.functional as F
    from torch import nn

    data_d = _resolve_data(data, image_size, n_train, n_test_pairs, pair, seed)
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    class SwinRegNet(nn.Module):
        def __init__(self, img_size, steps, stage):
            super().__init__()
            self.steps = steps
            common = dict(pretrained=False, in_chans=2, img_size=img_size)
            if stage is None:  # the notebook's 2x2 bottleneck (collapses, see the docstring)
                self.encoder = timm.create_model(
                    'swin_tiny_patch4_window7_224', num_classes=0, global_pool='', **common
                )
                channels = self.encoder.num_features
            else:
                self.encoder = timm.create_model(
                    'swin_tiny_patch4_window7_224',
                    features_only=True,
                    out_indices=(stage,),
                    **common,
                )
                channels = self.encoder.feature_info.channels()[-1]
            with torch.no_grad():
                enc = self._features(torch.zeros(1, 2, *img_size))
            if enc.ndim != 4 or enc.shape[1] != channels:
                raise RuntimeError(f'unexpected swin feature layout {tuple(enc.shape)}')
            self.decoder = nn.Sequential(
                nn.Conv2d(channels, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(size=img_size, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 3, padding=1),
            )
            nn.init.normal_(self.decoder[-1].weight, std=1e-5)
            nn.init.zeros_(self.decoder[-1].bias)

        def _features(self, x):
            enc = self.encoder(x)
            enc = enc[-1] if isinstance(enc, (list, tuple)) else enc  # features_only -> list
            return enc.permute(0, 3, 1, 2)  # timm >= 0.9 swin is NHWC; the decoder wants NCHW

        def _warp(self, src, flow):
            # Pull-back sampling of src at pixel (i, j) + flow. With align_corners=True
            # pixel i sits exactly at -1 + 2i/(n-1), so d px == 2d/(n-1) normalized.
            # (The notebook paired a linspace(-1, 1, n) grid with align_corners=False,
            # whose pixel centers are elsewhere — a +-0.5 px identity stretch that the
            # convention self-check in meta exposed: warp_rmse 2.4e-2 vs 3e-7 here.)
            B, _, H, W = src.shape
            gy, gx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=src.device),
                torch.linspace(-1, 1, W, device=src.device),
                indexing='ij',
            )
            grid = torch.stack([gx, gy], -1)[None].expand(B, -1, -1, -1)
            flow_norm = torch.stack([flow[:, 1] * (2 / (W - 1)), flow[:, 0] * (2 / (H - 1))], -1)
            return F.grid_sample(
                src, grid + flow_norm, mode='bilinear', padding_mode='border', align_corners=True
            )

        def forward(self, source, target):
            disp = self.decoder(self._features(torch.cat([source, target], 1)))
            if self.steps > 0:
                disp = disp / 2**self.steps
                for _ in range(self.steps):
                    disp = disp + self._warp(disp, disp)
            return disp, self._warp(source, disp)

    hw = tuple(int(n) for n in data_d['test'][0].shape)
    model = SwinRegNet(hw, integration_steps, feature_stage).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sample, (te_src, te_tgt) = _batches(data_d, device, torch)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    losses = []
    for _ in range(epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            src, tgt = sample(gen)
            opt.zero_grad()
            disp, warped = model(src, tgt)
            smooth = (
                (disp[:, :, 1:] - disp[:, :, :-1]).abs().mean()
                + (disp[:, :, :, 1:] - disp[:, :, :, :-1]).abs().mean()
            ) / 2
            loss = F.mse_loss(warped, tgt) + lambda_smooth * smooth
            loss.backward()
            opt.step()
            tot += loss.item()
        losses.append(tot / steps_per_epoch)

    model.eval()
    with torch.no_grad():
        disp, warped = model(te_src, te_tgt)  # (1, 2, H, W) = [dy, dx]
    disp = disp.cpu().numpy()[0].astype(np.float64)
    kind = 'direct' if integration_steps == 0 else 'diffeo'
    feats = 'bottleneck' if feature_stage is None else f'stage-{feature_stage} features'
    return _finish(
        f'TransMorph-style SwinRegNet ({kind}, {feats})',
        data_d['test'][0].astype(np.float64),
        disp,
        warped.cpu().numpy()[0, 0].astype(np.float64),
        losses,
        t0,
        pad='border',  # grid_sample padding_mode='border' in _warp
        seed=seed,
        **data_d['info'],
        feature_stage=feature_stage,
        integration_steps=integration_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lambda_smooth=lambda_smooth,
        device=device,
    )
