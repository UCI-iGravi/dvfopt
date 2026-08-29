"""Mechanism 3 for real: small learned registration networks, trained here.

The setups are the ``benchmarks/registration/{voxelmorph,transmorph}-registration``
notebooks' — synthetic ellipse images, 200 epochs x 50 steps, MSE + λ·smoothness
— but seeded, and returning the inference field on a held-out pair instead of
correcting it in place. ``integration_steps=0`` is a DIRECT displacement
regressor (the unregularized-network signature the paper's mechanism 3 is
about); ``7`` puts a scaling-and-squaring layer on top (a learned
diffeomorphism — folds, if any, are then mechanism 4's).

Needs torch (+ voxelmorph / timm), which the main venv does not carry — the
separate-venv recipe is in ``dvf_origins/README.md``. Without them the builders
raise ``ModuleNotFoundError`` and ``generate`` skips the rows. Every builder
records ``warp_rmse`` — the RMSE between the network's own warped source and a
pull-back resampling of the source by the RETURNED field — next to
``warp_rmse_swapped`` (same with the channels swapped), so the field convention
(``[dy, dx]``, ``moving(x + u(x))``) is checked on what is shipped, not assumed.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from dvf_origins._common import ROOT, pack2d
from dvf_origins.synthetic import _warp_image

# RegTools pipeline outputs (external, not in this repo): per brain the axis-aligned
# volume, the ANTs affine (fwd_transforms/ants_affine_1.mat) and the SyN result.
REGTOOLS_COHORT = Path(
    os.environ.get(
        'DVF_ORIGINS_REGTOOLS',
        'C:/Users/Andy/Documents/GitHub/UCI-XuLab/UCI-XuLab-RegTools/output/brain25_cohort',
    )
)
TEMPLATE = ROOT / 'data' / 'mouse_brain' / 'average_template_25.nii.gz'
COHORT_BRAINS = ('B0032', 'B0039', 'B0049', 'B0053', 'B0200', 'B0213', 'B0304')
CACHE = ROOT / 'data' / 'origins' / 'cache'


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
    """Block-mean downsample, percentile-normalise to [0, 1], centre-crop to ``crop``."""
    from skimage.transform import downscale_local_mean

    a = downscale_local_mean(a.astype(np.float64), (downsample, downsample))
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    H, W = a.shape
    h, w = crop
    y0, x0 = (H - h) // 2, (W - w) // 2
    return a[y0 : y0 + h, x0 : x0 + w].astype(np.float32)


def _cohort_planes(vol_arr, zs, downsample, crop):
    """Coronal planes ``i = z`` of a SimpleITK ``(k, j, i)`` array as ``(H, W) = (j, k)``."""
    return np.stack([_prep_plane(vol_arr[:, :, z].T, downsample, crop) for z in zs])


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
    deformed nonlinearly), coronal planes ``i = z``, paired with the template's
    plane at the same ``z``. Every brain except ``test_brain`` trains; the test
    pair is ``test_brain`` at ``test_z`` (default: the plane the real m1 / m4
    rows use). Planes are block-mean downsampled by ``downsample`` and centre-
    cropped to ``crop`` (multiples of 32: the VoxelMorph UNet has five levels;
    3 / 96x128 keeps ~85 % of the 320x456 field of view). Cached under
    ``data/origins/cache/`` (gitignored). Raises ``FileNotFoundError`` when the
    RegTools outputs (external) or the template are absent.
    """
    tag = f'{test_brain}_z{test_z}_ds{downsample}_{crop[0]}x{crop[1]}_n{len(zs)}'
    cache_file = CACHE / f'cohort_slices_{tag}.npz'
    if cache and cache_file.is_file():
        z = np.load(cache_file)
        info = json.loads(str(z['info']))
        return dict(
            train=(z['train_src'], z['train_tgt']),
            test=(z['test_src'], z['test_tgt']),
            paired=True,
            info=info,
        )
    import SimpleITK as sitk

    if not TEMPLATE.is_file():
        raise FileNotFoundError(f'template not found (data is gitignored): {TEMPLATE}')
    if not (REGTOOLS_COHORT / test_brain).is_dir():
        raise FileNotFoundError(
            f'RegTools cohort outputs not found: {REGTOOLS_COHORT} (set DVF_ORIGINS_REGTOOLS)'
        )
    tpl = sitk.ReadImage(str(TEMPLATE))
    tpl_arr = sitk.GetArrayFromImage(tpl)
    train_brains = [b for b in COHORT_BRAINS if b != test_brain]

    def aligned(brain):
        d = REGTOOLS_COHORT / brain
        mov = sitk.ReadImage(str(d / '01_axis_alignment' / 'axisAlignedData.nii.gz'))
        aff = sitk.ReadTransform(
            str(d / '02_nonlinear' / 'parameters' / 'fwd_transforms' / 'ants_affine_1.mat')
        )
        return sitk.GetArrayFromImage(
            sitk.Resample(mov, tpl, aff, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        )

    tgt = _cohort_planes(tpl_arr, zs, downsample, crop)
    train_src = np.concatenate(
        [_cohort_planes(aligned(b), zs, downsample, crop) for b in train_brains]
    )
    train_tgt = np.concatenate([tgt] * len(train_brains))
    test_src = _cohort_planes(aligned(test_brain), [test_z], downsample, crop)[0]
    test_tgt = _cohort_planes(tpl_arr, [test_z], downsample, crop)[0]
    info = dict(
        data='cohort',
        train_brains=train_brains,
        test_brain=test_brain,
        test_z=test_z,
        n_train=len(train_src),
        zs=list(zs),
        downsample=downsample,
        crop=list(crop),
    )
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            train_src=train_src,
            train_tgt=train_tgt,
            test_src=test_src,
            test_tgt=test_tgt,
            info=json.dumps(info),
        )
    return dict(train=(train_src, train_tgt), test=(test_src, test_tgt), paired=True, info=info)


def _resolve_data(data, image_size, n_train, n_test_pairs, pair, seed):
    if data is None:
        return _synthetic_data(image_size, n_train, n_test_pairs, pair, data_seed=42 + seed)
    if data == 'cohort':
        return cohort_data()
    return data


def _batches(data_d, device, torch):
    """Training tensors and a pair sampler honouring ``paired``, plus the test pair."""
    A, B = (
        torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(1).to(device)
        for x in data_d['train']
    )

    def sample(gen):
        idx = torch.randint(0, len(A), (2,), generator=gen)
        i, j = idx[0], (idx[0] if data_d['paired'] else idx[1])
        return A[i][None], B[j][None]

    test = tuple(
        torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(device)
        for x in data_d['test']
    )
    return sample, test


def _finish(tool, source, disp, warped, losses, t0, **params):
    phi = pack2d(disp[0], disp[1])
    field = phi[1:, 0]  # the RETURNED channels, so a swap in the packing would show here
    H, W = source.shape
    Y, X = np.mgrid[0:H, 0:W]
    sy, sx = Y + field[0], X + field[1]
    inside = (sy >= 0) & (sy <= H - 1) & (sx >= 0) & (sx <= W - 1)
    off_image = float(1.0 - inside.mean())

    def _rmse(f):  # over in-image samples only: the networks pad off-image samples
        d = _warp_image(source, f) - warped  # differently (zeros vs border), which is
        return float(np.sqrt((d[inside] ** 2).mean())) if inside.any() else float('nan')

    rmse, rmse_swapped = _rmse(field), _rmse(field[::-1])  # not a convention question
    meta = dict(
        source='learned',
        tool=tool,
        final_loss=float(losses[-1]) if losses else float('nan'),
        train_s=round(time.perf_counter() - t0, 1),
        warp_rmse=rmse,
        warp_rmse_swapped=rmse_swapped,
        # collapse detector: a network that shifts everything off-image samples only the
        # (black) border, gets a low MSE for free, and both RMSEs above read 0 — see transmorph
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
    ``None`` = the notebook's synthetic images, ``'cohort'`` = :func:`cohort_data`
    (real brains), or a dict of that shape."""
    import os

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
    ``integration_steps=0`` direct, ``7`` scaling-and-squaring.

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
            # (The notebook pairs a linspace(-1, 1, n) grid with align_corners=False,
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
        seed=seed,
        **data_d['info'],
        feature_stage=feature_stage,
        integration_steps=integration_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lambda_smooth=lambda_smooth,
        device=device,
    )
