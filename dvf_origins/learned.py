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

import time

import numpy as np

from dvf_origins._common import pack2d
from dvf_origins.synthetic import _warp_image


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


def _check_pair(pair, n_test_pairs):
    if not 0 <= pair < n_test_pairs:
        raise ValueError(f'pair={pair} out of range for n_test_pairs={n_test_pairs}')


def _finish(tool, source, disp, warped, losses, t0, **params):
    phi = pack2d(disp[0], disp[1])
    field = phi[1:, 0]  # the RETURNED channels, so a swap in the packing would show here
    rmse = float(np.sqrt(((_warp_image(source, field) - warped) ** 2).mean()))
    rmse_swapped = float(np.sqrt(((_warp_image(source, field[::-1]) - warped) ** 2).mean()))
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
    device=None,
):
    """VoxelMorph (``vxm.nn.models.VxmPairwise``, pytorch backend) as in the
    notebook; ``integration_steps=0`` direct, ``7`` diffeomorphic."""
    import os

    os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
    os.environ.setdefault('VXM_BACKEND', 'pytorch')
    import neurite as ne
    import torch
    import voxelmorph as vxm

    _check_pair(pair, n_test_pairs)
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train, test = _dataset(image_size, n_train, n_test_pairs, data_seed=42 + seed)
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
    tr = torch.from_numpy(train).float().unsqueeze(1).to(device)
    gen = torch.Generator().manual_seed(seed)
    kw = dict(return_warped_source=True, return_field_type='displacement')

    model.train()
    losses = []
    for _ in range(epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, len(tr), (2,), generator=gen)
            src, tgt = tr[idx[0]][None], tr[idx[1]][None]
            opt.zero_grad()
            disp, warped = model(src, tgt, **kw)
            loss = image_loss(tgt, warped) + lambda_smooth * grad_loss(disp)
            loss.backward()
            opt.step()
            tot += loss.item()
        losses.append(tot / steps_per_epoch)

    model.eval()
    te = torch.from_numpy(test).float().unsqueeze(1).to(device)
    with torch.no_grad():
        src, tgt = te[2 * pair][None], te[2 * pair + 1][None]
        disp, warped = model(src, tgt, **kw)  # disp (1, 2, H, W) = [dy, dx]
    disp = disp.cpu().numpy()[0].astype(np.float64)
    return _finish(
        f'VoxelMorph VxmPairwise ({"direct" if integration_steps == 0 else "diffeo"})',
        test[2 * pair].astype(np.float64),
        disp,
        warped.cpu().numpy()[0, 0].astype(np.float64),
        losses,
        t0,
        seed=seed,
        image_size=image_size,
        pair=pair,
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

    _check_pair(pair, n_test_pairs)
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train, test = _dataset(image_size, n_train, n_test_pairs, data_seed=42 + seed)

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
                enc = self._features(torch.zeros(1, 2, img_size, img_size))
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

    model = SwinRegNet(image_size, integration_steps, feature_stage).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = torch.from_numpy(train).float().unsqueeze(1).to(device)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    losses = []
    for _ in range(epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, len(tr), (2,), generator=gen)
            src, tgt = tr[idx[0]][None], tr[idx[1]][None]
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
    te = torch.from_numpy(test).float().unsqueeze(1).to(device)
    with torch.no_grad():
        src, tgt = te[2 * pair][None], te[2 * pair + 1][None]
        disp, warped = model(src, tgt)  # (1, 2, H, W) = [dy, dx]
    disp = disp.cpu().numpy()[0].astype(np.float64)
    kind = 'direct' if integration_steps == 0 else 'diffeo'
    feats = 'bottleneck' if feature_stage is None else f'stage-{feature_stage} features'
    return _finish(
        f'TransMorph-style SwinRegNet ({kind}, {feats})',
        test[2 * pair].astype(np.float64),
        disp,
        warped.cpu().numpy()[0, 0].astype(np.float64),
        losses,
        t0,
        seed=seed,
        image_size=image_size,
        pair=pair,
        feature_stage=feature_stage,
        integration_steps=integration_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lambda_smooth=lambda_smooth,
        device=device,
    )
