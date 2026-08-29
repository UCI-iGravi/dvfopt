"""Mechanism 3 for real: small learned registration networks, trained here.

The setups are the ``benchmarks/registration/{voxelmorph,transmorph}-registration``
notebooks' — synthetic ellipse images, 200 epochs x 50 steps, MSE + λ·smoothness
— but seeded, and returning the inference field on a held-out pair instead of
correcting it in place. ``integration_steps=0`` is a DIRECT displacement
regressor (the unregularized-network signature the paper's mechanism 3 is
about); ``7`` puts a scaling-and-squaring layer on top (a learned
diffeomorphism — folds, if any, are then mechanism 4's).

Needs torch (+ voxelmorph / timm), which the main venv does not carry::

    uv venv .venv-torch --python 3.12
    uv pip install --python .venv-torch/Scripts/python.exe --torch-backend=cpu \\
        -e . torch timm "voxelmorph @ git+https://github.com/voxelmorph/voxelmorph.git"
    .venv-torch/Scripts/python -m dvf_origins generate --mechanism 3

Without them the builders raise ``ModuleNotFoundError`` and ``generate`` skips
the rows. Every builder records ``warp_rmse`` — the RMSE between the network's
own warped source and a pull-back resampling of the source by the returned
field — next to ``warp_rmse_swapped`` (same with the channels swapped), so the
field convention (``[dy, dx]``, ``moving(x + u(x))``) is checked, not assumed.
"""

import time

import numpy as np
from scipy import ndimage

from dvf_origins._common import pack2d


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


def _pullback(img, dy, dx):
    H, W = img.shape
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    return ndimage.map_coordinates(img, [Y + dy, X + dx], order=1, mode='nearest')


def _convention_check(source, disp, warped):
    """RMSE of my pull-back warp vs the network's, for ``[dy, dx]`` and swapped."""
    ok = np.sqrt(((_pullback(source, disp[0], disp[1]) - warped) ** 2).mean())
    swapped = np.sqrt(((_pullback(source, disp[1], disp[0]) - warped) ** 2).mean())
    return float(ok), float(swapped)


def _finish(tool, source, disp, warped, losses, t0, **params):
    rmse, rmse_swapped = _convention_check(source, disp, warped)
    meta = dict(
        source='learned',
        tool=tool,
        final_loss=float(losses[-1]),
        train_s=round(time.perf_counter() - t0, 1),
        warp_rmse=rmse,
        warp_rmse_swapped=rmse_swapped,
        **params,
    )
    return pack2d(disp[0], disp[1]), meta


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
    device=None,
):
    """The notebook's ``SwinRegNet`` — a Swin-Tiny encoder (timm, 2-channel
    input, no pretraining) + small ConvNet decoder regressing a displacement;
    ``integration_steps=0`` direct, ``7`` scaling-and-squaring."""
    import timm
    import torch
    import torch.nn.functional as F
    from torch import nn

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train, test = _dataset(image_size, n_train, n_test_pairs, data_seed=42 + seed)

    class SwinRegNet(nn.Module):
        def __init__(self, img_size, steps):
            super().__init__()
            self.steps = steps
            self.encoder = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=False,
                in_chans=2,
                num_classes=0,
                global_pool='',
                img_size=img_size,
            )
            with torch.no_grad():
                enc = self.encoder(torch.zeros(1, 2, img_size, img_size))
            if enc.ndim == 3:  # (B, N, C) tokens
                self.enc_spatial = (int(enc.shape[1] ** 0.5),) * 2
                channels = enc.shape[2]
            elif enc.ndim == 4 and enc.shape[-1] != enc.shape[1]:  # (B, h, w, C) channels-last
                self.enc_spatial = None
                channels = enc.shape[-1]
            else:  # (B, C, h, w)
                self.enc_spatial = None
                channels = enc.shape[1]
            self.enc_channels_last = (
                enc.ndim == 4 and enc.shape[-1] == channels and enc.shape[1] != channels
            )
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
            enc = self.encoder(torch.cat([source, target], 1))
            if enc.ndim == 3:
                B, _, C = enc.shape
                enc = enc.permute(0, 2, 1).reshape(B, C, *self.enc_spatial)
            elif self.enc_channels_last:
                enc = enc.permute(0, 3, 1, 2)
            disp = self.decoder(enc)
            if self.steps > 0:
                disp = disp / 2**self.steps
                for _ in range(self.steps):
                    disp = disp + self._warp(disp, disp)
            return disp, self._warp(source, disp)

    model = SwinRegNet(image_size, integration_steps).to(device)
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
    return _finish(
        f'TransMorph-style SwinRegNet ({"direct" if integration_steps == 0 else "diffeo"})',
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
