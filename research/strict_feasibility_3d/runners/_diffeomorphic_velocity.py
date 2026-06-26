"""Diffeomorphic time-velocity reparameterization.

Idea: parameterize phi as exp(v) where v is a velocity field and
exp is integrated via scaling-and-squaring. Any sufficiently
smooth v has det(d exp(v)/dx) > 0 EVERYWHERE — the result is
fold-free by construction.

Method:
  phi_0 = v / 2**N
  phi_{k+1} = phi_k composed with phi_k (interpolated)
  phi_N approximates exp(v).

Loss: ||phi_N - phi_target||_1 + lambda * ||grad v||_2^2

The smoothness regularizer keeps v small/smooth enough that
phi_N stays a diffeomorphism. We solve via Adam on v with
PyTorch autograd. If exp(v) is far from phi_target it means the
target is non-diffeomorphic; the optimum is the closest fold-
free phi.

GOAL: explore whether the closest diffeomorphism to B0039 is
within reasonable L1 of the input, even if not the global L1
optimum.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import torch.nn.functional as F

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compose_phi(phi, phi_inner):
    """phi_out[x] = x_disp + phi[x + phi_inner[x] - x] where the
    coordinates use (z, y, x) ordering.

    phi shape: (1, 3, D, H, W) with displacement channels [dz, dy, dx].
    Returns the composition phi(phi_inner(x)) as displacement.

    grid_sample expects normalized coords in [-1, 1] with order
    (x, y, z) for 3D.
    """
    _, _, D, H, W = phi.shape
    # Build base grid of positions (z, y, x).
    zs = torch.arange(D, device=phi.device, dtype=phi.dtype)
    ys = torch.arange(H, device=phi.device, dtype=phi.dtype)
    xs = torch.arange(W, device=phi.device, dtype=phi.dtype)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing='ij')
    # Target positions = base + phi_inner displacement.
    pz = gz + phi_inner[0, 0]
    py = gy + phi_inner[0, 1]
    px = gx + phi_inner[0, 2]
    # Normalize to [-1, 1] for grid_sample (which takes x, y, z order).
    nx = 2.0 * px / max(W - 1, 1) - 1.0
    ny = 2.0 * py / max(H - 1, 1) - 1.0
    nz = 2.0 * pz / max(D - 1, 1) - 1.0
    grid = torch.stack([nx, ny, nz], dim=-1).unsqueeze(0)
    # grid_sample on phi (treat each channel as a separate scalar field).
    # phi shape (1, 3, D, H, W); grid shape (1, D, H, W, 3).
    sampled = F.grid_sample(
        phi, grid, mode='bilinear', padding_mode='border',
        align_corners=True,
    )
    # phi(x + phi_inner(x)) = sampled. Result is the displacement of the
    # composed map.
    return sampled + phi_inner


def scaling_and_squaring(v, N=6):
    """Approximate exp(v) by phi_0 = v / 2**N then N compositions."""
    phi = v / (2.0 ** N)
    for _ in range(N):
        phi = compose_phi(phi, phi)
    return phi


def grad_loss(v):
    """||grad v||_2^2 across all 3 axes for all 3 channels."""
    dz = v[:, :, 1:, :, :] - v[:, :, :-1, :, :]
    dy = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]
    dx = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]
    return (dz ** 2).sum() + (dy ** 2).sum() + (dx ** 2).sum()


def main():
    phi_np = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float32)
    V0 = six_tet_volumes_3d(phi_np)
    print(
        f'Start: shape={phi_np.shape}  n_neg={int((V0 <= 0).sum())}  '
        f'n<0.01={int((V0 < THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    # Target field, (1, 3, D, H, W).
    target = torch.from_numpy(phi_np[None]).to(DEVICE)

    # Initialize velocity = target (identity init at N=0 squaring).
    # If we naively start with v=phi, exp(v) ~= phi only if phi is
    # smooth; for our folded input, the optimization will pull v
    # toward a smoother diffeomorphic approximation.
    v = target.clone().detach().requires_grad_(True)

    LAMBDA_GRAD = 1e-3
    LR = 0.05
    EPOCHS = 800
    N_SQUARE = 6

    opt = torch.optim.Adam([v], lr=LR)
    t0 = time.time()
    print(f'\n=== Diffeomorphic optimization: N_squaring={N_SQUARE}, '
          f'lr={LR}, lambda_grad={LAMBDA_GRAD}, epochs={EPOCHS} ===', flush=True)

    best_n_neg = int((V0 <= 0).sum())
    for epoch in range(EPOCHS):
        opt.zero_grad()
        phi_pred = scaling_and_squaring(v, N=N_SQUARE)
        loss_l1 = (phi_pred - target).abs().mean()
        loss_grad = grad_loss(v)
        loss = loss_l1 + LAMBDA_GRAD * loss_grad
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                phi_p = phi_pred[0].cpu().numpy()
                V = six_tet_volumes_3d(phi_p)
                n_neg = int((V <= 0).sum())
                n_below = int((V < THRESHOLD - 1e-5).sum())
                L1 = float(np.abs(phi_p - phi_np).sum())
                print(
                    f'  epoch {epoch:4d}: L1_loss={float(loss_l1):.5f}  '
                    f'grad_loss={float(loss_grad):.1f}  '
                    f'n_neg={n_neg}  n<0.01={n_below}  '
                    f'min_T={float(V.min()):+.4f}  L1_from_input={L1:.1f}',
                    flush=True,
                )
                if n_neg < best_n_neg:
                    best_n_neg = n_neg

    # Final evaluation.
    with torch.no_grad():
        phi_final = scaling_and_squaring(v, N=N_SQUARE)[0].cpu().numpy()
    V = six_tet_volumes_3d(phi_final)
    n_neg = int((V <= 0).sum())
    n_below = int((V < THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(phi_final - phi_np).sum())
    wall = time.time() - t0
    print(
        f'\n=== Final ===\n'
        f'  n_neg={n_neg}  n<0.01={n_below}\n'
        f'  min_T={float(V.min()):+.6f}\n'
        f'  L1 from input={L1:.1f}\n'
        f'  best n_neg seen during opt: {best_n_neg}\n'
        f'  wall={wall:.1f}s\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_diffeo.npy', phi_final)


if __name__ == '__main__':
    main()
