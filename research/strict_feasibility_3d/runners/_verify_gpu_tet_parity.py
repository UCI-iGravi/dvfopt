"""Parity check: torch 6-tet volume kernel vs the canonical numpy kernel.

Gate for the 3D GPU-ALM untangler experiment (round 4). Mirrors the 2D
port's protocol: on random fields, the prototype torch kernel
(`algorithms/_gpu_untangle_3d._tet_volumes_torch`) must match
``dvfopt.jacobian.tetrahedron_sign.six_tet_volumes_3d`` to 1e-10 in
float64 — on CPU and (if available) CUDA — before any benchmarking.

Also cross-checks the packaged ``six_tet_volumes_3d_torch`` for reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from dvfopt.jacobian.tetrahedron_sign_torch import six_tet_volumes_3d_torch
from research.strict_feasibility_3d.algorithms._gpu_untangle_3d import _tet_volumes_torch


def check(phi_np, device):
    ref = six_tet_volumes_3d(phi_np)  # (6, D-1, H-1, W-1) numpy (numba)
    t = torch.tensor(phi_np, dtype=torch.float64, device=device)
    mine = _tet_volumes_torch(t[0], t[1], t[2], torch).cpu().numpy()
    pkg = six_tet_volumes_3d_torch(t).cpu().numpy()
    d_mine = float(np.abs(mine - ref).max())
    d_pkg = float(np.abs(pkg - ref).max())
    return d_mine, d_pkg


def main():
    rng = np.random.default_rng(0)
    ok = True
    for trial, (shape, scale) in enumerate(
        [((3, 9, 10, 11), 2.0), ((3, 5, 33, 17), 5.0), ((3, 16, 64, 64), 3.0)]
    ):
        phi = rng.normal(0.0, scale, size=shape).astype(np.float64)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            d_mine, d_pkg = check(phi, device)
            passed = d_mine < 1e-10
            ok &= passed
            print(
                f'trial {trial} shape={shape} scale={scale} dev={device}: '
                f'max|mine-numpy|={d_mine:.3e}  max|pkg-numpy|={d_pkg:.3e}  '
                f'{"PASS" if passed else "FAIL"}',
                flush=True,
            )
    # Identity field sanity: every tet volume must be exactly +1/6.
    phi_id = np.zeros((3, 4, 5, 6))
    t = torch.tensor(phi_id, dtype=torch.float64)
    v_id = _tet_volumes_torch(t[0], t[1], t[2], torch).numpy()
    d_id = float(np.abs(v_id - 1.0 / 6.0).max())
    ok &= d_id < 1e-12
    print(f'identity field: max|V - 1/6| = {d_id:.3e}  {"PASS" if d_id < 1e-12 else "FAIL"}')
    print('PARITY:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
