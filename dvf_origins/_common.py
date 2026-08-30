"""Shared bits for the harness: the repo root and the 2D field packer."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repo root (the harness is not installed)
ORIGINS = ROOT / 'data' / 'origins'  # the generated tree: <mechanism>/<case>.npy, manifest, cache


def pack2d(dy, dx):
    """``(dy, dx)`` 2D arrays -> the ``(3, 1, H, W)`` float64 ``[0, dy, dx]`` field."""
    dy = np.asarray(dy, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    return np.stack([np.zeros_like(dy), dy, dx])[:, None]
