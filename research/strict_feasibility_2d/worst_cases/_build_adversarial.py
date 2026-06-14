"""Generate hand-designed adversarial synthetic cases.

Run from the repo root:

    python research/strict_feasibility_2d/worst_cases/_build_adversarial.py

Outputs:

* ``synthetic/dense_bowtie_cluster_15x15.npz`` -- 15x15 grid with a
  3x3 cluster of bowtie pairs in the centre.
* ``synthetic/tiny_margin_10x10.npz`` -- 10x10 grid where many cells
  are folded but only by a small margin. Stresses linearisation since
  LP steps must be small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

OUTDIR = Path(__file__).parent / 'synthetic'


def _save_npz(name: str, phi_2hw: np.ndarray, title: str):
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    n_neg = int((np.minimum(T1, T2) <= 0).sum())
    min_T = float(np.minimum(T1, T2).min())
    path = OUTDIR / f'{name}.npz'
    np.savez(
        path,
        phi=phi_2hw.astype(np.float64),
        msample=np.empty((0, 3), dtype=np.float64),
        fsample=np.empty((0, 3), dtype=np.float64),
        init_n_neg=np.int64(n_neg),
        init_min_T=np.float64(min_T),
        shape=np.asarray(phi_2hw.shape[1:], dtype=np.int64),
        title=np.asarray(title),
        key=np.asarray(name),
    )
    print(f'  {name:<32s} {phi_2hw.shape}  n_neg={n_neg:3d}  min_T={min_T:+.4f}')


def build_dense_bowtie_cluster_15x15() -> np.ndarray:
    H, W = 15, 15
    phi = np.zeros((2, H, W), dtype=np.float64)
    # A 3-row band of alternating bowtie crossings in the middle.
    for r in (6, 7, 8):
        for c in (5, 7, 9):
            phi[1, r, c] = +1.2  # dx
            phi[1, r, c + 1] = -1.2
    return phi


def build_tiny_margin_10x10() -> np.ndarray:
    H, W = 10, 10
    phi = np.zeros((2, H, W), dtype=np.float64)
    # Row-alternating dy shift: even rows up by +0.55, odd rows down by
    # -0.55. Every interior cell straddles two such rows, so its top
    # corners sit at y=r+0.55 and bottom corners at y=(r+1)-0.55, leaving
    # min(T1, T2) approx -0.05 -- "just barely infeasible" stresses the
    # linearisation since LP steps must be small to stay feasible.
    phi[0, 0::2, :] = +0.55
    phi[0, 1::2, :] = -0.55
    return phi


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f'Writing adversarial cases to {OUTDIR}')
    _save_npz(
        'dense_bowtie_cluster_15x15',
        build_dense_bowtie_cluster_15x15(),
        'Dense bowtie cluster (15x15)',
    )
    _save_npz(
        'tiny_margin_10x10',
        build_tiny_margin_10x10(),
        'Tiny-margin alternating shear (10x10)',
    )


if __name__ == '__main__':
    main()
