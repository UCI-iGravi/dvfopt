"""Save the canonical 2-triangle 2D synthetic test cases to NPZ files.

Each case in :func:`dvfopt.testdata.canonical_2tri_2d` gets one NPZ at
``data/dvfs/canonical_2tri_2d/<key>.npz`` with these arrays:

* ``phi`` — ``(2, H, W)`` float64, channels ``[dy, dx]`` (the 2-tri pack
  convention used by every 2-triangle solver).
* ``msample`` — ``(N, 3)`` moving correspondences ``[z, y, x]``.
* ``fsample`` — ``(N, 3)`` fixed correspondences ``[z, y, x]``.
* ``init_n_neg`` — 0-d int, total negative-area triangles in the input.
* ``init_min_T`` — 0-d float64, minimum triangle area in the input.
* ``shape`` — ``(2,)`` int, the ``(H, W)`` grid shape.
* ``title`` — 0-d unicode string, human-readable case title.
* ``key`` — 0-d unicode string, the canonical case key.

Run from the repo root::

    python scripts/_save_canonical_2tri_2d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent

from dvfopt.testdata import canonical_2tri_2d

OUTDIR = _REPO / 'data' / 'dvfs' / 'canonical_2tri_2d'


def _bowtie_7x7_shoelace():
    """The 7×7 shoelace-artifact bowtie from
    ``notebooks/two-triangle-check/02_optimization.ipynb``.

    Pixels (3, 3) and (3, 4) swap in the dx channel — the two cells
    anchored at those positions get bowtie-shaped warped quads. The
    full grid has exactly **two folded 2-triangle cells** and **zero
    neg-Jdet pixels** (the central-difference stencil cancels the
    artifact across its 2Δ symmetry). The manuscript's smallest demo
    that 2-tri catches sub-pixel folds the Jdet-CD stencil misses.

    Returns ``(name, phi_2hw, meta)`` matching the
    :func:`dvfopt.testdata.canonical_2tri_2d` triple shape, with
    correspondence arrays empty (the field is hand-set, not built
    from correspondences).
    """
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

    H, W = 7, 7
    dy = np.zeros((H, W), dtype=np.float64)
    dx = np.zeros((H, W), dtype=np.float64)
    dx[3, 3] = +1.2
    dx[3, 4] = -1.2
    phi = np.stack([dy, dx])
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    min_T = float(min(T1.min(), T2.min()))
    meta = dict(
        shape=phi.shape[1:],
        title='Bowtie 7×7 — shoelace artifact (02_optimization.ipynb)',
        msample=np.empty((0, 3), dtype=np.float64),
        fsample=np.empty((0, 3), dtype=np.float64),
        init_n_neg=n_neg,
        init_min_T=min_T,
    )
    return 'bowtie_7x7_shoelace', phi, meta


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f'Saving canonical 2-tri 2D cases to {OUTDIR}')
    print()
    print(f'  {"key":<22s}  {"shape":>10s}  {"n_neg":>6s}  {"min_T":>9s}  file')
    print(f'  {"-" * 22}  {"-" * 10}  {"-" * 6}  {"-" * 9}  {"-" * 30}')
    # Add the hand-set 7×7 bowtie alongside the correspondence-derived
    # canonical suite — it's the manuscript's smallest illustration of
    # the "Jdet-CD misses sub-pixel folds that 2-tri catches" story.
    cases = list(canonical_2tri_2d()) + [_bowtie_7x7_shoelace()]
    for name, phi, meta in cases:
        path = OUTDIR / f'{name}.npz'
        np.savez(
            path,
            phi=phi.astype(np.float64),
            msample=meta['msample'].astype(np.float64),
            fsample=meta['fsample'].astype(np.float64),
            init_n_neg=np.int64(meta['init_n_neg']),
            init_min_T=np.float64(meta['init_min_T']),
            shape=np.asarray(meta['shape'], dtype=np.int64),
            title=np.asarray(meta['title']),
            key=np.asarray(name),
        )
        H, W = meta['shape']
        print(
            f'  {name:<22s}  {H:>4d}x{W:<4d}  {meta["init_n_neg"]:>6d}  '
            f'{meta["init_min_T"]:+9.4f}  {path.name}'
        )
    print()
    print(f'Wrote {len(list(OUTDIR.glob("*.npz")))} NPZ files.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
