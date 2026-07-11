"""Round-4 bench: 3D GPU-ALM untangle as a seed for correct_dvf_3d.

Question: does the 3D analogue of the 2D accuracy='max' GPU seed
(gpu_untangle_alm_2d) deliver a similar L1 win for the 3D 6-tet pipeline?

Protocol (per crop, real B0039 raw laplacian field, dense band z0-16):
  (a) baseline : correct_dvf_3d(crop)                 -> wall, L1, residual
  (b) seeded   : gpu_untangle_alm_3d(crop) [stage stats: folds before/
                 after, stall level, wall] then correct_dvf_3d(gpu_out)
                 -> total wall, L1 vs ORIGINAL crop, residual

Crops: three (3, 16, 128, 128) crops of graded fold density from the
z0-16 dense band (sparse ~1%, medium ~9%, dense ~31% of cubes folded).

Parity gate: runners/_verify_gpu_tet_parity.py must PASS first (it does:
max|torch - numpy| <= 2.3e-13 on random fields).

Usage:
  python _bench_gpu_untangle_3d.py [--crops sparse medium dense] [--arms a b]
  python _bench_gpu_untangle_3d.py --crops medium dense --arms gpu   # seed-stage-only probe
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import (
    n_neg_best_diagonal,
    six_tet_min_volume_3d,
)
from dvfopt.pipeline_3d import correct_dvf_3d
from research.strict_feasibility_3d.algorithms._gpu_untangle_3d import gpu_untangle_alm_3d

_DATA_CANDIDATES = [
    _REPO / 'data' / 'dvfs' / 'archive' / 'new_b0039_laplacian_deformation_field.npz',
    Path(
        r'C:\Users\Andy\Documents\GitHub\UCI-iGravi\deformation-field-processing'
        r'\data\dvfs\archive\new_b0039_laplacian_deformation_field.npz'
    ),
]

# (y0, x0) of each 128x128 in-plane crop; z = 0..15 (16 slices).
_CROPS = {
    'sparse': (0, 96),  # ~2.7k folded cubes (1.1%)
    'medium': (32, 128),  # ~22.7k folded cubes (9.4%)
    'dense': (96, 160),  # ~74.1k folded cubes (30.6%)
}


def _load_band():
    for p in _DATA_CANDIDATES:
        if p.exists():
            print(f'data: {p}', flush=True)
            return np.load(p)['arr'].astype(np.float64)[:, 0:16]
    raise FileNotFoundError('new_b0039_laplacian_deformation_field.npz not found')


def _stats(phi):
    mv = six_tet_min_volume_3d(phi)
    return dict(
        n_neg=int((mv <= 0).sum()),
        n_below=int((mv < 0.01 - 1e-5).sum()),
        min_T=float(mv.min()),
        n_cubes=int(mv.size),
    )


def _fmt_report(tag, rep, l1_vs_orig, wall):
    return (
        f'  {tag:<22s} feasible={rep.feasible!s:<5s} n_neg_out={rep.n_neg_out:6d}  '
        f'n_below={rep.n_below_out:6d}  min_T={rep.min_T_out:+.5f}  '
        f'L1_vs_orig={l1_vs_orig:>12.1f}  wall={wall:>8.1f}s  '
        f'bd_floor_out={rep.best_diag_floor_out}'
    )


def run_crop(name, crop, arms, gpu_kwargs):
    print(f'\n=== crop {name}: shape={crop.shape} ===', flush=True)
    st = _stats(crop)
    bd = n_neg_best_diagonal(crop, threshold=0.0)
    print(
        f'  input: n_neg={st["n_neg"]} ({st["n_neg"] / st["n_cubes"]:.1%})  '
        f'min_T={st["min_T"]:+.3f}  best_diag_floor={bd} ({bd / st["n_cubes"]:.1%})',
        flush=True,
    )
    results = {}

    if 'b' in arms or 'gpu' in arms:
        # ---- GPU-ALM seed stage (shared by arm b and the gpu-only probe) ----
        t0 = time.time()
        seeded = gpu_untangle_alm_3d(crop, verbose=1, **gpu_kwargs)
        gpu_wall = time.time() - t0
        st_g = _stats(seeded)
        l1_g = float(np.abs(seeded - crop).sum())
        print(
            f'  [gpu stage] n_neg {st["n_neg"]} -> {st_g["n_neg"]}  '
            f'min_T {st["min_T"]:+.3f} -> {st_g["min_T"]:+.5f}  '
            f'L1={l1_g:.1f}  wall={gpu_wall:.1f}s',
            flush=True,
        )
        # Post-seed pathology triage: does the seed pull the crop out of
        # the "feasible set ~empty" regime (best-diag floor) that the
        # baseline's escape-skip guard keys on?
        bd_g = n_neg_best_diagonal(seeded, threshold=0.0)
        print(
            f'  [gpu stage] best_diag_floor {bd} -> {bd_g} '
            f'({bd_g / st["n_cubes"]:.2%} of cubes)',
            flush=True,
        )
        results['gpu'] = dict(
            n_neg=st_g['n_neg'], min_T=st_g['min_T'], l1=l1_g, wall=gpu_wall, bd_floor=bd_g
        )

    if 'b' in arms:
        t0 = time.time()
        out_b, rep_b = correct_dvf_3d(seeded, verbose=1)
        b_wall = time.time() - t0
        l1_b = float(np.abs(out_b - crop).sum())
        print(_fmt_report('(b) gpu-seed+correct', rep_b, l1_b, gpu_wall + b_wall), flush=True)
        results['b'] = dict(
            rep=rep_b,
            l1=l1_b,
            wall=gpu_wall + b_wall,
            gpu_wall=gpu_wall,
            gpu_n_neg=st_g['n_neg'],
            gpu_min_T=st_g['min_T'],
            gpu_l1=l1_g,
        )

    if 'a' in arms:
        # ---- (a) baseline correct_dvf_3d on the raw crop ----
        t0 = time.time()
        out_a, rep_a = correct_dvf_3d(crop, verbose=1)
        a_wall = time.time() - t0
        l1_a = float(np.abs(out_a - crop).sum())
        print(_fmt_report('(a) correct_dvf_3d', rep_a, l1_a, a_wall), flush=True)
        results['a'] = dict(rep=rep_a, l1=l1_a, wall=a_wall)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', nargs='+', default=['sparse', 'medium', 'dense'])
    ap.add_argument('--arms', nargs='+', default=['b', 'a'])
    ap.add_argument('--n-outer', type=int, default=40)
    args = ap.parse_args()

    band = _load_band()
    all_results = {}
    for name in args.crops:
        y0, x0 = _CROPS[name]
        crop = band[:, :, y0 : y0 + 128, x0 : x0 + 128].copy()
        all_results[name] = run_crop(name, crop, args.arms, dict(n_outer=args.n_outer))

    print('\n=== SUMMARY ===', flush=True)
    for name, res in all_results.items():
        for arm in ('a', 'b'):
            if arm not in res:
                continue
            r = res[arm]
            rep = r['rep']
            print(
                f'{name:<7s} ({arm}) wall={r["wall"]:8.1f}s  L1={r["l1"]:12.1f}  '
                f'n_neg_out={rep.n_neg_out:6d}  n_below={rep.n_below_out:6d}  '
                f'feasible={rep.feasible}',
                flush=True,
            )


if __name__ == '__main__':
    main()
