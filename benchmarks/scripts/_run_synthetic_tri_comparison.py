"""Triangle-target benchmark on the synthetic test cases in data/dvfs/testcases.

Compares ``iterative_2d_tri_barrier`` under two coverage schemes:

* ``standard`` — original simplex (2D)-per-cell TR-BL split. Vertices (0,0) and
  (H-1, W-1) are each in only ONE triangle constraint.
* ``full_coverage`` — same plus two corner-patch triangles using the
  TL-BR diagonal at cells (0,0) and (H-2, W-2). Every vertex is now in
  ≥2 triangles.

Reports per-case:
- init tri_neg / min (standard + full-coverage)
- wall_s, final tri_neg / min, L2 displacement
- per-corner final triangle area at the four corner cells, so the
  coverage-gap effect is visible.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dvfopt.core.barrier.tri2d import iterative_2d_tri_barrier
from dvfopt.jacobian.triangle_sign import (
    _corner_patch_areas_2d,
    _triangle_areas_2d,
)

SYNTHETIC = [
    '01a_10x10_crossing.npy',
    '01b_10x10_opposite.npy',
    '01c_20x40_edges.npy',
    '01d_20x40_crossing.npy',
    '01e_20x20_random_spirals.npy',
    '01f_20x20_random_seed_42.npy',
    '03a_10x10_opposite.npy',
    '03a_10x10_random_seed_42.npy',
    '03b_10x10_crossing.npy',
    '03c_20x20_opposite.npy',
    '03c_20x20_random_seed_42.npy',
    '03d_20x20_crossing.npy',
]


def _stats(phi2):
    T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
    patches = _corner_patch_areas_2d(phi2[0], phi2[1])
    return dict(
        tri_neg=int((T1 <= 0).sum() + (T2 <= 0).sum()),
        tri_min=float(min(T1.min(), T2.min())),
        patches_min=float(patches.min()),
        full_neg=int((T1 <= 0).sum() + (T2 <= 0).sum() + (patches <= 0).sum()),
    )


def _silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def run_one(name, full_coverage):
    path = os.path.join(_REPO_ROOT, 'data', 'dvfs', 'testcases', name)
    arr = np.load(path)
    phi_init = np.stack([arr[1, 0], arr[2, 0]])
    init = _stats(phi_init)

    t0 = time.perf_counter()
    try:
        phi_out = _silent(
            iterative_2d_tri_barrier,
            phi_init.copy(),
            threshold=0.01,
            max_minimize_iter=300,
            verbose=0,
            full_coverage=full_coverage,
        )
    except Exception as exc:
        return dict(
            case=name,
            scheme='full_coverage' if full_coverage else 'standard',
            error=f'{type(exc).__name__}: {exc}',
            wall_s=time.perf_counter() - t0,
        )
    wall = time.perf_counter() - t0
    final = _stats(phi_out)
    l2 = float(np.linalg.norm(phi_out - phi_init))
    return dict(
        case=name,
        scheme='full_coverage' if full_coverage else 'standard',
        wall_s=wall,
        init_tri_neg=init['tri_neg'],
        init_tri_min=init['tri_min'],
        init_patches_min=init['patches_min'],
        final_tri_neg=final['tri_neg'],
        final_tri_min=final['tri_min'],
        final_patches_min=final['patches_min'],
        final_full_neg=final['full_neg'],
        l2=l2,
        # Feasibility under each criterion
        feas_standard=(final['tri_neg'] == 0),
        feas_full=(final['full_neg'] == 0),
    )


def main():
    rows = []
    header_fmt = (
        "{case:<32} {scheme:<14} {wall_s:>7} "
        "{init_tri_neg:>6} {final_tri_neg:>7} {final_tri_min:>10} "
        "{init_patches_min:>10} {final_patches_min:>10} "
        "{l2:>7} {feas_std:>9} {feas_full:>9}"
    )
    print(
        header_fmt.format(
            case='case',
            scheme='scheme',
            wall_s='wall_s',
            init_tri_neg='init_n',
            final_tri_neg='final_n',
            final_tri_min='final_min',
            init_patches_min='init_p',
            final_patches_min='final_p',
            l2='L2',
            feas_std='feas_std',
            feas_full='feas_full',
        )
    )
    print('-' * 145)
    for name in SYNTHETIC:
        for full in (False, True):
            r = run_one(name, full)
            rows.append(r)
            if 'error' in r:
                print(f"{name:<32} {r['scheme']:<14} {r['wall_s']:>7.2f} ERROR: {r['error']}")
                continue
            print(
                header_fmt.format(
                    case=name,
                    scheme=r['scheme'],
                    wall_s=f"{r['wall_s']:.2f}",
                    init_tri_neg=r['init_tri_neg'],
                    final_tri_neg=r['final_tri_neg'],
                    final_tri_min=f"{r['final_tri_min']:+.4f}",
                    init_patches_min=f"{r['init_patches_min']:+.3f}",
                    final_patches_min=f"{r['final_patches_min']:+.4f}",
                    l2=f"{r['l2']:.2f}",
                    feas_std=str(r['feas_standard']),
                    feas_full=str(r['feas_full']),
                )
            )
        print()

    # Save CSV
    out_dir = os.path.join(_REPO_ROOT, 'benchmarks', 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'synthetic_tri_comparison.csv')
    cols = [
        'case',
        'scheme',
        'wall_s',
        'init_tri_neg',
        'init_tri_min',
        'init_patches_min',
        'final_tri_neg',
        'final_tri_min',
        'final_patches_min',
        'final_full_neg',
        'l2',
        'feas_standard',
        'feas_full',
    ]
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            if 'error' in r:
                continue
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f"CSV: {csv_path}")

    # Per-scheme aggregate
    print(
        f"\n{'scheme':<14} {'avg_wall_s':>11} {'feas_std_rate':>15} {'feas_full_rate':>16} {'avg_L2':>9}"
    )
    print('-' * 70)
    for scheme in ('standard', 'full_coverage'):
        s_rows = [r for r in rows if r.get('scheme') == scheme and 'error' not in r]
        if not s_rows:
            continue
        avg_wall = np.mean([r['wall_s'] for r in s_rows])
        avg_l2 = np.mean([r['l2'] for r in s_rows])
        fs = sum(1 for r in s_rows if r['feas_standard']) / len(s_rows)
        ff = sum(1 for r in s_rows if r['feas_full']) / len(s_rows)
        print(
            f"{scheme:<14} {avg_wall:>11.2f} {fs * 100:>14.0f}% {ff * 100:>15.0f}% {avg_l2:>9.2f}"
        )


if __name__ == '__main__':
    main()
