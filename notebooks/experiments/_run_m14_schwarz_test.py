"""Benchmark the m14-Schwarz prototype vs global m14.

Five test cases:

1. Synthetic 30×30 with 3 separated fold clusters (sparse).
2. B0039 z=12 crop 30×30 / 379 folds (moderate-dense, single big cluster).
3. B0039 z=12 crop 30×30 / 1484 folds (near-saturated single cluster).
4. B0039 z=12 crop 60×60 around fold cluster (multi-cluster real data).
5. B0039 z=12 full slice 320×456 / 8978 folds (the wall test).

Per case we report: wall_s, n_neg, min_T, L1, L2, feasible, and whether
the prototype fell back to global m14.
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _m14_schwarz_proto import m14_schwarz, _stats
from dvfopt import iterative_2d_tri_refine_repair
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


THRESHOLD = 0.01


def _silent(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*a, **k)


def _plant_fold(arr, cy, cx, amp=0.8):
    arr[cy, cx] += amp
    arr[cy+1, cx] -= amp
    arr[cy, cx+1] -= amp
    arr[cy+1, cx+1] += amp


def synth_sparse(seed=0):
    np.random.seed(seed)
    H, W = 30, 30
    dy = np.random.normal(0, 0.05, (H, W))
    dx = np.random.normal(0, 0.05, (H, W))
    _plant_fold(dx, 5, 5)
    _plant_fold(dx, 5, 20)
    _plant_fold(dy, 22, 12)
    return np.stack([dy, dx])


def load_b0039(cy, cx, size):
    arr = np.load(os.path.join(_REPO_ROOT, 'data', 'dvfs', 'b0039',
                               'b0039_laplacian_deformation_field.npy'))
    dy = arr[1, 12, cy:cy+size, cx:cx+size].astype(np.float64).copy()
    dx = arr[2, 12, cy:cy+size, cx:cx+size].astype(np.float64).copy()
    return np.stack([dy, dx])


def load_b0039_full():
    arr = np.load(os.path.join(_REPO_ROOT, 'data', 'dvfs', 'b0039',
                               'b0039_laplacian_deformation_field.npy'))
    return np.stack([arr[1, 12].astype(np.float64).copy(),
                     arr[2, 12].astype(np.float64).copy()])


CASES = [
    ('synth_30x30_sparse_8', synth_sparse, dict()),
    ('z12_30x30_379',        lambda: load_b0039(120, 180, 30), dict()),
    ('z12_30x30_1484',       lambda: load_b0039(180, 180, 30), dict()),
    ('z12_60x60',            lambda: load_b0039(140, 160, 60), dict()),
    ('z12_full_320x456',     load_b0039_full, dict(time_budget_s=900.0)),
]


def run_one(method_label, fn, phi_in, **kwargs):
    t0 = time.time()
    err = ''
    extra = {}
    try:
        if method_label == 'm14_schwarz':
            phi_out, hist = fn(phi_in.copy(), threshold=THRESHOLD,
                                anchor='l1', verbose=0, record_history=True,
                                **kwargs)
            extra = dict(
                fallback=hist.get('fallback_to_global', False),
                n_clusters=len(hist.get('cluster_runs', [])),
                outer_rounds=len(hist.get('outer_rounds', [])),
            )
        else:
            phi_out = _silent(fn, phi_in.copy(),
                              threshold=THRESHOLD, anchor='l1', verbose=0,
                              **kwargs)
    except Exception as exc:
        return dict(method=method_label, wall_s=time.time()-t0,
                    error=f'{type(exc).__name__}: {exc}', feasible=False,
                    n_neg=-1, min_T=float('nan'), L1=float('nan'),
                    L2=float('nan'), **extra)
    wall = time.time() - t0
    n_neg, min_T = _stats(phi_out)
    diff = (phi_out - phi_in).ravel()
    return dict(
        method=method_label, wall_s=wall,
        n_neg=n_neg, min_T=min_T,
        L1=float(np.abs(diff).sum()),
        L2=float(np.sqrt(np.dot(diff, diff))),
        feasible=(n_neg == 0 and min_T >= THRESHOLD - 1e-5),
        error='', **extra,
    )


def main():
    print('Loading cases...', flush=True)
    cases = [(label, builder(), kw) for (label, builder, kw) in CASES]
    print(f'{"case":<22} {"shape":<10} {"init n_neg":>10} {"init min_T":>10}',
          flush=True)
    for label, phi, _ in cases:
        H, W = phi.shape[1], phi.shape[2]
        n_neg, min_T = _stats(phi)
        print(f'{label:<22} {H}x{W:<8} {n_neg:>10}  {min_T:>+10.3f}',
              flush=True)
    print(flush=True)

    rows = []
    for label, phi, kw in cases:
        print(f'\n=== {label} ===', flush=True)
        for method_label, fn in [
            ('m14_global',  iterative_2d_tri_refine_repair),
            ('m14_schwarz', m14_schwarz),
        ]:
            r = run_one(method_label, fn, phi, **kw)
            r['case'] = label
            rows.append(r)
            extras = ''
            if 'fallback' in r:
                extras = f"  fallback={r['fallback']}  clusters={r['n_clusters']}"
            tag = 'OK' if r['feasible'] else ('ERR' if r['error'] else 'FAIL')
            print(f'  [{tag:>4}] {method_label:<14}  wall={r["wall_s"]:>7.2f}s  '
                  f'n_neg={r["n_neg"]:>5}  min_T={r["min_T"]:+.4f}  '
                  f'L1={r["L1"]:>9.1f}  L2={r["L2"]:>8.2f}'
                  + extras
                  + (f'  err={r["error"]}' if r["error"] else ''),
                  flush=True)

    print('\n=== Summary: speedup (m14_global / m14_schwarz wall) ===',
          flush=True)
    print(f'{"case":<22} {"global wall":>11} {"schwarz wall":>13} '
          f'{"speedup":>8} {"global L1":>10} {"schwarz L1":>11} '
          f'{"L1 ratio":>9}', flush=True)
    for label, _, _ in cases:
        g = next((r for r in rows if r['case'] == label
                  and r['method'] == 'm14_global'), None)
        s = next((r for r in rows if r['case'] == label
                  and r['method'] == 'm14_schwarz'), None)
        if g is None or s is None:
            continue
        sp = g['wall_s'] / s['wall_s'] if s['wall_s'] > 0 else float('inf')
        l1_ratio = s['L1'] / g['L1'] if g['L1'] > 0 else float('nan')
        print(f'{label:<22} {g["wall_s"]:>10.2f}s {s["wall_s"]:>12.2f}s '
              f'{sp:>7.2f}x {g["L1"]:>10.1f} {s["L1"]:>11.1f} '
              f'{l1_ratio:>8.2f}x', flush=True)

    # CSV.
    out_csv = os.path.join(_REPO_ROOT, 'benchmarks', 'results',
                            'm14_schwarz_prototype.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ['case', 'method', 'wall_s', 'n_neg', 'min_T', 'L1', 'L2',
            'feasible', 'fallback', 'n_clusters', 'error']
    with open(out_csv, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            vals = [str(r.get(c, '')) for c in cols]
            f.write(','.join(vals) + '\n')
    print(f'\nCSV: {out_csv}', flush=True)


if __name__ == '__main__':
    main()
