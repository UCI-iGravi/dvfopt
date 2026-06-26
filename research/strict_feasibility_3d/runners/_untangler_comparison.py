"""Head-to-head comparison of fold-elimination methods on one common
cropped sub-volume of the B0039 dense band.

Methods compared (all on the IDENTICAL crop, identical free boundary,
so the relative comparison is fair):
  1. Garanzha chi_eps regularized-barrier untangler (conventional).
  2. TLC-style lifted-content untangler (conventional).
  3. M10Tet = HarmonicALMBarrier3D (our barrier/ALM bulk solver).
  4. Coupled k-ring SLSQP (cluster) + local M10Tet recovery (our escape).
  Plus: per-cell best-diagonal feasibility (free re-triangulation).

This answers the literature workflow's adversarial prediction: that
conventional global node-movement untanglers converge to the SAME
shared-corner local minimum our stack reaches. Crop chosen to contain
all dense-band folds with a generous padding ring so the fold region is
interior.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import (
    n_neg_best_diagonal,
    six_tet_volumes_3d,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01

# Crop: full z, padded y/x window around the fold cluster (y136-221, x191-283).
Z0, Z1 = 0, 16
Y0, Y1 = 116, 241
X0, X1 = 171, 303


def stats(phi, label, phi0=None):
    V = six_tet_volumes_3d(phi)
    n_neg = int((V <= 0).sum())
    n_below = int((V < THRESHOLD - 1e-5).sum())
    mn = float(V.min())
    bd = n_neg_best_diagonal(phi, threshold=0.0)
    l1 = '' if phi0 is None else f'  L1={float(np.abs(phi - phi0).sum()):.1f}'
    print(f'  {label}: n_neg={n_neg}  n<0.01={n_below}  best_diag_n_neg={bd}  '
          f'min_T={mn:+.6f}{l1}', flush=True)
    return dict(n_neg=n_neg, n_below=n_below, best_diag_n_neg=bd, min_T=mn)


def main():
    phi_full = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    crop0 = phi_full[:, Z0:Z1, Y0:Y1, X0:X1].copy()
    print(f'Crop shape: {crop0.shape} (from full {phi_full.shape})', flush=True)
    base = stats(crop0, 'CROP input')
    results = {'input': base}

    # ---- 1. Garanzha chi_eps ----
    print('\n=== Garanzha chi_eps untangler ===', flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'gz', str(_HERE / '_garanzha_untangle.py'))
    gz = importlib.util.module_from_spec(spec); spec.loader.exec_module(gz)
    t0 = time.time()
    out_gz = gz.run_garanzha(
        crop0, lam=1e-4, target=THRESHOLD,
        eps_schedule=(2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01), maxiter=200,
        verbose=0)
    w = time.time() - t0
    r = stats(out_gz, f'Garanzha (wall={w:.1f}s)', crop0); r['wall'] = w
    results['garanzha'] = r
    np.save(OUTPUT / 'cmp_garanzha.npy', out_gz)

    # ---- 2. TLC ----
    print('\n=== TLC lifted-content untangler ===', flush=True)
    spec = importlib.util.spec_from_file_location(
        'tlc', str(_HERE / '_tlc_untangle.py'))
    tlc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tlc)
    t0 = time.time()
    out_tlc = tlc.run_tlc(
        crop0, lam=1e-4, target=THRESHOLD,
        eps_schedule=(2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01), maxiter=200,
        verbose=0)
    w = time.time() - t0
    r = stats(out_tlc, f'TLC (wall={w:.1f}s)', crop0); r['wall'] = w
    results['tlc'] = r
    np.save(OUTPUT / 'cmp_tlc.npy', out_tlc)

    # ---- 3. M10Tet (our barrier/ALM) ----
    print('\n=== M10Tet (HarmonicALMBarrier3D) @ 0.012 ===', flush=True)
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )
    t0 = time.time()
    out_m10 = Solver(
        constraint=Tet6Constraint3D(shape=crop0.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.012,
    ).fit(crop0).corrected
    w = time.time() - t0
    r = stats(out_m10, f'M10Tet (wall={w:.1f}s)', crop0); r['wall'] = w
    results['m10tet'] = r
    np.save(OUTPUT / 'cmp_m10tet.npy', out_m10)

    # ---- 4. Coupled k-ring (cluster) + local recovery (our escape) ----
    print('\n=== Coupled k-ring (cluster, recover=True) ===', flush=True)
    from dvfopt import CoupledKRing3DStrategy
    t0 = time.time()
    out_ck = Solver(
        constraint=Tet6Constraint3D(shape=crop0.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=CoupledKRing3DStrategy(
            k_ring=2, feasibility_thr=1e-3, mode='cluster',
            n_workers=1, recover=True),
        threshold=0.01,
    ).fit(crop0).corrected
    w = time.time() - t0
    r = stats(out_ck, f'CoupledKRing+recover (wall={w:.1f}s)', crop0); r['wall'] = w
    results['coupled_kring'] = r
    np.save(OUTPUT / 'cmp_coupled_kring.npy', out_ck)

    # ---- 5. M10Tet then coupled k-ring (the full pipeline) ----
    print('\n=== M10Tet -> Coupled k-ring (full pipeline) ===', flush=True)
    t0 = time.time()
    mid = out_m10  # reuse the M10Tet result
    out_pipe = Solver(
        constraint=Tet6Constraint3D(shape=crop0.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=CoupledKRing3DStrategy(
            k_ring=2, feasibility_thr=1e-3, mode='cluster',
            n_workers=1, recover=True),
        threshold=0.01,
    ).fit(mid).corrected
    w = time.time() - t0
    r = stats(out_pipe, f'M10Tet+CoupledKRing (wall={w:.1f}s, +M10Tet above)', crop0)
    r['wall'] = w
    results['pipeline'] = r
    np.save(OUTPUT / 'cmp_pipeline.npy', out_pipe)

    # ---- Summary ----
    print('\n' + '=' * 78, flush=True)
    print('SUMMARY (crop ' + str(crop0.shape[1:]) + ')', flush=True)
    print('=' * 78, flush=True)
    print(f'{"method":<26} {"n_neg":>6} {"n<0.01":>8} {"bestdiag_neg":>13} '
          f'{"min_T":>11} {"wall(s)":>9}', flush=True)
    print('-' * 78, flush=True)
    order = ['input', 'garanzha', 'tlc', 'm10tet', 'coupled_kring', 'pipeline']
    for k in order:
        r = results.get(k)
        if not r:
            continue
        print(f'{k:<26} {r["n_neg"]:>6} {r["n_below"]:>8} '
              f'{r["best_diag_n_neg"]:>13} {r["min_T"]:>+11.5f} '
              f'{r.get("wall", 0):>9.1f}', flush=True)


if __name__ == '__main__':
    main()
