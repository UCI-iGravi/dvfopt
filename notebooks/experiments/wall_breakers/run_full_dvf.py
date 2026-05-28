"""Run one wall-breaker method on every z-slice of the DVF, in parallel.

Usage:
    python run_full_dvf.py --method harmonic_l2_polished
    python run_full_dvf.py --method svf_squaring --workers 8

Per-slice JSONs land in ``results_full_dvf/{method}/``; the corrected
slice is saved as ``slice_z{Z:03d}.npy`` next to its JSON. The script
prints a running progress line every N completions and writes a final
per-method CSV ``{method}__full_dvf.csv``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

# Disable CUDA in workers BEFORE torch is imported anywhere -- ProcessPoolExecutor
# on Windows spawns fresh interpreters that inherit this env. The wall-breaker
# methods that don't need GPU (m01-m04, m07-m10's main path) get a much lighter
# torch import, avoiding the 'paging file too small' DLL load failures we hit
# with 8 workers when each tried to load cufft64_11.dll.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

import harness as H


def _solve_one(args):
    """Run one method on one (z, slice). Returns metrics dict."""
    (z, mod_name, method_name, threshold, time_budget_s, out_dir,
     data_path) = args
    # Belt and braces: also set in the child in case parent's env wasn't inherited.
    import os as _os
    _os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    _os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))
    sys.path.insert(0, _HERE)
    import harness as H
    import importlib
    import numpy as np

    phi_in = H.load_slice_2d(data_path, z)
    init = H.metrics(phi_in)
    mod = importlib.import_module(mod_name)
    kwargs = {}
    if 'time_budget_s' in mod.solve.__code__.co_varnames:
        kwargs['time_budget_s'] = time_budget_s

    t0 = time.time()
    try:
        out = mod.solve(phi_in, threshold=threshold, **kwargs)
        phi_out = np.asarray(out['phi_out'])
        info = out.get('info', {}) or {}
        err = None
    except Exception as exc:
        return {
            'z': int(z), 'method': method_name,
            'init_tri_neg': init['tri_neg'], 'init_tri_min': init['tri_min'],
            'wall_s': time.time() - t0, 'error': f'{type(exc).__name__}: {exc}',
            'final_tri_neg': None, 'final_tri_min': None,
            'l2_delta': None, 'l1_delta': None,
            'l2_per_entry': None, 'l2_per_pixel': None,
            'l1_per_entry': None, 'l1_per_pixel': None,
            'H': phi_in.shape[1], 'W': phi_in.shape[2],
            'feasible_2tri': False, 'final_jdet_neg': None,
            'final_jdet_min': None, 'final_sho_neg': None, 'final_sho_min': None,
        }

    wall = time.time() - t0
    final = H.metrics(phi_out)
    l2 = H.l2_delta(phi_out, phi_in)
    l1 = H.l1_delta(phi_out, phi_in)
    Hh, Ww = phi_in.shape[1], phi_in.shape[2]
    n_entries = 2 * Hh * Ww          # dy AND dx scalars
    n_pixels = Hh * Ww               # corners (= dy,dx pairs)
    l2_per_entry = l2 / np.sqrt(n_entries)
    l2_per_pixel = l2 / np.sqrt(n_pixels)
    l1_per_entry = l1 / n_entries
    l1_per_pixel = l1 / n_pixels
    feas = H.is_2tri_feasible(final, threshold)

    # Save the corrected slice (compressed).
    np.save(os.path.join(out_dir, f'slice_z{z:03d}.npy'), phi_out)

    result = {
        'z': int(z), 'method': method_name,
        'init_tri_neg': init['tri_neg'], 'init_tri_min': init['tri_min'],
        'final_tri_neg': final['tri_neg'],
        'final_tri_min': final['tri_min'],
        'final_sho_neg': final['sho_neg'],
        'final_sho_min': final['sho_min'],
        'final_jdet_neg': final['jdet_neg'],
        'final_jdet_min': final['jdet_min'],
        'l2_delta': l2, 'l1_delta': l1,
        'l2_per_entry': l2_per_entry, 'l2_per_pixel': l2_per_pixel,
        'l1_per_entry': l1_per_entry, 'l1_per_pixel': l1_per_pixel,
        'H': Hh, 'W': Ww,
        'feasible_2tri': bool(feas),
        'wall_s': wall, 'error': None,
    }
    # Drop a small JSON per slice (no transient phi_out).
    with open(os.path.join(out_dir, f'slice_z{z:03d}.json'), 'w') as f:
        json.dump({**result, 'info_keys': list(info.keys())}, f, indent=2,
                  default=str)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True,
                    help='NAME of the method (e.g. harmonic_l2_polished)')
    ap.add_argument('--module', default=None,
                    help='module path; defaults to looking up by NAME')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--threshold', type=float, default=0.01)
    ap.add_argument('--time_budget_s', type=float, default=600.0)
    ap.add_argument('--z', type=int, nargs='*', default=None,
                    help='subset of z to run (default: all 528)')
    ap.add_argument('--out_root', default=os.path.join(_HERE,
                                                        'results_full_dvf'))
    ap.add_argument('--data_path', default=None,
                    help='override harness.DATA_PATH; .npy or .npz '
                         '(uses key "arr" or first array)')
    ap.add_argument('--progress_every', type=int, default=10)
    args = ap.parse_args()
    if args.data_path is None:
        args.data_path = H.DATA_PATH

    # Resolve module path from NAME if not given.
    if args.module is None:
        candidates = [
            'methods.m01_svf_projection', 'methods.m02_harmonic_extension',
            'methods.m03_augmented_lagrangian', 'methods.m04_paint_and_blend',
            'methods.m05_torch_full_grid', 'methods.m06_quasi_conformal',
            'methods.m07_tv_regularized', 'methods.m08_harmonic_seed_polish',
            'methods.m09_svf_polished', 'methods.m10_harmonic_l2_polished',
            'methods.m11_lbfgs_barrier',
            'methods.m12_l2_refine',
            'methods.m13_line_search',
            'methods.m14_l2_refine_repair',
            'methods.m14_l1',
            'methods.m_slsqp',
        ]
        for c in candidates:
            try:
                mod = importlib.import_module(c)
                if getattr(mod, 'NAME', '') == args.method:
                    args.module = c
                    break
            except Exception:
                continue
        if args.module is None:
            raise SystemExit(f'could not find module for method NAME={args.method!r}')

    out_dir = os.path.join(args.out_root, args.method)
    os.makedirs(out_dir, exist_ok=True)

    # z-list: default all D slices.
    if args.z is None:
        # Peek the volume just for its z-dimension (NPZ requires opening).
        if args.data_path.endswith('.npz'):
            with np.load(args.data_path) as z_file:
                key = 'arr' if 'arr' in z_file.files else z_file.files[0]
                D = int(z_file[key].shape[1])
        else:
            phi_full = np.load(args.data_path, mmap_mode='r')
            D = int(phi_full.shape[1])
        z_list = list(range(D))
    else:
        z_list = list(args.z)

    print(f'method={args.method}  module={args.module}  '
          f'workers={args.workers}  slices={len(z_list)}  '
          f'time_budget={args.time_budget_s}s  '
          f'data={os.path.basename(args.data_path)}', flush=True)

    base_work = [(z, args.module, args.method, args.threshold,
                  args.time_budget_s, out_dir, args.data_path) for z in z_list]

    # Bug #1: resilient pool. If a worker dies (OOM, segfault, BrokenProcessPool)
    # we rebuild the pool and resubmit only the slices that haven't completed.
    # Bug #2: per-future hard deadline. We wait up to (2 * time_budget_s + 60)
    # for each completion; if a future stays unfinished past that, we treat it
    # as timed-out and move on (the worker itself is orphaned but the pool
    # context manager exits and reclaims it).
    MAX_POOL_RETRIES = 4
    HARD_DEADLINE_S = 2.0 * args.time_budget_s + 60.0

    rows = []
    remaining_z = set(z[0] for z in base_work)
    t0 = time.time()
    feas_count = 0
    err_count = 0

    def _record(r):
        nonlocal feas_count, err_count
        rows.append(r)
        if r.get('feasible_2tri'):
            feas_count += 1
        if r.get('error'):
            err_count += 1
        remaining_z.discard(r['z'])

    def _progress(i, total, last):
        if (i + 1) % args.progress_every != 0 and (i + 1) != total:
            return
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-3)
        eta = (total - (i + 1)) / max(rate, 1e-6)
        feas_str = "FEAS" if last.get("feasible_2tri") else "fail"
        print(f'  [{i+1:4d}/{total}]  '
              f'feas={feas_count}  err={err_count}  '
              f'last z={last["z"]:3d} {feas_str} '
              f'tri_min={last.get("final_tri_min")}  '
              f'L2={last.get("l2_delta")}  '
              f'wall={last.get("wall_s", 0):.1f}s  '
              f'(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)', flush=True)

    total = len(base_work)
    for retry in range(MAX_POOL_RETRIES + 1):
        if not remaining_z:
            break
        cur_work = [w for w in base_work if w[0] in remaining_z]
        if retry > 0:
            print(f'[pool retry {retry}/{MAX_POOL_RETRIES}] '
                  f'{len(cur_work)} slices left after pool/worker failure',
                  flush=True)
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                fut_to_z = {pool.submit(_solve_one, w): w[0]
                             for w in cur_work}
                for fut in as_completed(fut_to_z, timeout=None):
                    z_done = fut_to_z[fut]
                    try:
                        r = fut.result(timeout=HARD_DEADLINE_S)
                    except BrokenProcessPool:
                        # The whole pool died -- break out and rebuild.
                        print(f'  pool died on z={z_done}; will rebuild',
                              flush=True)
                        raise
                    except Exception as exc:
                        # Worker raised (or hit our hard deadline).
                        r = {
                            'z': int(z_done), 'method': args.method,
                            'wall_s': time.time() - t0,
                            'error': f'{type(exc).__name__}: {exc}',
                            'final_tri_neg': None, 'final_tri_min': None,
                            'l2_delta': None, 'l1_delta': None,
                            'l2_per_entry': None, 'l2_per_pixel': None,
                            'l1_per_entry': None, 'l1_per_pixel': None,
                            'H': None, 'W': None,
                            'feasible_2tri': False,
                            'final_jdet_neg': None, 'final_jdet_min': None,
                            'final_sho_neg': None, 'final_sho_min': None,
                            'init_tri_neg': None, 'init_tri_min': None,
                        }
                    _record(r)
                    done_count = total - len(remaining_z)
                    _progress(done_count - 1, total, r)
        except BrokenProcessPool:
            # Loop back to retry with a fresh pool.
            continue
        except Exception as exc:
            print(f'[pool] unexpected outer exception: {type(exc).__name__}: {exc}',
                  flush=True)
            continue

    if remaining_z:
        # Record placeholder errors for slices that never completed.
        for z in sorted(remaining_z):
            _record({
                'z': int(z), 'method': args.method,
                'wall_s': 0.0,
                'error': 'pool_retries_exhausted',
                'final_tri_neg': None, 'final_tri_min': None,
                'l2_delta': None, 'l1_delta': None,
                'l2_per_entry': None, 'l2_per_pixel': None,
                'l1_per_entry': None, 'l1_per_pixel': None,
                'H': None, 'W': None, 'feasible_2tri': False,
                'final_jdet_neg': None, 'final_jdet_min': None,
                'final_sho_neg': None, 'final_sho_min': None,
                'init_tri_neg': None, 'init_tri_min': None,
            })

    # Write per-method CSV.
    import csv
    csv_path = os.path.join(args.out_root, f'{args.method}__full_dvf.csv')
    rows.sort(key=lambda r: r['z'])
    keys = ['z', 'method', 'feasible_2tri',
            'init_tri_neg', 'init_tri_min',
            'final_tri_neg', 'final_tri_min',
            'final_sho_neg', 'final_sho_min',
            'final_jdet_neg', 'final_jdet_min',
            'l2_delta', 'l1_delta',
            'l2_per_entry', 'l2_per_pixel',
            'l1_per_entry', 'l1_per_pixel',
            'H', 'W',
            'wall_s', 'error']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nwrote {csv_path}', flush=True)
    print(f'feasible: {feas_count}/{len(rows)} ({100*feas_count/len(rows):.1f}%)  '
          f'errors: {err_count}', flush=True)


if __name__ == '__main__':
    main()
