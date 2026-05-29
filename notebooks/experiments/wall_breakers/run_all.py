"""Benchmark every wall-breaker method on every test fixture.

Usage:
    python run_all.py --fixture crop --methods harmonic alm    # subset
    python run_all.py --fixture crop                            # all methods, crops only
    python run_all.py --fixture slice --z 12                    # full slice z=12
    python run_all.py                                            # everything

CSV + Markdown summaries land in ``results/``; per-result JSON +
corrected-slice ``.npy`` next to them.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

import harness as H

METHOD_MODULES = [
    'methods.m01_svf_projection',
    'methods.m02_harmonic_extension',
    'methods.m03_augmented_lagrangian',
    'methods.m04_paint_and_blend',
    'methods.m05_torch_full_grid',
    'methods.m06_quasi_conformal',
    'methods.m07_tv_regularized',
    'methods.m08_harmonic_seed_polish',
    'methods.m09_svf_polished',
    'methods.m10_harmonic_l2_polished',
]


def load_methods(filter_names: list[str] | None) -> list:
    mods = []
    for mod_name in METHOD_MODULES:
        m = importlib.import_module(mod_name)
        if filter_names and getattr(m, 'NAME', '') not in filter_names \
                and mod_name not in filter_names:
            continue
        mods.append(m)
    return mods


def build_fixtures(z_list, fixture_kinds, phi_full) -> list[tuple[str, int, np.ndarray]]:
    out = []
    for z in z_list:
        phi = H.get_slice(phi_full, z)
        for kind in fixture_kinds:
            if kind == 'crop':
                crop, _ = H.get_worst_component_crop(phi, pad=4)
                out.append(('crop', z, crop))
            elif kind == 'slice':
                out.append(('slice', z, phi))
    return out


def write_summary(rows: list[dict], out_dir: str):
    if not rows:
        return None, None
    keys = list({k for r in rows for k in r.keys()})
    # Schema is whatever harness.MethodResult.to_row produces -- keep this
    # list in sync with that for column ordering. The set-difference at
    # the end picks up any future fields without dropping them.
    keys_order = ['method', 'fixture', 'z', 'H', 'W',
                  'feasible_2tri',
                  'init_tri_neg', 'init_tri_min',
                  'final_tri_neg', 'final_tri_min',
                  'final_sho_neg', 'final_sho_min',
                  'final_jdet_neg', 'final_jdet_min',
                  'l2_delta', 'l1_delta',
                  'l2_per_entry', 'l2_per_pixel',
                  'l1_per_entry', 'l1_per_pixel',
                  'wall_s', 'error']
    keys = keys_order + sorted(k for k in keys if k not in keys_order)
    csv_path = os.path.join(out_dir, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = os.path.join(out_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Wall-breaker benchmark summary\n\n')
        # Per-fixture, per-z, per-method
        from collections import defaultdict
        by_fix_z = defaultdict(list)
        for r in rows:
            by_fix_z[(r['fixture'], r['z'])].append(r)
        for key in sorted(by_fix_z.keys()):
            fix, z = key
            f.write(f'## {fix} z={z}\n\n')
            f.write('| method | feasible | tri_neg | tri_min | jdet_min | L2_delta | wall (s) | error |\n')
            f.write('|---|---|---:|---:|---:|---:|---:|---|\n')
            for r in sorted(by_fix_z[key], key=lambda x: x['method']):
                tag = '✅' if r.get('feasible_2tri') else '❌'
                err = (r.get('error') or '')[:60]
                tmin = r.get('final_tri_min')
                jmin = r.get('final_jdet_min')
                tmin_s = f'{tmin:+.4f}' if tmin is not None else 'NA'
                jmin_s = f'{jmin:+.4f}' if jmin is not None else 'NA'
                f.write(
                    f"| {r['method']} | {tag} | "
                    f"{r.get('final_tri_neg', '')} | {tmin_s} | {jmin_s} | "
                    f"{r.get('l2_delta', 0):.2f} | "
                    f"{r.get('wall_s', 0):.1f} | {err} |\n")
            f.write('\n')
    return csv_path, md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z', type=int, nargs='*', default=list(H.WORST_SLICES))
    ap.add_argument('--fixture', nargs='*', default=['crop'],
                    choices=['crop', 'slice'])
    ap.add_argument('--methods', nargs='*', default=None,
                    help='subset of NAMEs; default all')
    ap.add_argument('--out', default=os.path.join(_HERE, 'results'))
    ap.add_argument('--save_npy', action='store_true',
                    help='save phi_out npy for every successful result')
    ap.add_argument('--time_budget_s', type=float, default=600.0)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    methods = load_methods(args.methods)
    print(f'methods: {[m.NAME for m in methods]}', flush=True)
    phi_full = H.load_volume()
    fixtures = build_fixtures(args.z, args.fixture, phi_full)
    print(f'fixtures: {len(fixtures)} '
          f'({len(args.z)} z-values, {len(args.fixture)} kinds)', flush=True)

    rows = []
    for fkind, z, phi_in in fixtures:
        for m in methods:
            print(f'\n>>> {m.NAME} on {fkind} z={z} shape={phi_in.shape}',
                  flush=True)
            kwargs = {}
            # Methods that accept a time budget honour it; others ignore.
            if 'time_budget_s' in getattr(m.solve, '__code__').co_varnames:
                kwargs['time_budget_s'] = args.time_budget_s
            if args.verbose and 'verbose' in m.solve.__code__.co_varnames:
                kwargs['verbose'] = 1
            res = H.run_method(m, phi_in, fixture=fkind, z=z, **kwargs)
            print(H.fmt_row(res), flush=True)
            H.save_result(res, args.out)
            if args.save_npy and not res.error and res.phi_out is not None:
                np.save(os.path.join(args.out,
                                     f'{res.method}__{fkind}__z{z:03d}.npy'),
                        res.phi_out)
            rows.append(res.to_row())

    csv_path, md_path = write_summary(rows, args.out)
    print(f'\nwrote {csv_path}\n      {md_path}', flush=True)


if __name__ == '__main__':
    main()
