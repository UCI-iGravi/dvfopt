"""Back-fill ``l2_per_entry``, ``l2_per_pixel``, ``l1_per_entry``,
``l1_per_pixel`` and ``H``/``W`` columns on existing full-DVF CSVs.

For runs that lacked l1_delta in the CSV (manuscript_slsqp,
harmonic, svf_squaring etc.), L1 is recomputed from the saved
slice .npy + the original input. Output CSV is written in place
(overwriting the old one).

All full-DVF slices are (2, 320, 456) so n_entries = 291840,
n_pixels = 145920.
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

import harness as H

H_GRID, W_GRID = 320, 456
N_ENTRIES = 2 * H_GRID * W_GRID
N_PIXELS = H_GRID * W_GRID


def _maybe_compute_l1(row, method_dir, vol):
    """Return L1 (using csv field if present, else compute from saved .npy)."""
    l1 = row.get('l1_delta')
    if l1 not in ('', None):
        try:
            return float(l1)
        except ValueError:
            pass
    z = int(row['z'])
    npy = os.path.join(method_dir, f'slice_z{z:03d}.npy')
    if not os.path.isfile(npy):
        # Fall back to the manuscript output location.
        alt = os.path.abspath(os.path.join(
            _HERE, '..', '..', 'manuscript', 'output',
            '2d_real_full', 'slices', f'slice_z{z:03d}.npy'))
        if os.path.isfile(alt):
            npy = alt
        else:
            return None
    phi_in = np.stack([vol[1, z].copy(), vol[2, z].copy()])
    phi_out = np.load(npy)
    if phi_out.ndim == 4 and phi_out.shape[0] == 3:    # manuscript (3,1,H,W)
        phi_out = np.stack([phi_out[1, 0], phi_out[2, 0]])
    elif phi_out.ndim == 3 and phi_out.shape[0] == 3:
        phi_out = np.stack([phi_out[1], phi_out[2]])
    return float(np.abs(phi_out - phi_in).sum())


def update_csv(csv_path: str, method_subdir: str | None = None) -> None:
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print(f'{csv_path}: empty')
        return
    vol = None
    # When the method_subdir is given, resolve it relative to the CSV's
    # parent (so we work with results_full_dvf, results_b0039_*, etc).
    csv_root = os.path.dirname(os.path.abspath(csv_path))
    full_method_dir = (os.path.join(csv_root, method_subdir)
                       if method_subdir else None)
    if full_method_dir is not None:
        vol = np.load(H.DATA_PATH, mmap_mode='r')

    # Augment each row.
    out_fields = list(rows[0].keys())
    for new_col in ('l1_delta', 'l2_per_entry', 'l2_per_pixel',
                     'l1_per_entry', 'l1_per_pixel', 'H', 'W'):
        if new_col not in out_fields:
            out_fields.append(new_col)

    for r in rows:
        l2 = r.get('l2_delta')
        try:
            l2_val = float(l2) if l2 not in ('', None) else None
        except ValueError:
            l2_val = None
        if full_method_dir is not None:
            l1_val = _maybe_compute_l1(r, full_method_dir, vol)
        else:
            l1_val = (float(r['l1_delta'])
                      if r.get('l1_delta') not in ('', None) else None)
        r['l1_delta'] = l1_val if l1_val is not None else ''
        r['l2_per_entry'] = l2_val / np.sqrt(N_ENTRIES) if l2_val is not None else ''
        r['l2_per_pixel'] = l2_val / np.sqrt(N_PIXELS) if l2_val is not None else ''
        r['l1_per_entry'] = l1_val / N_ENTRIES if l1_val is not None else ''
        r['l1_per_pixel'] = l1_val / N_PIXELS if l1_val is not None else ''
        r['H'] = H_GRID
        r['W'] = W_GRID

    out_path = csv_path
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'updated {csv_path}  ({len(rows)} rows)')


def main():
    """Auto-detect every CSV under ``results_*`` (handles the original
    DVF's ``results_full_dvf`` AND the b0039 ones in
    ``results_b0039_composed`` / ``results_b0039_laplacian``).

    For each CSV, the per-method subdir is inferred from the file basename:
    ``{NAME}__full_dvf.csv`` -> sibling ``{NAME}/`` directory under the
    same parent. If the subdir holds per-slice .npys we use them to
    recompute L1; otherwise we fall back to the manuscript path.
    """
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='*', default=None,
                    help='result roots to scan; default = all results_* sibling dirs')
    args = ap.parse_args()
    if args.roots is None:
        # Auto-detect: any results_*  directory next to this script.
        args.roots = sorted(d for d in glob.glob(os.path.join(_HERE, 'results_*'))
                             if os.path.isdir(d))
    if not args.roots:
        print('no result roots found')
        return
    for root in args.roots:
        csvs = sorted(glob.glob(os.path.join(root, '*__full_dvf.csv')))
        if not csvs:
            print(f'(no CSVs in {root})')
            continue
        print(f'--- {os.path.relpath(root, _HERE)} ---')
        for csv_path in csvs:
            base = os.path.basename(csv_path)
            method_name = base.replace('__full_dvf.csv', '')
            subdir = os.path.join(root, method_name)
            method_subdir = method_name if os.path.isdir(subdir) else None
            update_csv(csv_path, method_subdir=method_subdir)


if __name__ == '__main__':
    main()
