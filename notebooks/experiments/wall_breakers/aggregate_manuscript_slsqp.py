"""Aggregate the manuscript SLSQP cluster-based run (output/2d_real_full/
slices/slice_zNNN.npy) into a per-slice CSV with the same schema as
``run_full_dvf.py``'s output, so it can be compared head-to-head with
the wall-breaker methods.

The manuscript pipeline:

* SLSQP per connected fold component (windowed, frozen-edge ring) --
  see notebooks/manuscript/_run_2d_clusters.py.
* anchor: L2 by default; L1 polish on residuals; pad-boost on stall.
* threshold = 0.01 with margin 1e-3.

We never re-run this solver here; we just measure what it produced
slice by slice. Wall-time is NOT recovered from this aggregation
(the run was distributed) so the wall_s column is left blank for
this baseline.
"""
import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

import numpy as np
import harness as H

MANUSCRIPT_DIR = os.path.abspath(os.path.join(
    _HERE, '..', '..', 'manuscript', 'output', '2d_real_full', 'slices'))
METHOD_NAME = 'manuscript_slsqp'


def main():
    out_dir = os.path.join(_HERE, 'results_full_dvf', METHOD_NAME)
    os.makedirs(out_dir, exist_ok=True)
    phi_full = np.load(H.DATA_PATH, mmap_mode='r')
    n_z = phi_full.shape[1]

    rows = []
    found = 0
    for z in range(n_z):
        fn = os.path.join(MANUSCRIPT_DIR, f'slice_z{z:03d}.npy')
        phi_in = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
        init = H.metrics(phi_in)
        if not os.path.isfile(fn):
            rows.append({
                'z': z, 'method': METHOD_NAME,
                'init_tri_neg': init['tri_neg'],
                'init_tri_min': init['tri_min'],
                'final_tri_neg': None, 'final_tri_min': None,
                'final_sho_neg': None, 'final_sho_min': None,
                'final_jdet_neg': None, 'final_jdet_min': None,
                'l2_delta': None, 'feasible_2tri': False,
                'wall_s': None, 'error': 'manuscript-output-missing',
            })
            continue
        out = np.load(fn)
        if out.ndim == 4:                     # (3, 1, H, W)
            phi_out = np.stack([out[1, 0], out[2, 0]])
        elif out.ndim == 3 and out.shape[0] == 2:
            phi_out = out
        else:
            phi_out = np.stack([out[1], out[2]])
        final = H.metrics(phi_out)
        l2 = H.l2_delta(phi_out, phi_in)
        feas = H.is_2tri_feasible(final, 0.01)
        rows.append({
            'z': z, 'method': METHOD_NAME,
            'init_tri_neg': init['tri_neg'], 'init_tri_min': init['tri_min'],
            'final_tri_neg': final['tri_neg'],
            'final_tri_min': final['tri_min'],
            'final_sho_neg': final['sho_neg'],
            'final_sho_min': final['sho_min'],
            'final_jdet_neg': final['jdet_neg'],
            'final_jdet_min': final['jdet_min'],
            'l2_delta': l2,
            'feasible_2tri': bool(feas),
            'wall_s': None, 'error': None,
        })
        found += 1

    keys = ['z', 'method', 'feasible_2tri',
            'init_tri_neg', 'init_tri_min',
            'final_tri_neg', 'final_tri_min',
            'final_sho_neg', 'final_sho_min',
            'final_jdet_neg', 'final_jdet_min',
            'l2_delta', 'wall_s', 'error']
    csv_path = os.path.join(_HERE, 'results_full_dvf',
                             f'{METHOD_NAME}__full_dvf.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {csv_path}')
    feas = sum(1 for r in rows if r.get('feasible_2tri'))
    miss = sum(1 for r in rows if r.get('error') == 'manuscript-output-missing')
    print(f'aggregated {found}/{n_z} manuscript outputs  '
          f'feasible {feas}/{n_z}  missing {miss}')


if __name__ == '__main__':
    main()
