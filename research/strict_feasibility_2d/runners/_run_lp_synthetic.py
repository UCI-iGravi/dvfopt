"""Batch comparison on every synthetic worst case.

Writes:
    runners/output/comparison_synthetic.csv
    runners/output/corrected/<case>_<method>.npz

Cluster_pipeline rows record an ``error`` field (adapter not implemented).
"""

from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

from research.strict_feasibility_2d.runners._compare import METHOD_NAMES, run_method
from research.strict_feasibility_2d.worst_cases._load import load_synthetic_canonical

OUTDIR = _HERE / 'output'
CORR_DIR = OUTDIR / 'corrected'

CSV_FIELDS = [
    'case_id',
    'method',
    'shape',
    'init_n_neg_2tri',
    'init_min_T',
    'final_n_neg_2tri',
    'final_min_T',
    'feasible',
    'L1_dev',
    'L2_dev',
    'Linf_dev',
    'wall_s',
    'error',
]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / 'comparison_synthetic.csv'

    cases = load_synthetic_canonical()
    print(f'Found {len(cases)} synthetic cases, {len(METHOD_NAMES)} methods.', flush=True)

    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for case_id, phi_in, meta in cases:
            print(
                f'\n=== {case_id}  shape={meta["shape"]}  init_n_neg={meta["init_n_neg"]} ===',
                flush=True,
            )
            for method in METHOD_NAMES:
                try:
                    rec = run_method(method, phi_in)
                except Exception:
                    print(
                        f'  {method:<18s} UNCAUGHT -- {traceback.format_exc(limit=2)}',
                        flush=True,
                    )
                    continue
                row = {k: rec[k] for k in CSV_FIELDS if k in rec}
                row['case_id'] = case_id
                row['shape'] = f'{meta["shape"][0]}x{meta["shape"][1]}'
                writer.writerow(row)
                fh.flush()
                flag = 'OK ' if rec['feasible'] else 'INF'
                err = f'   err={rec["error"]}' if rec['error'] else ''
                print(
                    f'  {method:<18s} {flag}  n_neg={rec["final_n_neg_2tri"]:3d}  '
                    f'min_T={rec["final_min_T"]:+.4f}  L1={rec["L1_dev"]:.3f}  '
                    f'({rec["wall_s"]:.2f}s){err}',
                    flush=True,
                )
                np.savez(
                    CORR_DIR / f'{case_id}_{method}.npz',
                    phi_out=rec['phi_out'].astype(np.float64),
                )

    print(f'\nWrote {out_csv}')


if __name__ == '__main__':
    main()
