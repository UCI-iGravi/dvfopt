"""Batch comparison on selected B0039 slices.

Default slice set: z=12 (manuscript-canonical hard case) + a handful of
others to probe scale. Empirical-worst slice discovery comes later.

Writes:
    runners/output/comparison_b0039.csv
    runners/output/corrected/<case>_<method>.npz

Cluster_pipeline rows record an ``error`` field (adapter not implemented).
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

from research.strict_feasibility_2d.runners._compare import METHOD_NAMES, run_method
from research.strict_feasibility_2d.runners._run_lp_synthetic import CSV_FIELDS
from research.strict_feasibility_2d.worst_cases._load import load_b0039_slice

OUTDIR = _HERE / 'output'
CORR_DIR = OUTDIR / 'corrected'

DEFAULT_SLICES = (12, 100, 200, 300, 400)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--slices',
        type=int,
        nargs='+',
        default=list(DEFAULT_SLICES),
        help='Z-slice indices to run.',
    )
    p.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=list(METHOD_NAMES),
        help='Subset of methods to run (default: all).',
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / 'comparison_b0039.csv'

    print(f'Slices: {args.slices}', flush=True)
    print(f'Methods: {args.methods}', flush=True)

    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for z in args.slices:
            try:
                case_id, phi_in, meta = load_b0039_slice(z)
            except (IndexError, FileNotFoundError) as exc:
                print(f'\n[skip] z={z}: {exc}', flush=True)
                continue
            print(
                f'\n=== {case_id}  shape={meta["shape"]}  init_n_neg={meta["init_n_neg"]} ===',
                flush=True,
            )
            for method in args.methods:
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
                    f'  {method:<18s} {flag}  n_neg={rec["final_n_neg_2tri"]:4d}  '
                    f'min_T={rec["final_min_T"]:+.4f}  L1={rec["L1_dev"]:.1f}  '
                    f'({rec["wall_s"]:.1f}s){err}',
                    flush=True,
                )
                np.savez(
                    CORR_DIR / f'{case_id}_{method}.npz',
                    phi_out=rec['phi_out'].astype(np.float64),
                )

    print(f'\nWrote {out_csv}')


if __name__ == '__main__':
    main()
