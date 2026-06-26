"""Benchmark the shipped auto_slp (continuous scheduler default) on EVERY
B0039 2D slice — time + results (feasibility + L1) per slice.

Uses the real production path `run_method('auto_slp', ...)`, which now
routes large slices through cluster_slp_iter(scheduler='continuous').
Each slice is solved sequentially (clean per-slice wall timing; the slice
uses its own internal 16-worker pool).

RESUMABLE: results are appended to a CSV one row per slice (flushed), which
doubles as the checkpoint — re-running skips slices already in the CSV.
GUARDED for Windows spawn.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

_FIELDS = [
    'z',
    'init_n_neg',
    'final_n_neg',
    'feasible',
    'L1_dev',
    'final_min_T',
    'wall_s',
    'dispatch',
]


def main():
    from research.strict_feasibility_2d.runners._compare import run_method

    OUT = Path(__file__).parent / 'output'
    OUT.mkdir(exist_ok=True)
    CSV = OUT / 'all_slices_auto_slp_continuous.csv'
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    D = raw.shape[1]

    done = set()
    if CSV.exists():
        with open(CSV, newline='') as f:
            for row in csv.DictReader(f):
                done.add(int(row['z']))
        print(f'RESUME: {len(done)}/{D} slices already done', flush=True)
    else:
        with open(CSV, 'w', newline='') as f:
            csv.writer(f).writerow(_FIELDS)

    t_run = time.time()
    n_done_now = 0
    for z in range(D):
        if z in done:
            continue
        sl = raw[1:3, z].astype(np.float64)
        rec = run_method('auto_slp', sl)
        row = [
            z,
            rec.get('init_n_neg_2tri'),
            rec.get('final_n_neg_2tri'),
            int(bool(rec.get('feasible'))),
            f"{rec.get('L1_dev', float('nan')):.3f}",
            f"{rec.get('final_min_T', float('nan')):.6f}",
            f"{rec.get('wall_s', float('nan')):.2f}",
            rec.get('auto_dispatch', rec.get('error', '?')),
        ]
        with open(CSV, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        n_done_now += 1
        print(
            f'[z={z:3d}] init_n_neg={row[1]:>5} -> {row[2]:>3} '
            f'feasible={row[3]} L1={row[4]:>10} wall={row[6]:>7}s '
            f'(elapsed {(time.time() - t_run) / 3600:.2f}h, {n_done_now} this run)',
            flush=True,
        )

    # ---- Summary over the full CSV ----
    rows = []
    with open(CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    walls = np.array([float(r['wall_s']) for r in rows])
    l1s = np.array([float(r['L1_dev']) for r in rows])
    feas = np.array([int(r['feasible']) for r in rows])
    fin = np.array([int(r['final_n_neg']) for r in rows])
    order = np.argsort(walls)[::-1][:10]
    print(f'\n===== SUMMARY: {len(rows)} slices =====', flush=True)
    print(f'feasible (n_neg=0): {int(feas.sum())}/{len(rows)}', flush=True)
    print(f'total final folds:  {int(fin.sum())}', flush=True)
    print(
        f'total wall:         {walls.sum() / 3600:.2f}h  '
        f'(mean {walls.mean():.1f}s, median {np.median(walls):.1f}s, '
        f'max {walls.max():.1f}s)',
        flush=True,
    )
    print(f'total L1:           {l1s.sum():.0f}  (mean {l1s.mean():.1f})', flush=True)
    print(
        'slowest 10 slices (z: wall s):',
        [(int(rows[i]['z']), round(float(rows[i]['wall_s']), 1)) for i in order],
        flush=True,
    )


if __name__ == '__main__':
    main()
