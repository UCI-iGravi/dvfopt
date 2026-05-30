"""Summarise the :mod:`b0039_l1_comparison` benchmark CSV into
manuscript-grade tables.

Produces three artefacts under ``benchmarks/output/b0039_l1_comparison/``:

1. ``summary_by_method.csv`` — one row per ``(constraint, method_id)``
   pair, aggregated across slices: feasibility rate, mean / median
   L1 / L2 / wall_time, and a brief failure mode count.

2. ``per_slice_pivot.csv`` — wide table with rows = slice_z, columns =
   ``(constraint, method_id)``, values = L1. Easy to drop into a paper
   as a per-slice comparison.

3. ``best_per_slice.csv`` — for each slice, which method achieved the
   lowest L1 *among those that reached feasibility*. The win-count
   column at the bottom gives the headline "method X wins on N/22
   slices" number.

The intent is to make the comparison directly drop-into a paper:
``pd.read_csv`` -> rounded numbers -> figure / latex table.

Usage::

    python benchmarks/b0039_l1_summary.py
    python benchmarks/b0039_l1_summary.py --csv path/to/results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / 'benchmarks' / 'output' / 'b0039_l1_comparison' / 'results.csv'


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Coerce numeric columns; CSV writer may have written blanks for nan.
    num_cols = [
        'l1_total',
        'l1_mean',
        'l2_total',
        'linf',
        'final_n_neg',
        'final_min_T',
        'final_n_neg_jdet',
        'final_min_T_jdet',
        'final_n_neg_2tri',
        'final_min_T_2tri',
        'wall_time_s',
        'n_phases',
        'total_iter',
        'cluster_count',
        'init_n_neg_jdet',
        'init_n_neg_2tri',
        'init_min_T_jdet',
        'init_min_T_2tri',
        'threshold',
        'eps_l1',
        'time_budget_s',
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['feasible'] = df['feasible'].astype(str).str.lower().isin(('true', '1', 'yes'))
    df['has_error'] = df['error_kind'].fillna('').astype(str).str.len() > 0
    return df


def summary_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (constraint, method_id, strategy)."""
    groups = df.groupby(['constraint', 'method_id', 'strategy'], dropna=False)
    rows = []
    for (constraint, method_id, strategy), g in groups:
        n_total = len(g)
        n_err = int(g['has_error'].sum())
        n_ok = n_total - n_err
        n_feas = int((g['feasible'] & ~g['has_error']).sum())
        feas_runs = g[g['feasible'] & ~g['has_error']]
        # Failure-mode summary
        err_kinds = g[g['has_error']]['error_kind'].value_counts().to_dict()
        err_kinds_str = '; '.join(f'{k}:{v}' for k, v in err_kinds.items())

        # Residual-fold averages across all non-error runs — manuscript
        # needs these to call out the "Jdet-CD missed sub-pixel folds"
        # cases (where final_n_neg_jdet == 0 but final_n_neg_2tri > 0).
        ok = g[~g['has_error']]
        rows.append(
            dict(
                constraint=constraint,
                method_id=method_id,
                strategy=strategy,
                n_runs=n_total,
                n_errors=n_err,
                n_feasible=n_feas,
                feasibility_rate=round(n_feas / n_total, 3) if n_total else float('nan'),
                # L1 across feasible runs (the primary metric).
                L1_mean=round(feas_runs['l1_total'].mean(), 2) if len(feas_runs) else float('nan'),
                L1_median=round(feas_runs['l1_total'].median(), 2) if len(feas_runs) else float('nan'),
                L1_std=round(feas_runs['l1_total'].std(), 2) if len(feas_runs) else float('nan'),
                # Wall time across all *non-error* runs (feasible or not).
                wall_mean_s=round(ok['wall_time_s'].mean(), 2) if len(ok) else float('nan'),
                wall_median_s=round(ok['wall_time_s'].median(), 2) if len(ok) else float('nan'),
                # L2 + Linf as secondary metrics.
                L2_mean=round(feas_runs['l2_total'].mean(), 2) if len(feas_runs) else float('nan'),
                Linf_mean=round(feas_runs['linf'].mean(), 4) if len(feas_runs) else float('nan'),
                # Residual-fold stats under BOTH views, averaged over OK
                # runs (a method that "solves" under Jdet may still leave
                # folded triangles, and vice versa).
                final_n_neg_jdet_mean=round(ok['final_n_neg_jdet'].mean(), 1) if len(ok) else float('nan'),
                final_n_neg_2tri_mean=round(ok['final_n_neg_2tri'].mean(), 1) if len(ok) else float('nan'),
                final_min_T_jdet_mean=round(ok['final_min_T_jdet'].mean(), 4) if len(ok) else float('nan'),
                final_min_T_2tri_mean=round(ok['final_min_T_2tri'].mean(), 4) if len(ok) else float('nan'),
                error_modes=err_kinds_str,
            )
        )
    return pd.DataFrame(rows).sort_values(['constraint', 'method_id']).reset_index(drop=True)


def per_slice_pivot(df: pd.DataFrame, value: str = 'l1_total') -> pd.DataFrame:
    """Wide table: rows=case_id, cols=(constraint, method_id), values=L1.

    Only feasible runs populate values; everything else is NaN."""
    feas = df[df['feasible'] & ~df['has_error']].copy()
    feas['col'] = feas['constraint'] + ':' + feas['method_id']
    pivot = feas.pivot_table(index='case_id', columns='col', values=value, aggfunc='min')
    return pivot.round(2)


def best_per_slice(df: pd.DataFrame) -> pd.DataFrame:
    """For each case: which method achieved the lowest L1 among
    feasible runs? Includes the n_neg-2tri init density for context."""
    feas = df[df['feasible'] & ~df['has_error']].copy()
    if feas.empty:
        return pd.DataFrame()
    rows = []
    for case_id, g in feas.groupby('case_id'):
        idx = g['l1_total'].idxmin()
        best = g.loc[idx]
        init = df[df['case_id'] == case_id].iloc[0]
        rows.append(
            dict(
                case_id=case_id,
                dataset=init.get('dataset', ''),
                init_n_neg_2tri=int(init['init_n_neg_2tri']),
                init_n_neg_jdet=int(init['init_n_neg_jdet']),
                best_method=f'{best["constraint"]}:{best["method_id"]}',
                best_L1=round(float(best['l1_total']), 2),
                best_wall_s=round(float(best['wall_time_s']), 2),
                best_min_T=round(float(best['final_min_T']), 4),
            )
        )
    out = pd.DataFrame(rows).sort_values(['dataset', 'init_n_neg_2tri', 'case_id']).reset_index(drop=True)
    return out


def win_count(best_df: pd.DataFrame) -> pd.DataFrame:
    """How many slices each ``(constraint:method_id)`` wins on (lowest L1)."""
    if best_df.empty:
        return pd.DataFrame()
    counts = best_df['best_method'].value_counts().rename_axis('method').reset_index(name='wins')
    counts['win_rate'] = round(counts['wins'] / len(best_df), 3)
    return counts


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV,
        help=f'Path to the results CSV (default: {DEFAULT_CSV})',
    )
    p.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='Where to write the summary files. Default: same dir as the CSV.',
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not args.csv.exists():
        print(f'ERROR: CSV not found: {args.csv}', file=sys.stderr)
        return 2
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load(args.csv)
    print(f'Loaded {len(df)} rows from {args.csv}')
    print(f'  Cases: {df["case_id"].nunique()}')
    if 'dataset' in df.columns:
        for ds, dg in df.groupby('dataset'):
            print(f'    {ds}: {dg["case_id"].nunique()} cases / {len(dg)} runs')
    print(f'  Methods: {df["method_id"].nunique()}')
    print(f'  Errors: {df["has_error"].sum()} ({df["has_error"].mean() * 100:.1f}%)')
    print(f'  Feasible: {(df["feasible"] & ~df["has_error"]).sum()}')

    by_method = summary_by_method(df)
    pivot = per_slice_pivot(df)
    best = best_per_slice(df)
    wins = win_count(best)

    by_method.to_csv(out_dir / 'summary_by_method.csv', index=False)
    pivot.to_csv(out_dir / 'per_slice_pivot.csv')
    best.to_csv(out_dir / 'best_per_slice.csv', index=False)
    if not wins.empty:
        wins.to_csv(out_dir / 'win_counts.csv', index=False)

    print()
    print('=' * 80)
    print('summary_by_method (L1 across feasible runs):')
    print('=' * 80)
    with pd.option_context(
        'display.max_columns', None, 'display.width', None, 'display.max_colwidth', 40
    ):
        print(by_method.to_string(index=False))
    if not wins.empty:
        print()
        print('=' * 80)
        print('win counts (lowest L1 per slice):')
        print('=' * 80)
        print(wins.to_string(index=False))

    print()
    print(f'Wrote: {out_dir}/summary_by_method.csv')
    print(f'Wrote: {out_dir}/per_slice_pivot.csv')
    print(f'Wrote: {out_dir}/best_per_slice.csv')
    if not wins.empty:
        print(f'Wrote: {out_dir}/win_counts.csv')

    return 0


if __name__ == '__main__':
    sys.exit(main())
