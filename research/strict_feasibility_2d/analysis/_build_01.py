"""Generator for ``01_baseline_l1_gap.ipynb``.

Run from any directory:

    python research/strict_feasibility_2d/analysis/_build_01.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

CELLS = [
    (
        'md',
        """\
# 01 — Baseline L1 gap

Loads `comparison_synthetic.csv` (+ `comparison_b0039.csv` if present),
builds the headline table per the spec, and plots per-case L1 deviation
by method.

**Strict-feasibility rule:** rows with `feasible=False` are EXCLUDED
from the L1 ranking. No averaging over infeasible solutions.
""",
    ),
    (
        'code',
        """\
from pathlib import Path
import pandas as pd

# Notebook lives in research/strict_feasibility_2d/analysis/
# CSVs live in ../runners/output/
_HERE = Path.cwd()
OUTDIR = _HERE.parent / 'runners' / 'output' if (_HERE.name == 'analysis') else _HERE
# Fallback: locate by searching from HERE upward.
if not (OUTDIR / 'comparison_synthetic.csv').exists():
    for parent in [_HERE, *_HERE.parents]:
        cand = parent / 'research' / 'strict_feasibility_2d' / 'runners' / 'output'
        if (cand / 'comparison_synthetic.csv').exists():
            OUTDIR = cand
            break

print('CSV directory:', OUTDIR)
df_synth = pd.read_csv(OUTDIR / 'comparison_synthetic.csv')
df_b0039 = pd.read_csv(OUTDIR / 'comparison_b0039.csv') if (OUTDIR / 'comparison_b0039.csv').exists() else pd.DataFrame()
df = pd.concat([df_synth, df_b0039], ignore_index=True)
print(f'Loaded {len(df)} rows across {df.case_id.nunique()} cases and {df.method.nunique()} methods.')
df.head()
""",
    ),
    (
        'code',
        """\
# Per-method feasibility summary.
summary = df.groupby('method').agg(
    n_runs=('feasible', 'size'),
    n_feasible=('feasible', 'sum'),
).assign(
    feasible_frac=lambda d: d['n_feasible'] / d['n_runs'],
)
summary.sort_values('feasible_frac', ascending=False)
""",
    ),
    (
        'code',
        """\
# Headline table: L1 deviation per case x method (feasible-only).
df_feas = df[df['feasible']].copy()
pivot_L1 = df_feas.pivot_table(
    index='case_id', columns='method', values='L1_dev', aggfunc='first',
).round(4)

# Add an `L1_lp_oneshot` baseline column and the per-method gap vs m14.
if 'lp_oneshot' in pivot_L1.columns and 'm14' in pivot_L1.columns:
    pivot_L1['_gap_m14_minus_lp'] = pivot_L1['m14'] - pivot_L1['lp_oneshot']
pivot_L1
""",
    ),
    (
        'code',
        """\
# Wall-time table.
pivot_wall = df.pivot_table(
    index='case_id', columns='method', values='wall_s', aggfunc='first',
).round(2)
pivot_wall
""",
    ),
    (
        'md',
        """\
## Reading the headline table

* A blank cell means that (case, method) was infeasible at exact eval
  and excluded — see the feasibility summary cell above for which
  method failed where.
* `_gap_m14_minus_lp` = `L1_dev(m14) - L1_dev(lp_oneshot)` — positive
  numbers mean LP wins on L1, negative means M14 wins.
* `cluster_pipeline` rows are expected to be empty / errored — the
  adapter is not wired yet (see Task 9 note in the plan).
""",
    ),
    (
        'code',
        """\
import matplotlib.pyplot as plt
import numpy as np

# Per-case bar chart. Drop the gap column for plotting.
cases = pivot_L1.index.tolist()
methods = [m for m in pivot_L1.columns if not m.startswith('_')]
if cases and methods:
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(cases)), 5))
    n = len(methods)
    x = np.arange(len(cases))
    w = 0.8 / n
    for i, m in enumerate(methods):
        vals = pivot_L1[m].values
        ax.bar(x + (i - n / 2) * w, vals, width=w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=30, ha='right')
    ax.set_ylabel('L1 deviation from input')
    ax.set_title('L1 deviation by method (feasible runs only)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(_HERE / 'l1_per_case.png' if _HERE.name == 'analysis' else 'l1_per_case.png', dpi=150)
    plt.show()
else:
    print('No feasible rows to plot.')
""",
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == 'md' else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python'},
    }
    path = HERE / '01_baseline_l1_gap.ipynb'
    nbf.write(nb, str(path))
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
