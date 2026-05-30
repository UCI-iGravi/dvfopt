"""Generator for the parameterized-API tour notebook.

Run from this directory to produce ``quick_tour.ipynb``.
"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent

CELLS = [
    ('md', """
# DVFopt — Quick tour of the parameterized API

The package is organized around three orthogonal axes — **constraint**, **objective**, **strategy** — composed via a `Solver`.

This notebook covers:

1. The one-shot `correct_dvf` convenience function.
2. Explicit `Solver` composition (when you want control over the choice).
3. The `auto_strategy` heuristic.
4. The `DVFopt` per-slice facade (for 3D volumes + tabular reports).
5. Constraint families (2-tri, Jdet 2D, Jdet 3D) and how they share the same solvers.
"""),
    ('code', """
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import dvfopt
print(f'dvfopt {dvfopt.__version__}')

# Build a small planted-fold field for the demos.
rng = np.random.default_rng(7)
phi = np.stack([rng.normal(0, 0.4, (12, 12)),
                rng.normal(0, 0.4, (12, 12))])

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
T1, T2 = _triangle_areas_2d(phi[0], phi[1])
init_n_neg = int((np.minimum(T1, T2) <= 0).sum())
print(f'init n_neg = {init_n_neg}, min_T = {min(T1.min(), T2.min()):+.4f}')
"""),
    ('md', """
## 1. One-shot `correct_dvf`

Pass a field plus three string labels. `strategy='auto'` reads the initial fold count + min_T and picks an appropriate strategy.
"""),
    ('code', """
from dvfopt import correct_dvf

result = correct_dvf(
    phi,
    constraint='2tri',     # what makes feasibility: per-cell 2-triangle areas
    objective='l1',        # smoothed L1 anchor (sparse corrections)
    strategy='auto',       # auto-pick: 'barrier' here for moderate density
)
print(f'feasible: {result.feasible}')
print(f'init   n_neg={result.init_n_neg}  min_T={result.init_min_T:+.4f}')
print(f'final  n_neg={result.final_n_neg}  min_T={result.final_min_T:+.4f}')
print(f'wall: {result.wall_time:.3f}s')
"""),
    ('md', """
## 2. Explicit `Solver` composition

When you want to pin a particular constraint / objective / strategy. Each axis is a class.
"""),
    ('code', """
from dvfopt import (
    Solver,
    TriConstraint2D, TriConstraint2DFullCoverage,
    L1Objective, L2Objective,
    BarrierStrategy, SLSQPFullGridStrategy,
    M10Strategy, M14Strategy, M14SchwarzStrategy,
)

solver = Solver(
    constraint=TriConstraint2D(shape=(12, 12)),
    objective=L1Objective(eps=1e-4),
    strategy=BarrierStrategy(margin=1e-3, max_iter=200),
)
print(repr(solver))
result = solver.fit(phi.copy(), verbose=0)
print(f'final  n_neg={result.final_n_neg}  min_T={result.final_min_T:+.4f}  '
      f'feasible={result.feasible}')
"""),
    ('md', """
## 3. The `auto_strategy` heuristic

Given a `Constraint` and initial stats, returns the recommended strategy label.

Tiers (2-triangle constraint):

* `n_neg > 5000` or `init_min < -10` → wallbreakers (`m10` / `m14` / `m14_schwarz`)
* `n_neg > 100` or `init_min < -0.25` → `barrier`
* otherwise → `slsqp`
"""),
    ('code', """
from dvfopt.solver import auto_strategy
from dvfopt.constraints import TriConstraint2D, JdetConstraint2D, JdetConstraint3D

print('2-tri 2D:')
c = TriConstraint2D((100, 100))
print(f'  mild      ({1:>5}, {-0.05:+.2f}): {auto_strategy(c, 1, -0.05)}')
print(f'  moderate  ({300:>5}, {-0.4:+.2f}): {auto_strategy(c, 300, -0.4)}')
print(f'  extreme   ({6000:>5}, {-15.0:+.2f}, L1): {auto_strategy(c, 6000, -15, "l1")}')
print(f'  extreme   ({6000:>5}, {-15.0:+.2f}, L2): {auto_strategy(c, 6000, -15, "l2")}')
c_big = TriConstraint2D((320, 456))
print(f'  extreme   ({6000:>5}, {-15.0:+.2f}, L1, big slice): '
      f'{auto_strategy(c_big, 6000, -15, "l1")}')

print()
print('Jdet 2D (no wallbreakers):')
cj = JdetConstraint2D((100, 100))
print(f'  mild:     {auto_strategy(cj, 1, -0.1)}')
print(f'  dense:    {auto_strategy(cj, 700, -1.2)}')

print()
print('Jdet 3D:')
cj3 = JdetConstraint3D((10, 100, 100))
print(f'  any:      {auto_strategy(cj3, 50, -0.5)}')
"""),
    ('md', """
## 4. `DVFopt` per-slice facade

Wraps `Solver` with:
* 2D/3D auto-detection (`(2, H, W)`, `(3, H, W)`, `(3, D, H, W)`)
* per-slice loop with tabular + dataframe + plot helpers
* string-config-style configuration via `DVFoptConfig`

Use when you have a 3D volume and want tabular reports across slices.
"""),
    ('code', """
from dvfopt import DVFopt, DVFoptConfig

cfg = DVFoptConfig(
    constraint='2tri',
    solver='auto',         # let auto_strategy pick per-slice
    objective='l1',
    threshold=0.01,
    verbose=0,
)
res = DVFopt(cfg).fit(phi)
print(res.summary())
print()
print(res.to_dataframe())
"""),
    ('md', """
## 5. Same Solver, different constraint

The strategies are **constraint-agnostic** (barrier, SLSQP windowed) where possible. Strategies that embed constraint-specific reasoning — m10, m14, m14-Schwarz (harmonic extension assumes 2-tri geometry) — reject incompatible constraints at construction time.
"""),
    ('code', """
from dvfopt import Solver, JdetConstraint2D, L2Objective, BarrierStrategy

# Same BarrierStrategy works with Jdet:
res_jdet = Solver(
    constraint=JdetConstraint2D(shape=(12, 12)),
    objective=L2Objective(),
    strategy=BarrierStrategy(),
).fit(phi.copy(), verbose=0)
print(f'Jdet barrier: feasible={res_jdet.feasible}  '
      f'min_jdet={res_jdet.final_min_T:+.4f}')
"""),
    ('code', """
# But m10 only makes sense for the 2-tri constraint — rejected at solver construction.
from dvfopt import M10Strategy

try:
    Solver(
        constraint=JdetConstraint2D(shape=(12, 12)),
        objective=L2Objective(),
        strategy=M10Strategy(),
    )
except TypeError as e:
    print(f'expected TypeError: {e}')
"""),
    ('md', """
## Strategy zoo

| Strategy | Constraint required | 3D? | Best for |
|---|---|---|---|
| `BarrierStrategy` | any | yes | Moderate density (100-5000 folds) |
| `SLSQPFullGridStrategy` | 2-tri | no | Mild folds, want KKT semantics |
| `SLSQPWindowedStrategy` | any | yes | Legacy Jdet windowed path |
| `SchwarzStrategy` | 2-tri | no | Many small clusters across a big slice |
| `M10Strategy` | 2-tri | no | Extreme density, L2 anchor (feasibility-guaranteed) |
| `M14Strategy` | 2-tri | no | Extreme density, L1 anchor (smallest deviation) |
| `M14SchwarzStrategy` | 2-tri | no | Extreme density + large slice (>20K corners) |

The auto-resolver knows this matrix and picks the right one. For research / benchmarking, you can also pin the strategy directly.
"""),
]


def main():
    nb = nbf.v4.new_notebook()
    for kind, src in CELLS:
        src = src.strip('\n')
        if kind == 'md':
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    out = HERE / 'quick_tour.ipynb'
    nbf.write(nb, out)
    print(f'Wrote {out} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    main()
