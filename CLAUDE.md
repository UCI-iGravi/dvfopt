# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for correcting **negative Jacobian determinants** in 2D/3D deformation (displacement) fields. The installable `dvfopt/` package composes a solve out of three orthogonal axes — Constraint × Objective × Strategy — over a method-first `dvfopt/core/` (one package per algorithm family). Notebooks in `notebooks/` demonstrate each method; `benchmarks/` compares performance across registration algorithms.

**Adding a method / constraint / objective? Read [ARCHITECTURE.md](ARCHITECTURE.md) first** — it holds the dependency rules and the three extension checklists. This file is the map of what exists.

## Setup & Commands

```bash
# Install core package (editable)
pip install -e .

# Install with benchmark dependencies (itk-elastix, opencv, timm, torch, voxelmorph)
pip install -e ".[benchmarks]"

# Or install all dev dependencies (includes voxelmorph from GitHub, pandas, ipykernel)
pip install -r requirements-dev.txt
```

`uv` is the faster path (10-100x on the heavy torch/itk legs) and is what CI
uses. `uv pip install --system -e ".[dev]"` is a drop-in for the pip lines
above. `uv.lock` pins a full resolution (`uv sync --extra dev` reproduces the
exact env; `uv lock --check` verifies the lock still matches pyproject's
constraints — CI's test leg runs this, the coverage leg installs from the lock
via `uv sync --locked`). pyproject deps stay unpinned; the lock is the opt-in
reproducible snapshot. Re-lock after editing dependency constraints (`uv lock`).

`requires-python = ">=3.10"` and `scipy>=1.15,<1.19`: the traced SLSQP driver
(`dvfopt.core.primitives.slsqp`) vendors scipy's `_slsqplib` private internals
(the SLSQP C core), which exist only on scipy >=1.16 — itself requiring
Python >=3.11 — through 1.18, hence the upper pin. On scipy 1.15.x (what
Python 3.10 resolves to, and which lacks `_slsqplib`) the driver sets
`HAS_TRACED_SLSQP = False` and transparently falls back to scipy's own
`minimize(method='SLSQP')`: same numerics, no per-iteration trace.
`requirements-dev.txt` carries the same scipy pin. Ruff's `target-version`
deliberately trails at `py39` (see the comment in pyproject).

The install exposes a `dvfopt` CLI (`dvfopt {info, correct, gui}`, also `python -m dvfopt`) over the library — see [dvfopt/cli.py](dvfopt/cli.py); `correct` drives the solver / per-slice sweep / 2.5D / 3D pipelines and writes `summary.json` + `convergence.png` reports (the per-slice sweep parallelizes over slices with `--n-workers N`; inner solves stay serial, no nested pools). Exit codes: 0 feasible / 1 folds remain / 2 usage errors.

Solver progress output routes through the `dvfopt` logger (`dvfopt/_logging.py` — `vlog`/`log_info`/`log_warning`; `verbose=` semantics unchanged). A live-stdout handler is auto-installed (and propagation disabled, so lines print exactly once) only when no handler is attached to the `dvfopt` logger; attach your own handler to that logger to take over routing. Tests live in `tests/` and are run with `pytest` (`[tool.pytest.ini_options]` in pyproject scopes collection to `tests/`, so a bare `pytest` never over-collects the `notebooks/experiments/_run_*_test.py` scratch scripts). CI: `.github/workflows/ci.yml` runs the numba-tuned suite on Python 3.11/3.12; `.github/workflows/test.yml` runs Python 3.10/3.11/3.12 with `ruff check` + `ruff format --check` + `mypy` + the pytest suite + an installed-CLI smoke, plus a coverage job and a benchmark-import smoke. Additional validation is done through Jupyter notebooks.

```bash
# Run all tests. pytest-randomly shuffles order each run (reproduce a failure
# with --randomly-seed=<seed>); pytest-xdist parallelizes with -n auto.
pytest                     # or: pytest -n auto

# Run a specific test module
pytest tests/test_slp_strategy.py

# Lint + format (ruff pinned 0.16.3 in the dev extras AND .pre-commit-config.yaml)
ruff check dvfopt dvfopt_gui tests benchmarks
ruff format --check dvfopt dvfopt_gui tests benchmarks

# Static types (scoped to the cleanly-typed modules; see [tool.mypy]) + coverage
mypy
pytest tests/ --cov=dvfopt --cov-report=term-missing

# Or drive everything through the nox task runner (noxfile.py):
nox                        # default: lint + format_check + tests
nox -s typecheck           # mypy      | nox -s cov   # coverage

# Solver perf regression tracking (airspeed velocity; not run in CI):
asv run                    # asv_bench/benchmarks/, config in asv.conf.json
```

## Architecture

### Data conventions

- **Deformation fields:** `(3, 1, H, W)` numpy arrays with channels `[dz, dy, dx]`. For 2D work the z-slice dim is 1. Convention is pull-back (backward mapping).
- **3D fields:** `(3, D, H, W)` with `[dz, dy, dx]`.
- **Coordinates/correspondences:** always `[z, y, x]` ordering, shape `(N, 3)`.
- **Jacobian threshold:** `0.01` (from `dvfopt/_defaults.py`). Error tolerance `1e-5`.
- **SimpleITK interop:** arrays transposed `(3,1,H,W)` → `(1,H,W,3)` and axis-reordered `[2,1,0]` (zyx→xyz). See `dvfopt/jacobian/sitk_jdet.py`.

### Optimization internals

- **phi flattening — TWO conventions exist, do not cross-mix them:**
  - **`PhiPack.DX_FIRST`** — `phi[:N]` = `dx`, `phi[N:2N]` = `dy` (3D: also `phi[2N:]` = `dz`). Declared by `JdetConstraint2D`, `JdetConstraint3D` **and `SimplexConstraint3D`** (the simplex (3D) family packs x-first so it can share the 3D barrier plumbing with Jdet3D). Modules: `dvfopt/core/slsqp_windowed/*`, `dvfopt/core/primitives/{jdet2d,jdet3d}.py`, `dvfopt/core/barrier/{jdet2d,jdet3d,jdet3d_torch,tet3d_torch}.py`, `dvfopt/core/slsqp_fullgrid/tet3d.py`, `dvfopt/core/slp/{lp_direct_6tet,cluster_lp_6tet}.py`, the 3D wallbreakers, `dvfopt/jacobian/tetrahedron_sign.py`.
  - **`PhiPack.DY_FIRST`** — `phi[:N]` = `dy`, `phi[N:]` = `dx`. Declared by `SimplexConstraint2D` / `SimplexConstraint2DFullCoverage` / `SimplexConstraint2DBilinear` (2D triangle families only). Modules: `dvfopt/core/primitives/tri.py`, `dvfopt/core/barrier/tri2d.py`, `dvfopt/core/slsqp_fullgrid/tri2d.py`, `dvfopt/core/schwarz/{tri2d,_cluster}.py`, the 2D wallbreakers (m02/m03/m10/m12/m14), `dvfopt/core/slp/{lp_direct_2tri,cluster_lp_2tri,tri_linearize}.py`.
  - So the split is **not** "simplex vs Jdet" — it is `Constraint.pack`, and that attribute is the only thing to trust. `dvfopt/core/schwarz/_common.py` is pack-agnostic (slices `(C, *shape)` arrays, never flattens).
  - A flat phi vector from one pack CANNOT be passed to a helper of the other without channel-swapping. A helper that genuinely sees both — `dvfopt/core/marching/*` mixes a DX_FIRST simplex (3D) stack with a DY_FIRST simplex (2D) term — asserts on the pack lengths at the boundary; copy that pattern. Helpers in `dvfopt/core/primitives/tri.py` and `anchor_term` in `dvfopt/objectives.py` are the single sources of truth — reuse them rather than re-deriving partials.
- **Laplacian matrix:** uses `z*ny*nx + y*nx + x` flattening in `dvfopt/laplacian/utils.py`.
- **Windowed approach:** iterative SLSQP finds worst-Jdet pixel, computes bounding box of connected negative region + 1px positive border (min 3×3), runs `scipy.optimize.minimize(method='SLSQP')` on that sub-window with frozen edges. Grows window by 2 if needed.
- **Parallel variant:** `iterative_parallel()` batches non-overlapping windows into `ProcessPoolExecutor`. Falls back to serial for single windows (avoids Windows spawn overhead).

### Constraint modes

The 2D solver accepts `enforce_shoelace=True` (geometric quad-cell area) and `enforce_injectivity=True` (coordinate monotonicity) flags in addition to the default Jacobian determinant constraint. The 3D solver (`iterative_3d`) supports `enforce_injectivity=True` (axial monotonicity of deformed coordinates — linear rows, see `dvfopt/jacobian/monotonicity.py`); the 3D analogue of `enforce_shoelace` (geometric cell volume) is served by the dedicated simplex (3D) constraint family, and `SLSQPWindowedStrategy(enforce_shoelace=True)` on a 3D constraint raises accordingly. Both flags are exposed on `SLSQPWindowedStrategy`.

### Parameterized API (current)

The public surface is organized around three orthogonal axes composed via `Solver`:

```python
from dvfopt import Solver, SimplexConstraint2D, L1Objective, BarrierStrategy
result = Solver(
    constraint=SimplexConstraint2D(shape=(H, W)),
    objective=L1Objective(eps=1e-4),
    strategy=BarrierStrategy(),
).fit(phi)
```

**Robust one-call recipe** — the strongest from-raw fold clearer currently in the package:

```python
from dvfopt import correct_dvf
res = correct_dvf(phi, constraint='bilinear', strategy='isqp_windowed', objective='none')
```

Bilinear feasibility implies simplex feasibility (the 4 triangles/cell cover both diagonals, so the 2 simplex rows are a subset), and pure feasibility (`objective='none'`) frees the windowed isqp from the objective-basin traps a distance anchor pins it in. Verified on the three hard B0039 crops (`benchmarks/output/testcases/`, gitignored): simplex folds 645/598/0 → **0/0/0**, damage 0, 32s / 22s / 106s. Needs `osqp` (`pip install dvfopt[solvers]`).

**Constraints** ([dvfopt/constraints.py](dvfopt/constraints.py)) — `SimplexConstraint2D`, `SimplexConstraint2DFullCoverage`, `SimplexConstraint2DBilinear` (label `'bilinear'`: both diagonals, 4 triangles/cell = the `cell_min_jdet_2d` sub-pixel certificate; barrier / windowed / `SLSQPWindowedStrategy` accept it, the other simplex (2D)-specialised strategies reject it), `JdetConstraint2D`, `FiniteJdetConstraint2D`, `JdetConstraint3D`, `SimplexConstraint3D`. Each provides `values()`, `adjoint(v)`, optional `jacobian()` (SLSQP path only), plus `flatten/unflatten` between `(C, *shape)` arrays and the flat decision vector, and declares `pack` + `dim`. `@register_constraint('<label>')` + `make_constraint(name, shape)` back the string-label path. The pack convention is encoded in `Constraint.pack` — `PhiPack.DY_FIRST` for the 2D simplex pair, `PhiPack.DX_FIRST` for Jdet2D/Jdet3D **and simplex (3D)** (see the phi-flattening note above).

The **simplex metric** (labels `'simplex'` / `'simplex_standard'` / `'simplex_3d'`; formerly *2-tri* / *6-tet*) is the exact Jacobian determinant of the piecewise-linear interpolant on the fixed simplicial decomposition of the grid — 2 triangles per cell along the fixed BL–TR diagonal in 2D, 6 tetrahedra per cell in 3D — so feasibility is a genuine injectivity certificate for that interpolant. Strictness ordering: central-diff Jdet < `'finite'` (forward-diff = one triangle per cell) < simplex (both triangles). The old names *2-tri* / *6-tet* remain as registry aliases (`'2tri'`, `'2tri_standard'`, `'6tet'`, `'6tet_3d'`), and the old class names (`TriConstraint2D`, `TriConstraint2DFullCoverage`, `TriConstraint2DBilinear`, `Tet6Constraint3D`) remain importable.

**Objectives** ([dvfopt/objectives.py](dvfopt/objectives.py)) — `L1Objective(eps)`, `L2Objective()`, `NoneObjective()`, over the shared `anchor_term` (which lives in this module — pure numpy, must not import `dvfopt.core`). An `Objective` is `__call__(diff) -> (value, grad)` and nothing else; there is no operator composition. Every solver in the package takes `objective=` end-to-end (no string anchors left). Kernels that cannot call back into Python (numba wallbreakers, torch autograd) take the legacy `(kind, eps_l1)` pair from `objectives._kind_eps(objective)`.

**Strategies** ([dvfopt/strategies/](dvfopt/strategies/)) — `NMVFStrategy` (heuristic neighborhood-mean smoother, original method), `SLPStrategy` (sequential-LP / `auto_slp` — the L1-minimising strict-feasibility champion: per-cluster trust-region SLP + m14 seed + HiGHS L1 step, continuous parallel cluster scheduler; promoted from `research/strict_feasibility_2d` into `dvfopt/core/slp/`; also accepts the 3D simplex (3D) constraint via the promoted `lp_direct_6tet`/`cluster_lp_6tet` solvers with the research-validated `seed_3d='m10'` default), `BarrierStrategy`, `SLSQPFullGridStrategy`, `SLSQPWindowedStrategy`, `SchwarzStrategy`, `SchwarzWrapperStrategy(inner=…)` (generic Schwarz wrapper around any simplex (2D) or simplex (3D) inner — auto-detects 2D vs 3D), `WindowedWrapperStrategy(inner=…)` + `ISQPWindowedStrategy` (2D Jdet / standard simplex (2D) / bilinear / finite: the no-damage cluster-windowed engine with a label-selected window inner — `inner` is a string label (`'isqp'`/`'slsqp'`/`'slsqp+trust-constr'`), NOT a Strategy, because each window is a frozen-ring reduced problem; needs `osqp` for the isqp inner; knobs `no_tr_fallback=True` / `fallback_maxiter=200` / `qp_max_iter=2000` / `qp_max_iter_fallback=500` / `qp_backend='hybrid'` / `step_rule='exact_ls'` / `reanchor='none'` — see the windowed engine note below), `HarmonicALMBarrierStrategy` (alias `M10Strategy`), `HarmonicALMRefineRepairStrategy` (alias `M14Strategy`), `SchwarzHarmonicALMRefineRepairStrategy` (alias `M14SchwarzStrategy`). 3D analogues for the wallbreakers: `HarmonicALMBarrier3DStrategy` (alias `M10TetStrategy`), `HarmonicALMRefineRepair3DStrategy` (alias `M14TetStrategy`), `SchwarzHarmonicALMRefineRepair3DStrategy` (alias `M14Schwarz3DStrategy`). The class names are phase-stack-explicit: each algorithm in the pipeline (harmonic Laplacian extension, PHR-ALM, log-barrier polish, soft-penalty L2 refine, harmonic repair, Schwarz domain decomposition) appears in the name. The dedicated `Schwarz*` classes are equivalent to `SchwarzWrapperStrategy(inner=...)` with the inner pinned — both run through the shared `dvfopt.core.schwarz._common` core (one implementation of the Schwarz decomposition, not two). Each Strategy is a dataclass with strategy-specific knobs.

`accepts_constraints`, `accepts_objectives` and `supports_3d` class attrs declare compatibility (`None` = accept anything); `Solver.__init__` checks all three at construction and raises `IncompatibleConstraintError` / `IncompatibleObjectiveError` (both `dvfopt.exceptions`, both `TypeError` subclasses) rather than failing mid-solve. `SLPStrategy` declares `accepts_objectives = (L1Objective, NoneObjective)` — it is an L1 method and cannot honour an L2 anchor. `BarrierStrategy(objective_override=...)` lets a composed pipeline pin the barrier leg's objective independently of the Solver's.

**Solver** ([dvfopt/solver.py](dvfopt/solver.py)) — composes the three; provides `from_spec(constraint='simplex', ...)` string-based construction and one-shot `correct_dvf(phi, ...)`. `auto_strategy(constraint, init_n_neg, init_min, objective_label)` encodes the strategy-selection heuristic. **2D routing table** (every `isqp_windowed` row needs `osqp`; without it the row falls through to the tier heuristic):

| constraint | objective | strategy |
|---|---|---|
| `bilinear` | any | `isqp_windowed`, every fold tier |
| `simplex_standard` | `none` | `isqp_windowed`, every fold tier |
| `simplex*` | `l1` | `slp` (the L1 champion), every fold tier |
| `simplex*` | `l2` | density-tiered: `slsqp` mild → `barrier` → `m10` extreme |
| `jdet` / `finite` | any | `barrier` dense, `isqp_windowed` (else `slsqp_windowed`) mild |

`bilinear` + `isqp_windowed` + `none` is the measured robust 0-fold recipe — see [docs/recipe-2d-zero-folds.md](docs/recipe-2d-zero-folds.md). It is never silently substituted for an L1/L2 request (a different fidelity ask); `simplex`+`l1` keeps `'slp'` and logs a one-line hint. Note `'simplex'` = full-coverage has no windowed-engine locality entry, so only `'simplex_standard'` takes the `none` row. 3D routing is untouched: simplex (3D) tiers like 2D — extremes (n_neg > 5000 or min < -10) route to the 3D wallbreakers (`m10_3d` for L2, `m14_schwarz_3d` on >200K-voxel volumes, else `m14_3d`); everything else keeps `barrier`.

**SolveInfo** — every `Strategy.solve` normalises its return through `_build_solve_info`, so callers always get `(phi_out, SolveInfo)` with `.phases` / `.total_iter` / `.extras`. With `record_history=True` the SLSQP strategies also lift each run's per-major-iteration trace (from the traced driver) to the stable path `SolveInfo.extras['slsqp_trace']` — a list of `{'phase': ..., 'iters': [...]}` — so the GUI and reports never reach into per-phase `PhaseInfo.extras`.

**DVFopt facade** ([dvfopt/unified.py](dvfopt/unified.py)) — per-slice orchestration over `Solver`: 2D/3D auto-detection, tabular reports, plots. Use when you want `DVFoptConfig` string-based config and per-slice analysis across a 3D volume. `DVFoptConfig(n_workers=N)` solves the z-slices of a volume in a `ProcessPoolExecutor` (each slice is an independent solve, so it scales near-linearly with cores); `None`/`0`/`1` — or a single slice — stays serial. Spawn-safe (module-level worker, picklable args), but a *script* calling `fit()` with `n_workers > 1` on Windows/macOS must guard the call under `if __name__ == '__main__':`. Pool workers are pinned to **one compute thread each** — `dvfopt.core._pool.pin_worker_threads()` at the top of every pool worker plus `pinned_thread_env()` around every pool's submits (children inherit the env at spawn, the only point early enough for OpenBLAS/MKL); serial paths are untouched (the helper no-ops outside a child process). Without it numpy and scipy each start a full-width BLAS pool per worker (measured: 53 OS threads per worker on a 24-logical-core box). **Keep `n_workers` small — 2-4, NOT the core count** (same for the CLI's `--n-workers`): the solves are memory-bandwidth bound, so measured throughput peaks at ~2.6x around 4 workers and declines past it. Pinning is resource hygiene, not a scaling fix — it buys 9-25% at n_workers >= 12 and is neutral below. Measured tables in the CHANGELOG.

**Windowed engine knobs** (`windowed_correct` and `ISQPWindowedStrategy`, isqp inner only — defaults measured on the hard B0039 crops):

- `no_tr_fallback=True` — a window that fails to reach its target is retried ONCE, same box, with the trust region OFF (legacy backtracking line search) before grow-on-failure. The TR ratio test freezes on sliver-scale violations (~1e-4, inside OSQP's own noise) the line search still clears. Warm-started from the failed iterate; keeps whichever result has the higher constraint minimum (never worse).
- `fallback_maxiter=200` — SQP iteration budget for that retry (the line search otherwise runs far past convergence; sliver crop 90-105s vs 183s at 800).
- `qp_max_iter=2000` / `qp_max_iter_fallback=500` — OSQP ADMM iteration cap per subproblem, normal / fallback solves (`None` = OSQP's 8000 default). ~2x faster overall at unchanged feasibility. Escalation ladders over these caps were measured and are *worse* — do not re-add one.
- `giant_tile=64` / `giant_max_sweeps=8` — tile size / sweep cap for the overlapping-tile Schwarz decomposition of an over-`max_window_area` region (tiles step by `giant_tile - (2*ring + 2)`; the sweep loop exits early once the region is clear or stops improving). Bigger tiles = fewer Schwarz seams = fewer sweeps. Measured on a full raw B0039 z16 slice (bilinear, objective `none`): **tile 64 = 362 s / 22 windows / 1 round / no mop vs tile 32 = 685 s / 264 windows / 3 rounds / 4 mop** — 1.9x faster, 0 simplex folds and damage 0 either way, and a *smaller* move (L2 316 vs 404). The tiler was ~81% of full-slice wall time; 64 became the default in that measurement (behavior change vs the promoted 32).
- `giant_tile_fit=True` — treat `giant_tile` as a *target*: the effective tile for a region of bbox `(h, w)` is `ceil(max(h,w) / ceil(max(h,w) / giant_tile))`, clamped to `[0.75, 1.5] x giant_tile` (`_fit_tile`), so an integer number of near-equal tiles covers the longest side. Tile size acts on cost through grid **alignment** — the Schwarz sweep-round count — not through the size itself: a tile leaving a thin remainder strip needs an extra round to propagate through it. Measured on the raw B0039 z16 giant (a 125x152 box): tile 64 happens to align (1 round, 374 s) while 56 and 80 do not (2 rounds, ~600 s); the fitted 51 aligns by construction (1 round, 345 s). `False` = literal `giant_tile` (byte-identical to pre-fit behavior).
- `coarse_to_fine=True` / `coarse_factor=4` — **coarse-grid warm start**, prepended to the round loop. The same problem is solved on a `coarse_factor`x box-averaged field (displacements rescaled to coarse pixel units), the correction is bilinearly prolongated back (rescaled up), and the fine solve starts from `phi + delta` instead of cold — the fine windows then converge in far fewer SQP iterations. The prolongated delta is **masked to the free boxes `find_windows` opens on the fine fold mask**, so the warm start can only move pixels the engine was going to free anyway: no-damage is preserved by construction, and the final damage accounting still runs against the ORIGINAL input. Skipped — leaving the path byte-identical to `coarse_to_fine=False` — on a fold-free field or when `min(H, W) < 4 * giant_tile` (a coarse problem that small is not a useful preview and its own solve is not amortised), which is why every small-crop benchmark and the whole test suite still exercise the cold path. Measured on raw B0039 z16 (3890 simplex folds, bilinear rows, objective `none`, maxiter 600): **205 s / 909 SQP iterations (841 fine + a 16 s, 68-iteration coarse solve) vs 283 s / 1320 cold** — -28% wall, -31% iterations, 0 folds and damage 0 either way, at a slightly *smaller* move (L2 320.6 vs 325.1), so the speed is not bought with fidelity. The coarse solve itself cleared 1054 -> 0 coarse folds; the warmed fine field still had 2840 folds, i.e. the win is a better basin for the fine windows, not folds pre-removed. `report.coarse_solve_s` / `coarse_folds_before` / `coarse_folds_after` / `coarse_iters` / `warm_folds` record the stage (`-1` = skipped). Factor sweep on the same slice (deterministic): `coarse_factor=2` 189 s / 841 fine iters / L2 321; **`4` (default) 182-184 s / 762 / L2 280 (-13% move)**; `8` 167 s / 733 / L2 319 (the pure-speed setting). Multilevel cascades (4->2->1, 8->4->2->1) trade wall for fidelity and never beat a single 4x level on both axes.
- `tr_delta=2.0` / `tr_max=16.0` — initial radius / cap of the isqp inner's trust region (grid units), previously hard-coded in `isqp_solve`. The defaults are unchanged, so every prior measurement stands. `tr_delta=1.0` is the speed/fidelity dial: raw B0039 z16 **267 s / 1022 SQP iterations at L2 move 344**, vs 300 s / 1320 / L2 325 at 2.0 — -11% wall and -23% iterations for a visibly bigger departure from the input, which is why 2.0 stays the default and coarse-to-fine (which costs no fidelity) is the preferred speedup. `tr_max` never binds on the measured B0039 windows.
- `step_rule='exact_ls'` (default; `'tr'` restores the ratio test byte for byte) — how the isqp inner turns a QP step into an iterate. Instead of accepting or rejecting the whole step by the trust-region ratio test, `'exact_ls'` takes the **exact minimiser of the merit along the step**. It is exact because every 2D row family here (2tri, bilinear, jdet, finite) is a BILINEAR form in `(dy, dx)`, so along the line a row is exactly quadratic: `c(a) = c + a (J d) + a^2 q` with `q = cons(x + d) - c - J d` — and that `cons(x + d)` is the evaluation the ratio test **already makes**, so the model costs no extra constraint evaluation and needs no per-family Hessian table. The merit is then piecewise quadratic with breakpoints at the rows' roots, and its global minimiser on `[0, 1]` is a vectorised breakpoint sweep (`isqp._exact_line_min`). The objective along the line is fitted from `obj` at `a = 0, 1/2, 1` — exact for `NoneObjective` / `L2Objective`, approximate for the eps-smoothed L1 — so the TRUE merit at `a*` is checked before stepping and the iteration falls back to the `'tr'` acceptance if it did not decrease; `'exact_ls'` therefore can never regress a window. The ratio test's own futility threshold (achieved <= 1e-3 x predicted) is kept as the `tr-collapse` trigger: an exact minimiser always finds SOME decrease, so without it a hopeless window grinds instead of handing off to the escalation ladder (measured load-bearing: z0_sliver 229 s -> 139 s). Measured raw B0039 z16: **200 s / 563 SQP iterations vs 244 s / 780** (-18% / -28%), 0 folds, damage 0, *smaller* move (L2 268 vs 280); over a 9-real-slice B0039 sample (z16 + every 48th from z=64, fold counts 835-3890) **9/9 wall wins AND 9/9 iteration wins, -19% wall / -27% iterations in total**, and a smaller L2 move on every slice. The one crop regression is `z0_sliver` (351 s vs 77 s) — a case with 0 simplex folds to begin with and only ~1e-4-scale violations, i.e. built entirely at OSQP's noise floor, where four mathematically-equivalent framings of the same method span 139-287 s: a chaos detector with no counterpart on real slices. The rule applies on the no-trust-region fallback rung too (scoping it out of that rung measured *worse*). **2D only** — a 6-tet volume row is trilinear, hence cubic along a line — and rejected at `windowed_correct`'s entry otherwise. A maximal fold-free step cap was measured and **REFUTED** (`a_max` ~1e-3-1e-1 on real windows strangles the elastic mechanism, ending at violation 40-84 vs the baseline's 0.027); do not add one.
- **QP backend** `qp_backend='hybrid'` (default) / `ip_cold=True` / `ip_after_admm_iters=800` — which QP solver backs each isqp subproblem. `'hybrid'` runs interior-point **Clarabel** on a window's COLD first solve and on the solve after any ADMM run of `>= ip_after_admm_iters` iterations (the stale-warm-start tail signal), and warm-started OSQP everywhere else; the IP point seeds OSQP's warm start, and any IP failure falls through to ADMM. `'osqp'` reproduces the pre-hybrid behavior **byte for byte**, and is also what `'hybrid'` degrades to when `clarabel` is missing (logged once at DEBUG). Why hybrid and not IP-always: on real giant-tile QPs (16k vars, 27k rows) Clarabel takes ~0.25 s / 15-25 iterations at ~1e-9 feasibility vs OSQP's 0.4-2.2 s / 700-4000 ADMM iterations at ~1e-3 — but in-engine *warm-started* OSQP averages 0.175 s/solve, so Clarabel-always is **slower** (raw B0039 z16 381 s vs 300 s, and 34% more SQP iterations). Hybrid: **z16 262 s vs 300 s (-13%)**, 0 simplex folds, damage 0, smaller move (L2 325 vs 346). Policy sweep: cold-only 296 s, threshold 400 -> 289 s, **800 -> 262 s (best)**, 1500 -> 269 s, no-cold/800 -> 281 s. The IP legs also change the SQP trajectory, and on some windows steer it into a basin with no escape, so the window escalation ladder gains a **backend rung**: a real window (never a giant tile) that ends GENUINELY folded (`cons < -margin_delta`, not merely short of the margin-shifted target) is retried whole on plain OSQP from its ORIGINAL start state, ahead of grow-on-failure — `report.backend_fallbacks` / `WindowRec.backend_fallback` count it. Without the rung the z0_cluster crop ends one triangle inverted at -1.2e-4; with it all three hard crops reach 0 simplex folds and raw z16 is unchanged (264.5 s, L2 325.1, 0 fallbacks — every full-slice retry candidate there is a tile, which the Schwarz sweep and the mop already cover; retrying tiles cost 505 s for no fold gain).
- `reanchor='none'` (default — off) / `reanchor_maxiter=60` / `reanchor_sweeps=3` / `reanchor_tile=48` — optional **post-feasibility re-anchor stage**, appended after the mop. The robust recipe solves with `objective='none'` (pure feasibility keeps the inner out of the objective-basin traps a distance anchor pins it in), so the correction is close to the input only by construction. When the field comes out fold-free — with no fold left to trap the inner — this stage tiles the MOVED region (`reanchor_tile` px, overlapping by 8), re-solves each tile against the distance-to-INPUT objective (`'l2'` or `'l1'`, `reanchor_maxiter` inner iterations, same constraint rows / frozen ring / paste-back as any window), and keeps a tile only if every enforced row stays at or above `threshold` AND the tile actually reduced the distance — otherwise it reverts. Up to `reanchor_sweeps` sweeps, stopping once a sweep buys < 1% of the L2 move. **The no-damage invariant is unaffected**: each tile's free set is intersected with the moved mask (`build_subproblem(..., free_extra=)`), so the stage only ever moves pixels the main solve already moved — the moved set can shrink, never grow — and those pixels and the rows they influence are inside `touched` by construction. A global re-check after the stage reverts the whole thing and logs if a fold appeared anyway (unreachable given the per-tile verification, but guarded). Off by default because fidelity is a concern separate from the zero-fold certificate. `report.reanchor_sweeps_run` / `reanchor_tiles` / `reanchor_accepted` / `reanchor_l2_before` / `reanchor_l2_after` record it. Measured on the raw B0039 z16 slice (bilinear rows, objective `none`, maxiter 600): `'l1'` takes the L2 move **268.4 -> 208.1 (-22%)** and the L1 move **24812 -> 12876 (-48%)** at 0 simplex folds and damage 0 (50 of 57 tiles accepted, 3 sweeps); `'l2'` -5% / -10% (16 of 38 tiles, 2 sweeps). **`'l1'` is the stronger re-anchor even measured in L2** — its Gauss-Newton diagonal floors at 0.1 against a unit gradient, so each SQP iteration proposes a steady sign-directed step the trust region caps, where L2's `2 d` gradient against a 2.0 diagonal proposes the whole jump to the reference and the constraints clip it hard. The stage roughly doubles-to-triples a slice's wall time — that, and the fact that a 0-fold field is already correct, is why it is off by default.

### Implementation modules (internal — strategies delegate to these)

The legacy `iterative_*` functions are no longer part of the public API but remain as internal implementations the strategies call into:

| Strategy | Delegates to |
|---|---|
| `NMVFStrategy` (Jdet 2D) | `dvfopt.core.nmvf.nmvf_correct_2d` |
| `SLPStrategy` (simplex (2D) + simplex (3D)) | 2D: `dvfopt.core.slp.cluster_slp_iter` (large) / `slp_iter` (small); 3D: `cluster_slp_iter_3d` / `slp_iter_3d`; auto-routes by pixel/voxel count |
| `BarrierStrategy` (any constraint) | `dvfopt.core.barrier._core.run_penalty_barrier_lbfgs` |
| `SLSQPFullGridStrategy` (simplex (2D)) | `dvfopt.core.slsqp_fullgrid.tri2d.iterative_2d_tri_slsqp` |
| `SLSQPFullGrid3DStrategy` (simplex (3D)) | `dvfopt.core.slsqp_fullgrid.tet3d.iterative_3d_tet_slsqp` |
| `SLSQPWindowedStrategy` (Jdet) | `dvfopt.core.slsqp_windowed.iterative.iterative_serial` / `.iterative3d.iterative_3d` |
| `ISQPWindowedStrategy` (Jdet / simplex (2D) / bilinear / finite, 2D) | `dvfopt.core.windowed.windowed_correct` (inner: `core.primitives.isqp`) |
| `SchwarzStrategy` (simplex (2D)) | `dvfopt.core.schwarz.tri2d.iterative_2d_tri_schwarz` |
| `HarmonicALMBarrierStrategy` (alias `M10Strategy`) | `dvfopt.core.wallbreakers.iterative_2d_tri_harmonic_polished` |
| `HarmonicALMRefineRepairStrategy` (alias `M14Strategy`) | `dvfopt.core.wallbreakers.iterative_2d_tri_refine_repair` |
| `SchwarzHarmonicALMRefineRepairStrategy` (alias `M14SchwarzStrategy`) | builds a pinned `HarmonicALMRefineRepairStrategy` inner from its own knobs, then calls `dvfopt.core.schwarz._common.cluster_schwarz_2d_tri` directly — the same core `SchwarzWrapperStrategy` uses |
| `SchwarzWrapperStrategy(inner=...)` | `dvfopt.core.schwarz._common.cluster_schwarz_2d_tri` / `cluster_schwarz_3d_tet` directly, calling `inner.solve` per cluster |
| `HarmonicALMBarrier3DStrategy` (alias `M10TetStrategy`) | `dvfopt.core.wallbreakers._alm_3d` (harmonic + ALM-3D + polish) |
| `HarmonicALMRefineRepair3DStrategy` (alias `M14TetStrategy`) | `dvfopt.core.wallbreakers._refine_repair_3d` |
| `SchwarzHarmonicALMRefineRepair3DStrategy` (alias `M14Schwarz3DStrategy`) | builds a pinned `HarmonicALMRefineRepair3DStrategy` inner from its own knobs, then calls `dvfopt.core.schwarz._common.cluster_schwarz_3d_tet` directly |

### Building blocks (still public, still useful for custom pipelines)

| Function | Module | Purpose |
|----------|--------|---------|
| `harmonic_extension_2d()` (m02) | `dvfopt.core.wallbreakers._harmonic` | Laplacian extension over fold cores |
| `augmented_lagrangian_2d()` (m03) | `dvfopt.core.wallbreakers._alm` | PHR-ALM with L-BFGS-B |
| `l2_refine_2d()` (m12) | `dvfopt.core.wallbreakers._l2_refine` | Soft-penalty refinement of a feasible seed |
| `solve_cluster_2tri_2d()` | `dvfopt.core.schwarz._cluster` | Per-cluster SLSQP with frozen-edge interior mask |
| `tri_areas_flat()` / `tri_grad_T_v()` | `dvfopt.core.primitives.tri` | Canonical simplex (2D) constraint + adjoint |
| `anchor_term()` | `dvfopt.objectives` | Shared anchor math behind `L1/L2/NoneObjective` |
| `run_penalty_barrier_lbfgs()` | `dvfopt.core.barrier._core` | Shared penalty→barrier homotopy engine |
| `cluster_schwarz_2d_tri()` / `cluster_schwarz_3d_tet()` | `dvfopt.core.schwarz._common` | Shared Schwarz decomposition engine |
| `windowed_correct()` | `dvfopt.core.windowed` | Shared cluster-windowed no-damage engine (frozen rings, no-TR retry then grow-on-failure, giant tiling, terminal mop) |
| `isqp_solve()` / `colored_jacobian()` | `dvfopt.core.primitives.isqp` / `.coloring` | Elastic-QP SQP inner (OSQP; `HAS_OSQP` gate) + CPR-coloring sparse Jacobians |
| `minimize_slsqp_traced()` / `ineq_dict()` | `dvfopt.core.primitives.slsqp` | Traced C-SLSQP driver (scipy's own core) + its old-style ineq-constraint dict helper — the single driver behind all 10 SLSQP call sites |
| `fold_stats()` / `constraint_fold_stats()` / `FoldStats` | `dvfopt.metrics` | Canonical fold statistics (n_neg / n_below / min / severity) shared by pipelines, CLI, reports |
| `load_dvf()` / `save_dvf()` | `dvfopt.io.fields` | Field I/O — `.npy`/`.npz` + NIfTI/MetaImage/NRRD (moved from `dvfopt_gui.io_formats`) |

### 2.5D marching (3D fold *prevention*)

`correct_dvf_25d()` ([dvfopt/pipeline_25d.py](dvfopt/pipeline_25d.py)) prevents
inter-layer simplex (3D) folds instead of repairing them. **Precondition: `dz ≡ 0`**
(i.e. the input is per-slice 2D-corrected) — the inter-layer simplex (3D) volume then
depends only on adjacent slices' `dy/dx`. The pipeline validates this and raises
if `dz ≠ 0`; it never writes `phi[0]`.

It auto-picks the mildest inter-layer as a frozen seed (no layer is cold-started
against raw data), sweeps outward in both directions repairing each slice against
its already-repaired neighbour (`march_slice`, elastic LP over the free plane's
interior with a frozen ring), then runs a frozen-rim 3D-interior mop
(`mop_interior_3d`) for folds that need *both* slices of a pair to move — which
the single-frozen-plane sweep structurally cannot fix.

On the full 528-slice B0039 volume this took the 3D fold count from **1,058,831 →
33** (99.997%). The residual ~33 are the **geometric floor** of the fixed-diagonal
simplex (3D) decomposition (no feasible tet split exists), not a solver limitation — an
exact-feasibility solver with escalating freedom cannot move them.

| Function | Module | Purpose |
|----------|--------|---------|
| `correct_dvf_25d()` / `Correct25DReport` | `dvfopt.pipeline_25d` | End-to-end 2.5D marching orchestrator |
| `march_slice()` / `layer_min_v()` | `dvfopt.core.marching` | Per-slice sweep repair + inter-layer min-volume |
| `mop_interior_3d()` | `dvfopt.core.marching` | Frozen-rim 3D-interior elastic-SLP residual mop |

**Other primitives:**

| Function | Module | Purpose |
|----------|--------|---------|
| `correct_dvf_3d()` / `Correct3DReport` | `dvfopt.pipeline_3d` | End-to-end true-3D fold-*repair* orchestrator (simplex (3D) feasibility); complements the 2.5D *prevention* pipeline above |
| `jacobian_det2D()` / `jacobian_det3D()` | `dvfopt.jacobian.numpy_jdet` | Fast numpy Jacobian determinant |
| `ift_radius_2d()` / `ift_radius_3d()` / `cell_min_jdet_2d()` | `dvfopt.jacobian.injectivity_radius` | Quantitative-IFT injectivity-radius *estimate* maps (tight second-difference stencils, windowed-Lipschitz ladder saturating at `max_window`; `max_window=0` = pointwise estimate; orientation-blind, NOT a certificate) + exact bilinear cell min-Jdet certificate (2D only — no trilinear analogue, simplex (3D) covers 3D). References + caveats in the module docstring |
| `injectivity_stats()` / `InjectivityStats` | `dvfopt.metrics` | Sub-pixel injectivity diagnostics over those maps (also `dvfopt info --ift`; with `--check`, 2D bilinear-folded cells exit 1) |
| `solveLaplacianFromCorrespondences()` | `dvfopt.laplacian.solver` | Build DVF from correspondences |
| `sliceToSlice3DLaplacian()` | `dvfopt.laplacian.correspondence` | Full slice-to-slice Laplacian registration pipeline |
| `make_deformation()` / `make_random_dvf()` / `SYNTHETIC_CASES` | `dvfopt.testdata` | Generate test deformation fields (`from dvfopt.testdata import ...`) |

### Directory layout

- `dvfopt/` — the installable package; **one package, no sibling top-level packages** since 0.5.0. `core/` is method-first: one sub-package per algorithm family — `primitives/` (shared constraint math + the traced SLSQP driver, zero method logic), `nmvf/`, `barrier/` (`_core.py` is the shared homotopy engine), `slsqp_windowed/`, `slsqp_fullgrid/`, `schwarz/` (`_common.py` is the shared decomposition engine), `wallbreakers/`, `slp/`, `marching/`. Alongside `core/`: `jacobian/`, `dvf/`, `viz/`, `io/`, `utils/`, plus the absorbed `laplacian/` and `testdata/`.
- `dvfopt_gui/` — PyQtGraph live-solver GUI (`app.py` + the `LiveSolverWindow` mixins `_win_fileio.py`/`_win_render.py`/`_win_run.py` and shared helpers `_shared.py`, plus `worker.py`, `convergence.py`, `history.py`, `persistence.py`, `demo.py`, `overview.py`, `strategy_params.py`, `logdock.py`; displacement-field I/O now lives in the library at [dvfopt/io/fields.py](dvfopt/io/fields.py) — the GUI imports the `*_sitk` loaders from there). The GUI also supports a **true-3D mode**: load a `(3, D, H, W)` volume and pick the `simplex (3D) (3D)` or `Jdet (3D)` constraint to solve the whole volume with the 3D pipelines (M14Tet/M14-Schwarz3D/M10Tet/SLSQP-fullgrid-3D, or Barrier/SLSQP-windowed for Jdet3D). 3D wallbreaker runs stream per-phase snapshots and honor Stop at phase boundaries; the viewer renders the simplex (3D) min-volume slice of the current z. The method menu now includes **SLP (default simplex (2D) champion; also in the tet3d menu as SLP-3D)** and an **Auto** picker (`auto_strategy`, available for the 2D families AND the 3D constraints); each explicit method-menu entry constructs through the dvfopt strategy registry (`worker._MID_TO_LABEL` → `make_strategy`, parity-tested against the menus in [tests/test_gui_strategy_parity.py](tests/test_gui_strategy_parity.py) — a new strategy needs a registry label + a menu-spec row + a table row); the **Pipeline ▾** button runs `correct_dvf_25d` (2.5D marching, needs dz≡0 — a violation prompts an explicit, undoable consent dialog to zero the dz channel before running) or the one-click **full pipeline** (per-slice 2D → 2.5D). The tet3d menu adds the **full 3D pipeline** (`correct_dvf_3d`) and a torch-gated GPU barrier. Loads accept NIfTI/MetaImage/NRRD displacement fields via SimpleITK (and export back to `.npy`/`.nii.gz`); loads are threaded and reject non-finite fields. The feasibility threshold is editable (`thr:` spinbox), 3D metrics are cached (fast z-scrub/hover), the undo stack is byte-budgeted, a clickable per-slice fold strip sits under the plot, every strategy's dataclass knobs — spanning the 2D, tet3d, and jdet3d families — are editable via Params → Strategy, and "Run section" works on 3D sub-volumes (Rect ROI + z-range). Solver-path runs record their SolveInfo: the convergence chart marks pipeline-stage boundaries (stage names ride on the history snapshots and survive save/load), View → "Save convergence report…" renders `plot_solve_info`, and a View → "Solver log" dock streams the dvfopt logger live (its level drives the worker's `verbose`). An **Injectivity gap (min axial)** view mode renders the monotonicity-gap map in 2D and 3D, and the Params dialog renders `float | None` knobs (e.g. `injectivity_threshold`) as checkbox-enabled overrides — the 2D windowed method exposes exactly its constraint-mode toggles.
- `dvfopt/laplacian/` — Laplacian interpolation (matrix construction, CG/LGMRES solvers, contour correspondence matching). Was the top-level `laplacian/` package before 0.5.0 — import as `from dvfopt.laplacian import ...`.
- `dvfopt/testdata/` — test case definitions and builders (synthetic, random DVF, real-data slices). Was the top-level `test_cases/` package before 0.5.0 — import as `from dvfopt.testdata import ...`.
- `notebooks/` — canonical experiment notebooks
- `benchmarks/` — performance comparison notebooks, grouped into subfolders:
  - `solvers/slsqp/` — SLSQP windowed solver comparisons (serial vs parallel, constraint modes, windowed vs fullgrid, 3D correction)
  - `solvers/barrier/` — penalty/barrier L-BFGS solver (3D barrier, CPU vs GPU)
  - `scaling/` — performance vs grid size, folding severity, L2-Jdet correlation
  - `registration/` — external registration methods (Elastix, VoxelMorph, TransMorph, ANTs, OpenCV) + post-hoc correction
  - `pipelines/` — end-to-end 3D slice-wise correction pipelines
  - `benchmark_utils.py` — shared helpers (plotting, metrics, run-dir I/O) + the brain-cohort loader (`list_cohort`, `load_cohort_field`, `load_cohort_section`, `load_cohort_correspondences`); notebooks add `..` to sys.path to import it
  - `cohort_benchmark.py` — folding benchmark over the in-repo brain cohort (`data/dvfs/brain25_cohort_corrected/`, gitignored). `run_cohort_2d_sections` (isolated 2D z-slices, `n_workers` for parallel solves) and `run_cohort_benchmark` (whole-field) each write a timestamped run dir (`results.csv`, `summary.json`, `figures/`, `report.html`) with before→after Jdet / simplex (2D) / simplex (3D) fold metrics; `interactive=True` emits the interactive report instead of static figures
  - `interactive_report.py` — self-contained interactive HTML report: per-field pan/zoom Jacobian-map `<canvas>` (hover exact Jdet + dy/dx, before/after toggle, displacement-vector overlay, correspondence overlay), severity-ranked fold-cluster ROI table (click-to-locate) — base64 float arrays + vanilla JS, no external assets
  - `correspondence_analysis.py` — per-slice Laplacian boundary-condition diagnostics: prescribed displacement, registration residual (fit) before/after correction, and outlier flags (large-disp / high-residual / incoherent). Convention (verified): `field[:, fixed] == moving − fixed` (backward map)
- `scripts/` — image generation scripts for docs
- `data/` — real data NIfTI files and `.npy` test case arrays (all gitignored). Includes `dvfs/brain25_cohort_corrected/<brain>/<variant>/` — 7 real brains (Laplacian field + ANTs warp + correspondences) copied from the sibling RegTools project, consumed by `benchmarks/cohort_benchmark.py`
- `archive/` — historical notebooks (not canonical)
