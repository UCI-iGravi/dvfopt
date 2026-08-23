# Library Architecture Unification — Design

**Date:** 2026-08-22
**Branch:** `refactor/library-architecture` (worktree, branched from `origin/main` @ 6f27dc7, post-PR #64)
**Status:** approved approach (A: method-first core), spec for review

## Goal

Reorganize the repo into one clean, importable library with three honest axes
(constraints / objectives / strategies) and a `core/` where **one solver method =
one package**, so a new method, constraint, or objective drops in without
touching unrelated code. Clean break: no shims; all in-repo consumers updated in
the same PR.

## Decisions (locked with user)

1. **Clean break** — move/rename freely; update `dvfopt_gui/`, `benchmarks/`,
   `tests/`, `scripts/`, `asv_bench/`, canonical `notebooks/` in the same PR.
   `research/` and `archive/` are frozen provenance (never edited; they pin the
   old layout in git history).
2. **One package** — absorb top-level `laplacian/` → `dvfopt.laplacian` and
   `test_cases/` → `dvfopt.testdata`. `dvfopt_gui/` stays a separate top-level
   app package.
3. **Objectives made real** — strategies pass `Objective` objects down; the
   engines call `objective(diff)`. No more `anchor=objective.label,
   eps_l1=getattr(...)` unwrapping.
4. **Approach A** — method-first `core/`; polymorphism only where the math is
   generic (barrier/ALM engine, constraints); specialized champions stay
   specialized behind the uniform `Strategy.solve` contract. No rewrite of
   validated numerics.
5. **Traced SLSQP driver** — `benchmarks/slsqp_traced.py` is promoted into the
   library and replaces every `scipy.optimize.minimize(method='SLSQP')` call
   site (byte-identical with `trace=None`; pyslsqp-grade tracing when recording
   history).
6. **Delivery** — executed in this worktree, subagent-driven, PR'd, review
   findings fixed pre-merge, squash-merge on green.

## Target tree

```
dvfopt/
├── constraints.py          # unchanged role; imports flow ONLY to core/primitives
├── objectives.py           # real Objective axis (see §Objectives)
├── strategies/             # thin adapters + registry; + accepts_objectives attr
├── solver.py  unified.py  pipeline_25d.py  pipeline_3d.py
├── core/
│   ├── primitives/         # shared math + engines-without-method-logic
│   │   ├── tri.py
│   │   ├── jdet2d.py
│   │   ├── jdet3d.py
│   │   ├── constraint_values.py
│   │   └── slsqp.py        # vendored traced C-SLSQP driver
│   ├── nmvf/
│   ├── barrier/            # generic penalty→barrier engine + per-family drivers
│   ├── slsqp_windowed/     # the windowed Jdet family (2D+3D) incl. coordinators
│   ├── slsqp_fullgrid/     # full-grid 2-tri + 6-tet SLSQP
│   ├── schwarz/            # Schwarz decomposition engine + 2-tri driver + cluster solver
│   ├── wallbreakers/       # m02/m03/m10/m12/m14 pipelines, 2D + 3D
│   ├── slp/                # unchanged
│   ├── marching/           # unchanged
│   └── _pool.py            # shared process-pool infra
├── laplacian/              # ← top-level laplacian/
├── testdata/               # ← top-level test_cases/
└── jacobian/ io/ viz/ dvf/ utils/ metrics.py validation.py exceptions.py
    cli.py _defaults.py _logging.py _plots.py
```

### Structural rules the tree encodes

* **One method = one package**, named for the method. No flat
  `iterative2d_tri_*.py` siblings; no package whose name says less than it
  contains (`slsqp/` today actually means "windowed Jdet").
* **Dependency direction:** `constraints.py` → `core/primitives` only (today it
  reaches into three method modules for private helpers — fixed by the
  primitive extractions below). Method packages → `primitives` + at most
  another method's *engine* (`wallbreakers` → `barrier._core` and
  `schwarz._common` are legitimate: those pipelines really run a barrier polish
  and a Schwarz decomposition).
* Strategies remain thin adapters: `strategies/<name>.py` ↔ `core/<name>/`.

## Module mapping (complete)

| New location | From | Notes |
|---|---|---|
| `core/primitives/tri.py` | `core/tri_primitives.py` + `_build_full_grid_tri_jac` (from `core/iterative2d_tri_slsqp.py`) | canonical 2-tri values/adjoint/jacobian; numba fast-path unchanged |
| `core/primitives/jdet2d.py` | `_jdet_2d_flat`, `_jdet_grad_T_v_2d` extracted from `core/iterative2d_barrier.py` | so `JdetConstraint2D` stops importing a solver module |
| `core/primitives/jdet3d.py` | `core/barrier_objective.py` | rename kills the "objective" misnomer (it's constraint math) |
| `core/primitives/constraint_values.py` | `core/_internal/constraint_values.py` | generic per-cell constraint maps (used by `_plots`, `unified`, GUI) |
| `core/primitives/slsqp.py` | `benchmarks/slsqp_traced.py` | see §Traced SLSQP driver |
| `core/nmvf/__init__.py` | `core/_nmvf.py` | |
| `core/barrier/_core.py` | `core/_barrier_core.py` | generic penalty→barrier homotopy engine; gains `objective=` (see §Objectives) |
| `core/barrier/jdet2d.py` | `core/iterative2d_barrier.py` | minus the extracted primitives |
| `core/barrier/jdet3d.py` | `core/iterative3d_barrier.py` | |
| `core/barrier/jdet3d_torch.py` | `core/iterative3d_barrier_torch.py` | |
| `core/barrier/tri2d.py` | `core/iterative2d_tri_barrier.py` | |
| `core/barrier/tet3d_torch.py` | `core/iterative3d_tet_barrier_torch.py` | |
| `core/slsqp_windowed/{constraints,constraints3d,gradients,gradients3d,spatial,spatial3d,iterative,iterative3d,parallel,_grad_op}.py` | `core/slsqp/*` (same names) | |
| `core/slsqp_windowed/coordinator.py` | `core/solver.py` | renamed to say what it is (serial per-pixel fix loop + adaptive outer) |
| `core/slsqp_windowed/coordinator3d.py` | `core/solver3d.py` | |
| `core/slsqp_windowed/{_io,_metrics,_window}.py` | `core/_internal/{io,metrics,window}.py` | windowed-family private helpers come home; `core/_internal/` dissolves |
| `core/slsqp_fullgrid/tri2d.py` | `core/iterative2d_tri_slsqp.py` | minus the tri-jac builder |
| `core/slsqp_fullgrid/tet3d.py` | `core/iterative3d_tet_slsqp.py` | |
| `core/schwarz/tri2d.py` | `core/iterative2d_tri_schwarz.py` | |
| `core/schwarz/_common.py` | `core/wallbreakers/_schwarz_common.py` | one Schwarz home; wallbreaker Schwarz variants + `SchwarzWrapperStrategy` import from here |
| `core/schwarz/_cluster.py` | `core/_cluster_2tri.py` | only consumer is the Schwarz path |
| `core/wallbreakers/` | unchanged minus `_schwarz_common.py` | |
| `core/slp/`, `core/marching/`, `core/_pool.py` | unchanged | |
| `dvfopt/laplacian/{solver,correspondence,utils}.py` | `laplacian/` | public exports unchanged |
| `dvfopt/testdata/` | `test_cases/` | `from dvfopt.testdata import SYNTHETIC_CASES, load_slice, make_deformation, ...` |

**Deleted:** `core/objective.py` (`objective_euc` ≡ `anchor_term(d,'l2')` ≡
`L2Objective.__call__`; call sites use the objective object), `core/_internal/`
(dissolved), top-level `laplacian/` and `test_cases/`, `benchmarks/slsqp_traced.py`
(promoted; benchmarks import from the library).

## Objectives made real

* `Objective.__call__(diff) -> (value, grad)` becomes the consumed contract.
  `anchor_term` survives in `core/barrier/_core.py` as the private
  implementation the three built-ins call.
* `core/barrier/_core.py` engine functions take `objective: Objective` instead
  of `(anchor: str, eps_l1: float)`. All ~15 unwrap sites across
  `strategies/` + `wallbreakers/` + barrier drivers become `objective=objective`
  pass-throughs (default `L2Objective()` at impl level for direct callers).
* Windowed SLSQP's per-window subproblem calls `objective(diff)` (its
  `minimize(..., jac=True)` needs exactly `(value, grad)`). Default stays L2 —
  behavior unchanged — but the "your objective is ignored" `UserWarning` is
  replaced by honest plumbing.
* **New `Strategy.accepts_objectives: tuple[type, ...] | None`** class attr,
  symmetric with `accepts_constraints`, checked in `Solver.__init__`.
  `SLPStrategy` declares `(L1Objective, NoneObjective)` (its LP is structurally
  L1). Construction-time `IncompatibleObjectiveError` (new, sibling of
  `IncompatibleConstraintError`) replaces runtime warnings.
* Extension: a new objective = one `Objective` subclass; works everywhere the
  generic engines run; no registry or dispatch edits. `make_objective` keeps
  accepting `'l1'/'l2'/'none'` strings for config paths.

## Traced SLSQP driver

`core/primitives/slsqp.py` (ex `benchmarks/slsqp_traced.py`): vendored
scipy ≥ 1.15 SLSQP driver loop over `scipy.optimize._slsqplib.slsqp` (C core).
`trace=None` ⇒ arithmetic path identical to `minimize(method='SLSQP')`;
`trace={}` ⇒ per-major-iteration records (objective, max violation, KKT
optimality, step α, line-search count, BFGS resets, inconsistent-QP flag,
nfev/ngev, multipliers). Parity vs real pyslsqp is established by
`benchmarks/trace_parity_check.py` (stays in benchmarks; needs the optional
py ≤ 3.12 pyslsqp env).

All ten in-library scipy-SLSQP call sites swap to it:

| Call site (old path) | Sites |
|---|---|
| `core/iterative2d_tri_slsqp.py` → `slsqp_fullgrid/tri2d.py` | 2 |
| `core/iterative3d_tet_slsqp.py` → `slsqp_fullgrid/tet3d.py` | 1 |
| `core/_cluster_2tri.py` → `schwarz/_cluster.py` | 2 |
| `core/wallbreakers/_coupled_kring_3d.py` | 1 |
| `core/_internal/window.py` + `core/solver3d.py` → `slsqp_windowed/` | 4 |

Sites building `NonlinearConstraint` switch to the driver's old-style dict
constraints directly (all have analytic jacs in hand, which the driver
requires; no dependency on scipy's private `new_constraint_to_old`).
The wallbreakers' L-BFGS-B `minimize` calls are not SLSQP and are untouched.

**`method_name` knob preserved:** the windowed family publicly accepts
`method_name="SLSQP"` (default). The traced driver is used when
`method_name == 'SLSQP'`; any other value falls back to plain
`scipy.optimize.minimize(method=method_name)` unchanged.

**Tracing → SolveInfo:** with `record_history=True`, SLSQP-family strategies
thread a trace dict into the driver; records land in `SolveInfo`
(phases/extras) so `plot_solve_info`, the convergence report, and the GUI
convergence dock get real solver internals for the SLSQP families, matching
what the barrier family already reports. `record_history=False` ⇒ `trace=None`
⇒ zero overhead, scipy-identical.

## Consumer migration (clean break)

Import rewrites (mechanical, no logic changes) in: `dvfopt_gui/`,
`benchmarks/` (~30 modules incl. `windowed_isqp.py`, `slsqp_variants.py`,
`trace_parity_check.py` importing the driver from the library),
`tests/` (85 modules), `scripts/`, `asv_bench/`, canonical `notebooks/`
(nbformat-aware rewrite for `.ipynb`). `from test_cases import …` →
`from dvfopt.testdata import …`; `from laplacian import …` →
`from dvfopt.laplacian import …`. `research/`, `archive/`,
`notebooks/experiments/` scratch scripts: untouched.

## Packaging / CI / tooling

* `pyproject.toml`: `packages.find.include = ["dvfopt*", "dvfopt_gui*"]`;
  version → **0.5.0**; `scipy>=1.15,<1.19` (vendored-internals contract; the
  driver already raises loudly outside it).
* mypy scope paths + ruff target paths updated for moved modules; both CI
  workflows (now uv-based, post-#64) updated for renamed paths/commands.
* `CHANGELOG.md`: 0.5.0 entry with an old→new import mapping table.
* `CLAUDE.md`: rewritten to match the new layout; the false "objective
  composition (`+`, `*`)" claim is removed.
* New **`ARCHITECTURE.md`** (repo root): the three axes, the dependency rules,
  and three checklists — *add a method* (create `core/<name>/`; add
  `strategies/<name>.py` dataclass with `@register_strategy` + the three compat
  attrs; one test module; if GUI-exposed: registry label + menu row, enforced
  by `tests/test_gui_strategy_parity.py`), *add a constraint*, *add an
  objective*.

## Verification gates (per phase and final)

* Full suite: `pytest -n auto` (85 modules), `ruff check`,
  `ruff format --check`, `mypy`, GUI strategy-parity test, installed-CLI smoke.
* **New driver byte-identity test:** `trace=None` vs
  `scipy.optimize.minimize(method='SLSQP')` on a fixed windowed case — same
  final x, nit, exit mode. CI canary for future scipy bumps.
* Explicit risk checks: `ProcessPoolExecutor` pickling of moved functions
  (`slsqp_windowed/parallel.py`, `_pool.py` — spawn resolves by qualified
  name); import-cycle check at the constraints ↔ primitives seam
  (`python -c "import dvfopt"` in a fresh interpreter per phase).

## Execution phases (each gated on the full suite)

1. Primitive extractions + `core/primitives/` (incl. driver promotion).
2. Method packages (moves/renames; `core/_internal` dissolution).
3. Traced-driver swap at all ten call sites + byte-identity test.
4. Objectives plumbing + `accepts_objectives`.
5. Absorb `laplacian/` + `test_cases/`; pyproject/packaging.
6. Consumer import rewrites (GUI, benchmarks, tests, scripts, asv, notebooks).
7. Docs (ARCHITECTURE.md, CLAUDE.md, CHANGELOG) + final full gate.

## Out of scope

* Any numerical/behavioral change to solvers (beyond the plumbed objective
  defaults, which are chosen to preserve current behavior).
* Rewriting specialized solvers to be constraint-polymorphic (rejected
  Approach B).
* `research/`, `archive/`, GUI feature work, notebook content beyond imports.
