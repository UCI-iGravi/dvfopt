# Changelog

Tracks user-visible changes to `dvfopt`. Format inspired by
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`dvfopt.jacobian.tetrahedron_sign_torch`** — torch forward for the
  6-tet signed-volume check
  ([dvfopt/jacobian/tetrahedron_sign_torch.py](dvfopt/jacobian/tetrahedron_sign_torch.py)).
  Bit-exact parity with the numpy forward; autograd through it matches
  the analytical adjoint to 4e-16. Building block for a future
  GPU-accelerated barrier-on-tet path; the full windowed barrier
  integration (mirroring
  [iterative3d_barrier_torch.py](dvfopt/core/iterative3d_barrier_torch.py))
  is deferred — torch is in the `[benchmarks]` extra, not core.

### Changed
- **`_compute_constraint_2d` consolidated.** Two near-duplicate copies
  (in [`dvfopt/_plots.py`](dvfopt/_plots.py) and
  [`dvfopt/unified.py`](dvfopt/unified.py)) now share a single helper in
  [`dvfopt/core/_internal/constraint_values.py`](dvfopt/core/_internal/constraint_values.py).
  An `include_patches` flag selects the right behavior per call site
  (plotting code wants `False` to keep the reshape happy; stats code
  wants `True` to match what the solver sees).
- **Type hints added** on the new tet primitives
  (`six_tet_volumes_3d`, `six_tet_fold_classification`,
  `tet_volumes_flat`, `tet_grad_T_v`) and the viz overview functions
  (`plot_fold_overview`, `plot_fold_overview_3d`, `plot_before_after`,
  `plot_before_after_3d`, `plot_solver_comparison`, `jdet_norm`).
- **Lint clean.** Full `ruff check dvfopt/ tests/` pass with 0 errors.
- **Tet primitives now re-exported from `dvfopt.jacobian`**:
  `six_tet_volumes_3d`, `six_tet_fold_classification`,
  `tet_volumes_flat`, `tet_grad_T_v`.

### Fixed
- **CI lint failure in [PR #10](https://github.com/UCI-iGravi/dvfopt/pull/10).**
  `ruff check` failed on Ubuntu (Python 3.10/3.11/3.12) with
  `RUF100: Unused noqa directive (non-enabled: F401)` at
  [tests/test_tetrahedron_sign.py:110](tests/test_tetrahedron_sign.py#L110).
  Replaced the `try/except ImportError + # noqa: F401` pattern with
  `pytest.importorskip('torch')` — same semantics, no noqa needed.

  Root cause: local runs were `ruff check dvfopt/ tests/`; CI runs
  `ruff check dvfopt tests benchmarks` (note the third directory).
  The unused-noqa rule's interaction with the active rule set differed
  enough that the directive was flagged on CI but not locally.

  **New helper** to prevent recurrence:
  [scripts/check_ci.py](scripts/check_ci.py) replays the CI workflow
  steps locally (ruff check + ruff format check + benchmark py_compile
  smoke + pytest). README has a "Development" section pointing at it.

- **`plot_step_snapshot` + `plot_deformed_quads_colored` theme conflict**
  closed — both functions now use `fig.colorbar(im, ax=ax)` + drop the
  manual `tight_layout`/`plt.show` calls. The 2 xfail markers in
  [tests/test_viz_smoke.py](tests/test_viz_smoke.py) are removed.
- **`plot_problematic_triangles` theme warning** silenced — dropped
  the `fig.tight_layout()` call that conflicted with the theme's
  `constrained_layout=True` default.
- **`auto_strategy` for `Tet6Constraint3D`** — added explicit tet-family
  branch that always returns `'barrier'` (the only strategy that
  currently supports tet). Previously fell through to
  `'slsqp_windowed'` which doesn't accept tet constraints and crashed.

## [0.3.0] — 2026-05-29

### Changed (breaking)
- **`'2tri'` now resolves to `TriConstraint2DFullCoverage`**, not
  `TriConstraint2D`. The standard TR-BL split leaves the two
  diagonally-opposite grid corners `(0, 0)` and `(H-1, W-1)` in only
  one triangle each — an asymmetric coverage gap. The full-coverage
  variant adds two opposite-diagonal corner patches at those cells so
  every grid vertex is in ≥ 2 triangles. The patches are 2 scalar
  constraints + 6 gradient terms total — measurable in microseconds.

  - **What changed**: the registry alias `'2tri'` resolves to the
    full-coverage class; the previous standard behavior is preserved
    as `'2tri_standard'`.
  - **`'2tri_full'` removed.** It was a transitional alias for the
    full-coverage class; with `'2tri'` now being full-coverage, it's
    redundant. Existing callers using `'2tri_full'` should switch to
    `'2tri'`. The internal back-compat guards in `_plots.py` and
    `unified.py` were also removed.
  - **Migration**: most code keeps working — `correct_dvf(..., constraint='2tri', ...)`
    and `DVFopt(constraint='2tri', ...)` continue to compile and run,
    just with 2 extra constraints. To exactly reproduce numbers from
    benchmarks recorded before this change, switch to
    `constraint='2tri_standard'`.
  - **What might shift**: L-BFGS-B / SLSQP iteration paths can differ
    slightly because the multiplier vector grew by 2 entries.
    Convergence quality and feasibility verdict are unchanged in
    practice (the corner patches are virtually never folded in real
    data).
  - The class names `TriConstraint2D` and `TriConstraint2DFullCoverage`
    are unchanged — only the registry mapping flipped.

### Added
- **`dvfopt.viz.theme`** — central matplotlib + seaborn theme
  ([dvfopt/viz/theme.py](dvfopt/viz/theme.py)). Single source of
  truth for fonts/spines/dpi/colormaps; previously each plot
  function set its own ad-hoc style. Public:

  - `apply_theme(context='paper')` — idempotent, applies seaborn
    `ticks` style + paper context + custom rcParams (dpi=130,
    no top/right spines, `RdBu_r` default cmap, `savefig.dpi=200`).
  - `reset_theme()` — restore matplotlib defaults.
  - `PALETTE` / `Palette` — curated palette with semantic colors
    (`fold`, `feasible`, `anchor`, `grid_warp`, `grid_ref`) and
    canonical colormaps (`cmap_jdet='RdBu_r'`,
    `cmap_severity='YlOrRd'`, `cmap_magnitude='magma'`).
  - `jdet_norm(jdet_arrays, threshold=0.01)` — TwoSlopeNorm
    builder centered on 0 for diverging Jdet panels.

- **`dvfopt.viz.overview`** — high-impact "money shot" plots
  ([dvfopt/viz/overview.py](dvfopt/viz/overview.py)):

  - `plot_fold_overview(phi)` — 4-panel figure for a folded 2D DVF:
    Jdet heatmap with fold contour, warped grid with **per-triangle
    fold classification** (TR-BL diagonal drawn, T1/T2 shaded
    orange for single-flip / deep red for both-flipped), Jdet
    distribution with threshold marker, per-row/column fold
    counts.
  - `plot_before_after(phi_before, phi_after)` — side-by-side
    Jdet panels with a shared norm + a correction-magnitude
    panel.
  - `plot_solver_comparison(phi_in, results={'slsqp': ..., ...})`
    — N+1 panel comparison of solvers on the same input,
    shared norm.
  - `plot_fold_overview_3d(phi)` — 3D analogue: 3D Jdet scatter,
    worst-z slice heatmap, Jdet distribution with embedded
    **per-voxel tet-flip histogram** (0-6 tets flipped per voxel
    cell, using the new 6-tet decomposition), and per-axis fold
    projections (folds vs Z / Y / X).
  - `plot_before_after_3d(phi_before, phi_after)` — pair of
    3D Jdet scatters with a shared norm.

  All accept the same input layouts as the validator and apply
  the theme automatically; all accept `save_path=...`.

- **`dvfopt.jacobian.tetrahedron_sign`** — 6-tetrahedron signed
  volumes per voxel ([dvfopt/jacobian/tetrahedron_sign.py](dvfopt/jacobian/tetrahedron_sign.py)).
  Decomposes each voxel cell into 6 tetrahedra sharing the main
  diagonal `C0`→`C7`; identity field yields exactly `+1/6` per
  tet, `+1.0` total volume. Public:

  - `six_tet_volumes_3d(phi)` → `(6, D-1, H-1, W-1)` signed
    volumes per tet.
  - `six_tet_fold_classification(phi)` → `(D-1, H-1, W-1)`
    int8 count of flipped tets per voxel cell.
  - `tet_volumes_flat(phi_flat, D, H, W)` — flat-pack form for
    the constraint system.
  - `tet_grad_T_v(phi_flat, D, H, W, v)` — analytical
    `J^T @ v` adjoint via the cross-product form
    `V = sgn * (1/6) * (B-A) · ((C-A) × (D-A))`. Verified to
    1e-10 against central-difference gradient.

- **`Tet6Constraint3D`** — 3D analogue of `TriConstraint2D`
  ([dvfopt/constraints.py](dvfopt/constraints.py)). Enforces every
  per-tet signed volume ≥ threshold; smoother than the per-voxel
  Jdet constraint at fold boundaries. Phi pack `[dx, dy, dz]` (DX_FIRST,
  shared with `JdetConstraint3D`). Registered as `'6tet'` /
  `'6tet_3d'`. Works end-to-end with `BarrierStrategy` — small
  folded fields are feasibilised through the standard penalty →
  log-barrier homotopy.

- **2-triangle primitive home moved.** The flat-pack
  `tri_areas_flat` / `tri_grad_T_v` / *_full_coverage variants
  now live in [dvfopt/core/tri_primitives.py](dvfopt/core/tri_primitives.py)
  (matching its docstring). Old underscore-prefixed names in
  `iterative2d_tri_barrier.py` are aliases for back-compat —
  the 16 callers that imported them keep working without
  changes.

- **Existing 3D viz now uses the theme.** All six functions in
  [dvfopt/viz/fields3d.py](dvfopt/viz/fields3d.py)
  (`plot_jdet_slices`, `plot_jdet_3d`, `plot_jdet_3d_before_after`,
  `plot_neg_voxels_before_after`, `plot_deformation_grid_3d`,
  `plot_grid_before_after_3d`) call `apply_theme()` and use the
  theme's `RdBu_r` cmap + `PALETTE` colors. The
  `constrained_layout` warnings from `subplots_adjust` calls are
  gone.

- **`seaborn`** added as a runtime dependency in
  [pyproject.toml](pyproject.toml).

- **`dvfopt.validation`** — single-source-of-truth input validation
  ([dvfopt/validation.py](dvfopt/validation.py)). Every entry point
  (`Solver.fit`, `DVFopt.fit`, `correct_dvf`, `Constraint.coerce`)
  routes user input through `validate_dvf()`. Public helpers:
  `validate_dvf`, `validate_finite`, `validate_spatial_min_size`,
  `coerce_to_ndarray`.

### Fixed
- **Input handling is now graceful at the boundary**, not 5 frames
  deep:
  - Lists / tuples / array-likes are accepted (auto-`asarray`).
  - `(2, 1, H, W)` and `(3, 1, H, W)` singleton-D layouts accepted.
  - NaN/Inf rejected with a count, before the solver starts.
  - Sub-minimum spatial sizes (H/W < 3, zero-size axes) rejected
    with an actionable message naming the bad axis.
  - Wrong channel counts (`(4, H, W)`, `(H, W)`) rejected with a
    list of accepted layouts.
  - All shape/finite errors raise `SolverConfigError` / `ValueError`
    (never raw numpy errors).
- `int16` / `float32` inputs silently up-promote to `float64` (was
  already true; now documented).
- `DVFopt.fit` defensively copies — the input array is guaranteed
  not to be mutated.

### Changed
- **`dvfopt/strategies.py` split into a `dvfopt/strategies/` subpackage**
  with one file per strategy:

  - [`base.py`](dvfopt/strategies/base.py) — `Strategy` ABC, registry,
    `_build_solve_info` helper
  - [`barrier.py`](dvfopt/strategies/barrier.py) — `BarrierStrategy`
  - [`slsqp.py`](dvfopt/strategies/slsqp.py) — `SLSQPFullGridStrategy`,
    `SLSQPWindowedStrategy`
  - [`schwarz.py`](dvfopt/strategies/schwarz.py) — `SchwarzStrategy`
  - [`wallbreakers.py`](dvfopt/strategies/wallbreakers.py) — `M10Strategy`,
    `M14Strategy`, `M14SchwarzStrategy`

  Public imports are unchanged: `from dvfopt import BarrierStrategy`
  and `from dvfopt.strategies import BarrierStrategy` both work
  via re-export. Strategies register themselves via
  `@register_strategy('label')` at module import time.
- **Private utilities moved under `dvfopt/core/_internal/`**:
  `_io.py`, `_metrics.py`, `_window.py` (the windowed-SLSQP loop's
  internal helpers) now live at
  `dvfopt/core/_internal/{io,metrics,window}.py`. Signals "do not
  import from user code" more strongly than a single underscore.
  `dvfopt.core.solver` still re-exports them for back-compat.

### Deferred (honest)
- **Pulling `iterative_*.py` algorithm bodies into Strategy classes**
  is genuinely multi-day. Each legacy implementation file has 11+
  test / notebook / benchmark importers; cleaning that migration is
  not "polish." The current 2-layer split (Strategy → function) costs
  ~30–50 lines of indirection per strategy and hasn't actively caused
  problems.
- **Full `dvfopt/core/` reorganization** into
  `_math/_loop/_solvers/` subpackages is similarly deferred. Current
  layout (private modules underscore-prefixed, public modules at the
  top level) is readable and stable. The aesthetic gain from deeper
  subpackaging doesn't justify the import-path churn across the test
  / notebook / benchmark fleet.

- **Strategies build `SolveInfo` directly** via the new
  ``_build_solve_info`` helper in :mod:`dvfopt.strategies`. The
  back-compat normalization in :meth:`Solver.fit` still exists but is
  rarely hit — external strategies that haven't migrated continue to
  work transparently.
- Type hints on `Objective` composition classes (`SumObjective`,
  `ScaledObjective`). The remainder of the public surface
  (`Constraint`, `Strategy`, `Solver`, `SolveResult`, `SolveInfo`,
  `PhaseInfo`) is now fully annotated.
- `__all__` added to `dvfopt/unified.py`. CI YAML quotes the `"on":`
  key to avoid YAML-1.1 → bool coercion (cosmetic; GitHub Actions
  handled it either way).

### Added
- **Exception hierarchy** ([dvfopt/exceptions.py](dvfopt/exceptions.py)):
  `DVFoptError` (base), `SolverConfigError` (sub-`ValueError`),
  `IncompatibleConstraintError` (sub-`TypeError`), `FeasibilityError`,
  `BudgetExhaustedError` (sub-`FeasibilityError`). Existing `except
  ValueError` / `except TypeError` handlers continue to work; user
  code can now catch DVFopt-specific failures uniformly.
- **Package logger** at `dvfopt.logger` plus
  `dvfopt.enable_default_handler()`. All solver progress can be
  routed through Python's standard `logging` module; callers control
  verbosity via the normal logging API.
- `SolveInfo.from_legacy_history()` adapter — every strategy's
  free-form `info` dict is normalized into a populated `SolveInfo`
  with `phases: list[PhaseInfo]`. The contract is now used in earnest,
  not just declared.
- Top-level `dvfopt._plots` module (visualization helpers extracted
  from `unified.py`). Matplotlib stays out of `unified.py`'s import
  graph until a plot is actually called.
- `benchmarks/_run_canonical_2tri_suite.py` — demonstrates the
  declarative `BenchmarkSuite` workflow as the migration path from
  hand-written `_run_*.py` scripts.
- New test files: `test_solve_info_and_exceptions.py` (10 tests) and
  `test_logging_setup.py` (5 tests).

### Changed
- `Solver.fit()` now normalizes every strategy's `info` return value
  into a `SolveInfo` instance via a new `_normalize_info` helper.
  Strategies that produce list-of-dicts, dict-with-history, or
  stage-keyed dicts all converge on the same `SolveInfo.phases`
  output. The unified `Result.history_df()` and `plot_convergence`
  consume this uniform shape.
- `Strategy._check_constraint` now raises
  `IncompatibleConstraintError` (instead of plain `TypeError`).
- `DVFopt._validate` now raises `SolverConfigError` (instead of plain
  `ValueError`).
- `unified.py` shrunk from 686 → 538 lines by extracting plot methods
  to `dvfopt/_plots.py`. The `Result.plot_*` methods now delegate to
  the extracted functions.

### Removed
- Committed CSVs and PNGs under `benchmarks/results/` — these are
  regenerated artifacts and now gitignored. Use `git add -f` to
  commit one explicitly when needed.

### Added
- `BenchmarkSuite` — declarative harness in
  [benchmarks/benchmark_suite.py](benchmarks/benchmark_suite.py).
  Replaces hand-written `_run_*.py` scripts with a `(cases, solvers)`
  dict + `.run()` returning a pandas DataFrame. Streams CSV row-by-row.
- `register_constraint` and `register_strategy` decorators for plugin
  extensibility. External packages can register new constraint families
  / strategies and make them available via `make_constraint('foo', shape)` /
  `make_strategy('bar')`.
- `PhaseInfo` and `SolveInfo` dataclasses (in `dvfopt.solver`) —
  standardized history container for cross-strategy comparability.
  Strategies opt in by populating `SolveInfo`; legacy free-form `info`
  dicts still supported.
- Property tests for constraint adjoints via Hypothesis
  ([tests/test_constraint_properties.py](tests/test_constraint_properties.py)).
  Randomizes shape / seed / amplitude over 60+ examples per test,
  catching boundary cases fixed-seed tests miss.
- CI workflow ([.github/workflows/test.yml](.github/workflows/test.yml))
  — runs ruff + pytest on every push to main / nightly across
  Python 3.10 / 3.11 / 3.12.
- Pre-commit hooks ([.pre-commit-config.yaml](.pre-commit-config.yaml))
  — ruff, ruff-format, trailing whitespace, YAML/TOML/merge-conflict
  checks, large-file guard, nbstripout.
- Ruff config in [pyproject.toml](pyproject.toml) with project-tuned
  ignores (E501/E701/E702 for math-heavy code, RUF001-3 for unicode
  in docstrings).

### Changed
- `Constraint.coerce()` — new method on the base class that accepts
  loose input shapes (`(2, H, W)`, `(3, H, W)`, `(3, 1, H, W)`) and
  returns the canonical `(C, *shape)` float64 ndarray. Strategies no
  longer need to do their own coercion. Subclasses override for
  family-specific accommodations (e.g. `TriConstraint2D` accepts the
  legacy 3-channel layout).
- `Strategy.accepts_constraints` (tuple of accepted `Constraint`
  subclasses) replaces the previous `requires_2tri` bool. Documents
  precisely what each strategy accepts and works correctly for future
  constraint types.
- `SliceResult` now extends `SolveResult` rather than duplicating its
  fields. Legacy field names (`init_min`, `final_min`) are kept as
  properties for backward compatibility with the dataframe + plot
  code.
- `DVFoptConfig` slimmed: strategy-specific knobs (`lam_schedule`,
  `mu_schedule`, `barrier_max_iter`, etc.) removed from the dataclass.
  Pass a pre-built `Strategy` instance to `solver=...` for non-default
  knobs, or set them via `strategy_kwargs={...}`.

### Fixed
- m14-Schwarz `fallback_size_ratio` check: previously compared
  cell-space span against corner counts (off by one + units mismatch),
  delaying or skipping the global-m14 fallback for near-full clusters.
  Now uses inclusive cell-space comparison.

### Removed
- Legacy `iterative_*` exports at the top-level `dvfopt` namespace.
  Solver implementations remain accessible from `dvfopt.core.*` for
  internal use but are no longer the public API. Migrate to `Solver` /
  `correct_dvf` / `DVFopt`.

---

## [0.2.0] — Parameterized solver refactor

### Added
- Parameterized public API around three orthogonal axes:
  - `Constraint`: `TriConstraint2D`, `TriConstraint2DFullCoverage`,
    `JdetConstraint2D`, `JdetConstraint3D` — flattening, sparse
    Jacobian, adjoint, all with FD validation.
  - `Objective`: `L1Objective`, `L2Objective`, `NoneObjective` (+
    composition via `+` / `*`).
  - `Strategy`: `BarrierStrategy`, `SLSQPFullGridStrategy`,
    `SLSQPWindowedStrategy`, `SchwarzStrategy`, `M10Strategy`,
    `M14Strategy`, `M14SchwarzStrategy` — uniform interface, wraps the
    existing solver implementations.
  - `Solver` composes the three and returns a `SolveResult`.
- `correct_dvf(phi, constraint=..., objective=..., strategy=...)`
  one-shot convenience.
- `auto_strategy(constraint, init_n_neg, init_min, objective_label)`
  heuristic — picks barrier/m10/m14/m14_schwarz/slsqp based on fold
  density.
- `iterative_2d_tri_refine_repair_schwarz` — cluster-localized m14.
  ~5× faster than global m14 on the full B0039 z=12 slice with ~11%
  lower L1. Includes a final global barrier polish to recover the
  safety margin if Schwarz overlap nicks it.
- `max_grow_iters` parameter on m10 and m14 — speed-vs-L1 tuning knob
  for the harmonic-extension stage. Was previously hardcoded to 8.
- Canonical 2D 2-triangle benchmark suite
  (`test_cases.canonical_2tri_2d`) — the six synthetic correspondence
  cases promoted from notebook 14. Used as the standardized smoke test
  for any new solver.
- `quick_tour.ipynb` demonstrating the parameterized API end-to-end.

### Changed
- `DVFopt` rewired to dispatch through `Solver` instead of the legacy
  `_run_*` methods.
- `_resolve_solver` heuristic moved from `DVFopt._resolve_solver` to
  `dvfopt.solver.auto_strategy`. Now considers slice size to pick
  `m14_schwarz` over `m14` for large slices.

### Removed
- `iterative_2d_tri_smoothmin` — wall-test showed it's strictly inferior
  to barrier in every regime (worse than baseline SLSQP at 400+ folds,
  fails outright at 30×30/379).
- `continuation_steps` parameter on `iterative_2d_tri_slsqp` — marginal
  benefit (~3-5% L1) at API complexity cost.
- `DVFopt._run_trust_constr` — experimental and unmaintained; trust-constr
  can be plugged back in later as a Strategy subclass.

---

## [0.1.0] — Initial release

Initial SLSQP-based correction of negative Jacobian determinants in
2D and 3D deformation fields. Includes:

- Windowed iterative SLSQP (`iterative_serial`, `iterative_parallel`,
  `iterative_3d`).
- Penalty / log-barrier L-BFGS-B (`iterative_2d_barrier`,
  `iterative_2d_tri_barrier`, `iterative_3d_barrier`).
- 2-triangle constraint family (full-grid SLSQP, Schwarz hybrid,
  wallbreakers m02/m03/m10/m12/m14).
- `DVFopt` high-level facade with per-slice tabular reports and plots.
- Canonical synthetic test cases + B0039 real-data slice fixtures.
