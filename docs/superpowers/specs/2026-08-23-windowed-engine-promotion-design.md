# Windowed engine promotion — design (stage 1)

2026-08-23. Promotes the benchmarks windowed fold-corrector
(`benchmarks/windowed_isqp.py` + the isqp-osqp inner from
`benchmarks/slsqp_variants.py`, PRs #61–64) into the library as dvfopt's
**third shared engine** (`core/windowed/`) plus a wrapper strategy family —
following the same promote-then-delete path the traced SLSQP driver took in
0.5.0 (PR #65).

Evidence base: 528/528 B0039 slices cleared on jdet/finite (100%), 99%+ on
2-tri, **damage = 0 on all 2178 tasks**; the isqp inner is 3–5× faster than
scipy-SLSQP and the only solver that survives dense-cluster feasibility.

## Goals / non-goals

**Goals (this PR):** the windowing engine + a generic `WindowedWrapperStrategy`
+ a zero-arg `ISQPWindowedStrategy`; the isqp elastic-QP solver and CPR
coloring as primitives; `FiniteJdetConstraint2D` as a real registered
constraint; osqp packaged and gated; the no-damage test suite running in CI
(today it silently skips on every leg — no CI leg installs osqp).

**Non-goals (stage 2, separate ticket):** 3D windowing; porting
shoelace/injectivity/triangles constraint modes; collapsing
`core/slsqp_windowed/` into a thin wrapper pin (which would close its
`constraints.py:147-152` FOLLOW-UP seam gap for every caller); window-level
parallelism via `core/_pool`; promoting the escape modes beyond the `weighted`
hook (blocked on the z=0 monotone-ladder experiment).

## Why an engine + wrapper, not a fourth method package

ARCHITECTURE.md names exactly two shared engines (`barrier/_core.py`,
`schwarz/_common.py`) that "carry no method logic of their own." Windowing is
the same species: domain decomposition orthogonal to the inner solve. The
empirical proof is already in: three different inners ran through one
windowing machine with the no-damage invariant holding for all — the
invariant lives in the decomposition, not the solver. This is not the
":113-115 variant rule" case (a knob on `SLSQPWindowedStrategy`): the two
drivers have different correctness models (below) and different inner
contracts.

## Decision 1 — the inner contract is a frozen-ring reduced problem, NOT `Strategy.fit`

`SchwarzWrapperStrategy` hands its inner a padded crop and a full `Strategy`;
overlapping sweeps provide convergence. The windowed engine's guarantee is
stronger — **no-damage by construction**: ring variables are hard-frozen
inside the solve, every constraint row a free pixel influences is enforced
with global-matching evaluation, and only free pixels are pasted back. A
`Strategy.fit(crop)` cannot express frozen variables or row restriction;
shoehorning it reproduces exactly the optimize-the-ring-then-discard-it seam
gap `core/slsqp_windowed/` documents in its own FOLLOW-UP comment.

The inner protocol is therefore the existing `_Sub` contract:

```python
class WindowInner(Protocol):
    def solve(self, sub: WindowSub, maxiter: int, trace: dict | None) -> tuple[np.ndarray, int, bool]:
        ...  # (x_full, n_iter, feasible); frozen vars MUST stay at sub.flat0

@dataclass(frozen=True)
class WindowSub:  # built by the engine, one per window
    constraint: Constraint      # patch-shaped clone
    flat0: np.ndarray           # patch flattened in the constraint's own pack
    cons: Callable              # enforced rows only, target = threshold + margin_delta
    cons_jac: Callable          # sparse (n_enforced, n_vars)
    obj: Callable; obj_grad: Callable; hess_diag: Callable
    free_idx: np.ndarray; free_mask: np.ndarray
    patch_box: tuple; n_enforced: int
```

Shipped inners (registry keyed by string, mirroring the benchmark's):
`'isqp'` (default — the elastic-QP SQP), `'slsqp'` (traced C-SLSQP driver on
the reduced problem), `'slsqp+trust-constr'` (escalation, keep-better). A
`StrategyInnerAdapter` for masked-solve-capable strategies (SLP) is stage 2.

## Decision 2 — locality lives in an engine-side registry, not on `Constraint` (yet)

The engine needs per-constraint locality knowledge: ring width (jdet 2 /
2tri 1 / finite 1), the fold map on the pixel grid, influenced-rows for a
free mask, and evaluation-validity at patch edges (central-diff rows on an
interior cut disagree with the global field; true image borders are exempt).
No such concept exists on `Constraint`, and the add-a-constraint checklist
has no slot for it. Rather than widen the base contract in the same PR that
adds an engine, locality is a small adapter registered per constraint type
inside the engine:

```python
# core/windowed/_locality.py
@dataclass(frozen=True)
class WindowLocality:
    ring: int
    min_field: Callable[[np.ndarray], np.ndarray]        # (H, W) fold map (+inf pad on cell grids)
    influenced_rows: Callable[..., np.ndarray]           # free_mask, borders -> enforced row idx
    jac_of: Callable[..., "sparse.csr_matrix"]           # patch jacobian (colored or analytic)

LOCALITY: dict[type, WindowLocality] = {JdetConstraint2D: …, TriConstraint2D: …,
                                        TriConstraint2DFullCoverage: …, FiniteJdetConstraint2D: …}
```

`WindowedWrapperStrategy.accepts_constraints` is derived from
`tuple(LOCALITY)` — one source of truth. Folding locality into `Constraint`
attributes is the stage-2 refactor once 3D forces the question.

## Decision 3 — the Objective→(obj, grad, hess_diag) adapter stays solver-side

ARCHITECTURE.md closes the Objective contract at `__call__(diff) -> (value,
grad)` + `label`; there is no `hess_diag` and we do not add one. The engine
keeps an adapter keyed off `objectives._kind_eps(objective)` (the same escape
hatch the numba/torch kernels use): L2 → constant 2.0 diagonal; L1 →
eps-smoothed GN diagonal floored at 0.1 (the floor is load-bearing — see
`slsqp_variants.py`'s rationale); `'none'` → zero objective with a flat 2.0
diagonal to keep the QP positive-definite. Values and gradients themselves
come from `objective(diff)` — the adapter adds only the curvature model.

## File manifest

**New in `dvfopt/`:**

| path | contents |
|---|---|
| `core/windowed/__init__.py` | re-exports `windowed_correct` (engine entry) |
| `core/windowed/_common.py` | the engine: `find_windows`, `build_subproblem`, round loop, grow-on-failure, giant tiling, mop, damage accounting (from `benchmarks/windowed_isqp.py`, ~intact) |
| `core/windowed/_locality.py` | `WindowLocality` + registry + the (family, shape) coloring cache |
| `core/windowed/_inners.py` | `WindowSub`, inner registry (`isqp`/`slsqp`/`slsqp+trust-constr`) |
| `core/primitives/isqp.py` | `isqp_solve` (ex `_isqp_solve_osqp`) + `_backtrack`; module-level `HAS_OSQP` flag; **drops the `constraint=` kwarg** (primitives may not import `dvfopt.constraints`; callable `cons_jac` is the seam — the full-grid coloring path moves to the caller) |
| `core/primitives/coloring.py` | `dense_jacobian`, `jacobian_coloring`, `colored_jacobian` (duck-typed `.adjoint`, no constraint import) |
| `core/primitives/finite_jdet.py` | flat forward-diff det + analytic sparse jacobian (checklist item 5: primitives hold the math) |
| `strategies/windowed.py` | `WindowedWrapperStrategy` (`@register_strategy('windowed_wrapper')`, `inner=` required) + `ISQPWindowedStrategy` (`@register_strategy('isqp_windowed')`, zero-arg, knobs: `margin=3, maxiter=400, max_rounds=8, margin_delta=1e-3, max_window_area=3000, mop_margin=25, trust_region=True, time_budget_s=None`). Both declare `accepts_constraints = tuple(LOCALITY)`, `accepts_objectives = (L1Objective, L2Objective, NoneObjective)`, `supports_3d = False` |

**Changed:** `constraints.py` (+`FiniteJdetConstraint2D`, real subclass:
`pack=DX_FIRST`, `dim=2`, `coerce` via the `TriConstraint2D.coerce` unbound
idiom, `jacobian()` implemented, `@register_constraint('finite')`;
`metrics.py` docstring list gains `'finite'` — the `'auto'` branch is
untouched), `strategies/__init__.py` + `dvfopt/__init__.py` (imports,
`__all__`), `solver.py` (auto_strategy Jdet mild tier → `'isqp_windowed'`
when `importlib.util.find_spec('osqp')` else `'slsqp_windowed'`; fix the
`:486` docstring drift), `dvfopt_gui/_shared.py` + `worker.py` +
`strategy_params.py` (menu/label/params rows for 2tri + jdet tabs; finite
has no GUI tab — API/CLI only), `pyproject.toml` (new extra
`solvers = ["osqp"]`; `osqp` added to `dev` so CI's `[dev,gui]` legs run the
suite; `uv lock` re-run), ARCHITECTURE.md (engine list gains
`core/windowed/_common.py`; constraint checklist footnote on the locality
registry), CLAUDE.md (delegation-table + building-blocks rows), CHANGELOG
(0.6.0 entry).

**Deleted from `benchmarks/` (the slsqp_traced precedent):**
`windowed_isqp.py`, `finite_jdet.py`, the isqp core + coloring out of
`slsqp_variants.py`. **Retained as harnesses, imports repointed:**
`fullslice_bench.py`, `windowed_bench.py`, `escape_bench.py`,
`windowed_escape.py` (imports the engine; still hosts penalty/twophase +
`repair_residuals`), `b0039_isqp_bench.py`, `comprehensive_bench.py`,
`slsqp_variants.py` (SOLVERS crop harness), `trace_parity_check.py`.

## Engine conventions (mirroring `schwarz/_common.py`)

- Module docstring with an explicit **inner-contract section**.
- Entry: `windowed_correct(phi_in, inner, *, constraint, objective, threshold,
  margin=3, maxiter=400, max_rounds=8, margin_delta=1e-3,
  max_window_area=3000, mop_margin=25, time_budget_s=None, verbose=1,
  record_history=False, step_callback=None)`.
- Return `(phi_out, history) if record_history else phi_out`; `history` is a
  plain dict whose per-phase entries carry `n_iter/n_neg/min_T/wall_s` +
  `extras={damage, n_windows, giant_regions, mop_cleared, l1_move, l2_move}`
  so `SolveInfo.from_legacy_history` maps it for free — the strategy ends
  with `self._finish(...)`. (Strictly better than `slsqp_windowed`, which
  returns bare `phi` and emits an empty SolveInfo.)
- `step_callback({'phi': …, 'stage': …})` fired per round / giant / mop;
  `KeyboardInterrupt` propagates as the documented Stop. (Fixes the two
  Schwarz-wrapper gaps: callback is forwarded, and per-window isqp traces are
  lifted to `SolveInfo.extras['isqp_trace']` — a distinct key from
  `slsqp_trace`, whose record shape is pinned by tests.)
- Per-window failures contained (`log_warning` + continue); logging via
  `dvfopt._logging`; no prints.
- `time_budget_s` honored at round/window boundaries (GUI toolbar contract).
- The giant-region tiler stays inside this engine — it is windowing-specific
  (ring-inset tiles + damage accounting) and does NOT reuse
  `schwarz/_common.py`, whose crop-Strategy contract cannot freeze rings.
- Pack handling: the engine flattens exclusively through the constraint's own
  `flatten`/`unflatten` — pack-agnostic by delegation.

## osqp packaging and gating

`osqp` joins two extras: user-facing `solvers` and `dev` (so `test.yml`'s
`[dev,gui]` legs and the coverage leg actually run the invariant suite —
today every osqp test skips in CI). Gating: `core/primitives/isqp.py` sets
`HAS_OSQP` module-level (try/except, the `HAS_TRACED_SLSQP` idiom);
`ISQPWindowedStrategy.solve` raises a friendly
`ImportError("isqp inner requires osqp — pip install dvfopt[solvers]")` when
missing (the `slp accuracy='max'` idiom); the GUI menu rows gate via an
`_osqp_available()` probe, visible-but-disabled (the torch idiom, keeps
parity-test set-equality). `ci.yml`'s `[fast]` numba legs stay osqp-free —
they exercise the skip path, which is itself contract.

## Declared behavior deltas

1. `trust_region=True` is the promoted default (PR #64's fix for the
   line-search stall); `trust_region=False` reproduces the legacy path. The
   528-slice campaign numbers were measured pre-TR.
2. `margin_delta` (constraints driven to `threshold + 1e-3`) is engine
   semantics, distinct from `slsqp_windowed`'s plain threshold.
3. `auto_strategy`'s Jdet mild tier routes to `'isqp_windowed'` only when
   osqp is importable; otherwise unchanged.
4. Everything else is a pure move: the engine code is the tested PR #61–64
   driver, and the promotion PR must show the ported no-damage suite green.

## Test plan

- Port `tests/test_windowed_isqp.py` invariants to the promoted API (module
  skip on `HAS_OSQP` stays); add the checklist minimum for the strategy:
  construction through `Solver`, resolution by `'isqp_windowed'` label,
  rejection of an unsupported constraint (`Tet6Constraint3D`) and objective.
- `FiniteJdetConstraint2D`: analytic-vs-numeric jacobian property test (move
  the self-check), `constraint_fold_stats(..., constraint='finite')` smoke,
  registry round-trip.
- Wrapper: inner-required error, non-Strategy-inner `TypeError` analogue for
  unknown inner labels, no-damage with each shipped inner (parametrized).
- GUI parity: the three rows (menu spec + `_MID_TO_LABEL` + params table);
  `isqp_windowed` must construct zero-arg through `_build_strategy`.
- ImportError-path test with osqp masked (`monkeypatch` + `sys.modules`).

## Stage-2 ticket (filed alongside, not implemented)

3D windowing (6-tet/Jdet3D locality); constraint modes
(shoelace/injectivity/triangles) as extra enforced-row sets; fold locality
into `Constraint`; `slsqp_windowed` → thin wrapper pin (closes its FOLLOW-UP
halo gap); window-level parallelism on `core/_pool`; `StrategyInnerAdapter`;
promote remaining escape modes after the monotone-ladder experiment.
