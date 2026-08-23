# Architecture

Rules for extending `dvfopt`, not a module census. For *what exists*, read
[CLAUDE.md](CLAUDE.md); for *where it moved in 0.5.0*, read
[CHANGELOG.md](CHANGELOG.md).

## Three axes

A solve is the composition of three independent choices:

```
              Constraint                 Objective              Strategy
     "what must stay feasible"      "what to minimise"      "how to get there"
   TriConstraint2D / Tet6Con...    L1 / L2 / NoneObjective   SLP, Barrier, SLSQP*, ...
                 │                          │                       │
                 └──────────────┬───────────┴───────────────────────┘
                                ▼
                       Solver(constraint=, objective=, strategy=).fit(phi)
                                ▼
                          (phi_out, SolveInfo)
```

Any triple that type-checks is runnable. A Strategy declares what it can
actually take (`accepts_constraints`, `accepts_objectives`, `supports_3d`) and
`Solver.__init__` rejects a bad triple at construction with
`IncompatibleConstraintError` / `IncompatibleObjectiveError` — never mid-solve.

`Solver.from_spec(constraint='2tri', objective='l1', strategy='slp')` builds the
same thing from string labels via the three registries.

## Dependency rules

Imports flow one way. Breaking these is what re-tangles the package.

| Layer | May import | Must NOT import |
|---|---|---|
| `dvfopt/objectives.py` | numpy only | anything in `dvfopt.core` |
| `dvfopt/constraints.py` | `core.primitives.*`, `jacobian.*` | any `core/<method>/` package |
| `dvfopt/core/primitives/` | numpy/scipy, `dvfopt.objectives` | any `core/<method>/`, `strategies`, `solver` |
| `dvfopt/core/<method>/` | `core.primitives.*`, the shared engines, `dvfopt.objectives` | a *sibling* method package, `strategies`, `solver` |
| `dvfopt/strategies/<name>.py` | `constraints`, `objectives`, its own `core/<method>/` | another strategy's core package |
| `dvfopt/solver.py`, `unified.py`, `pipeline_*.py`, `cli.py` | all of the above | — |

- **Method packages are siblings, not a chain.** `core/barrier/`,
  `core/slsqp_windowed/`, `core/slsqp_fullgrid/`, `core/schwarz/`,
  `core/wallbreakers/`, `core/slp/`, `core/nmvf/`, `core/marching/` never
  import each other. Anything two of them need moves to `core/primitives/` or
  to one of the two shared engines.
- **The two shared engines** are `core/barrier/_core.py`
  (`run_penalty_barrier_lbfgs`, the penalty→barrier homotopy) and
  `core/schwarz/_common.py` (`cluster_schwarz_2d_tri` / `cluster_schwarz_3d_tet`,
  the domain decomposition). They are the exception to "no cross-method
  imports" — a method package may import an engine, and the wallbreakers do.
- **Objectives are pure.** An `Objective` is `(diff) -> (value, grad)` and
  nothing else — no state, no constraint knowledge, no I/O. Kernels that cannot
  call back into Python (numba, torch autograd) take the legacy
  `(kind, eps_l1)` pair from `objectives._kind_eps(objective)` instead.
- **Strategies own no math.** A strategy is argument marshalling plus one call
  into its `core/<method>/` package, then `self._finish(...)` to normalise the
  return into `(phi_out, SolveInfo)`.

## phi-pack conventions

Two flat layouts exist. Crossing them silently swaps dy/dx and produces a
plausible-looking wrong answer, so the layout is declared on the Constraint
(`Constraint.pack`) and every helper assumes exactly one of them.

| Pack | Layout | Declared by | Modules |
|---|---|---|---|
| `PhiPack.DY_FIRST` | `phi[:N]=dy`, `phi[N:]=dx` | `TriConstraint2D`, `TriConstraint2DFullCoverage` | `core/primitives/tri.py`, `core/barrier/tri2d.py`, `core/slsqp_fullgrid/tri2d.py`, `core/schwarz/{tri2d,_cluster}.py`, the 2D `core/wallbreakers/*`, `core/slp/{lp_direct_2tri,cluster_lp_2tri,tri_linearize}.py` |
| `PhiPack.DX_FIRST` | `phi[:N]=dx`, `phi[N:2N]=dy` (3D: `phi[2N:]=dz`) | `JdetConstraint2D`, `JdetConstraint3D`, **`Tet6Constraint3D`** | `core/slsqp_windowed/*`, `core/primitives/{jdet2d,jdet3d}.py`, `core/barrier/{jdet2d,jdet3d,jdet3d_torch,tet3d_torch}.py`, `core/slsqp_fullgrid/tet3d.py`, `core/slp/{lp_direct_6tet,cluster_lp_6tet}.py`, the 3D `core/wallbreakers/*`, `jacobian/tetrahedron_sign.py` |

Note the split is **not** "2-tri/6-tet vs Jdet": the 6-tet 3D family packs
`[dx, dy, dz]` so it can share the 3D barrier plumbing with `JdetConstraint3D`.
`core/schwarz/_common.py` is pack-agnostic (it slices `(C, *shape)` arrays and
never flattens).

A flat vector from one family cannot be handed to the other without a channel
swap. A helper that genuinely sees both — `core/marching/*` mixes a DX_FIRST
6-tet stack with a DY_FIRST 2-tri term — asserts on the pack lengths at the
boundary. Copy that pattern; do not bridge silently.

## Add a method

1. **`dvfopt/core/<name>/`** — a package, not a loose module. Public entry
   point(s) re-exported from its `__init__.py`; helpers underscore-prefixed.
   Constraint math comes from `core/primitives/`; reuse an engine if the method
   is a penalty/barrier homotopy or a Schwarz decomposition. No sibling-method
   imports.
2. **`dvfopt/strategies/<name>.py`** — a `@dataclass` `Strategy` subclass
   decorated with `@register_strategy('<label>')`, its knobs as dataclass
   fields. Declare `accepts_constraints`, `accepts_objectives`, `supports_3d`
   (omit an `accepts_*` to accept anything). `solve()` marshals arguments, calls
   into `core/<name>/`, and returns `self._finish(out, record_history, threshold)`.
   Import it from `dvfopt/strategies/__init__.py` so the registration runs.
3. **`tests/test_<name>.py`** — at minimum: direct composition through
   `Solver`, resolution by string label, and rejection of a constraint or
   objective the strategy declares it does not accept.
4. **GUI (optional)** — three rows, all parity-tested by
   `tests/test_gui_strategy_parity.py`: a `worker._MID_TO_LABEL` entry mapping
   menu id → registry label, the matching menu-spec row, and the table row.
   Miss one and that test fails.
5. **Docs** — a row in the CLAUDE.md strategy→impl delegation table.

If the method is a variant of an existing one (a different seed, a different
schedule), it is a **dataclass field on the existing Strategy**, not a new
package.

## Add a constraint

1. Subclass `Constraint` in `dvfopt/constraints.py`.
2. Implement `values()`, `adjoint(v)`, `flatten(arr)`, `unflatten(phi)`.
   Add `jacobian()` only if an SLSQP strategy will consume it (SLSQP needs the
   sparse rows; barrier/SLP need only the adjoint).
3. Declare `pack` (`PhiPack.DY_FIRST` or `PhiPack.DX_FIRST`) and `dim`.
   The pack declaration is the contract every strategy trusts.
4. `@register_constraint('<label>')` so `from_spec` / the CLI / the GUI can
   name it.
5. Put the flat constraint evaluation + analytic adjoint in
   `core/primitives/`, not in the constraint class — solvers call the primitive
   directly on hot paths.

## Add an objective

Subclass `Objective`, implement `__call__(diff) -> (value, grad)`, set `label`.
That is the whole contract — no other method is called on an objective.

If the objective must reach the numba/torch inner kernels, it also needs a
`label` (and `eps`, if smoothed) that `_kind_eps` can map to a kind string;
those kernels dispatch on an integer flag and cannot evaluate Python.
