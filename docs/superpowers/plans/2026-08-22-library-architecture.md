# Library Architecture Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize dvfopt into one clean package: method-first `core/`, real Objective axis, traced C-SLSQP driver at every SLSQP call site, `laplacian`/`test_cases` absorbed.

**Architecture:** Pure moves + mechanical import rewrites gated by the existing 85-module suite; genuinely new code (driver helpers, objective plumbing, `accepts_objectives`) is TDD'd. Each task = move/change → repo-wide import fix → full gate → commit.

**Tech Stack:** Python 3.10–3.12, numpy/scipy(1.15–1.18)/numba, pytest(-xdist/-randomly), ruff 0.16.3, mypy, git worktree.

**Spec:** `docs/superpowers/specs/2026-08-22-library-architecture-design.md` (same branch — read it first; the mapping tables there are normative).

## Global Constraints

- **Work ONLY in the worktree:** `c:\Users\Andy\Documents\GitHub\UCI-iGravi\deformation-field-processing\.claude\worktrees\library-architecture` (branch `refactor/library-architecture`). Never touch the main checkout.
- **Interpreter:** `.venv\Scripts\python.exe` inside the worktree (created in Task 0). All `pytest`/`ruff`/`mypy` commands below mean `.venv\Scripts\python -m pytest` etc.
- **Gate (every task, before its commit):** `python -c "import dvfopt"` → `pytest tests/ -n auto -q` (0 failures; compare count to Task 0 baseline) → `ruff check dvfopt dvfopt_gui tests benchmarks` → `ruff format dvfopt dvfopt_gui tests benchmarks` (then re-run check).
- **mypy** runs at Tasks 0, 5, 14, 16 only (its `[tool.mypy]` file list moves with the modules; keep it green at those checkpoints).
- **No numerical/behavioral changes** except those the spec names (driver swap = byte-identical; objective plumbing defaults preserve current behavior).
- **After every move task:** `git grep -n "<old dotted path>" -- . ":!research" ":!archive" ":!notebooks" ":!data" ":!writing" ":!docs"` must return nothing (notebooks are handled once, in Task 15). Also grep `pyproject.toml`, `.github/workflows/`, `noxfile.py`, `asv_bench/` for the old *file* paths and fix hits.
- **Frozen, never edit:** `research/`, `archive/`, `writing/`, `notebooks/experiments/` scratch scripts, anything under `data/`.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

## Canonical module mapping (single source of truth for all rewrites)

Rewrites replace **dotted module paths** (word-boundary), longest-old-first:

| # | Old | New |
|---|---|---|
| 1 | `dvfopt.core.tri_primitives` | `dvfopt.core.primitives.tri` |
| 2 | `dvfopt.core.barrier_objective` | `dvfopt.core.primitives.jdet3d` |
| 3 | `dvfopt.core._internal.constraint_values` | `dvfopt.core.primitives.constraint_values` |
| 4 | `dvfopt.core._barrier_core` | `dvfopt.core.barrier._core` |
| 5 | `dvfopt.core.iterative2d_barrier` | `dvfopt.core.barrier.jdet2d` |
| 6 | `dvfopt.core.iterative3d_barrier_torch` | `dvfopt.core.barrier.jdet3d_torch` |
| 7 | `dvfopt.core.iterative3d_barrier` | `dvfopt.core.barrier.jdet3d` |
| 8 | `dvfopt.core.iterative2d_tri_barrier` | `dvfopt.core.barrier.tri2d` |
| 9 | `dvfopt.core.iterative3d_tet_barrier_torch` | `dvfopt.core.barrier.tet3d_torch` |
| 10 | `dvfopt.core._internal.io` | `dvfopt.core.slsqp_windowed._io` |
| 11 | `dvfopt.core._internal.metrics` | `dvfopt.core.slsqp_windowed._metrics` |
| 12 | `dvfopt.core._internal.window` | `dvfopt.core.slsqp_windowed._window` |
| 13 | `dvfopt.core.solver3d` | `dvfopt.core.slsqp_windowed.coordinator3d` |
| 14 | `dvfopt.core.solver` | `dvfopt.core.slsqp_windowed.coordinator` |
| 15 | `dvfopt.core.objective` | `dvfopt.core.slsqp_windowed._objective` *(interim; module deleted in Task 12)* |
| 16 | `dvfopt.core.slsqp` | `dvfopt.core.slsqp_windowed` |
| 17 | `dvfopt.core.iterative2d_tri_slsqp` | `dvfopt.core.slsqp_fullgrid.tri2d` |
| 18 | `dvfopt.core.iterative3d_tet_slsqp` | `dvfopt.core.slsqp_fullgrid.tet3d` |
| 19 | `dvfopt.core.iterative2d_tri_schwarz` | `dvfopt.core.schwarz.tri2d` |
| 20 | `dvfopt.core.wallbreakers._schwarz_common` | `dvfopt.core.schwarz._common` |
| 21 | `dvfopt.core._cluster_2tri` | `dvfopt.core.schwarz._cluster` |
| 22 | `dvfopt.core._nmvf` | `dvfopt.core.nmvf` |
| 23 | `laplacian` | `dvfopt.laplacian` |
| 24 | `test_cases` | `dvfopt.testdata` |
| 25 | `slsqp_traced` | `dvfopt.core.primitives.slsqp` *(benchmarks-local import name)* |

Rewrite sweep directories: `dvfopt dvfopt_gui tests benchmarks scripts asv_bench tools` plus (until absorbed) `laplacian test_cases`. **Never** `research archive notebooks data writing docs`.

---

### Task 0: Worktree environment + green baseline

**Files:** none (env only)

**Interfaces:**
- Produces: `.venv` in the worktree; recorded baseline test count all later tasks compare against.

- [ ] **Step 1: Create venv + editable install** (from the worktree root)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -U pip
.venv/Scripts/python -m pip install -e ".[dev,fast,gui]"
```

- [ ] **Step 2: Baseline gate**

Run: `.venv/Scripts/python -m pytest tests/ -n auto -q` → record the `N passed, M skipped` line in the task report.
Run: `.venv/Scripts/python -m ruff check dvfopt dvfopt_gui tests benchmarks` → clean.
Run: `.venv/Scripts/python -m mypy` → clean.
Expected: all green (main just merged #64 green-CI). If NOT green, STOP and report — do not proceed on a red baseline.

- [ ] **Step 3: Create the rewrite tool** — Create `tools/rewrite_imports.py`:

```python
"""One-shot dotted-module-path rewriter for the 0.5.0 reorg.

Usage: python tools/rewrite_imports.py OLD=NEW [OLD=NEW ...] [--dirs d1 d2 ...]
Replaces word-bounded dotted paths in *.py under the given dirs (default: the
sweep set). Longest OLD first so prefixes never clobber (solver3d before solver).
"""

import re
import sys
from pathlib import Path

DEFAULT_DIRS = ["dvfopt", "dvfopt_gui", "tests", "benchmarks", "scripts", "asv_bench", "tools"]


def main() -> None:
    args = sys.argv[1:]
    dirs = DEFAULT_DIRS
    if "--dirs" in args:
        i = args.index("--dirs")
        dirs, args = args[i + 1 :], args[:i]
    pairs = sorted(
        (a.split("=", 1) for a in args), key=lambda p: len(p[0]), reverse=True
    )
    pats = [(re.compile(rf"(?<![\w.]){re.escape(o)}(?![\w])"), n) for o, n in pairs]
    changed = 0
    for d in dirs:
        for f in Path(d).rglob("*.py"):
            text = orig = f.read_text(encoding="utf-8")
            for pat, new in pats:
                text = pat.sub(new, text)
            if text != orig:
                f.write_text(text, encoding="utf-8")
                changed += 1
                print(f"rewrote {f}")
    print(f"{changed} files changed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add tools/rewrite_imports.py
git commit -m "chore(reorg): venv baseline + import-rewrite tool"
```

---

### Task 1: `core/primitives/` package (tri, jdet2d, jdet3d, constraint_values)

**Files:**
- Create: `dvfopt/core/primitives/__init__.py`, `dvfopt/core/primitives/jdet2d.py`
- Move: `dvfopt/core/tri_primitives.py` → `dvfopt/core/primitives/tri.py`; `dvfopt/core/barrier_objective.py` → `dvfopt/core/primitives/jdet3d.py`; `dvfopt/core/_internal/constraint_values.py` → `dvfopt/core/primitives/constraint_values.py`
- Modify: `dvfopt/core/iterative2d_tri_slsqp.py` (move `_build_full_grid_tri_jac` out), `dvfopt/core/iterative2d_barrier.py` (move `_jdet_2d_flat`, `_jdet_grad_T_v_2d` out), `dvfopt/constraints.py` (imports now point ONLY at primitives)

**Interfaces:**
- Produces: `dvfopt.core.primitives.tri` exporting everything `tri_primitives` did **plus** `build_full_grid_tri_jac(H, W, full_coverage) -> Callable[[ndarray], sp.csr_matrix]` (public rename of `_build_full_grid_tri_jac`; keep `_build_full_grid_tri_jac = build_full_grid_tri_jac` alias in `slsqp_fullgrid` consumers' rewrite). `dvfopt.core.primitives.jdet2d` exporting `jdet_2d_flat(phi_flat, H, W)` and `jdet_grad_T_v_2d(phi_flat, H, W, v)` (public renames; old underscore names kept as module-level aliases so the sweep is purely mechanical).

- [ ] **Step 1: Moves**

```bash
mkdir -p dvfopt/core/primitives
git mv dvfopt/core/tri_primitives.py dvfopt/core/primitives/tri.py
git mv dvfopt/core/barrier_objective.py dvfopt/core/primitives/jdet3d.py
git mv dvfopt/core/_internal/constraint_values.py dvfopt/core/primitives/constraint_values.py
```

Create `dvfopt/core/primitives/__init__.py` with only a docstring:

```python
"""Shared constraint/objective math + engines with zero method logic.

Modules: tri (2-triangle), jdet2d, jdet3d (Jacobian-determinant flat forms
and adjoints), constraint_values (per-cell maps for reporting), slsqp
(traced C-SLSQP driver, added later in the reorg).
"""
```

- [ ] **Step 2: Extract the tri jacobian builder.** Cut the whole `_build_full_grid_tri_jac` function from `dvfopt/core/iterative2d_tri_slsqp.py` and paste it into `dvfopt/core/primitives/tri.py` unchanged, renamed `build_full_grid_tri_jac`, adding at the bottom of `tri.py`:

```python
# Back-compat name used across the fullgrid/schwarz call sites.
_build_full_grid_tri_jac = build_full_grid_tri_jac
```

In `iterative2d_tri_slsqp.py` add `from dvfopt.core.primitives.tri import _build_full_grid_tri_jac` and remove any imports the extracted function alone needed.

- [ ] **Step 3: Extract the jdet2d primitives.** Cut `_jdet_2d_flat` and `_jdet_grad_T_v_2d` (and only the imports they need) from `dvfopt/core/iterative2d_barrier.py` into new `dvfopt/core/primitives/jdet2d.py` with public names + aliases:

```python
"""Flat 2D Jacobian-determinant forward form + adjoint (phi pack: [dx, dy])."""

# (paste the two functions here, renamed jdet_2d_flat / jdet_grad_T_v_2d,
#  bodies byte-identical)

_jdet_2d_flat = jdet_2d_flat
_jdet_grad_T_v_2d = jdet_grad_T_v_2d
```

In `iterative2d_barrier.py` replace the removed defs with `from dvfopt.core.primitives.jdet2d import _jdet_2d_flat, _jdet_grad_T_v_2d`.

- [ ] **Step 4: Repo-wide rewrite (mapping rows 1–3)**

```bash
.venv/Scripts/python tools/rewrite_imports.py \
  dvfopt.core.tri_primitives=dvfopt.core.primitives.tri \
  dvfopt.core.barrier_objective=dvfopt.core.primitives.jdet3d \
  dvfopt.core._internal.constraint_values=dvfopt.core.primitives.constraint_values
```

Then fix `dvfopt/core/_internal/__init__.py` by hand (drop its `constraint_values` re-export if present). Verify `dvfopt/constraints.py` now imports only from `dvfopt.core.primitives.*` and `dvfopt.jacobian.*` (its `iterative2d_barrier` / `iterative2d_tri_slsqp` imports must be gone — point them at `primitives.jdet2d` / `primitives.tri`).

- [ ] **Step 5: Gate** (Global Constraints gate; also `git grep` rows 1–3 old paths → empty)

- [ ] **Step 6: Commit** — `git commit -am "refactor(core): primitives package — tri/jdet2d/jdet3d/constraint_values"`

---

### Task 2: `core/barrier/` package

**Files:**
- Move: `_barrier_core.py`→`barrier/_core.py`, `iterative2d_barrier.py`→`barrier/jdet2d.py`, `iterative3d_barrier.py`→`barrier/jdet3d.py`, `iterative3d_barrier_torch.py`→`barrier/jdet3d_torch.py`, `iterative2d_tri_barrier.py`→`barrier/tri2d.py`, `iterative3d_tet_barrier_torch.py`→`barrier/tet3d_torch.py` (all under `dvfopt/core/`)
- Create: `dvfopt/core/barrier/__init__.py` (docstring only, like Task 1)

**Interfaces:**
- Produces: identical public names at the new paths; `run_penalty_barrier_lbfgs` + `anchor_term` now live at `dvfopt.core.barrier._core` (until Task 12 moves `anchor_term` to `dvfopt.objectives`).

- [ ] **Step 1: Moves** (`mkdir -p dvfopt/core/barrier` + six `git mv`, names per Files above; create `__init__.py`)
- [ ] **Step 2: Rewrite rows 4–9** (single `tools/rewrite_imports.py` call with those six pairs — the tool orders longest-first so `iterative3d_barrier_torch` rewrites before `iterative3d_barrier`)
- [ ] **Step 3: Gate** (+ `git grep` rows 4–9 → empty; check `pyproject.toml`/workflows for old file paths)
- [ ] **Step 4: Commit** — `git commit -am "refactor(core): barrier package — engine + per-family drivers"`

---

### Task 3: `core/slsqp_windowed/` package + `core/nmvf/`

**Files:**
- Move: `dvfopt/core/slsqp/` → `dvfopt/core/slsqp_windowed/` (whole dir); `core/solver.py`→`slsqp_windowed/coordinator.py`; `core/solver3d.py`→`slsqp_windowed/coordinator3d.py`; `core/_internal/{io,metrics,window}.py`→`slsqp_windowed/{_io,_metrics,_window}.py`; `core/objective.py`→`slsqp_windowed/_objective.py`; `core/_nmvf.py`→`core/nmvf/__init__.py`
- Delete: `dvfopt/core/_internal/` (now empty)
- Modify: `dvfopt/core/__init__.py` (re-exports `iterative_serial`/`iterative_parallel`/`iterative_3d` — update source module)

**Interfaces:**
- Produces: `dvfopt.core.slsqp_windowed.{iterative,iterative3d,parallel,constraints,constraints3d,gradients,gradients3d,spatial,spatial3d,_grad_op,coordinator,coordinator3d,_io,_metrics,_window,_objective}`; `dvfopt.core.nmvf.nmvf_correct_2d`. `dvfopt.core` re-exports unchanged in name.

- [ ] **Step 1: Moves**

```bash
git mv dvfopt/core/slsqp dvfopt/core/slsqp_windowed
git mv dvfopt/core/solver.py dvfopt/core/slsqp_windowed/coordinator.py
git mv dvfopt/core/solver3d.py dvfopt/core/slsqp_windowed/coordinator3d.py
git mv dvfopt/core/_internal/io.py dvfopt/core/slsqp_windowed/_io.py
git mv dvfopt/core/_internal/metrics.py dvfopt/core/slsqp_windowed/_metrics.py
git mv dvfopt/core/_internal/window.py dvfopt/core/slsqp_windowed/_window.py
git mv dvfopt/core/objective.py dvfopt/core/slsqp_windowed/_objective.py
mkdir -p dvfopt/core/nmvf && git mv dvfopt/core/_nmvf.py dvfopt/core/nmvf/__init__.py
git rm -r dvfopt/core/_internal
```

(If `_internal/__init__.py` still exists with content, delete it — Task 1 already stripped its only cross-package re-export.)

- [ ] **Step 2: Rewrite rows 10–16 + 22** (one tool call; longest-first ordering makes `_internal.io`/`solver3d`/`objective` safe before the bare `dvfopt.core.slsqp` prefix row and `dvfopt.core.solver` row)
- [ ] **Step 3: Windowed docstring in `coordinator.py`** — its module docstring references `dvfopt.core._io` etc.; update the three names to `dvfopt.core.slsqp_windowed._io/_metrics/_window`.
- [ ] **Step 4: Gate** (+ `git grep -n "core._internal\|core.slsqp\b\|core\.solver\b\|core\.solver3d\|core\.objective\b\|core\._nmvf"` filtered per Global Constraints → empty; ProcessPool check: `pytest tests/test_parallel_internals.py tests/test_pool_and_multiscale.py -q` explicitly green — pickling of moved functions is this task's risk)
- [ ] **Step 5: Commit** — `git commit -am "refactor(core): slsqp_windowed + nmvf packages; dissolve _internal"`

---

### Task 4: `core/slsqp_fullgrid/` package

**Files:**
- Move: `dvfopt/core/iterative2d_tri_slsqp.py`→`dvfopt/core/slsqp_fullgrid/tri2d.py`; `dvfopt/core/iterative3d_tet_slsqp.py`→`dvfopt/core/slsqp_fullgrid/tet3d.py`
- Create: `dvfopt/core/slsqp_fullgrid/__init__.py` (docstring only)

- [ ] **Step 1: Moves** (mkdir + 2 × `git mv` + `__init__.py`)
- [ ] **Step 2: Rewrite rows 17–18**
- [ ] **Step 3: Gate** (+ grep old paths → empty)
- [ ] **Step 4: Commit** — `git commit -am "refactor(core): slsqp_fullgrid package"`

---

### Task 5: `core/schwarz/` package (+ mypy checkpoint)

**Files:**
- Move: `dvfopt/core/iterative2d_tri_schwarz.py`→`dvfopt/core/schwarz/tri2d.py`; `dvfopt/core/wallbreakers/_schwarz_common.py`→`dvfopt/core/schwarz/_common.py`; `dvfopt/core/_cluster_2tri.py`→`dvfopt/core/schwarz/_cluster.py`
- Create: `dvfopt/core/schwarz/__init__.py` (docstring only)
- Modify: `pyproject.toml` `[tool.mypy]` file list — update every entry that names a module moved in Tasks 1–5.

- [ ] **Step 1: Moves** (mkdir + 3 × `git mv` + `__init__.py`)
- [ ] **Step 2: Rewrite rows 19–21**
- [ ] **Step 3: mypy checkpoint** — open `pyproject.toml`, map every `[tool.mypy]`/`[tool.ruff]` path through the canonical table; run `mypy` → clean.
- [ ] **Step 4: Gate** (+ grep rows 19–21 → empty; `pytest tests/test_m14_schwarz.py tests/test_schwarz_wrapper.py tests/test_tri_schwarz.py -q` named explicitly)
- [ ] **Step 5: Commit** — `git commit -am "refactor(core): schwarz package — one home for the decomposition"`

---

### Task 6: Promote the traced SLSQP driver (TDD)

**Files:**
- Move: `benchmarks/slsqp_traced.py` → `dvfopt/core/primitives/slsqp.py`
- Create: `tests/test_slsqp_driver.py`
- Modify: `benchmarks/windowed_isqp.py`, `benchmarks/slsqp_variants.py`, `benchmarks/trace_parity_check.py`, `tests/test_slsqp_variants.py` (import from the library), `pyproject.toml` (scipy pin)

**Interfaces:**
- Produces: `dvfopt.core.primitives.slsqp.minimize_slsqp_traced(func, x0, jac, constraints=(), bounds=None, maxiter=100, ftol=1e-6, trace=None, save_x=False) -> OptimizeResult` and **new helper** `ineq_dict(fun, jac, lb=0.0) -> dict` (old-style ineq constraint: shifts by `lb`, densifies sparse jacs). All swap tasks (7–10) consume exactly these two names.

- [ ] **Step 1: Write the failing test** — `tests/test_slsqp_driver.py`:

```python
"""Byte-identity + trace contract for the vendored traced C-SLSQP driver."""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


def _problem():
    rng = np.random.default_rng(7)
    n = 30
    x0 = rng.normal(0, 1, n)
    tgt = rng.normal(0, 1, n)
    A = rng.normal(0, 1, (8, n))
    b = A @ tgt + np.abs(rng.normal(0, 1, 8))  # active constraints at optimum

    def f(x):
        d = x - tgt
        return float(d @ d), 2.0 * d

    cons = [{"type": "ineq", "fun": lambda x: A @ x - b, "jac": lambda x: A}]
    return f, x0, cons


class TestByteIdentity:
    def test_identical_to_scipy_slsqp(self):
        from dvfopt.core.primitives.slsqp import minimize_slsqp_traced

        f, x0, cons = _problem()
        ref = minimize(
            f, x0, jac=True, method="SLSQP", constraints=cons,
            options={"maxiter": 100, "ftol": 1e-8},
        )
        r = minimize_slsqp_traced(
            lambda x: f(x)[0], x0, jac=lambda x: f(x)[1],
            constraints=cons, maxiter=100, ftol=1e-8, trace=None,
        )
        assert (r.status, r.nit) == (ref.status, ref.nit)
        assert np.array_equal(r.x, ref.x)

    def test_trace_records_majors(self):
        from dvfopt.core.primitives.slsqp import minimize_slsqp_traced

        f, x0, cons = _problem()
        tr: dict = {}
        r = minimize_slsqp_traced(
            lambda x: f(x)[0], x0, jac=lambda x: f(x)[1],
            constraints=cons, maxiter=100, ftol=1e-8, trace=tr,
        )
        assert tr["iters"] and tr["nit"] == r.nit
        last = tr["iters"][-1]
        assert last["max_viol"] < 1e-8
        assert {"obj", "opt", "alpha", "nfev"} <= set(last)


class TestIneqDict:
    def test_lb_shift_and_sparse_densify(self):
        from dvfopt.core.primitives.slsqp import ineq_dict

        fun = lambda x: np.array([x[0], x[1] * 2.0])
        jac = lambda x: sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        d = ineq_dict(fun, jac, lb=0.5)
        assert d["type"] == "ineq"
        np.testing.assert_allclose(d["fun"](np.array([1.0, 1.0])), [0.5, 1.5])
        J = d["jac"](np.array([1.0, 1.0]))
        assert isinstance(J, np.ndarray)
        np.testing.assert_allclose(J, [[1.0, 0.0], [0.0, 2.0]])
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_slsqp_driver.py -q` → FAIL (`ModuleNotFoundError: dvfopt.core.primitives.slsqp`).

- [ ] **Step 3: Promote + extend.** `git mv benchmarks/slsqp_traced.py dvfopt/core/primitives/slsqp.py`. Note the driver's `func`/`jac` are separate callables (its signature, unchanged). Append to the module (before the `__main__` block):

```python
def ineq_dict(fun, jac, lb=0.0):
    """Old-style ineq constraint dict for :func:`minimize_slsqp_traced`.

    Wraps ``fun(x) >= lb`` as ``fun(x) - lb >= 0`` and densifies sparse
    jacobians (the C core is dense — scipy's own SLSQP densifies too).
    """
    import scipy.sparse as _sp

    def _fun(x, *args):
        return np.asarray(fun(x), dtype=np.float64) - lb

    def _jac(x, *args):
        J = jac(x)
        return J.toarray() if _sp.issparse(J) else np.asarray(J, dtype=np.float64)

    return {"type": "ineq", "fun": _fun, "jac": _jac}
```

Run the rewrite for row 25 **limited to benchmarks/tests**: `python tools/rewrite_imports.py slsqp_traced=dvfopt.core.primitives.slsqp --dirs benchmarks tests` — then hand-check the four modified files (their old import was path-hack `from slsqp_traced import ...`; make it `from dvfopt.core.primitives.slsqp import ...` and delete any `sys.path` insertion that existed only for it).

- [ ] **Step 4: scipy pin.** In `pyproject.toml` dependencies: `"scipy"` → `"scipy>=1.15,<1.19"`.

- [ ] **Step 5: Run tests** — `pytest tests/test_slsqp_driver.py tests/test_slsqp_variants.py -q` → PASS. Full gate.

- [ ] **Step 6: Commit** — `git commit -am "feat(core): promote traced C-SLSQP driver to primitives + ineq_dict helper + scipy pin"`

---

### Task 7: Swap fullgrid tri2d to the traced driver

**Files:**
- Modify: `dvfopt/core/slsqp_fullgrid/tri2d.py` (2 call sites)

**Interfaces:**
- Consumes: `minimize_slsqp_traced`, `ineq_dict` (Task 6 signatures).

- [ ] **Step 1: Swap.** In `tri2d.py`, replace the `NonlinearConstraint` + both `minimize(...)` calls. Old shape (cold run; warm run is identical but with `z_warm`, `warm_max_iter`, `warm_ftol`):

```python
nlc = NonlinearConstraint(_constr, lb=threshold, ub=np.inf, jac=jac_func)
res = minimize(_obj, z_anchor.copy(), jac=True, method='SLSQP', constraints=[nlc],
               options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': verbose >= 3})
```

New (the driver takes func and jac separately; `_obj` returns `(value, grad)`):

```python
from dvfopt.core.primitives.slsqp import ineq_dict, minimize_slsqp_traced

cons = [ineq_dict(_constr, jac_func, lb=threshold)]
trace_cold: dict | None = {} if record_history else None
res = minimize_slsqp_traced(
    lambda z: _obj(z)[0], z_anchor.copy(), jac=lambda z: _obj(z)[1],
    constraints=cons, maxiter=max_iter, ftol=1e-9, trace=trace_cold,
)
```

Mirror for the warm call with `trace_warm`. Remove the now-unused `NonlinearConstraint`/`minimize` imports. Keep the history dicts exactly as they are (they read `res.nit/status/success`, all present on the driver's `OptimizeResult`), and when `record_history`, add `'trace': trace_cold` / `'trace': trace_warm` keys to the respective history entries (Task 11 surfaces them).

`disp` has no driver equivalent — drop it (tracing replaces it, per the driver's docstring).

- [ ] **Step 2: Verify equivalence via existing suite** — `pytest tests/ -n auto -q -k "fullgrid or tri_slsqp or integration"` then the full gate. Numerical results must be unchanged (byte-identical driver); any test diff = bug in the swap.
- [ ] **Step 3: Commit** — `git commit -am "feat(slsqp-fullgrid): tri2d on traced driver"`

---

### Task 8: Swap fullgrid tet3d + coupled-kring

**Files:**
- Modify: `dvfopt/core/slsqp_fullgrid/tet3d.py` (1 site), `dvfopt/core/wallbreakers/_coupled_kring_3d.py` (1 site)

**Interfaces:**
- Consumes: Task 6 names.

- [ ] **Step 1: tet3d swap** — same transform as Task 7 (one cold call; `_obj` returns `(value, grad)`; `jac_func = build_tet_sparse_jac(D, H, W)` is sparse — `ineq_dict` densifies): `cons = [ineq_dict(_constr, jac_func, lb=threshold)]`, `trace = {} if record_history else None`, driver call with `maxiter=max_iter, ftol=ftol`.
- [ ] **Step 2: kring swap** — its constraint is already an old-style dict, but `jac` is only present when `use_analytical_jacobian=True` (default False) and the driver REQUIRES jac. Replace the `minimize` call with a branch:

```python
if use_analytical_jacobian:
    from dvfopt.core.primitives.slsqp import minimize_slsqp_traced

    res = minimize_slsqp_traced(
        obj, x0, jac=obj_grad, constraints=[constraint_dict],
        maxiter=maxiter, ftol=ftol,
    )
else:
    # ponytail: FD-jacobian path stays on scipy — the traced driver
    # deliberately requires analytic jacs; upgrade when kring defaults flip.
    res = minimize(obj, x0, jac=obj_grad, constraints=[constraint_dict],
                   method='SLSQP', options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})
```

(`obj`/`obj_grad` are already separate callables here — no lambda splitting needed.)
- [ ] **Step 3: Gate** (explicitly `pytest tests/test_coupled_kring_3d.py tests/test_integration_3d.py -q` + full gate)
- [ ] **Step 4: Commit** — `git commit -am "feat(slsqp): tet3d fullgrid + kring(analytic) on traced driver"`

---

### Task 9: Swap schwarz per-cluster solver

**Files:**
- Modify: `dvfopt/core/schwarz/_cluster.py` (2 sites: `obj_l2` L2 passes, `obj_l1` polish)

- [ ] **Step 1: Swap both calls.** `nl` is the existing `NonlinearConstraint(cons_fun, lb=threshold, ...)`-style object — locate its construction in `_cluster.py`, build `cons = [ineq_dict(<its fun>, <its jac>, lb=<its lb>)]` once next to it, and replace both `minimize(obj_x, z_init, jac=True, method='SLSQP', constraints=[nl], options={...})` calls with `minimize_slsqp_traced(lambda z: obj_x(z)[0], z_init, jac=lambda z: obj_x(z)[1], constraints=cons, maxiter=<same>, ftol=<same or default>)`. The L2 call passes `maxiter=l2_max_iter` (no ftol option today → use the driver default `ftol=1e-6`... **NO** — scipy's default for SLSQP is also 1e-6, so omitting preserves behavior; the L1 call passes `ftol=1e-9` as today.)
- [ ] **Step 2: Gate** (`pytest tests/test_tri_schwarz.py tests/test_m14_schwarz.py -q` + full)
- [ ] **Step 3: Commit** — `git commit -am "feat(schwarz): per-cluster solves on traced driver"`

---

### Task 10: Swap the windowed family (with `method_name` knob preserved)

**Files:**
- Modify: `dvfopt/core/slsqp_windowed/_window.py` (2 sites), `dvfopt/core/slsqp_windowed/coordinator3d.py` (2 sites)

**Interfaces:**
- Produces: private helper `_window_minimize(obj, x0, constraints, maxiter, method_name, disp=False)` in `_window.py`, imported by `coordinator3d.py` — ALL four sites route through it.

- [ ] **Step 1: One shared shim** in `_window.py` (root-cause rule: four identical call shapes → one helper):

```python
from scipy.optimize import NonlinearConstraint

from dvfopt.core.primitives.slsqp import ineq_dict, minimize_slsqp_traced


def _window_minimize(obj, x0, constraints, maxiter, method_name, disp=False):
    """Route a window subproblem: traced C-SLSQP when possible, scipy otherwise.

    The traced driver requires analytic constraint jacobians; the windowed
    constraint builders attach them for every constraint family, but the
    ``method_name`` knob (public API) may select a non-SLSQP method — those
    and any jac-less constraint fall back to ``scipy.optimize.minimize``.
    """
    cons = []
    if method_name == "SLSQP":
        for c in constraints:
            if isinstance(c, NonlinearConstraint) and callable(c.jac):
                lb = float(np.min(c.lb)) if np.ndim(c.lb) else float(c.lb)
                cons.append(ineq_dict(c.fun, c.jac, lb=lb))
            elif isinstance(c, dict) and callable(c.get("jac")):
                cons.append(c)
            else:
                cons = None
                break
        if cons is not None:
            return minimize_slsqp_traced(
                lambda z: obj(z)[0], x0, jac=lambda z: obj(z)[1],
                constraints=cons, maxiter=maxiter,
            )
    return minimize(
        obj, x0, jac=True, constraints=constraints,
        options={"maxiter": maxiter, "disp": disp}, method=method_name,
    )
```

- [ ] **Step 2: Route the four call sites** through it — in `_window.py` (`_full_grid_step`, `_optimize_single_window`) and `coordinator3d.py` (both `result = minimize(...)` blocks): `result = _window_minimize(lambda phi1: objective_euc(phi1, <anchor>), <x0>, constraints, max_minimize_iter, method_name, disp=<as today>)` — keep each site's exact anchor/x0 variables.
- [ ] **Step 3: Gate.** Windowed family has the deepest coverage — run the full suite; explicitly confirm `tests/test_slp_strategy.py tests/test_integration.py tests/test_parallel_internals.py` green. Results must be numerically identical on SLSQP defaults.
- [ ] **Step 4: Commit** — `git commit -am "feat(slsqp-windowed): route window solves through traced driver (method_name fallback kept)"`

---

### Task 11: SLSQP tracing → SolveInfo

**Files:**
- Modify: `dvfopt/strategies/slsqp.py` (fullgrid 2-tri + tet3d strategies), `dvfopt/core/slsqp_fullgrid/tri2d.py`, `dvfopt/core/slsqp_fullgrid/tet3d.py` (already emit `'trace'` in history from Tasks 7–8 — verify), `tests/test_convergence.py` or new `tests/test_slsqp_tracing.py`

**Interfaces:**
- Produces: with `record_history=True`, `SolveInfo.extras['slsqp_trace']` = list of per-phase trace dicts (`[{'phase': 'cold', 'iters': [...], ...}, ...]`) for `SLSQPFullGridStrategy` and the tet3d fullgrid strategy. Windowed/parallel: deliberately not traced (`# ponytail:` comment at `_window_minimize`: per-window traces omitted; add if the GUI grows a per-window inspector).

- [ ] **Step 1: Failing test** — `tests/test_slsqp_tracing.py`:

```python
"""record_history=True surfaces per-major-iteration SLSQP traces in SolveInfo."""

import numpy as np

from dvfopt import L1Objective, Solver, SLSQPFullGridStrategy, TriConstraint2D


def _folded_field(h=8, w=8):
    rng = np.random.default_rng(3)
    phi = rng.normal(0, 0.8, (2, h, w))  # [dy, dx], strong enough to fold
    return phi


def test_fullgrid_trace_in_solveinfo():
    phi = _folded_field()
    solver = Solver(
        constraint=TriConstraint2D(shape=phi.shape[1:]),
        objective=L1Objective(),
        strategy=SLSQPFullGridStrategy(),
    )
    res = solver.fit(phi, record_history=True)
    traces = res.info.extras.get("slsqp_trace")
    assert traces, "expected slsqp_trace in SolveInfo.extras"
    assert traces[0]["iters"], "trace must contain major-iteration records"
    assert {"obj", "max_viol", "opt", "alpha"} <= set(traces[0]["iters"][0])
```

(Adapt the import names to the actual public exports — `from dvfopt import ...` already exposes strategies; check `res.info`/`res` attribute naming against `SolveResult` in `dvfopt/solver.py` and use its real field, e.g. `res.info` or `res.solve_info`, whichever exists.)

- [ ] **Step 2: Run to verify failure** — trace key absent → assertion fails.
- [ ] **Step 3: Implement.** In `slsqp_fullgrid/tri2d.py`/`tet3d.py`: ensure the history entries carry `'trace'` (Tasks 7–8 added them). In `strategies/slsqp.py`, after `self._finish(...)` builds the SolveInfo for those two strategies, lift traces: `info.extras['slsqp_trace'] = [{'phase': h.get('phase', f'run{i}'), **h['trace']} for i, h in enumerate(raw_history) if isinstance(h, dict) and h.get('trace')]` — implement by capturing the raw history before `_finish` (the strategy already holds it when `record_history=True`).
- [ ] **Step 4: Run tests** — new test green + full gate.
- [ ] **Step 5: Commit** — `git commit -am "feat(slsqp): per-major-iteration traces surfaced in SolveInfo.extras"`

---

### Task 12: Objectives made real (engine + wallbreakers + windowed)

**Files:**
- Modify: `dvfopt/objectives.py` (absorb `anchor_term`), `dvfopt/core/barrier/_core.py` (engine takes `objective=`), all `dvfopt/core/wallbreakers/*.py` with `anchor`/`eps_l1` params, `dvfopt/core/slsqp_fullgrid/{tri2d,tet3d}.py`, `dvfopt/core/slsqp_windowed/{_window,coordinator3d}.py` (use `objective(diff)`), `dvfopt/strategies/{barrier,slsqp,wallbreakers,schwarz_wrapper}.py` (drop unwrapping)
- Delete: `dvfopt/core/slsqp_windowed/_objective.py`
- Test: existing suite + `tests/test_objective.py` updated to target `L2Objective`

**Interfaces:**
- Produces: `dvfopt.objectives.anchor_term(diff, kind, eps_l1=1e-4)` (moved verbatim from `_core`; re-export removed there). Engine signature: `run_penalty_barrier_lbfgs(..., objective: Objective | None = None)` — `None` means `L2Objective()`. Every wallbreaker impl signature `anchor: str = 'l2', eps_l1: float = 1e-4` becomes `objective: Objective | None = None`. Conversion rule for ALL call sites (benchmarks/tests included): `anchor='l1', eps_l1=X` → `objective=L1Objective(eps=X)`; `anchor='l2'` → `objective=L2Objective()`; `anchor='none'` → `objective=NoneObjective()`; variable pair `anchor=a, eps_l1=e` → `objective=make_objective(a, eps_l1=e)`.

- [ ] **Step 1: Move `anchor_term`** from `dvfopt/core/barrier/_core.py` into `dvfopt/objectives.py` verbatim (delete the `objectives → core` import; `objectives.py` becomes pure numpy). In `_core.py`: `from dvfopt.objectives import Objective`. Then sweep every remaining importer: `git grep -n "anchor_term" -- dvfopt dvfopt_gui tests benchmarks scripts` → point all imports at `dvfopt.objectives.anchor_term` (CLAUDE.md's building-blocks table row updates in Task 16).
- [ ] **Step 2: Engine.** In `_core.py`, `_penalty_objective`/`_barrier_objective`/`run_penalty_barrier_lbfgs` replace the `anchor, eps_l1` params with `objective`; the line `val, grad = anchor_term(diff, anchor, eps_l1)` becomes `val, grad = objective(diff)`; `run_penalty_barrier_lbfgs` defaults `objective=None` → `objective = objective or L2Objective()` at the top (import `L2Objective`).
- [ ] **Step 3: Wallbreakers + fullgrid.** Apply the signature change and conversion rule through `_alm.py`, `_alm_3d.py`, `_harmonic_polished.py`, `_l2_refine.py`, `_l2_refine_3d.py`, `_refine_repair.py`, `_refine_repair_3d.py`, `_m14_schwarz.py`, `_m14_schwarz_3d.py`, `slsqp_fullgrid/tri2d.py` (`_obj` becomes `lambda z: objective(z - z_anchor)` semantics: `def _obj(z): return objective(z - z_anchor)`), `slsqp_fullgrid/tet3d.py` (same). Sweep for stragglers: `git grep -n "anchor=\|eps_l1=" -- dvfopt benchmarks tests scripts` and convert every hit by the rule (leave `eps_l1` where it is a *constructor* arg of `L1Objective`/`make_objective`).
- [ ] **Step 4: Windowed.** Replace `objective_euc(phi1, anchor_flat)` at the four `_window_minimize` call sites with a bound objective: `obj = lambda z, _a=<anchor_var>: objective(z - _a)` where `objective` is threaded down from the strategy (add `objective: Objective | None = None` param through `iterative_serial`/`iterative_3d`/`iterative_parallel` → coordinator/_window functions; `None` → `L2Objective()`). Delete `slsqp_windowed/_objective.py`; update `tests/test_objective.py`/`test_formula_and_boundary.py`/`test_edge_cases.py` references to assert the same formulas via `L2Objective()(diff)`.
- [ ] **Step 5: Strategies.** Delete every `anchor=objective.label or 'l2'` / `eps_l1=getattr(objective, 'eps', 1e-4)` pair in `strategies/*.py` → pass `objective=objective`. In `strategies/slsqp.py` delete the `UserWarning` block (Task 13 adds the construction-time check). `BarrierStrategy.anchor_override` knob: keep, now `objective_override: Objective | None = None` (rename; it existed to force a kind — an Objective instance is the honest form now). After the rename run `git grep -n "anchor_override" -- dvfopt_gui tests benchmarks` and update hits (the GUI Params dialog renders dataclass knobs generically, but persisted param dicts or tests may name it).
- [ ] **Step 6: Gate** — full suite; run `pytest tests/test_barrier_core.py tests/test_algorithm_logic.py tests/test_invariants.py -q` explicitly. Numerical outputs must be unchanged (same formulas, new plumbing).
- [ ] **Step 7: Commit** — `git commit -am "refactor(objectives): Objective plumbed end-to-end; anchor/eps_l1 string pairs removed"`

---

### Task 13: `accepts_objectives` + `IncompatibleObjectiveError` (TDD)

**Files:**
- Modify: `dvfopt/exceptions.py`, `dvfopt/strategies/base.py`, `dvfopt/strategies/slp.py`, `dvfopt/solver.py`
- Test: `tests/test_constraints_and_params.py` (or new `tests/test_accepts_objectives.py`)

**Interfaces:**
- Produces: `dvfopt.exceptions.IncompatibleObjectiveError(DVFoptError, TypeError)`; `Strategy.accepts_objectives: tuple[type, ...] | None = None`; `Strategy._check_objective(objective)`; `Solver.__init__` calls it after `_check_constraint`. `SLPStrategy.accepts_objectives = (L1Objective, NoneObjective)`.

- [ ] **Step 1: Failing test** — `tests/test_accepts_objectives.py`:

```python
import pytest

from dvfopt import L1Objective, L2Objective, Solver, TriConstraint2D
from dvfopt.exceptions import IncompatibleObjectiveError
from dvfopt.strategies import SLPStrategy


def test_slp_rejects_l2_at_construction():
    with pytest.raises(IncompatibleObjectiveError, match="SLPStrategy"):
        Solver(
            constraint=TriConstraint2D(shape=(8, 8)),
            objective=L2Objective(),
            strategy=SLPStrategy(),
        )


def test_slp_accepts_l1():
    Solver(
        constraint=TriConstraint2D(shape=(8, 8)),
        objective=L1Objective(),
        strategy=SLPStrategy(),
    )  # must not raise
```

- [ ] **Step 2: Run to verify failure** (ImportError on `IncompatibleObjectiveError`).
- [ ] **Step 3: Implement.** `exceptions.py`: `class IncompatibleObjectiveError(DVFoptError, TypeError): """Strategy × objective mismatch (e.g. SLP with an L2 objective)."""` + docstring listing update. `strategies/base.py`: add the class attr and, mirroring `_check_constraint`:

```python
def _check_objective(self, objective) -> None:
    from dvfopt.exceptions import IncompatibleObjectiveError

    if self.accepts_objectives is not None and not isinstance(
        objective, self.accepts_objectives
    ):
        accepted = ', '.join(t.__name__ for t in self.accepts_objectives)
        raise IncompatibleObjectiveError(
            f'{type(self).__name__} requires one of ({accepted}); '
            f'got {type(objective).__name__}'
        )
```

`slp.py`: `accepts_objectives = (L1Objective, NoneObjective)` (import from `dvfopt.objectives`). `solver.py` `__init__`: `self.strategy._check_objective(self.objective)` right after the constraint check.
- [ ] **Step 4: GUI check.** `dvfopt_gui/worker.py` defaults `objective_id='l1'` — SLP paths fine. Verify `auto_strategy` never returns `'slp'` for an `'l2'` label (read `dvfopt/solver.py::auto_strategy`; per CLAUDE.md it routes 2-tri+L1 only — if any branch can pair slp with l2, guard it there). Run `pytest tests/test_gui_strategy_parity.py tests/test_gui_logic.py -q`.
- [ ] **Step 5: Gate + commit** — `git commit -am "feat(solver): accepts_objectives construction-time check"`

---

### Task 14: Absorb `laplacian/` + `test_cases/`; packaging to 0.5.0

**Files:**
- Move: `laplacian/` → `dvfopt/laplacian/`; `test_cases/` → `dvfopt/testdata/`
- Modify: `pyproject.toml` (packages, version), every importer (rewrite rows 23–24)

- [ ] **Step 1: Moves**

```bash
git mv laplacian dvfopt/laplacian
git mv test_cases dvfopt/testdata
```

- [ ] **Step 2: Rewrite rows 23–24** — `python tools/rewrite_imports.py laplacian=dvfopt.laplacian test_cases=dvfopt.testdata` then **hand-audit** the diff: the bare words `laplacian`/`test_cases` appear in prose/comments and in `dvfopt/laplacian`'s own intra-package imports (`from laplacian.utils import ...` → `from dvfopt.laplacian.utils import ...` is CORRECT; but a comment like "the laplacian solver" must NOT have become "the dvfopt.laplacian solver" — revert prose hits; the tool only touches dotted-path word boundaries, so bare-word prose is untouched, but single-word `import laplacian` lines DO rewrite — that is intended). Data paths inside `testdata/_builders.py` are cwd-relative (`data/...`) — verified, leave untouched.
- [ ] **Step 3: Packaging.** `pyproject.toml`: `include = ["dvfopt*", "dvfopt_gui*"]`; `version = "0.5.0"`. Check `[tool.mypy]`/`[tool.ruff]`/workflows/`noxfile.py` for `laplacian`/`test_cases` path mentions and update (ruff/mypy target lists shrink — the absorbed dirs ride along under `dvfopt`).
- [ ] **Step 4: Reinstall + gate.** `pip install -e ".[dev,fast,gui]"` (metadata changed), then the full gate + mypy. `python -c "from dvfopt.testdata import SYNTHETIC_CASES; from dvfopt.laplacian import solver"` smoke.
- [ ] **Step 5: Commit** — `git commit -am "refactor!: absorb laplacian + test_cases into dvfopt; version 0.5.0"`

---

### Task 15: Notebooks + leftover sweep

**Files:**
- Modify: `notebooks/*.ipynb` and `benchmarks/**/*.ipynb` code cells (imports only)

- [ ] **Step 1: Notebook rewrite script** (run from worktree root; throwaway, do not commit):

```python
# scratch: rewrite_nbs.py — run with .venv python, then delete
import json
import re
from pathlib import Path

PAIRS = [  # full canonical table, rows 1-25, longest-old-first
    ("dvfopt.core._internal.constraint_values", "dvfopt.core.primitives.constraint_values"),
    ("dvfopt.core.iterative3d_tet_barrier_torch", "dvfopt.core.barrier.tet3d_torch"),
    ("dvfopt.core.iterative3d_barrier_torch", "dvfopt.core.barrier.jdet3d_torch"),
    ("dvfopt.core.iterative2d_tri_schwarz", "dvfopt.core.schwarz.tri2d"),
    ("dvfopt.core.iterative2d_tri_barrier", "dvfopt.core.barrier.tri2d"),
    ("dvfopt.core.iterative2d_tri_slsqp", "dvfopt.core.slsqp_fullgrid.tri2d"),
    ("dvfopt.core.iterative3d_tet_slsqp", "dvfopt.core.slsqp_fullgrid.tet3d"),
    ("dvfopt.core.wallbreakers._schwarz_common", "dvfopt.core.schwarz._common"),
    ("dvfopt.core.iterative3d_barrier", "dvfopt.core.barrier.jdet3d"),
    ("dvfopt.core.iterative2d_barrier", "dvfopt.core.barrier.jdet2d"),
    ("dvfopt.core.barrier_objective", "dvfopt.core.primitives.jdet3d"),
    ("dvfopt.core.tri_primitives", "dvfopt.core.primitives.tri"),
    ("dvfopt.core._internal.metrics", "dvfopt.core.slsqp_windowed._metrics"),
    ("dvfopt.core._internal.window", "dvfopt.core.slsqp_windowed._window"),
    ("dvfopt.core._internal.io", "dvfopt.core.slsqp_windowed._io"),
    ("dvfopt.core._barrier_core", "dvfopt.core.barrier._core"),
    ("dvfopt.core._cluster_2tri", "dvfopt.core.schwarz._cluster"),
    ("dvfopt.core.solver3d", "dvfopt.core.slsqp_windowed.coordinator3d"),
    ("dvfopt.core.objective", "dvfopt.core.slsqp_windowed._objective"),
    ("dvfopt.core.solver", "dvfopt.core.slsqp_windowed.coordinator"),
    ("dvfopt.core._nmvf", "dvfopt.core.nmvf"),
    ("dvfopt.core.slsqp", "dvfopt.core.slsqp_windowed"),
    ("test_cases", "dvfopt.testdata"),
    ("slsqp_traced", "dvfopt.core.primitives.slsqp"),
    ("laplacian", "dvfopt.laplacian"),
]
PATS = [(re.compile(rf"(?<![\w.]){re.escape(o)}(?![\w])"), n) for o, n in PAIRS]

for nb_path in list(Path("notebooks").rglob("*.ipynb")) + list(Path("benchmarks").rglob("*.ipynb")):
    if "experiments" in nb_path.parts:
        continue
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    dirty = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        new = src
        for pat, repl in PATS:
            new = pat.sub(repl, new)
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
            dirty = True
    if dirty:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("rewrote", nb_path)
```

**Caution:** the bare-word rows (`laplacian`, `test_cases`) will hit notebook code like `import laplacian` (intended) — but NOT comments/markdown (script only touches code cells, dotted word boundaries). Skim the git diff of every touched notebook before committing; revert false positives (e.g. a local variable named `laplacian`, plain `sys.path` hacks now redundant — delete those lines).

- [ ] **Step 2: Leftover sweep.** `git grep -nE "dvfopt\.core\.(slsqp\b|solver|objective\b|_internal|_barrier_core|_cluster_2tri|_nmvf|iterative[23]d|tri_primitives|barrier_objective|wallbreakers\._schwarz_common)|^(from|import) (laplacian|test_cases|slsqp_traced)\b" -- . ":!research" ":!archive" ":!writing" ":!data" ":!docs" ":!notebooks/experiments"` → must be empty.
- [ ] **Step 3: Gate + notebook import smoke.** Full gate; plus `python -c "import nbformat"`-based spot check is overkill — instead execute the lightest canonical notebook's import cell manually if any doubt. CLI smoke: `.venv/Scripts/dvfopt info` (or `python -m dvfopt info`) exits 0.
- [ ] **Step 4: Commit** — `git commit -am "refactor: migrate notebooks + final import sweep"`

---

### Task 16: Docs, CHANGELOG, CI — final gate

**Files:**
- Create: `ARCHITECTURE.md`
- Modify: `CLAUDE.md`, `CHANGELOG.md`, `.github/workflows/{ci,test}.yml`, `README.md` (import examples, if any old paths), `AGENTS.md` (if it names old paths)

- [ ] **Step 1: `ARCHITECTURE.md`.** Sections: (1) the three axes + Solver composition diagram (text); (2) dependency rules (constraints→primitives only; strategies→core/<method>; methods→primitives + engines `barrier._core`/`schwarz._common`; objectives pure); (3) phi-pack conventions table (from CLAUDE.md, updated module names); (4) **Add a method** checklist: `core/<name>/` package → `strategies/<name>.py` dataclass, `@register_strategy('<label>')`, declare `accepts_constraints`/`accepts_objectives`/`supports_3d` → `tests/test_<name>.py` → optional GUI: registry label in `worker._MID_TO_LABEL` + menu-spec row + table row (parity-tested by `tests/test_gui_strategy_parity.py`); (5) **Add a constraint** checklist (subclass `Constraint`, implement values/adjoint/flatten/unflatten (+jacobian for SLSQP), `@register_constraint`, pack declared); (6) **Add an objective** checklist (subclass `Objective.__call__` returning `(value, grad)`; nothing else). Keep it under ~200 lines; it documents rules, not modules.
- [ ] **Step 2: `CLAUDE.md` rewrite.** Update: architecture section module paths (use the canonical table), strategy→impl delegation table (new paths), building-blocks table (`anchor_term` now `dvfopt.objectives`; `tri_areas_flat` etc. at `core.primitives.tri`; add `minimize_slsqp_traced`/`ineq_dict` row), directory layout (laplacian/test_cases now inside dvfopt; `dvfopt.testdata`), **delete** the false "Composition (`+`, `*`) supported" sentence, phi-flattening section module lists, and note `accepts_objectives`. Keep unaffected sections (2.5D, GUI, benchmarks) except path fixes.
- [ ] **Step 3: `CHANGELOG.md`** 0.5.0 entry: headline bullets (method-first core; Objective plumbed; traced C-SLSQP driver at all SLSQP sites — byte-identical, adds tracing; one-package absorption; `scipy>=1.15,<1.19`) + the full old→new import table (copy the canonical table).
- [ ] **Step 4: CI.** Open both workflows; fix any explicit path (`laplacian`, `test_cases`, moved core files) in lint/mypy/test/smoke steps. The benchmark-import smoke must still pass (benchmarks now import `dvfopt.core.primitives.slsqp`).
- [ ] **Step 5: FINAL GATE.** `pytest tests/ -n auto -q` (compare to Task 0 baseline — same-or-more passed, zero failures) + `ruff check` + `ruff format --check` + `mypy` + `pytest tests/test_gui_strategy_parity.py -q` + CLI smoke + `python -c "import dvfopt; print(dvfopt.__version__ if hasattr(dvfopt,'__version__') else 'ok')"`.
- [ ] **Step 6: Commit** — `git commit -am "docs: ARCHITECTURE.md + CLAUDE.md/CHANGELOG for 0.5.0 reorg"`

---

## Post-plan (orchestrator, not a task)

Push `refactor/library-architecture`, open the PR against `main` on **UCI-iGravi/dvfopt** (never the heemmanshuu remote), run /code-review, fix findings, squash-merge on green CI. PR body: goal, canonical mapping table, "byte-identical driver" note, breaking-changes list.
