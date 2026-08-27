# The 3D analogue of the both-diagonals (bilinear) constraint

Research note, branch `research-3d-all-tets`. Nothing here is merged into the
library; the prototype lives in
[`benchmarks/tet_all_constraint.py`](../../../benchmarks/tet_all_constraint.py).

**Motivating 2D lesson.** The fixed-diagonal 2-triangle metric is blind to folds
on the *other* diagonal: a 2-tri-clean z0 slice of B0039 still hid 5518 folds
under the both-diagonals (bilinear) metric, and enforcing all 4 triangles per
cell was what made the twisted cells solvable — the constraint's blind spot was
exactly where the hard cells lived. Question: what is the 3D analogue, and does
it matter on real data?

Short answer: **the analogue is 24 distinct tets per cell, and yes, it matters —
for the same reason it did in 2D.** On raw B0039 the fixed-diagonal metric
under-reports folds by 23% volume-wide. Worse, a `BarrierStrategy` run that
drives a real block to *strict* 6-tet feasibility (0 folds, floor 0, above
threshold — a field the library reports as perfectly corrected) leaves **557
inverted cells, 1.87%, worst -8.05** that the metric cannot see — four times
deeper than the worst fold in the input it corrected. The 2D case that motivated
this hid 3.8%; this is the same phenomenon at the same order of magnitude. And
at **matched wall-clock**, solving under the 24 rows instead beats it on every
axis: 70x fewer real folds, a 1800x shallower worst case, and a *smaller* move.

Two caveats that shape the recommendation: `correct_dvf_3d`'s full chain happens
to land nearly clean anyway (1-29 hidden cells across two blocks), so this is a
property of the *solver*, not a universal; and no tet set is an exact certificate
in 3D the way the 2D 4-triangle set is — adding the 5-tet splits finds yet
another comparable batch. Recommendation in §6.

---

## 1. Derivation: how many distinct tets?

A hexahedral cell (8 corners, indexed `i -> (oz, oy, ox) = ((i>>2)&1,
(i>>1)&1, i&1)`) has **4 body diagonals**: `(0,7) (1,6) (2,5) (3,4)`. Each one
defines a Kuhn/Freudenthal 6-tet split — the fan of 6 tets sharing that
diagonal, one per cube edge disjoint from both endpoints. Four splits x 6 tets
= **24 tets, all distinct**.

*Why the dedup is a no-op.* Every tet in the fan around diagonal `(s, e)` has
vertex set `{s, e, a, b}` where `(a, b)` is a cube **edge**. For a tet to belong
to two different fans it would need to contain two body diagonals, i.e. one of
them would have to be the `(a, b)` pair — but a body diagonal is not an edge.
So no tet is shared between fans. (Asserted in the prototype's self-check.)

The cube also admits **two 5-tet splits**: the central tet is one parity class
of vertices (`{0,3,5,6}` or `{1,2,4,7}`), plus 4 corner tets, each a vertex with
its three edge-neighbours. Those add 10 more distinct tets — 2 central + 8
corner — for **34 distinct tets** over all six decompositions. The 8 corner tets
are exactly the **8 corner Jacobians of the trilinear map** (`6 * V_corner` =
det of the three edge vectors at that vertex), which are 8 of the 27 Bernstein
coefficients of the trilinear cell's degree-(2,2,2) Jacobian determinant. None
of them appear among the 24.

### What the 24 rows certify — and what they don't

This is the point where the 2D analogy stops being exact, and it matters:

| | 2D (bilinear, 4 rows/cell) | 3D (24 rows/cell) |
|---|---|---|
| Interpolant Jdet | biaffine on the cell | degree-(2,2,2) polynomial |
| Cell minimum | attained at a corner | **not** attained at a corner |
| Rows | 4 triangles = the 4 corner Jdets (x1/2) | 24 tets, none of which are corner Jdets |
| Statement | `min(rows) = 0.5 * cell_min_jdet_2d` — an **exact** certificate of bilinear injectivity | all four PL interpolants simultaneously orientation-preserving — **not** a trilinear certificate |

So `AllTetConstraint3D` is a *decomposition-independent* strengthening (it stops
depending on the arbitrary choice of main diagonal), and a necessary condition
for the 34-row / trilinear statements. It is not the sub-cell exactness result
the 2D bilinear rows give. Getting that in 3D means the Bernstein/Bezier
coefficients of the degree-(2,2,2) Jacobian (27 rows/cell, a genuinely different
constraint family), not more tets.

### A three-level ladder (useful framing)

The library already has both ends of this; the prototype fills the middle:

| Metric | Per-cell value | Meaning |
|---|---|---|
| best-diagonal (`n_neg_best_diagonal`) | `max_d min_k V[d,k]` | optimistic — *some* split works ("geometric floor" when it fails) |
| 6-tet (`six_tet_min_volume_3d`) | `min_k V[0,k]` | today's metric — one arbitrary fixed split |
| **24-tet (this prototype)** | `min_d min_k V[d,k]` | pessimistic — *every* split works |

## 2. Prototype and verification

`benchmarks/tet_all_constraint.py`:

- `build_tet_table(include_5tet)` -> `(tets, signs)`, `K = 24` or `34`; signs
  normalised so the identity field gives a positive volume on every row. Rows
  `0..5` are exactly `SimplexConstraint3D`'s split, in its order.
- `AllTetConstraint3D(shape, include_5tet=False)` — `Constraint` with
  `pack = DX_FIRST`, `dim = 3`, `n_constraints = K * (D-1)(H-1)(W-1)`,
  `values` / `adjoint` / `flatten` / `unflatten`. It **delegates** the
  flatten/coerce plumbing to a held `SimplexConstraint3D` rather than
  subclassing it, deliberately: the 6-tet-only strategies gate with
  `accepts_constraints = (SimplexConstraint3D,)`, so a subclass would be
  silently accepted and then silently solved on 6 of its 24 rows.
- No sparse `jacobian()` — barrier and SLP need only the adjoint, and SLSQP is
  already impractical at 3D scale on 6 rows/cell.

Self-check (`python benchmarks/tet_all_constraint.py`), all passing:

```
derivation: 4 diagonals x 6 tets = 24 raw -> 24 distinct
            + 2 five-tet splits (10 more) -> 34 distinct
            8 corner tets (= trilinear corner Jacobians) absent from the 24  OK
identity field: every row positive; 24-row set all = +1/6  OK
shared 6 tets: max |diff| = 0 (exact)  OK
per-cell min == six_tet_volumes_all_diagonals(...).min(0)  OK
rows exactly cubic along a line: max |resid| = 2.78e-16  OK
AllTetConstraint3D(shape=(5, 5, 5), n_tets=24): FD gradient rel err = 4.621e-10  OK
AllTetConstraint3D(shape=(5, 5, 5), n_tets=34): FD gradient rel err = 7.837e-10  OK
```

- Values against `SimplexConstraint3D` on the shared 6 tets: **bit-exact**
  (max |diff| = 0), and the flat decision vector is `array_equal`.
- Per-cell min matches the library's own `six_tet_volumes_all_diagonals(...)
  .min(0)` to < 1e-15 — an independent check of the sign normalisation.
- Central-difference gradient check of `adjoint` vs `d(v . values)/dphi` on 40
  random coordinates: **rel err ~5e-10 / ~8e-10** for the 24- and 34-row
  variants (target was < 1e-6).
- A row is **exactly cubic** along a line (fit at `a in {0, 1/3, 2/3, 1}`,
  predict `a = 1/2`, residual 2.8e-16) — the fact §5.5's exact line search
  rests on.

Cost (pure numpy; this box has no numba, so all tet kernels run their fallback
path), 48^3 block: `values` 175 ms vs the 6-tet 61 ms, `adjoint` 682 ms vs
182 ms — about 4x, i.e. linear in the row count. A table-parameterised numba
kernel (the existing `_all_diag_min_kernel` already fuses all 24 for the min)
would close that; not needed for a prototype.

## 3. The blind spot on real data (B0039)

`data/dvfs/b0039/b0039_laplacian_deformation_field.npy`, `(3, 528, 320, 456)`,
`dz == 0` (per-slice Laplacian field). Fold = per-cell min volume `<= 0`.

### Whole volume (76.5 M cells), raw

| Metric | Folded cells |
|---|---|
| 6-tet (fixed diagonal) | **1 190 710** |
| 24-tet (all diagonals) | **1 464 374** |

**+273 664 folds, +23.0%.** So the fixed-diagonal metric under-reports the raw
volume's fold burden by roughly a quarter.

### Worst 64^3 block, raw — origin (z, y, x) = (0, 160, 160), 250 047 cells

| Metric | folds (<= 0) | below 0.01 | min |
|---|---|---|---|
| best-diagonal (some split works) | 46 820 | 56 390 | -248.34 |
| 6-tet (fixed) | 48 952 | 58 101 | -248.34 |
| 24-tet (all) | 51 589 | 60 217 | -269.19 |
| 34-tet (+5-tet splits) | 54 801 | 61 142 | -306.61 |

Cells that are 6-tet-clean but 24-tet folded: **2 637** (1.3% of the 201 095
6-tet-clean cells). Cells 24-tet-clean but 34-tet folded: **3 212** — the 5-tet
splits' corner tets find a comparable amount again, which is the concrete
evidence that "more tets" has no natural stopping point in 3D.

### How much does a 6-tet-clean field hide? (the "2-tri-clean hides 5518" question)

Block B: 32^3 at origin (224, 64, 64), 29 791 cells, 791 raw folds. Every run
below uses threshold 0.01, objective L1(1e-4). "hidden" = cells the 6-tet metric
calls clean that the 24-tet metric calls folded. L2 is the move from the raw
input.

| solve | 6-tet folds (min) | 24-tet folds (min) | hidden | wall | L2 |
|---|---|---|---|---|---|
| *(raw input)* | 791 (-1.8238) | 883 (-1.8238) | 92 (0.31%) | — | 0 |
| `BarrierStrategy` 6 rows, 200 it | 47 (-0.0142) | 580 (-4.3190) | 533 (1.79%) | 177 s | 27.5 |
| `BarrierStrategy` 6 rows, 600 it | **0** (+0.0110) | **557** (**-8.0501**) | **557 (1.87%)** | 598 s | 38.0 |
| `BarrierStrategy` **24 rows**, 200 it | 5 (-0.0037) | **8** (-0.0044) | 3 (0.01%) | 528 s | 36.0 |
| `correct_dvf_3d` (6 rows, full chain) | **0** (+0.0145) | 1 (-0.0029) | 1 (0.003%) | 1135 s | 64.7 |

**Row 3 is the answer, and it is the 2D result again.** A barrier run that
reaches *strict* 6-tet feasibility — 0 folds, `best_diag_floor` 0, min tet volume
above threshold, a field the library reports as perfectly corrected — contains
**557 inverted cells (1.87%)** that the fixed-diagonal metric cannot see, with the
worst at **-8.05** against an identity tet volume of +1/6. That is *four times
deeper* than the worst fold in the raw input it was correcting. Compare 2D, where
a 2-tri-clean B0039 z0 slice hid 5518 of 145 145 cells — **3.8%**. Same
phenomenon, same order of magnitude.

Two corollaries worth separating:

- **It is not a convergence artifact.** The stalled run (row 2, 47 residual
  folds) and the fully feasible one (row 3, 0 residual folds) hide essentially
  the same number, 533 vs 557 — and pushing to feasibility made the worst hidden
  violation *deeper*, -4.32 -> -8.05. A fixed-diagonal optimizer buys feasibility
  on its own diagonal by transporting the fold onto the three it does not track,
  and the harder you push it the more it transports.
- **It is solver-dependent, not solver-proof.** `correct_dvf_3d` (row 5) happens
  to land almost clean — 1 hidden cell — because its escape and multiscale
  re-seed stages untwist cells rather than trading diagonals. A second block
  (D, 32^3 at (192, 160, 416), 46 raw folds) gave `correct_dvf_3d` 0 6-tet folds
  and **29 hidden** (0.10%), worst -0.1432. So the good behaviour is a property
  of that particular chain, at a wide spread, not a guarantee of the metric.

## 4. Can a solver drive the all-tet constraint?

`BarrierStrategy` is constraint-generic (it only calls `values` / `adjoint`), so
`AllTetConstraint3D` drops straight in. SLP-3D, M10Tet/M14Tet and
SLSQP-fullgrid-3D are 6-tet-specific in their kernels and gate on
`accepts_constraints = (SimplexConstraint3D,)`, so they reject it at `Solver`
construction — which is why the prototype deliberately does not subclass
`SimplexConstraint3D` (see section 2).

It runs, it behaves, and **at matched wall-clock it is decisively the better
formulation.** The 24-row constraint costs 4x per iteration in pure numpy, so
the honest comparison is against a 6-row run given 3x the iteration budget —
`max_iter=600` vs `200`, 598 s vs 528 s:

| barrier run | rows/cell | wall | 6-tet folds (min) | 24-tet folds (min) | L2 move |
|---|---|---|---|---|---|
| `SimplexConstraint3D`, 600 it | 6 | 598 s | **0** (+0.0110) | 557 (**-8.0501**) | 37.97 |
| **`AllTetConstraint3D`, 200 it** | **24** | **528 s** | 5 (-0.0037) | **8** (**-0.0044**) | **36.03** |

Equal work, and the 24-row run delivers **70x fewer decomposition-independent
folds**, a worst-case violation **1800x shallower** (-0.0044 vs -8.05), and a
**smaller** move (L2 36.0 vs 38.0) — for the price of 5 residual folds on the
6-tet metric it was not exclusively optimizing. The 6-row run's "0 folds" is the
metric telling the solver what it wants to hear.

(The un-controlled comparison is in §3's table: at *equal iterations*, 24 rows
gives 5/8 folds against 6 rows' 47/580.)

Enforcing every diagonal removes the escape route the fixed-diagonal formulation
leaves open: the iterate cannot buy cheap progress by twisting the cell onto an
untracked split, so the barrier path stays in a basin that is good under both
metrics — and, because it never has to undo a twist it created, it ends up
moving the field *less*. That is the 2D "the blind spot is exactly where the
hard cells live" lesson, transferred.

Where it stalls: nowhere new. The failure mode is the same
plateau-near-the-boundary the 6-row barrier has (min_T flattening around
-4e-3 across the last three penalty phases), just an order of magnitude further
in. Cost is the one honest downside — 4x the rows is 4x the `values`/`adjoint`
work in pure numpy, and a fused kernel would recover most of it (the existing
`_all_diag_min_kernel` already evaluates all 24 in one pass for the min).


## 5. Design sketch: a 3D windowed engine (NOT attempted here)

Out of scope for this note; recorded so the next person does not re-derive it.
The 2D engine is `dvfopt/core/windowed/` (`windowed_correct` + `_locality.py` +
`_inners.py`, inner `core/primitives/isqp.py`). Piece by piece:

1. **Windows and frozen rings -> frozen shells.** Mechanically the same: label
   the fold map's connected components (3D structure), take bounding boxes, pad,
   free the interior, freeze the shell. Ring width for the tet families is
   **1**, same argument as 2-tri in 2D: a tet volume is an *exact* function of
   its cell's 8 corners, so a cell evaluates correctly iff all 8 corners are
   in-patch, which holds for every cell incident to a voxel 1 in from the patch
   edge. `JdetConstraint3D` needs 2 (central differences). The no-damage
   invariant carries over unchanged and is still the main reason to want this.
2. **`WindowLocality` must go dimension-generic.** Today it is `(2, H, W)` /
   `(ph, pw)` shaped and registered per 2D constraint type; the module docstring
   already flags folding it into `Constraint` as the stage-2 refactor. 3D is
   what forces it.
3. **Cost scaling is the real change.** A side-`s` window has `3 s^3` unknowns
   and `K (s-1)^3` rows. At `s = 16`: 12 288 vars, 20 250 rows for 6-tet,
   **81 000 for 24-tet**. That is still a small QP, but it grows as `s^3` where
   2D grew as `s^2`, so 3D windows must stay small (`s <= 12-16`) and the
   cluster count carries the work. The giant tiler's default tile should be
   ~16-24, not 64, and `giant_tile_fit` becomes a per-axis fit.
4. **Sparse patch Jacobians.** The structural pattern already exists in
   `jacobian/tetrahedron_sign.build_tet_sparse_jac` (each row touches 4 corners
   x 3 components = 12 columns); generalise it over the tet table and slice by
   the enforced-row set, exactly as the 2D triangle path builds its pattern by
   index arithmetic (never by dense probing — that was a 19 GB bug in 2D).
5. **Exact line search: cubic, still closed-form.** A tet volume is the
   determinant of three difference vectors, each affine in the displacements, so
   along `x + a d` a row is a **cubic** in `a` (2D's bilinear rows were only
   quadratic). Four coefficients, and two of them are free: `c = cons(x)` and
   the linear term `g = J d` are already computed. Fit `q, r` from
   `cons(x + d/2)` and `cons(x + d)` — i.e. exactly **one extra constraint
   evaluation** over the 2D path, which already evaluates `cons(x + d)` for the
   ratio test. The merit is `obj + sum_i rho_i max(0, thr - V_i(a))`; its
   breakpoints are the real roots of each row's cubic in `[0, 1]` (Cardano,
   closed form, <= 3 per row), and on each interval the merit is a cubic whose
   stationary points solve a quadratic. So `_exact_line_min`'s structure
   survives; only the root finder changes. Caveat: 24 rows/cell x 3 roots is
   ~18x more breakpoints than 2D bilinear, so the candidate sort becomes the
   line search's cost driver — prefilter to rows near the active band.
6. **Coarse-to-fine** carries over verbatim (trilinear prolongation, same
   free-box masking that preserves no-damage).
7. **Hybrid QP backend** carries over in principle, but expect the crossover to
   move toward warm-started OSQP: 3D stencils fill in far worse than 2D ones, so
   Clarabel's interior-point factorisations cost more.

## 6. Recommendation

**The 2D lesson does transfer. Build it — but build the cheap half first.**
Four findings, in the order they should change decisions:

1. *A 6-tet-feasible field is not a fold-free field.* The `BarrierStrategy` run
   that reached strict feasibility on block B hides 557 inverted cells (1.87%)
   at up to -8.05 — deeper than anything in the raw input. The 2D case that
   motivated this hid 3.8%. Same phenomenon, same order.
2. *It is not a convergence artifact.* Stalled (47 residual folds) and fully
   feasible (0) hide 533 and 557 respectively; pushing to feasibility made the
   worst hidden violation twice as deep. Pushing harder on the wrong constraint
   makes the hidden problem worse, not better.
3. *At matched wall-clock the 24-row constraint wins outright.* 528 s / 24 rows
   vs 598 s / 6 rows: 70x fewer decomposition-independent folds, a 1800x
   shallower worst case, and a **smaller** move (L2 36.0 vs 38.0). There is no
   cost/benefit trade to argue about here once the work is matched — only the
   4x per-iteration constant, which a fused kernel removes.
4. *But it is still not a certificate, and "more tets" has no stopping point.*
   6 -> 24 finds 2 637 extra folded cells on the worst 64^3 block; 24 -> 34 finds
   3 212 more. In 2D the 4 triangle rows **are** the 4 Bernstein coefficients of
   the degree-(1,1) bilinear Jacobian, and biaffine corner-positivity is
   equivalent to cell-positivity — hence exactness. In 3D the trilinear Jacobian
   is degree (2,2,2), so the exact analogue is its **27 Bernstein coefficients**
   (Johnen/Remacle-style hex validity): positivity of all 27 is *sufficient* and
   tightenable to exact by subdivision. 8 of them are the corner Jacobians the
   5-tet splits surface; none of them is a tet.

What to build, in order:

- **First — report the honest number (hours, not days).** Surface the
  all-diagonal fold count next to the fixed-diagonal one in `metrics.fold_stats`,
  the CLI `summary.json`, `Correct3DReport`, and the GUI. Zero new math:
  `six_tet_volumes_all_diagonals` is already a fused kernel, and `grep` shows its
  only non-test consumer today is the thread-pinning warm-up in `core/_pool.py`.
  Note the current asymmetry — `pipeline_3d` reports `best_diag_floor`, the
  **optimistic** end of the §1 ladder, and nothing anywhere reports the
  pessimistic end. Until this ships, every 3D "0 folds" result in the repo is
  unverified in the direction that matters. Highest value per line in this note.
- **Second — promote the constraint.** `AllTetConstraint3D` -> `dvfopt/constraints.py`
  as `SimplexConstraintAllDiag3D` (label `'simplex_alldiag_3d'`), with the
  values/adjoint pair moved into `core/primitives/` as a table-parameterised
  generalisation of the existing 6-tet kernels (the tet table, the sign
  normalisation, and the numba scatter pattern are all already there — this is a
  loop bound and a table argument, not new math). `BarrierStrategy` accepts it as
  written; that alone is a usable, better-behaved 3D solve. Do **not** ship the
  34-row variant: it costs 40% more rows for a strictly weaker story than the
  Bernstein family would give.
- **Third — consider `TrilinearBezierConstraint3D`** (27 fixed rows/cell) if a
  real sub-cell injectivity claim is ever needed, e.g. for a paper asserting the
  *resampled* field is injective rather than the PL one. Fixed row count, an
  actual certificate, and the natural terminus of finding 4. Not urgent.
- **Fourth — the 3D windowed engine.** Worth building for its own reason (the
  frozen-shell no-damage decomposition is the only 3D approach that scales) and
  it is constraint-agnostic, so it will carry whichever constraint wins above.
  Prerequisites in order: (a) the fused kernel from step 2 — the engine is
  call-bound and the pure-numpy 24-row adjoint is 4x the 6-tet cost; (b) the
  dimension-generic `WindowLocality` refactor `_locality.py` already anticipates;
  (c) the cubic exact line search in §5.5. Design sketch in §5 so nobody
  re-derives it. Do not start here.
- **Loose thread:** every existing 3D benchmark number in the repo was measured
  under the fixed-diagonal metric. Step 1 makes them re-measurable; expect some
  of them to move.

---

## Appendix: reproducing the numbers

No numba in the environment used, so every tet kernel ran its pure-numpy
fallback; wall times are that path, on a shared box, with
`OMP` / `OPENBLAS` / `MKL` / `RAYON_NUM_THREADS=1`. Fold counts are exact and
implementation-independent.

Volume scan (~6 min) — per cube-layer, both metrics from one fused call:

```python
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_all_diagonals
a = np.load(B0039_PATH, mmap_mode='r')            # (3, 528, 320, 456)
n6 = n24 = 0
for cz in range(a.shape[1] - 1):
    ad = six_tet_volumes_all_diagonals(np.asarray(a[:, cz:cz + 2], float))[:, 0]
    n6 += int((ad[0] <= 0).sum())                 # fixed diagonal
    n24 += int((ad.min(0) <= 0).sum())            # all four
# -> 1_190_710 and 1_464_374
```

Block solves — the constraint is the only thing that changes between the two
barrier rows of section 4:

```python
from dvfopt import BarrierStrategy, L1Objective, SimplexConstraint3D, Solver
from benchmarks.tet_all_constraint import AllTetConstraint3D
Solver(constraint=AllTetConstraint3D(phi.shape[1:]),   # or SimplexConstraint3D
       objective=L1Objective(eps=1e-4),
       strategy=BarrierStrategy(max_iter=200),         # 600 for the 6-row control
       threshold=0.01).fit(phi, verbose=1)
```

Blocks used (origins into the raw volume, all 32^3 unless noted):
B `(224, 64, 64)`, D `(192, 160, 416)`, worst-64^3 `(0, 160, 160)`. Block
selection: per-32x32-tile fold counts from the volume scan, then the argmax
(worst) or a target-count bisect (B ~ 800, D ~ 60).
