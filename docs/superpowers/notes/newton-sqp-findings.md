# Newton-type SQP in the windowed isqp engine — findings

Research prototype on `proto-newton-sqp`. Code: `benchmarks/isqp_newton.py` (patched
driver) + `benchmarks/newton_sqp_proto.py` (checks + harness). No library change.

**Verdict: DO NOT PROMOTE.** _(numbers below)_

---

## 1. The derivation, and why it is exact

Every 2-tri / bilinear constraint row is a signed triangle area

```
c = -1/2 [ (x_B - x_A)(y_C - y_A) - (y_B - y_A)(x_C - x_A) ]
```

with `x_P = ref_x_P + dx_P`, `y_P = ref_y_P + dy_P`. The reference grid is constant, so
`c` is an inhomogeneous **bilinear form** in the decision variables: exactly quadratic,
with all curvature in the x–y cross terms. Expanding and differentiating twice:

```
H[y_Q, x_P] = -1/2 * cyc(P, Q),      cyc(A,B) = cyc(B,C) = cyc(C,A) = +1
H[y_Q, y_P] = H[x_Q, x_P] = 0        (including the diagonal)
```

Six nonzeros of ±1/2 per row, **constant** (independent of the iterate). In the driver's
flat DY_FIRST packing (`phi[:N] = dy`, `phi[N:] = dx`) the y-var of pixel `p` is `p` and
the x-var is `p + N`, so every entry sits in the strictly-upper triangle already.

**Row → triangle map** (the task's measured layout, re-confirmed): rows are BLOCK-MAJOR,
`row = b*(H-1)*(W-1) + i*(W-1) + j`, with ordered vertices

| block | rows | ordered (A, B, C) |
|---|---|---|
| b=0 | T1 | (TR, BL, BR) |
| b=1 | T2 | (TL, BL, TR) |
| b=2 | U1 | (TL, BL, BR) |
| b=3 | U2 | (TR, TL, BR) |

The U pair is the x-mirror of the T pair (`tri_areas_flat_bilinear` mirrors the field);
mirroring swaps TL↔TR and BL↔BR and flips the area sign, which is where those orderings
come from. The engine's enforced-row set is recomputed with the engine's OWN locality
adapter (`_locality_of(c).influenced(...)`) so the Hessian rows line up with the rows
`sub.cons` / `sub.cons_jac` return, and the Hessian is restricted to `free_idx`
(a principal submatrix — correct, since frozen variables cannot move).

### Verification (`newton_sqp_proto.py --check`)

```
[1] OSQP dual sign: x*=0.999593  y*=-0.999593  (lower bound -> y <= 0) OK
[2] 2tri      rows= 24 vars= 40  max|H_fd - H_analytic| / max|H| = 4.441e-12  (< 1e-6) OK
[2] bilinear  rows= 48 vars= 40  max|H_fd - H_analytic| / max|H| = 4.441e-12  (< 1e-6) OK
[3] per-row spectrum = +-0.866025 (x2) and 0 (x2); PSD projection: min eig(P+) and
    min eig(P+ - lam*Hc) >= -1.2e-15 OK
[4] gershgorin nf=24 m=80 nnz=    82 tau=  13.485  assembly err=7.17e-18  min eig(P)=+11.1383 OK
[4] psd_row    nf=24 m=80 nnz=   140 tau=   0.000  assembly err=2.74e-16  min eig(P)=+2.7815 OK
```

Check [2] is **exhaustive**, not sampled: every row × every variable pair, by central
second differences of `constraint.values` on a 4×5 grid, for both the 2-tri and the
bilinear families. Rel err 4.4e-12 (roundoff floor of `h=1e-2` second differences), well
inside the 1e-6 requirement. It also asserts the analytic pattern really is x–y-only with
a zero diagonal.

### Dual sign convention

OSQP: `min ½z'Pz + q'z s.t. l ≤ Az ≤ u`, stationarity `Pz + q + A'y = 0`, and `y ≤ 0` on a
row held at its **lower** bound. The linearized rows are `c + Jd + s ≥ 0` → `l = -c`,
`u = +inf`, so only the lower bound can bind and `y ≤ 0` there (check [1] confirms it on a
one-row QP). The NLP Lagrangian for `c(x) ≥ 0` is `L = f - μ'c`, `μ ≥ 0`; matching the two
KKT systems gives `μ = -y`, hence

```
Hess(L) = H_obj - Σ μ_i Hc_i = H_obj + Σ y_i Hc_i
```

so OSQP's raw `y` enters with a **plus** sign. Positive `y` on these rows is impossible at
an optimum, so noise is clipped (`lam = min(y, 0)`). Multipliers come from the PREVIOUS
QP solve; iteration 1 uses `lam = 0`, i.e. exactly today's behaviour.

The hybrid backend's interior-point leg had to be extended: stock `_HybridQP._solve_ip`
returns only `x`, which would starve the Newton term precisely on the cold and
stale-warm-start solves. `_HybridQPY` maps Clarabel's cone duals back to OSQP's
convention (`y = -z` for finite-`l` rows, `y = +z` for finite-`u` rows; the linearized
rows are the first finite-`l` rows, so they sit at offset `fu.sum()`).

### Convexification

Each `Hc_i` is `[[0, M], [M', 0]]` with `M = -M'` the cross-product matrix of
`w = (½, ½, ½)`, so its spectrum is `{+s, +s, 0, 0, -s, -s}`, `s = √3/2 ≈ 0.866` —
indefinite with a symmetric spectrum. OSQP needs `P ⪰ 0`, so two modes were implemented:

- **`gershgorin`** (global shift). `S = Σ lam_i Hc_i` has a zero diagonal and couples only
  y-vars to x-vars, so under the (y | x) split `S = [[0, B], [B', 0]]` and
  `λ_min(S) = -σ_max(B) ≥ -√(‖B‖₁‖B‖∞)`. Shift `τ = √(‖B‖₁‖B‖∞)`. Tighter than plain
  Gershgorin on the full matrix (`max(‖B‖₁, ‖B‖∞)`) and still O(nnz). 6 extra nnz/row.
- **`psd_row`** (per-row PSD projection, no global shift). The positive part of `lam·Hc` is
  `P₊ = ½[[|lam|·s·(I - ŵŵ'), lam·M], [lam·M', |lam|·s·(I - ŵŵ')]]` — **half** the true
  off-diagonal coupling plus a shift local to that row's six variables. 18 nnz/row.

Both keep a **fixed sparsity pattern**: the union pattern (Newton entries ∪ the objective
diagonal, slack block empty) is built once with explicit zeros, and only `P.data` is
refreshed per iteration via a precomputed COO→CSC position map (`np.bincount(pos, vals)`),
so `update(Px=...)` never sees a pattern change. The driver asserts the pattern is
unchanged before every update — this is the failure mode a previous attempt hit.

---

## 2. The structural problem the derivation exposes

Two facts kill the Newton QP for this engine, independent of implementation quality.

**(a) With `NoneObjective` the true multipliers are zero.** The hard-case recipe solves a
pure feasibility problem (`objective=NoneObjective`, `g ≡ 0`). At a feasible point,
stationarity is `J'μ = 0` with `μ ≥ 0`, so `μ = 0` and `Hess(L) = H_obj` exactly. There is
no second-order constraint information to add near the solution — the Lagrangian Hessian
*is* the objective Hessian.

**(b) Away from feasibility the multipliers are the elastic big-M, not multipliers.** In
the elastic QP, a row whose slack is strictly positive has its dual pinned at exactly
`-rho` by complementarity with the slack's linear cost (`rho = 1e3`). So on a folded
window the "multipliers" are ~1e3 across the board — a penalty weight, not a Lagrange
multiplier. The resulting `Σ lam_i Hc_i` has entries up to ~500 and row-abs-sums up to
~8000 (a variable sits in up to 4 cells × 4 blocks = 16 enforced triangles).

Against `hess_diag = 2.0`, the convexification shift is then **three to four orders of
magnitude larger than the objective curvature**, which turns each QP into a near-trivial
proximal step. This is not fixable by picking a better bound: `τ ≈ 8|lam|` (gershgorin) or
a local shift of `≈ 7|lam|` accumulated (psd_row) versus a true coupling magnitude of
`0.87|lam|` — the convexification necessarily adds ~8-10× the curvature it is trying to
model, because every `Hc_i` has a symmetric ± spectrum and there is no PSD part to keep
without adding an equal-and-opposite shift.

---

## 3. Measurements

Setup for everything below: `SimplexConstraint2DBilinear` + `NoneObjective`,
`threshold=0.01`, engine defaults (`coarse_to_fine=True`, `qp_backend='hybrid'`,
`giant_tile_fit=True`), OMP/OPENBLAS/MKL/NUMEXPR/RAYON pinned to 1. Baseline is the
SAME instrumented driver with the Lagrangian term switched off — `--parity` proves that
path is byte-identical to the stock engine (`z0_cluster`: stock 387 SQP iterations,
patched 387, `np.array_equal(out) == True`).

Variants: `newton` = gershgorin shift, raw multipliers; `newton-psd` = per-row PSD
projection, raw multipliers; `newton-psd-capN` = PSD projection with `|lam| <= N`;
`psd-cap1-nocoupling` = the CONTROL that keeps `psd_row`'s diagonal blocks and zeroes
the x–y coupling (all of the regularization, none of the second-order information);
`ls-salvage` = no Hessian at all, just salvaging a rejected trust-region direction with
the legacy backtracking line search instead of discarding it.

### 3a. Per-window (the claim under test, isolated)

Same frozen-ring sub-problem, same start point, straight into the driver — no engine
retry cascade on top. `maxiter=150`. This is where step count is actually decided.

**z16_twist, window 0** (patch 48×50, 4700 free vars, 9212 enforced rows, worst viol 54.6)

| variant | iters | feasible | max_viol | exit | wall s | ADMM/QP | tau_mean |
|---|---|---|---|---|---|---|---|
| baseline | 108 | no | 0.0270 | tr-collapse | 27.3 | 440 | — |
| newton | 150 | no | **0.1910** | maxiter | 29.7 | 320 | **11920** |
| newton-psd | 150 | no | 0.0463 | maxiter | 37.4 | 413 | — |
| newton-psd-cap1 | 101 | no | 0.0265 | tr-collapse | 28.0 | 460 | — |
| newton-psd-cap3 | 83 | **yes** | 0.0 | model-flat | 24.2 | 510 | — |
| psd-cap1-nocoupling | 96 | no | 0.0343 | tr-collapse | 23.3 | 397 | — |
| ls-salvage | **49** | no | 0.0414 | tr-collapse | **13.6** | 513 | — |

**z16_twist, window 1** (patch 14×12, 286 free, 572 rows)

| variant | iters | feasible | wall s | ADMM/QP |
|---|---|---|---|---|
| baseline | 27 | yes | 0.4 | 548 |
| newton | 150 | **no** (viol 0.038) | 1.2 | 171 |
| newton-psd | 67 | yes | 0.8 | 412 |
| newton-psd-cap1 | **9** | yes | 0.2 | 1095 |
| newton-psd-cap3 | 15 | yes | 0.2 | 725 |
| psd-cap1-nocoupling | 15 | yes | 0.2 | 346 |
| ls-salvage | **9** | yes | 0.1 | 604 |

**z0_cluster, window 0** (patch 24×42, 1932 free, 3772 rows, worst viol 87.6)

| variant | iters | max_viol | exit | wall s | ADMM/QP | tau_mean |
|---|---|---|---|---|---|---|
| baseline | 115 | 0.0243 | tr-collapse | 14.6 | 708 | — |
| newton | 150 | **11.6151** | maxiter | 5.2 | 91 | **11920** |
| newton-psd | 113 | 0.0112 | tr-collapse | 14.2 | 672 | — |
| newton-psd-cap0.1 | 113 | 0.0111 | model-flat | 18.1 | 1019 | — |
| newton-psd-cap1 | 75 | 0.0112 | model-flat | 12.1 | 988 | — |
| newton-psd-cap3 | 63 | 0.0111 | model-flat | 10.7 | 1092 | — |
| psd-cap1-nocoupling | **56** | 0.0208 | tr-collapse | **8.8** | 960 | — |
| psd-cap1-2xcoupling | 68 | 0.0111 | model-flat | 10.6 | 869 | — |
| ls-salvage | 91 | 0.0325 | model-flat | 11.9 | 756 | — |

**z0_sliver, window 0** (patch 28×21, 988 free, 2160 rows, worst viol 0.0114) — the
TR-acceptance regression case. Every variant freezes at the sliver (0.0114) in 8–13
iterations; the no-TR retry is what clears it. No discrimination here.

### 3b. Full engine

_(filled in below)_

### 3c. The ADMM-cost side effect, measured

The a-priori worry ("a denser / less diagonal P can cost ADMM iterations") **did not
happen**. `P` stays extremely sparse — the Lagrangian term adds 6 (gershgorin) or 18
(psd_row) nonzeros per enforced row, against a Jacobian block that already has 12 per
row — and ADMM iterations per QP are statistically unchanged for `psd_row`
(397–513/QP on z16 w0 vs the baseline's 440; 672–1092/QP on z0_cluster vs 708). For
`gershgorin` the per-QP ADMM count actually **drops sharply** (91–320/QP), because the
enormous diagonal shift makes each subproblem trivially well-conditioned — the QPs get
easier and the steps get worthless. The cost is entirely in step COUNT and step
QUALITY, never in the QP solve.

---

## 4. Verdict and what to do instead

_(filled in below)_
