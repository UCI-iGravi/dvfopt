# Dual warm-starting in the windowed isqp engine

Research prototype, branch `proto-dual-warmstart`. Harness:
[`benchmarks/dual_warmstart_proto.py`](../../../benchmarks/dual_warmstart_proto.py)
(monkeypatches only — no library edit, so the baseline path is the shipped
default path byte for byte).

Question: the engine warm-starts the QP **primal** twice over (OSQP's `x` across
SQP iterations, and the prolongated coarse correction into the fine field), but
starts the **duals** cold. Does warm-starting the duals help — (a) across SQP
iterations inside a window, (b) from the coarse level to the fine level?

---

## TL;DR

| | verdict |
|---|---|
| (a) within a window, across SQP iterations | **already happening**, so nothing to add — but far from worthless. OSQP keeps `y` across `update()`+`solve()` on the one QP object per window, and the dual is the *load-bearing half* of that warm start. |
| (b) coarse level → fine level | **don't promote.** The coarse solve's terminal duals are all zero to 6e-09 — a converged SQP's last QP is the trivial `d = 0` one, so there is literally nothing to transfer. Injecting the one variant that does carry content measures **+4.6 %** ADMM. |
| (c) *found on the way*: interior point → ADMM, inside a window | **real gap, still don't promote.** `_HybridQP._solve_ip` hands OSQP `warm_start(x=…)` only, stranding the dual on solves that carry **52 %** of all fine ADMM work. Supplying it cuts ADMM-per-SQP-iteration **−35 %** — and costs **+39 %** SQP iterations (+62 % for the narrowed variant). Net total ADMM: −10.5 % / −0.2 %. |

Nothing here is a feasibility or fidelity question: **every** variant lands at 0
simplex folds, 0 bilinear folds, damage 0, and within 0.3 % of the same L2 move
(the sole exception is `zeroy`, deliberately handicapped, at +4.4 %). The whole
question is cost, and the answer is that the engine is already collecting the dual
warm start that matters.

---

## What the driver already does with `y`

`isqp_solve` creates **one** QP object per window and then keeps it:

```python
prob = None
while it < maxiter:
    ...
    if prob is not None and same_pattern:
        prob.update(q=q, l=lo, u=up, Px=p.data, Ax=a.data)   # values only
    else:
        prob = _make_qp(...)
        prob.setup(..., warm_starting=True, polishing=True, ...)
    res = prob.solve()
```

The Jacobian sparsity pattern is fixed across SQP iterations, so `same_pattern`
holds and the object survives the whole window. With `warm_starting=True` OSQP
carries **both** `x` and `y` (and its internal `z`) from one `solve()` to the
next. So (a) is already in place for free — no code change would add anything.

But "already happening" is not the same as "worth nothing". Isolating the two
halves on a synthetic QP of the engine's shape (`P` diagonal, `A` sparse,
lower-bounded rows, `polishing=True`):

| what is reset before the re-solve | ADMM iterations |
|---|---|
| nothing (the engine's actual behaviour) | 25 |
| `warm_start(x=0)` — primal only | 25 |
| `warm_start(y=0)` — dual only | 50 |
| `warm_start(x=0, y=0)` — fully cold | 50 |

**Zeroing the dual costs the full cold price; zeroing the primal costs nothing.**
For this problem class the ADMM warm start is essentially *all* dual.

Across a *changing* QP (the real situation — each SQP iteration re-linearizes, so
`q`, `l`, `Ax` all move) the benefit survives while the perturbation is small and
inverts when it is large:

| per-iteration perturbation | keep `y` (6 solves) | zero `y` (6 solves) |
|---|---|---|
| 0.01 | **150** | 225 |
| 0.05 | **200** | 300 |
| 0.20 | 375 | **150** |

So a *stale* dual is worse than a cold one. This matters for (c) below: it is an
argument for seeding from the immediately preceding solve, not from a distant one.

---

## Row structure (verified)

`SimplexConstraint2DBilinear` lays out its `4*(H-1)*(W-1)` rows **block-major**,
not cell-major — worth stating because "cell-major with 4 rows per cell" was the
expectation:

```
row = b * (H-1)*(W-1)  +  i * (W-1) + j          b in {T1, T2, U1, U2}
```

`values()` returns `concatenate([T1.ravel(), T2.ravel(), U1.ravel(), U2.ravel()])`
(`tri_areas_flat_bilinear`), and the engine's enforced-row set is built the same
way — `_influenced_2tri` does `concatenate([b*m + cell_flat for b in range(k)])`.
Asserted in `--selfcheck` by perturbing single pixels and checking which rows move.

The QP OSQP actually sees stacks three row groups (trust region on):

```
[0    : m]        J d + s >= -c        <- the constraint duals of interest
[m    : 2m]       0 <= s <= s_up
[2m   : 2m+nf]    -delta <= d <= delta
```

so an injected `y` has length `2m + nf` with only the first `m` entries set.
Constraint rows are lower-bounded with `u = +inf`, so their duals are **≤ 0** in
OSQP's convention, with magnitude capped at `rho = 1e3`.

## Dual scaling under restriction

`_restrict` box-averages by `factor` **and divides displacements by `factor`**, so
the coarse field lives in coarse pixel units and the same `threshold` means the
same thing there. Triangle areas are therefore *dimensionless* on both grids
(`--selfcheck` asserts both mean ≈ 0.5): `c` and `J` are O(1) either way, and only
the step rescales, `d_coarse = d_fine / factor`. QP stationarity
`H d = Jᵀ y_c + y_tr` then gives

> **`y_fine ≈ factor · y_coarse`**  (factor = `coarse_factor` = 4)

for rows whose dual is strictly interior. It does **not** apply to the majority of
rows, because the elastic formulation is bang-bang: a row whose slack `s_i` is
strictly inside `[0, s_up]` has `y_i = -rho` exactly by complementarity, and an
inactive row has `y_i = 0`. Both are scale-free.

---

## (b) Coarse → fine: there is nothing to transfer

Measured on raw B0039 z16, default engine (`coarse_factor=4`), capturing every
coarse window's duals. Over the whole coarse row grid (35 708 rows):

| coarse dual, taken at | frac zero | frac at `-rho` | frac interior | max abs | mean abs |
|---|---|---|---|---|---|
| **the coarse solution** (`last_y`) | **1.000** | 0.000 | 0.000 | **5.8e-09** | 5.0e-13 |
| the coarse cold QP (`first_y`) | 0.990 | 2.8e-05 | 0.0103 | 1000.0 | 0.130 |

**Every** dual at the coarse solution is zero to 6e-09. This is structural, not
incidental: a converged SQP's final QP is the one where the step is zero and every
constraint sits at or above its margin-shifted target, so nothing is active and
the duals vanish. Prolongating them onto the fine rows injects a vector of zeros —
which is precisely OSQP's cold start. **The transfer is provably a no-op.**

This is not a mapping bug to be fixed with a better restriction operator or a
better scale factor. The coarse level's information has *already* been moved
across, through the primal (`_prolongate`); no dual residual is left behind.

That leaves the coarse **cold** QP's duals as the only candidate with content.
They do not predict the fine windows' needs either — regressing each fine window's
first-QP dual on its mapped coarse counterpart over 143 696 paired rows:

```
corr = -0.0025          ls slope = 0.0014          coarse nonzero on those rows = 8.3%
```

Correlation zero to three decimals. The `c2ffirst` run below confirms the
prediction from these numbers.

> **The "previous round's duals" proxy has no data here.** The suggested cheaper
> fallback — reuse the last round's duals for the same window — needs the round
> loop to re-open windows. On raw z16 the baseline reports `rounds: 1`,
> `mop_windows: 0`, `backend_fallbacks: 0`: every window is solved once and the
> slice reaches zero folds in a single pass. There is no second visit to warm-start
> from, so the proxy is not merely worse than the real thing, it is undefined on
> this slice.

## (c) Sizing the interior-point → ADMM gap

The same baseline run, split by what preceded each ADMM solve:

| fine-stage ADMM solves | solves | ADMM iterations | avg/solve | share of all fine ADMM |
|---|---|---|---|---|
| directly after an interior-point solve | 142 | **153 150** | 1079 | **52.5 %** |
| ↳ of those, the window's *first* ADMM solve (dual is literal zero) | 23 | 31 875 | 1386 | 10.9 % |
| all fine ADMM solves | ~609 | 291 850 | 479 | 100 % |

(The second row is a subset of the first — with `ip_cold=True` a window's opening
solve is the interior-point one, so its first ADMM solve is also a post-IP solve.
It is broken out because its dual starts at exactly zero rather than merely stale,
and it is the most expensive kind at 1386 iterations.)

A post-IP solve costs **2.3× the average solve**, and 23 % of the solves account
for 52 % of all ADMM work. Those are exactly the solves whose dual `_solve_ip`
leaves behind, because it hands OSQP `warm_start(x=…)` only. The interior-point
legs themselves are cheap — 153 IP solves for 2917 IP iterations, 19 each.

So the hypothesis has a large, precisely-located target: half the fine ADMM budget
sits on solves that are being handed a good primal and a discarded dual.

---

## Numbers

Raw B0039 z16 (`vol[1:, 16]`, 320×456, 3890 simplex folds / 4022 bilinear),
`SimplexConstraint2DBilinear`, `NoneObjective`, `threshold=0.01`, `maxiter=600`,
engine defaults otherwise. `OMP/OPENBLAS/MKL/NUMEXPR/RAYON/VECLIB = 1`.

> **Read the iteration counts, not the wall clock.** Three things make wall time
> unusable here: the box was shared with an unrelated multi-worker job for the
> whole campaign; `RAYON_NUM_THREADS=1` single-threads Clarabel, which the hybrid
> backend leans on; and the four modes were run *concurrently* to fit the budget.
>
> That last one is safe for the metric that matters. The engine has no randomness
> and OSQP/Clarabel are deterministic given identical inputs and settings, so
> **every iteration count below is exactly reproducible and completely independent
> of load** — only the seconds columns move. The verdict rests on iterations,
> folds, damage and move; wall is reported only for order-of-magnitude context.

Baseline sanity: the instrumented `base` run reproduces the known default-path
numbers exactly — **762 fine SQP iterations, L2 move 280.3, 0 folds, 0 damage** —
so the monkeypatches are transparent.

### Headline

| mode | fine SQP iters | fine ADMM iters | post-IP ADMM | cold ADMM | IP iters | simplex folds | damage | L2 move | wall (contended) |
|---|---|---|---|---|---|---|---|---|---|
| `base` | 762 | 291 850 | 153 150 | 31 875 | 2917 | 0 | 0 | 280.3 | 278 s |
| `c2ffirst` (b) | 677 | 305 200 | 165 125 | 30 300 | 2896 | 0 | 0 | 281.5 | 357 s |
| `ipdual` (c) | 1057 | **261 200** | **108 825** | **19 775** | 2478 | 0 | 0 | 281.0 | 327 s |
| `ipdualcold` (c, narrow) | 1235 | 291 350 | 140 175 | 22 050 | 3110 | 0 | 0 | 280.7 | 314 s |
| `zeroy` (a, dual removed) | 1064 | 682 650 | 414 975 | 30 775 | 5302 | 0 | 0 | 292.7 | 602 s |

Every mode reaches **0 simplex folds, 0 bilinear folds, damage 0**, and moves the
field by within 0.5 % of the same L2 — nothing here trades feasibility or fidelity
for speed. The whole question is cost.

### (a) `zeroy`: what the free dual warm start is actually worth

Resetting `y` to zero before every ADMM solve — the only change — costs:

| | base | zeroy | |
|---|---|---|---|
| fine ADMM iterations | 291 850 | **682 650** | **+134 %** |
| ADMM iterations per SQP iteration | 383 | 642 | +68 % |
| fine SQP iterations | 762 | 1064 | +40 % |
| post-IP ADMM iterations | 153 150 | 414 975 | +171 % |
| windows opened | 28 | 39 | +39 % |
| L2 move | 280.3 | 292.7 | +4.4 % |
| wall (contended) | 278 s | 602 s | +117 % |

**2.3× the QP work and a 4.4 % larger move.** So (a) is not a no-op that happens to
already be handled — it is one of the larger single effects in the engine, and it
arrives free from keeping one QP object per window alive.

One internal check worth noting: `fine_cold_admm_iters` barely moves (31 875 →
30 775, −3.5 %). That is exactly right — a window's *first* ADMM solve already
starts from a zero dual in the baseline, so zeroing it changes nothing there. The
instrumentation agrees with the mechanism it claims to measure.

### The ladder

Per-solve cost, ordered by how good the dual handed to ADMM is:

| dual the ADMM solve starts from | post-IP solve, avg ADMM iters | window's cold solve, avg |
|---|---|---|
| zeroed every solve (`zeroy`) | 1446 | 1184 |
| coarse-injected, then retained (`c2ffirst`) | 1139 | 1377 |
| **retained across SQP iterations (shipped)** | 1079 | 1386 |
| retained **+ the IP leg's own dual** (`ipdual`) | **899** | **791** |

Monotone in dual quality, as it should be. The cold-solve column has a ±15 %
trajectory noise floor (compare `zeroy` 1184 vs `base` 1386, which are the same
situation — a zero dual — reached by different window layouts); `ipdual`'s 791 is
well outside it.

### (b) `c2ffirst`: injection lands, and changes nothing

The mapping works — 28 windows injected, 192 708 rows, **coarse-cell hit fraction
1.00** (every fine row found its parent coarse cell). And the result is noise:
SQP iterations −11 %, ADMM iterations **+4.6 %**, post-IP ADMM +7.8 %, L2 +0.4 %.
Exactly what `corr = −0.0025` predicted. The ±10 % swings are trajectory jitter,
not signal.

### (c) `ipdual`: the QP-level mechanism works, the SQP loop eats the gain

Per-solve, the dual handoff does precisely what the micro-benchmark said it would:

| | base | ipdual | |
|---|---|---|---|
| ADMM iterations per SQP iteration | 383 | **247** | **−35 %** |
| cold ADMM solve, avg iterations | 1386 | **791** | **−43 %** |
| post-IP ADMM solve, avg iterations | 1079 | **899** | −17 % |
| coarse stage: ADMM iterations | 9175 | **3550** | **−61 %** |
| coarse stage: SQP iterations | 18 | 11 | −39 % |

But the outer loop needs **more iterations**: 762 → 1057 fine SQP iterations
(**+39 %**). Cheaper QPs, more of them. Net on total fine ADMM: only −10.5 %.

That −10.5 % does not survive contact with the rest of an SQP iteration. Every
extra iteration also pays for a colored Jacobian rebuild, a `bmat` re-assembly, a
merit evaluation and an OSQP `update` — costs the ADMM counter does not see. 295
extra iterations against 30 650 saved ADMM iterations is a bad trade, and the wall
column agrees in sign (+18 %) even though its magnitude is untrustworthy here.

The mechanism for the regression is the obvious one: OSQP terminates on its own
tolerance, so a better-warm-started solve stops at a *different* point, not just
sooner. That perturbs each step, and the trust-region ratio test then accepts and
rejects differently.

### `ipdualcold`: narrowing the intervention does not rescue it

The natural fix is to seed only where the payoff is biggest and the perturbation
smallest — the window's cold first IP solve, leaving the tail-triggered ones as
shipped. It fails, and instructively:

| | base | `ipdual` (136 seeds) | `ipdualcold` (48 seeds) |
|---|---|---|---|
| ADMM iterations per SQP iteration | 383 | **247** | **236** |
| cold ADMM solve, avg | 1386 | 791 | 882 |
| post-IP ADMM solve, avg | 1079 | 899 | 960 |
| **fine SQP iterations** | **762** | 1057 (+39 %) | **1235 (+62 %)** |
| **total fine ADMM iterations** | **291 850** | 261 200 (−10.5 %) | **291 350 (−0.2 %)** |
| windows opened | 28 | 28 | 40 |

A third of the seeds, the same per-QP win (236 vs 247 ADMM per SQP iteration) —
and a *worse* outcome: 62 % more SQP iterations, 40 windows instead of 28, and the
total ADMM saving wiped out entirely. Fewer, better-targeted interventions did not
buy a calmer trajectory.

That is the real result here. The isqp loop's cost is governed by **how many steps
it takes**, not by what each QP costs, and the step count responds chaotically to
*any* change in where OSQP stops — the sign and size of the response has no
relation to how much of the QP work was saved.

---

## Verdict

**(a) Within a window — nothing to do, and nothing to undo.** OSQP already carries
`y` across every SQP iteration of a window, because `isqp_solve` deliberately
keeps one QP object alive and only `update`s its values. The finding worth
remembering is the *converse*: the dual is the load-bearing half of that warm
start (zeroing `y` costs the full cold price; zeroing `x` costs nothing), so any
future refactor that drops the shared QP object — or resets the dual "to be
safe" — pays **2.3× the QP work and a 4.4 % larger move** for it. That is a
regression risk to protect, not an opportunity to chase.

**(b) Coarse → fine — don't promote.** Three independent reasons, in order of how
conclusive they are:

1. *Structural.* The coarse solve's terminal duals are **all zero to 6e-09**
   (35 708 rows, `frac_zero = 1.000`). A converged SQP ends on the trivial `d = 0`
   QP; nothing is active; there is no dual to move. Prolongating them injects
   zeros, which is what OSQP starts from anyway.
2. *Predictive.* The one coarse dual that does carry content — the cold QP's — has
   correlation **−0.0025** with what the fine windows' first QPs actually need.
3. *Measured.* Injecting it anyway (`c2ffirst`, hit fraction 1.00 across 28
   windows and 192 708 rows) moves total fine ADMM **+4.6 %**.

The coarse level's information reaches the fine level through the **primal**
(`_prolongate`) and is fully spent there. This is not a mapping bug and no scale
factor rescues it. The suggested "previous round's duals" fallback is moot on this
slice — the baseline solves it in `rounds: 1`, so no window is ever revisited.

**(c) Interior point → ADMM — a real gap, correctly diagnosed, and still
don't promote.** `_HybridQP._solve_ip` genuinely does strand the dual on the solves
that carry 52 % of all fine ADMM work, and supplying it works exactly as the theory
says: ADMM per SQP iteration **383 → 247 (−35 %)**, cold solves −43 %, the coarse
stage −61 %. But the saving does not reach the bottom line, because the isqp loop
pays by the *step*, not by the QP, and its step count moves chaotically with any
change in the QP's termination point: +39 % SQP iterations for the broad version,
+62 % for the narrow one. Net total fine ADMM: −10.5 % and −0.2 %. Wall goes up in
both. Feasibility and fidelity are untouched throughout (0 folds, 0 damage, L2
within 0.3 %), so this is purely a cost question, and the cost answer is no.

**What would have to change first.** The −35 % per-QP win is real and reproduces
across both variants. It is unbankable only because the SQP step depends on *where
ADMM stopped*, not just on the QP's solution. Make the step independent of that —
tighten `osqp_eps` and/or raise `qp_max_iter` until consecutive solves land on the
same point regardless of warm start — and the per-QP saving would drop straight to
the bottom line. That is a different and more expensive operating regime (the
current 2000-iteration cap exists precisely to avoid it), and it is its own
experiment. Until someone runs it, the dual handoff is a lever with nothing to
push against.

---

## Appendix: the engine change for (c) — NOT recommended today

Kept because the diagnosis is sound and the hunk is the thing to reach for if the
`osqp_eps` / `qp_max_iter` question above ever gets answered. **Do not merge this
on its own** — as measured it is a net loss.

One hunk in `dvfopt/core/primitives/isqp.py`, inside `_HybridQP._solve_ip`. `fu` /
`fl` are already in scope, and it stays inside the existing `try/except`, so any
failure still falls through to ADMM exactly as today — the backend can still only
be faster, never less feasible.

```diff
             x = np.asarray(sol.x, dtype=np.float64)
             if str(sol.status) != "Solved" or not np.all(np.isfinite(x)):
                 return None
-            self._real.warm_start(x=x)
+            # Hand OSQP the DUAL as well as the primal. Clarabel's cone multiplier
+            # z >= 0 maps back through the same fu/fl split that built A_ip:
+            # stationarity Px + q + A[fu]'z_u - A[fl]'z_l = 0 vs OSQP's
+            # Px + q + A'y = 0, hence y[fu] += z_u, y[fl] -= z_l.
+            z = np.asarray(sol.z, dtype=np.float64)
+            y = np.zeros(self._a.shape[0])
+            nu = int(fu.sum())
+            y[fu] += z[:nu]
+            y[fl] -= z[nu:]
+            if np.all(np.isfinite(y)):
+                self._real.warm_start(x=x, y=y)
+            else:
+                self._real.warm_start(x=x)
```

No new knob, no API change, no `_InnerOpts` field — it is strictly a better
handoff on a path that already exists. The docstring line "An IP solve seeds
OSQP's warm start with its own solution" becomes true of the dual too.

**Test to port.** `--selfcheck`'s `_check_ip_dual_map` builds the engine's own row
shape (`m` lower-bounded-only rows over `n` two-sided box rows), solves it with
both backends, and asserts the mapped Clarabel dual equals OSQP's own to `1e-6`
(measured residual `2.5e-10`) with non-positive duals on the lower-bounded rows.
That is the unit test for the hunk.
