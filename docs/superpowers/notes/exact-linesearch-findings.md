# Exact 1-D line search in the windowed isqp engine — findings

Research prototype on `proto-exact-linesearch`. Code: `benchmarks/isqp_exactls.py`
(patched driver) + `benchmarks/exact_linesearch_proto.py` (correctness gate +
harness). No library change.

**Verdict: PROMOTE the exact merit line minimiser (B) as a `step_rule` on
`isqp_solve`; DO NOT promote the feasibility-preserving cap (A).**

One-paragraph summary: the constraint rows are exactly quadratic along any line, so
the merit function along the QP step is an exact piecewise quadratic whose global
minimiser is available in closed form. Replacing the trust-region ratio test with
that minimiser is the first candidate on this axis that **wins on the real B0039
z16 slice** — 151-167 s / 530-582 SQP iterations against the baseline's 195.8 s /
762, with a *smaller* departure from the input (L2 268 vs 280), 0 folds and damage
0 — where the Newton variants and `ls-salvage` were flat or worse. It also turns
the hardest window in the crop set from a 108-iteration **failure** into a
46-iteration **success**. The promotion design needs none of the Hessian machinery
the derivation suggested: `q = c(x+d) - c(x) - J d` is exact for any quadratic row
family and is paid for by an evaluation the stock ratio test already makes.
The one crop-level regression is `z0_sliver` (2.3-4.8x), and that case is
measurably chaotic — four mathematically-equivalent framings of the same method
span 139-287 s on it. Tool (A), the maximal feasibility-preserving step, is dead
on arrival: on real windows `a_max` is ~1e-3-1e-1, which strangles the elastic
SQP's whole mechanism.

---

## 1. The derivation, and the correctness gate

Every 2-tri / bilinear row is a signed triangle area, i.e. an inhomogeneous
**bilinear form** in the decision variables. So along any line it is *exactly*
quadratic:

```
c_i(x + a d) = c_i(x) + a (J d)_i + a^2 q_i(d),     q_i(d) = 1/2 d' Hc_i d
```

with `Hc_i` the constant per-row Hessian derived and verified to 4.4e-12 in
`docs/superpowers/notes/newton-sqp-findings.md`. In the DY_FIRST pack this is six
`+-1/2` x-y cross terms, so

```
q_i(d) = sum over the 6 (Q, P, v) pairs of  v * d_yQ * d_xP
```

Three numbers per row describe the whole line — **no evaluation of `cons` needed**.

Two closed-form tools follow.

**(A) Maximal feasibility-preserving step.** For a row that currently holds
(`c_i >= 0`), the first `a > 0` at which it breaks is a root of a quadratic;
`a_max` is the min over such rows (linear rows, `q_i = 0`, reduce to `-c/g`). This
is the *nonlinear* version of what `monotone=True` does linearly.

**(B) Exact merit line minimiser.** `m(a) = f(a) + sum_i w_i max(0, -c_i(a))` is
piecewise quadratic on `[0, a_hi]`: `f` is quadratic (`NoneObjective`: 0; L2:
quadratic — fitted exactly from three points), and each hinge `max(0, -c_i)`
switches on/off only at a root of `c_i`. Sort the `<= 2m` roots, sweep the active
set with a cumulative sum over `(w*-c0, w*-g, w*-q)`, and take the best of every
interval's two endpoints and its parabola vertex. O(m log m), fully vectorised,
and — unlike the ratio test — it needs **no** `cons(x + d)` evaluation at all.

### Verification (`exact_linesearch_proto.py --check`)

```
[1] z0_cluster  rows= 3772 free= 1932  max rel err |model - cons(x+a d)| = 2.600e-15  (< 1e-8) OK
[1] z16_twist   rows= 9212 free= 4700  max rel err |model - cons(x+a d)| = 6.161e-16  (< 1e-8) OK
[2] z0_cluster  obj=NoneObjective  |d|= 0.2 events=  211 a*=0.02164  m(0)=7.20932e+06 -> m(a*)=7.20918e+06  (grid min 7.20918e+06, backtrack min 7.20919e+06) OK
[2] z0_cluster  obj=NoneObjective  |d|= 1.0 events= 1081 a*=0.00413  m(0)=7.20932e+06 -> m(a*)=7.20927e+06  (grid min 7.20927e+06, backtrack min 7.20927e+06) OK
[2] z16_twist   obj=L2Objective    |d|= 0.2 events= 1561 a*=0.00829  m(0)=940269 -> m(a*)=940174  (grid min 940174, backtrack min 940175) OK
[2] z16_twist   obj=L2Objective    |d|= 1.0 events= 4838 a*=0.00012  m(0)=940269 -> m(a*)=940269  (grid min 940269, backtrack min 940269) OK
[3] z0_cluster  |d|= 0.2 a_max=0.043365  min satisfied row at a_max = -2.266e-15 >= 0  (min at 1.01*a_max = -3.794e-04 < 0) OK
[3] z16_twist   |d|= 1.0 a_max=0.000540  min satisfied row at a_max = +3.157e-16 >= 0  (min at 1.01*a_max = -2.152e-05 < 0) OK
[4] z0_cluster  |d|= 0.2  max rel |q_table - (c(x+d) - c - J d)| = 5.304e-13  (< 1e-10) OK
[4] z16_twist   |d|= 1.0  max rel |q_table - (c(x+d) - c - J d)| = 3.207e-15  (< 1e-10) OK
```

- **[1]** is the gate the task specified: the quadratic line model against direct
  evaluation of `cons(x + a d)` at random `a` on real windowed sub-problems.
  Measured rel err **2.6e-15**, seven orders inside the 1e-8 requirement.
- **[2]** is a stronger gate than asked for: the closed-form minimiser against a
  **dense brute-force scan of the true merit** (2001 real `cons` evaluations) —
  the model merit at `a*` equals the true merit there, and no scanned point beats
  it. Note the `backtrack min` column: on random directions plain backtracking
  (`1, 1/2, 1/4, ...`) lands close, because the minimiser often sits at small `a`;
  the separation shows up on real QP directions (§2), not on random ones.
- **[3]** `a_max` is *exactly* maximal: at `a_max` the worst satisfied row sits at
  ~1e-15, and at `1.01 * a_max` at least one is strictly negative.
- **[4]** the table-free identity `q = c(x + d) - c(x) - J d` reproduces the
  constant-Hessian table to 1e-13. **This is the finding that shrinks the
  promotion design** (see §5).

`--parity` proves the patched driver at `step_rule='tr'` is the stock path:
`z0_cluster` stock 387 SQP iterations, patched 387, `np.array_equal(out) == True`.

---

## 2. The micro gate (per-window, the claim under test in isolation)

Same frozen-ring sub-problem, same start point, straight into the driver — no
engine retry cascade on top. `maxiter=150`, hybrid backend, `osqp_max_iter=2000`.
`baseline` is the same instrumented driver at `step_rule='tr'`.

**z16_twist, window 0** (patch 48x50, 4700 free, 9212 rows, worst viol 54.6)

| variant | iters | feasible | max_viol | exit | wall s | ADMM/QP | mean a* |
|---|---|---|---|---|---|---|---|
| baseline | 108 | no | 0.0270 | tr-collapse | 27.3 | 440 | — |
| **ls_exact** | **46** | **YES** | **0.0** | step-tol | **21.5** | 1119 | 0.436 |
| ls_cap | 14 | no | **40.36** | tr-collapse | 3.5 | 411 | — |
| ls_both | 150 | no | **39.68** | maxiter | 10.7 | 60 | 0.410 |

**z16_twist, window 1** (patch 14x12, 286 free, 572 rows)

| variant | iters | feasible | wall s | ADMM/QP | mean a* |
|---|---|---|---|---|---|
| baseline | 27 | yes | 0.4 | 548 | — |
| **ls_exact** | **8** | yes | 0.1 | 919 | 0.925 |
| ls_cap | 17 | yes | 0.2 | 365 | — |
| ls_both | 11 | yes | 0.2 | 611 | 0.700 |

**z0_cluster, window 0** (patch 24x42, 1932 free, 3772 rows, worst viol 87.6)

| variant | iters | feasible | max_viol | exit | wall s | ADMM/QP | mean a* |
|---|---|---|---|---|---|---|---|
| baseline | 115 | no | 0.0243 | tr-collapse | 14.6 | 708 | — |
| **ls_exact** | **96** | no | **0.0111** | step-tol | 15.6 | 1071 | 0.429 |
| ls_cap | 54 | no | **83.70** | tr-collapse | 1.6 | 79 | — |
| ls_both | 150 | no | **82.42** | maxiter | 4.0 | 55 | 0.458 |

**z0_sliver, window 0** (patch 28x21, 988 free, 2160 rows, worst viol 0.0114)

| variant | iters | feasible | max_viol | exit | wall s | mean a* |
|---|---|---|---|---|---|---|
| baseline | 11 | no | 0.011436 | tr-collapse | 0.5 | — |
| ls_exact | 30 | no | 0.011006 | step-tol | 1.5 | 0.256 |

Reading:

- **(B) alone passes the gate decisively.** On the hardest window in the whole
  crop set it converts a 108-iteration *failure* (`tr-collapse` at violation
  0.027) into a 46-iteration **feasible** solve. On `z0_cluster` w0 it is -17%
  iterations at *half* the residual violation. On the small window it is 8
  iterations against 27. No Newton variant reached feasibility on z16_twist w0
  except `newton-psd-cap3` (83 iterations).
- **(A) fails the gate hard, and the reason is structural.** On real windows
  `a_max` averages 0.06 (z16 w0), 0.14-0.79 elsewhere — because a fold cluster of
  depth 50+ can only be pushed out by *temporarily* breaking neighbouring rows,
  which is exactly the trade the elastic slack exists to permit. Forbidding it
  nonlinearly strangles the method: `ls_cap` and `ls_both` both end at violation
  **40-84** (baseline: 0.024-0.027). `a_max` also collapses toward 0 whenever a
  satisfied row sits near its target, so the cap self-blocks (`n_cap_skip` fires
  on 41/150 iterations at the 1e-4 floor). This is the nonlinear analogue of
  `monotone=True`, and it is *stricter* than the linear version that already
  exists — a knob the engine deliberately does not default on.

Per the gate rule, only `ls_exact` advanced to the engine runs.

---

## 3. Engine runs (`windowed_correct`, stock knobs, `maxiter=600`)

`SimplexConstraint2DBilinear` + `NoneObjective`, `threshold=0.01`, engine defaults
(`coarse_to_fine=True`, `qp_backend='hybrid'`, `giant_tile_fit=True`),
OMP/OPENBLAS/MKL/NUMEXPR/RAYON pinned to 1. **Every row below has 0 simplex folds
and damage 0** — the only axes that move are cost and L2 move.

Variants: `ls_exact` = (B) everywhere; `ls_exact_tr` = (B) only on the
trust-region path (legacy backtracking kept on the engine's no-TR fallback rung);
`ls_exact_bail` = (B) plus the ratio test's own futility threshold as a
*termination* signal (see §4); `ls_exact_bail_eval` = the same, with the
**table-free** `q` (§5).

| case | variant | wall s | SQP iters | ADMM/QP | rejects | L2 move | backend fallbacks |
|---|---|---|---|---|---|---|---|
| z16_twist | baseline | 29.8 | 128 | 411 | 16 | 125.7 | 0 |
| z16_twist | ls_exact | **20.7** | **48** | 1117 | 0 | **103.4** | 0 |
| z16_twist | ls_exact_bail | **20.6** | **48** | 1117 | 0 | **103.4** | 0 |
| z16_twist | ls_exact_bail_eval | **20.1** | **48** | 1050 | 0 | **103.4** | 0 |
| z0_cluster | baseline | 44.1 | 387 | 630 | 84 | 542.7 | 9 |
| z0_cluster | ls_exact | 46.5 | 363 | 705 | 0 | 535.2 | 9 |
| z0_cluster | ls_exact_bail | 45.3 | 386 | 671 | 0 | 535.2 | 9 |
| z0_cluster | ls_exact_bail_eval | 44.1 | 402 | 604 | 0 | 535.1 | 9 |
| z0_sliver | baseline | **59.4** | **540** | 415 | 124 | 25.3 | 6 |
| z0_sliver | ls_exact | 228.8 | 1391 | 466 | 0 | 28.6 | 7 |
| z0_sliver | ls_exact_tr | 286.7 | 2780 | 385 | 0 | 27.6 | 15 |
| z0_sliver | ls_exact_bail | 138.7 | 1253 | 446 | 0 | 25.3 | 10 |
| z0_sliver | ls_exact_bail_eval | 256.1 | 2429 | 371 | 0 | 28.1 | 9 |
| **rawz16** | baseline | 195.8 | 762 (+18 coarse) | 484 | 92 | 280.3 | 0 |
| **rawz16** | ls_exact | 167.5 | 582 (+11) | 585 | 0 | 268.0 | 0 |
| **rawz16** | ls_exact_bail | **157.2** | **530** (+11) | 654 | 0 | **268.1** | 0 |
| **rawz16** | ls_exact_bail_eval | **151.0** | 582 (+11) | 520 | 0 | **268.1** | 0 |

The baseline reproduces the recorded reference exactly (rawz16: 195.8 s / 762
fine iterations / L2 280.3 / 0 folds vs the recorded ~187-196 s / 762 / 280.3 / 0;
crops 29.8 / 44.1 / 59.4 s vs the recorded 30 / 45 / 59), so these rows are
directly comparable to the existing engine numbers and to the Newton table.

**The real slice is the headline.** Against a baseline of 195.8 s / 762 SQP
iterations, all three exact-LS framings land at 151-167 s and 530-582 iterations —
**-14% to -23% wall, -24% to -30% iterations** — with a *smaller* move (L2 268 vs
280) and identical quality. For contrast, on the same case:

| candidate | wall s | fine iters | L2 move |
|---|---|---|---|
| baseline | 195.8 | 762 | 280.3 |
| Newton `newton-psd-cap3` | 176.3 | **836** (+10%) | 294.7 |
| `ls-salvage` | 185.8 | **811** (+6%) | 278.7 |
| **exact LS (this work)** | **151.0-167.5** | **530-582** (-24 to -30%) | **268.0** |

Both previous candidates needed *more* iterations than the baseline on the real
slice; this one needs 24-30% fewer. The `rejects` column shows the mechanism: the
baseline throws away 92 QP directions on rawz16 (16, 84, 124 on the crops) and
pays a whole extra QP solve to re-derive each one. The exact minimiser never
rejects — every solved QP produces a step.

---

## 4. The one crop regression, and why its numbers are not trustworthy

`z0_sliver` regresses 2.3x-4.8x. The mechanism is clean and was isolated
per-window: the window starts at violation 0.0114 and **neither** variant can
clear it. The baseline discovers that in 11 iterations (`tr-collapse`) and hands
the window straight to the engine's escalation ladder; the exact minimiser grinds
30 iterations to `step-tol` and reaches essentially the same violation (0.011006
vs 0.011436) before the ladder runs anyway. **An exact minimiser always finds
*some* decrease, so it never fires the fast bail-out that the ratio test uses to
give up on a hopeless window.** Multiplied over 15 windows x 2 rounds x the
escalation ladder, 11 -> 30 becomes 540 -> 1391.

Two framings were tried against that, and then the tuning stopped:

- **Scoping (B) out of the no-TR fallback rung** (`ls_exact_tr`) — the hypothesis
  that the exact LS was hijacking a rung tuned for backtracking. **Disproven**:
  286.7 s / 2780 iterations, *worse*, with 15 backend fallbacks vs the baseline's
  6.
- **Restoring the bail-out as a pure termination test** (`ls_exact_bail`): keep
  taking the exact step, but *also* evaluate the ratio test's own futility
  condition (`achieved <= 1e-3 x the QP's predicted decrease` — the existing
  constant, no new knob) and shrink/collapse the trust region on it. **Helps**:
  228.8 -> 138.7 s. It costs nothing on the winners (z16_twist unchanged at
  20.6 s / 48; rawz16 *improves* to 157.2 s / 530).

Still 2.3x. But the case's own measurements do not support a 2.3x claim:

> Four framings of the *same* method span **138.7 / 228.8 / 256.1 / 286.7 s** and
> **1253 / 1391 / 2429 / 2780** iterations on `z0_sliver`. The 138.7 and 256.1
> rows differ **only** in how `q` is computed — the constant-Hessian table versus
> the algebraic identity `c(x+d) - c - J d`, which §1 check [4] shows agree to
> **1e-13**. A 1e-13 perturbation produces a 1.8x wall difference.

That is the step-count-chaos signature, and it belongs to the *case*, not to the
method: `z0_sliver` has 0 simplex folds to begin with (min +0.0110 against a
0.01 threshold) and only ~1e-4-scale bilinear violations, i.e. every decision in
it is made at the level of OSQP's own noise floor, and the escalation ladder
amplifies the outcome ~10x. The same 1e-13 perturbation moves rawz16 by 4%
(157.2 vs 151.0 s) and z16_twist by 2%. **So the crop is a chaos detector, not a
benchmark**, and a "2.3x regression" measured on it carries an error bar of about
its own size. The honest statement is: on sliver-scale windows the exact LS spends
more iterations before the engine escalates, by an amount this crop cannot pin
down — which is why §7 goes to real slices instead.

---

## 5. The design shrinks: no Hessian table needed

The derivation reached `q_i(d)` through the constant per-row Hessians, which needs
the row -> triangle map, the block layout and the pack convention — ~120 lines of
family-specific machinery (`LineModel`, `triangle_abc`, `_PAIRS`).

It does not need any of that. Since the row *is* quadratic,

```
q = c(x + d) - c(x) - J d          exactly
```

and the stock ratio test **already evaluates `cons(x + d)` every iteration** — so
the identity form is free, and family-agnostic: every 2D family the windowed
engine supports (2tri, bilinear, jdet, finite) is a bilinear form in `(dy, dx)`,
so it holds for all of them with no per-family code. Check [4] confirms agreement
to 1e-13, and `ls_exact_bail_eval` measures the same wins (rawz16 151.0 s / 582
iterations; z16_twist 20.1 s / 48).

Caveat for the eventual 3D question: a 6-tet volume is *trilinear*, hence cubic
along a line, so neither the identity nor the quadratic model transfers. The
windowed engine is 2D-only today, so this is a guard to write, not a problem to
solve.

---

## 6. Verdict and promotion design

**PROMOTE (B). DO NOT promote (A).**

(A) is refuted on its own terms: `a_max` on real windows is small enough to
strangle the elastic mechanism, ending at violations three orders worse than the
baseline. It is a stricter nonlinear version of the `monotone=True` knob the
engine already has and deliberately leaves off.

(B) is the first change on this axis that wins on real data, and it wins on the
metric the engine actually pays for (SQP iterations) with a smaller move and
unchanged quality. Recommended shape:

1. **`dvfopt/core/primitives/isqp.py`** — add `step_rule: str = 'tr'` to
   `isqp_solve`, valid `{'tr', 'exact_ls'}`. Under `'exact_ls'`, the QP, the
   elastic slack, the trust-region box and the `tr_delta` bookkeeping are all
   unchanged; only the accept/reject block changes:
   - `g = J d` (already formed), `q = cons(x + d) - c - g` (the evaluation the
     ratio test already makes — so `'exact_ls'` costs *no* extra `cons` call);
   - `a*` from the piecewise-quadratic sweep (`line_events` + `exact_line_min`,
     ~60 lines of vectorised numpy, no new dependency);
   - take `x + a* d`; grow `tr_delta` when `a* >= 0.9` and `|d| >= 0.9 delta`,
     shrink when `a* < 0.25`, and keep the existing futility test
     (`achieved <= 1e-3 x pred`) as the `tr-collapse` trigger — that test is what
     stops a hopeless window from grinding (§4).
   The objective along the line is fitted from `obj` at `a = 0, 1/2, 1` — exact
   for `NoneObjective` / `L2Objective`; for `L1Objective` it is a convex quadratic
   approximation on the segment, which should be stated in the docstring (or the
   rule refused for L1 until measured).
2. **`dvfopt/core/windowed/_common.py`** — one `step_rule` field on `_InnerOpts`,
   threaded through `solve_window_inner` to the isqp inner exactly like
   `tr_delta`. `ISQPWindowedStrategy` gains the matching dataclass knob.
3. **Default.** See §7 — on the 8-slice real-data sample the exact rule wins the
   population, so flipping the default is defensible; a full 528-slice sweep
   before the flip is the conservative option, and the knob makes that a
   one-line experiment rather than a re-prototype.
4. **Tests.** `step_rule='tr'` must stay byte-identical (the `--parity` check,
   promoted to a test). Add the model gate as a unit test: on a small folded
   field, `cons(x + a d)` vs `c + a (J d) + a^2 (cons(x+d) - c - J d)` to 1e-10,
   and the closed-form minimiser against a 201-point brute-force scan. Keep the
   engine's existing no-damage assertions — both `step_rule`s must pass them
   (they do: damage 0 in every run above).

### What did not happen

The a-priori worry for a line search — that it would need extra constraint
evaluations — is inverted: the exact rule needs *fewer* nonlinear evaluations than
the ratio test it replaces (the ratio test's `mfun(x + d)` call is the only one,
and `'exact_ls'` reuses that same `cons(x + d)` for `q`). ADMM iterations per QP
do rise (484 -> 520-654 on rawz16) because a longer accepted step makes the next
warm start staler — but total ADMM work still falls with the iteration count
(301k -> 255k on rawz16).

---

## 7. Real-slice sample (the population that decides the default)

Three curated crops cannot settle a default � �4 shows one of them cannot even
resolve a 2x difference. So the same comparison was run on **eight real B0039
slices** spanning the volume (every 48th from z=64, fold counts 835-3167),
`baseline` vs `ls_exact_bail_eval` (the promotable form: table-free `q`, futility
bail retained), same engine defaults, same thread pinning.

| slice | folds in | baseline wall / iters | exact_ls wall / iters | d wall | d iters | L2 base / exact |
|---|---|---|---|---|---|---|
| z64 | 1392 | 125.7 s / 726 | 124.5 s / 799 | -1% | +10% | 72.0 / 67.9 |
| z112 | 835 | 48.4 s / 181 | 40.0 s / 156 | -17% | -14% | 27.6 / 27.1 |
| z160 | 1193 | 63.5 s / 369 | 43.1 s / 256 | -32% | -31% | 42.8 / 41.2 |
| z208 | 2560 | 223.4 s / 850 | 186.7 s / 773 | -16% | -9% | 77.1 / 73.4 |
| z256 | 2413 | 256.0 s / 1163 | 221.6 s / 826 | -13% | -29% | 76.9 / 75.2 |
| z304 | 3167 | 396.1 s / 1476 | 346.0 s / 1244 | -13% | -16% | 95.3 / 91.7 |
| z400 | 2238 | 123.0 s / 572 | 109.3 s / 493 | -11% | -14% | 79.1 / 76.2 |
| z496 | 2052 | 308.8 s / 1366 | 261.7 s / 1123 | -15% | -18% | 85.7 / 78.5 |
| **total** | | **1544.9 s / 6703** | **1332.9 s / 5670** | **-14%** | **-15%** | |

Adding the reference slice from �3 (z16: 195.8 -> 151.0 s, 762 -> 582) makes it
**9 real slices, 9 wall wins** (eight clear, one flat) and **8 of 9 iteration
wins**. Every slice has **0 simplex folds and damage 0** for both variants, and
the exact rule produces a **smaller L2 move on every single slice** � it is not
buying speed with a larger departure from the input.

The one non-win (z64, -1% wall / +10% iterations) is the shape �4 describes: more
iterations, same answer, no quality loss. Nothing in the sample looks like the
`z0_sliver` blow-up � the crop's 2.3-4.8x has no counterpart on real data, which
is consistent with it being a chaos artifact of a case built entirely at the
solver's noise floor.

**This settles the default question.** The exact rule wins the population by 14%
wall / 15% SQP iterations at strictly-not-worse quality, so `step_rule='exact_ls'`
is defensible as the windowed engine's default once the knob lands; the
conservative alternative (land the knob at `'tr'`, flip after a full 528-slice
sweep) is now a formality rather than a real risk, and the knob makes that sweep a
one-line experiment instead of another prototype.
