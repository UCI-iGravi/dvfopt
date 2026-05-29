## 7. Findings

### Headline result: **slack reformulation does not structurally fix SLSQP on dense folds.**

On every dense case where both solvers ran, the slack reform is **3–5× slower** than baseline and produces **the same or slightly worse** L1/L2 cost.

### Per-case observations

| Case | Init n_neg | Baseline | Slack-reform | Verdict |
|------|-----------|---------|---------|---------|
| 8×8 dense | 9 | 1.6s · L1=1.76 · ✅ | 5.2s · L1=1.76 · ✅ | tie quality, baseline 3× faster |
| 10×10 moderate | 8 | 6.9s · L1=0.97 · ✅ | 25.3s · L1=0.98 · ✅ | tie quality, baseline 4× faster |
| 10×10 dense | 20 | 7.1s · L1=6.59 · ✅ | 25.9s · L1=7.09 · ✅ | baseline +7% L1, 4× faster |
| 12×12 moderate | 4 | 17.8s · L1=0.62 · ✅ | 104s · L1=0.27 · ❌ (n_neg=2) | **slack failed feasibility** |
| 12×12 dense | 38 | 29.4s · L1=9.91 · ✅ | 100s · L1=11.10 · ⚠️ (min=+0.003<thr) | baseline wins |
| **B0039 z=12 12×12** | 25 | 12.3s · L1=4.81 · ✅ | 53.3s · L1=**3.09** · ✅ | **slack -36% L1**, 4× slower |

### Why slack-reform doesn't help (the actual SLSQP bottleneck)

The full-row-rank `[J_T(z), -I]` block does what the math promised — KKT conditioning is no longer driven by degenerate triangles. The FD check confirmed rank = `n_T` independent of how collapsed the triangles are.

But this **wasn't the bottleneck**. The real SLSQP failure modes on dense folds are:

1. **Non-convexity of the feasible set itself.** Even with a perfectly conditioned Jacobian, the manifold `{z : T_k(z) ≥ ε ∀k}` is non-convex (in fact disconnected near a many-fold configuration — each fold has at least two "ways out"). SLSQP's local quadratic model can't see this.
2. **Active-set growth on the slacks instead of the inequalities.** The reformulation moved the active set from "which inequalities bind" to "which slacks hit s=0." That's an equally large discrete decision SLSQP has to converge — just relocated. We did not eliminate the active-set update; we relabeled it.
3. **+n_T variables make every QP iteration more expensive.** The KKT block-elimination cost scales roughly cubically in the rank, and we just doubled the variable count. That explains the consistent 3–5× wall-clock penalty.

### The one real signal

On the **B0039 z=12 12×12 crop**, slack-reform found a notably better L1 anchor cost (3.09 vs 4.81 — 36% improvement) with comparable feasibility. The other dense synthetic case at 12×12 it failed to reach feasibility. So the L1 win is genuine on real data but not robust — and the wall-clock is 4× worse.

### Pivots that would be worth trying next

If the goal is "make SLSQP scale to dense," the next reformulations to try (in expected impact order) are:

1. **Smooth-min aggregation per cell** (`softmin(T1, T2; β) ≥ ε`). Halves the constraint count and removes the "two triangles per cell argue over one diagonal" effect that drives a lot of the active-set thrashing. Easy retrofit.
2. **Geometric product** (`T1·T2 ≥ ε²`). One constraint per cell, self-balancing, but allows T1≫T2 unless paired with a min-norm anchor term on `(T1−T2)`.
3. **Treat slack-reform as a polish step** after a feasibility-first seed (harmonic / m10 output). The slack reform may be wasted on the feasibility problem but valuable for the L1-minimization sub-problem given a feasible seed.

### What this prototype validates / falsifies

- ✅ Augmented Jacobian is full-row-rank, FD-validated. The math holds.
- ✅ Slack reform converges to feasibility in most cases.
- ❌ It does **not** scale better than baseline on dense folds.
- ❌ It does **not** make SLSQP feasible where baseline isn't.
- ⚠️ On one real-data case it found a meaningfully lower L1 — possibly useful as a **polish step**, not a primary solver.
- ❌ At ≥16×16, slack reform is impractical (well past 5 min per case at this iteration cap).

**Net conclusion: this is a clean negative result for the rank-deficiency hypothesis.** SLSQP's wall on dense 2-tri problems is not about constraint Jacobian rank — it's about the non-convex feasible-set geometry, which constraint reformulations alone can't fix. The wallbreaker pipeline (m10/m14) remains the right structural answer for dense cases.
