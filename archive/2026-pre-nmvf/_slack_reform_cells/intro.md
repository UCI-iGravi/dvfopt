# Slack-variable reformulation of the 2-tri SLSQP

**Premise.** The current `iterative_2d_tri_slsqp` enforces
`T1, T2 >= threshold` per cell as inequality constraints. On dense
folds (many `T_k` simultaneously near zero), the active-set update
thrashes and the constraint Jacobian rows become nearly parallel —
SLSQP's QP subproblem becomes ill-conditioned and the line search
collapses.

**Reformulation.** Introduce slack variables `s_k >= 0` such that

    T_k(z) - s_k = threshold       (equality, k = 1..n_T)
    s_k >= 0                       (bound)
    z   unrestricted

The objective is unchanged (`anchor(z - z_anchor)`, independent of `s`).
The equality-constraint Jacobian is

    dC/dw = [ J_T(z) , -I ]        shape (n_T, 2HW + n_T)

The `-I` block guarantees the equality Jacobian has full row rank
**always**, regardless of how degenerate the triangles are. The active
set now lives on the slacks (`s_k = 0` vs `s_k > 0`) rather than the
constraint values themselves; SLSQP handles bound active sets much
more robustly than inequality active sets because there is no
rank-deficiency interaction with the constraint linearization.

**This notebook.**

1. Build the slack-reform SLSQP solver and FD-verify the augmented
   Jacobian.
2. Compare wall clock, feasibility, and L2/L1 cost against the
   baseline `iterative_2d_tri_slsqp` on:
   - Synthetic 20×20 random fields at three fold densities.
   - Synthetic 30×30 dense fields.
   - A 40×40 crop from B0039 z=200 (moderate folds).
   - A 40×40 crop from B0039 z=12 (very dense — the canonical hard
     case).
3. Reference m10 (`iterative_2d_tri_harmonic_polished`) as the
   "always-feasibility" baseline — slack-reform should aim to match
   m10's feasibility while keeping closer to the anchor.

We do **not** run on full B0039 z=12 (320×456): both the baseline and
slack-reform have variable counts that put them outside SLSQP's
practical envelope (O(n³) QP per iter). The wallbreakers exist for
that regime regardless of how we reformulate the constraint.
