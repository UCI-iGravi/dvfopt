# m14-Schwarz: cluster-localized refinement prototype

**Problem.** Global m14 processes every corner of the field, even when
only ~5% of cells are folded. On a 320×456 slice with sparse fold
clusters this is mostly wasted compute — the optimizer is iterating on
cells that were already at the anchor and already feasible.

**Idea.** Detect fold clusters first (connected components of
`min(T1, T2) <= 0`, dilated by `merge_dilation=2` to group nearby
clusters). For each cluster, extract a bounding-box crop with `pad`
cells of context, run global m14 on the crop, splice back. Schwarz-style
domain decomposition — each subproblem is tiny, independent of slice
size.

If splicing introduces new folds at crop boundaries, do another round.
After `max_outer_iters` rounds, fall back to global m14 if there's
still progress to be had. Also fall back immediately when a single
cluster spans most of the field (no point in cropping).

**This notebook.**

1. Run the prototype on five cases ranging from very sparse (3 separated
   clusters on a 30×30 synthetic field) to fully saturated (B0039 z=12
   full slice with 8978 folds).
2. Compare wall clock, L1, L2 against global m14.
3. Visualize the fold-cluster decomposition + the per-cluster bounding
   boxes for a representative case.
