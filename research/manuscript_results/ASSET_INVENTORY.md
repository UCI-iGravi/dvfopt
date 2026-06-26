# Manuscript asset inventory (figures & tables)

Status legend: **EXISTS** = file already in `writing/images/dvfopt/`; **DEFERRED** =
to be generated in the user's later result pass (Phase B), inserted now as a
LaTeX placeholder; **REUSE-TEXT** = data already in the committed manuscript.

## Figures

| Asset | Status | Path / source | Used in §  |
|---|---|---|---|
| signed_areas.pdf (incl. central-diff-misses-fold panel c) | EXISTS | writing/images/dvfopt/signed_areas.pdf | §3, §4 (teaser candidate) |
| corner_point_notation.pdf (triangle assignment) | EXISTS | writing/images/dvfopt/corner_point_notation.pdf | §4 |
| geometric_primitives.pdf | EXISTS | writing/images/dvfopt/geometric_primitives.pdf | §3/§4 |
| tet_decomposition.pdf (Freudenthal–Kuhn) | EXISTS | writing/images/dvfopt/tet_decomposition.pdf | §8 |
| per_vertex_tetrahedra.pdf | EXISTS | writing/images/dvfopt/per_vertex_tetrahedra.pdf | §8 |
| objective_comparison2.png (L1/L2) | EXISTS | writing/images/dvfopt/objective_comparison2.png | §5 |
| F-new-1 untangleable_cell.pdf (fold under all 4 diagonals) | DEFERRED (Task B8) | placeholder | §8 |
| F-new-2 pareto_frontier.pdf (L1 vs wall) | DEFERRED (Task B7) | placeholder | §7 |
| F-new-3 density_ceiling.pdf | DEFERRED (Task B8) | placeholder | §8 |
| distribution_boxplots.pdf | DEFERRED (Task B5) | placeholder | §7 |
| runtime-vs-gridsize / runtime-vs-#folds | REUSE-TEXT (commented figs in draft) | draft fig:runtime_gridsize / fig:runtime_jacs | §6/§7 |

## Tables

| Asset | Status | Source | Used in § |
|---|---|---|---|
| T1 method taxonomy | author-fillable now (from spec §6 / CLAUDE.md strategy table) | — | §6 |
| T2 main 2D benchmark (methods × feas/min_T/L1/wall) | DEFERRED (Task B3/B5) | placeholder | §7 |
| T3 multi-source correction (+Dice/NCC/SDlogJ) | DEFERRED (Task B3/B4) | placeholder | §7 |
| T4 strict-feasibility & geometric ceiling | DEFERRED (Task B8) | placeholder | §8 |
| T5 central-diff blindness false-negative rates | DEFERRED (Task B2) | placeholder | §3/§7 |
| old tab:jac_results (windowed-SLSQP numbers) | REUSE-TEXT | already in draft | §6 (relabel/keep as baseline) |

## Placeholder convention (writing tasks)
Insert deferred tables/figures as a compiling stub:
```
% RESULT PLACEHOLDER (deferred, Task B_): <what fills this>
\begin{table}[ht]\centering
\caption{TODO (deferred result): <caption>.}
\label{tab:...}
\end{table}
```
Writing subagents must NOT invent numbers for DEFERRED assets.
