"""Shared constraint/objective math + engines with zero method logic.

Modules: tri (2-triangle), jdet2d, jdet3d (Jacobian-determinant flat forms
and adjoints), finite_jdet (forward-diff Jdet flat form + analytic sparse
jacobian), constraint_values (per-cell maps for reporting), slsqp
(traced C-SLSQP driver, added later in the reorg), isqp (elastic-QP I-SLSQP
solver, osqp-gated), coloring (CPR constraint-Jacobian coloring).
"""
