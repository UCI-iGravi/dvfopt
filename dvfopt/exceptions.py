"""Exception hierarchy for DVFopt.

All package-raised exceptions derive from :class:`DVFoptError`, so
downstream code can catch any DVFopt-specific failure with::

    try:
        result = solver.fit(phi)
    except dvfopt.DVFoptError as exc:
        ...

Concrete subclasses signal the kind of failure precisely:

* :class:`FeasibilityError` — the solver finished but failed to bring
  every constraint to ``>= threshold - err_tol``. The returned iterate
  is the best the solver could do; ``.result`` attribute carries the
  full :class:`SolveResult`.
* :class:`BudgetExhaustedError` — solver hit its iteration / time
  budget before reaching feasibility. Subclass of ``FeasibilityError``
  since it's a more specific reason for the same failure.
* :class:`IncompatibleConstraintError` — a Strategy was given a
  Constraint type it cannot handle (e.g. ``HarmonicALMBarrierStrategy``
  with ``JdetConstraint2D``). Subclass of ``TypeError`` so existing
  ``except TypeError`` handlers still catch it.
* :class:`SolverConfigError` — config validation failure. Subclass of
  ``ValueError`` for the same back-compat reason.

Strategies are *encouraged* to raise these (instead of plain
``RuntimeError``) when they detect a known failure mode, but plain
exceptions still leak through unchanged.
"""

from __future__ import annotations


class DVFoptError(Exception):
    """Base class for all DVFopt-raised exceptions."""


class SolverConfigError(DVFoptError, ValueError):
    """Bad configuration — invalid constraint/objective/strategy name,
    or an incompatible combination."""


class IncompatibleConstraintError(DVFoptError, TypeError):
    """A :class:`Strategy` was given a :class:`Constraint` type it does
    not support (declared via ``accepts_constraints``)."""


class FeasibilityError(DVFoptError):
    """Solver ran to completion but the final iterate is not strictly
    feasible (``min_T < threshold - err_tol`` or ``n_neg > 0``).

    Attributes
    ----------
    result : SolveResult | None
        The full :class:`dvfopt.solver.SolveResult` if available (the
        best-effort iterate plus diagnostics).
    """

    def __init__(self, message: str, *, result=None):
        super().__init__(message)
        self.result = result


class BudgetExhaustedError(FeasibilityError):
    """Solver hit ``max_iter`` / ``time_budget_s`` / ``warm_max_iter``
    before reaching feasibility. The iterate may still be partially
    converged — inspect ``.result``."""


__all__ = [
    'BudgetExhaustedError',
    'DVFoptError',
    'FeasibilityError',
    'IncompatibleConstraintError',
    'SolverConfigError',
]
