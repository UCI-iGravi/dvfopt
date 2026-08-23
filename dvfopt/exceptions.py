"""Exception hierarchy for DVFopt.

All package-raised exceptions derive from :class:`DVFoptError`, so
downstream code can catch any DVFopt-specific failure with::

    try:
        result = solver.fit(phi)
    except dvfopt.DVFoptError as exc:
        ...

Concrete subclasses signal the kind of failure precisely:

* :class:`IncompatibleConstraintError` — a Strategy was given a
  Constraint type it cannot handle (e.g. ``HarmonicALMBarrierStrategy``
  with ``JdetConstraint2D``). Subclass of ``TypeError`` so existing
  ``except TypeError`` handlers still catch it.
* :class:`IncompatibleObjectiveError` — a Strategy was given an
  Objective it cannot handle (e.g. ``SLPStrategy`` with
  ``L2Objective``). Also a subclass of ``TypeError``.
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


class IncompatibleObjectiveError(DVFoptError, TypeError):
    """Strategy × objective mismatch (e.g. SLP with an L2 objective)."""


__all__ = [
    'DVFoptError',
    'IncompatibleConstraintError',
    'IncompatibleObjectiveError',
    'SolverConfigError',
]
