"""Objective functions (anchor distances) for DVFopt.

Each objective is a smooth function of ``phi_flat - phi_anchor`` that
returns ``(value, gradient)``. The objective is independent of the
constraint family — every solver in the package accepts an
``Objective`` (or its string label via :func:`make_objective`).

The underlying math is the existing ``anchor_term`` in
:mod:`dvfopt.core._barrier_core`; these classes are a thin OO wrapper,
1:1 with the three legacy string options ``'l2' / 'l1' / 'none'``.
Solvers consume an objective by its ``label`` (and ``eps`` for L1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dvfopt.core._barrier_core import anchor_term


class Objective(ABC):
    """Smooth scalar objective ``f(phi - phi_anchor)``.

    Subclasses implement :meth:`__call__` returning
    ``(value: float, gradient: ndarray)``. The gradient has the same
    shape as the diff vector.
    """

    label: str = ''

    @abstractmethod
    def __call__(self, diff: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate at ``diff = phi - phi_anchor``."""

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


# ---------------------------------------------------------------------------
# Primitive objectives — match the existing `anchor_term` kinds.
# ---------------------------------------------------------------------------


class L2Objective(Objective):
    """``f(d) = 0.5 * ||d||_2^2``, gradient ``d``.

    The L2 anchor produces smoothly-spread corrections: every cell is
    pulled fractionally back toward the anchor. Default objective for
    the barrier and SLSQP families.
    """

    label = 'l2'

    def __call__(self, diff: np.ndarray) -> tuple[float, np.ndarray]:
        return anchor_term(diff, 'l2')


class L1Objective(Objective):
    """Smoothed-L1 anchor ``f(d) = sum sqrt(d^2 + eps^2) - eps``.

    Produces **concentrated** corrections — a few cells with high
    per-cell deviation rather than a smooth spread. The
    ``iterative_2d_tri_slsqp`` warm-restart pipeline (notebook 14) was
    designed around this objective.
    """

    label = 'l1'

    def __init__(self, eps: float = 1e-4):
        self.eps = float(eps)

    def __call__(self, diff: np.ndarray) -> tuple[float, np.ndarray]:
        return anchor_term(diff, 'l1', self.eps)

    def __repr__(self) -> str:
        return f'L1Objective(eps={self.eps!r})'


class NoneObjective(Objective):
    """``f(d) = 0`` everywhere. Used when only feasibility matters."""

    label = 'none'

    def __call__(self, diff: np.ndarray) -> tuple[float, np.ndarray]:
        return anchor_term(diff, 'none')


# ---------------------------------------------------------------------------
# Registry / convenience
# ---------------------------------------------------------------------------


def make_objective(spec, eps_l1: float = 1e-4) -> Objective:
    """Construct an Objective from a string label or a class instance.

    Accepts:
      * ``'l2'`` / ``'l1'`` / ``'none'`` (legacy string labels)
      * An existing :class:`Objective` instance (passed through)
    """
    if isinstance(spec, Objective):
        return spec
    if spec == 'l2':
        return L2Objective()
    if spec == 'l1':
        return L1Objective(eps=eps_l1)
    if spec == 'none':
        return NoneObjective()
    raise ValueError(f'unknown objective: {spec!r}')


__all__ = [
    'L1Objective',
    'L2Objective',
    'NoneObjective',
    'Objective',
    'make_objective',
]
