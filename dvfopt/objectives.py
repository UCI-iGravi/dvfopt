"""Objective functions (anchor distances) for DVFopt.

Each objective is a smooth function of ``phi_flat - phi_anchor`` that
returns ``(value, gradient)``. The objective is independent of the
constraint family — every solver in the package accepts an
``Objective`` (or its string label via :func:`make_objective`).

The underlying math is :func:`anchor_term` below; these classes are a
thin OO wrapper, 1:1 with the three legacy string options
``'l2' / 'l1' / 'none'``. Solvers take an ``Objective`` and call it on
``phi - phi_anchor``; the numba/torch inner kernels, which dispatch on an
integer anchor flag and cannot call back into Python, take the
``(kind, eps_l1)`` pair from :func:`_kind_eps` instead.

This module is pure numpy — it must not import from :mod:`dvfopt.core`
(the engine imports *from here*).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def anchor_term(diff: np.ndarray, kind: str, eps_l1: float = 1e-4):
    """Return ``(value, gradient)`` of the anchor term.

    ``kind`` is one of ``'l2'``, ``'l1'`` (smoothed), ``'none'``.
    """
    if kind == 'l2':
        return 0.5 * float(diff @ diff), diff.copy()
    if kind == 'l1':
        s = np.sqrt(diff * diff + eps_l1 * eps_l1)
        return float((s - eps_l1).sum()), diff / s
    if kind == 'none':
        return 0.0, np.zeros_like(diff)
    raise ValueError(f"unknown anchor kind: {kind!r}")


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


def _kind_eps(objective: Objective) -> tuple[str, float]:
    """Legacy ``(kind, eps_l1)`` pair for kernels that can't call Python.

    The numba fused kernels (wallbreakers) and the torch autograd path
    dispatch on an integer/string anchor flag and cannot evaluate an
    :class:`Objective` on their tensors, so they take this pair instead.
    Mirrors the historical ``anchor=objective.label or 'l2'`` unwrapping
    exactly — a custom subclass with no ``label`` falls back to ``'l2'``.
    """
    return objective.label or 'l2', float(getattr(objective, 'eps', 1e-4))


__all__ = [
    'L1Objective',
    'L2Objective',
    'NoneObjective',
    'Objective',
    'anchor_term',
    'make_objective',
]
