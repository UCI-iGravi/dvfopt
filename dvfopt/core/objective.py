"""Objective functions for SLSQP optimisation."""

import numpy as np


def objectiveEuc(phi, phi_init):
    """L2 norm objective with analytical gradient.

    Returns ``(value, gradient)`` for use with ``minimize(..., jac=True)``.
    The gradient of ``||phi - phi_init||`` is ``(phi - phi_init) / ||phi - phi_init||``.
    """
    diff = phi - phi_init
    sq_norm = np.dot(diff, diff)
    if sq_norm == 0.0:
        return 0.0, np.zeros_like(phi)
    norm = np.sqrt(sq_norm)
    diff /= norm  # in-place: avoids allocating a second array
    return norm, diff
