"""Objective functions for SLSQP optimisation."""

import numpy as np


def objectiveEuc(phi, phi_init):
    """L2-squared objective with analytical gradient.

    Minimises ``0.5 * ||phi - phi_init||^2``.  The gradient is
    ``phi - phi_init`` — linear, zero at the optimum, well-conditioned
    everywhere.  Preferred over the raw L2 norm because that has a kink
    at ``phi = phi_init`` where the Hessian is ill-conditioned, causing
    SLSQP to take erratic steps in the final convergence phase.

    Returns ``(value, gradient)`` for use with ``minimize(..., jac=True)``.
    """
    diff = phi - phi_init
    return 0.5 * np.dot(diff, diff), diff
