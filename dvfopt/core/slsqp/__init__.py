"""SLSQP-based iterative correction (windowed sub-problem optimisation).

Public entry points::

    from dvfopt.core.slsqp import iterative_serial, iterative_parallel, iterative_3d
"""

from dvfopt.core.slsqp.iterative import iterative_serial
from dvfopt.core.slsqp.parallel import iterative_parallel
from dvfopt.core.slsqp.iterative3d import iterative_3d

__all__ = [
    "iterative_serial",
    "iterative_parallel",
    "iterative_3d",
]
