"""SLSQP-based iterative correction (windowed sub-problem optimisation).

Public entry points::

    from dvfopt.core.slsqp_windowed import iterative_serial, iterative_parallel, iterative_3d
"""

from dvfopt.core.slsqp_windowed.iterative import iterative_serial
from dvfopt.core.slsqp_windowed.iterative3d import iterative_3d
from dvfopt.core.slsqp_windowed.parallel import iterative_parallel

__all__ = [
    "iterative_3d",
    "iterative_parallel",
    "iterative_serial",
]
