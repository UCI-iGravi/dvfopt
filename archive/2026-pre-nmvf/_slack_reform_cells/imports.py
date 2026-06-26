import sys, os, time, warnings
sys.path.insert(0, os.path.abspath('../..'))

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.core._barrier_core import anchor_term
from dvfopt.core.tri_primitives import tri_areas_flat
from dvfopt.core.iterative2d_tri_slsqp import (
    iterative_2d_tri_slsqp,
    _build_full_grid_tri_jac,
)
from dvfopt.core.wallbreakers import iterative_2d_tri_harmonic_polished
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)
