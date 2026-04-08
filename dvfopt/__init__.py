"""dvfopt — Deformation Vector Field Optimizer.

Correction of negative Jacobian determinants in 2D (and 3D) deformation
(displacement) fields via SLSQP-based optimisation.

Public API
----------
Core solvers::

    from dvfopt import iterative_serial, iterative_parallel, iterative_3d

Jacobian computation::

    from dvfopt import jacobian_det2D, jacobian_det3D, sitk_jacobian_determinant

DVF utilities::

    from dvfopt import generate_random_dvf, scale_dvf

Laplacian interpolation (separate ``laplacian_interp`` package)::

    from laplacian_interp import slice_to_slice_3d_laplacian, laplacian_a_3d

Visualisation (imports matplotlib)::

    from dvfopt.viz import plot_deformations, plot_grid_before_after

Test cases (separate ``test_cases`` package)::

    from test_cases import SYNTHETIC_CASES, make_deformation
"""

# -- Package metadata -------------------------------------------------------
__version__ = "0.1.0"

# -- Core solvers ------------------------------------------------------------
from dvfopt.core import (
    iterative_serial,
    iterative_parallel,
    iterative_3d,
)

# -- Jacobian computation ---------------------------------------------------
from dvfopt.jacobian import (
    jacobian_det2D,
    jacobian_det3D,
    sitk_jacobian_determinant,
    shoelace_det2D,
    shoelace_constraint,
    injectivity_constraint,
)

# -- DVF generation / scaling ------------------------------------------------
from dvfopt.dvf import (
    generate_random_dvf,
    generate_random_dvf_3d,
    scale_dvf,
    scale_dvf_3d,
)

# -- I/O ---------------------------------------------------------------------
from dvfopt.io import load_nii_images

# -- Defaults ----------------------------------------------------------------
from dvfopt._defaults import DEFAULT_PARAMS
