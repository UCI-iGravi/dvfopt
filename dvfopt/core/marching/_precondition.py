"""Shared 2.5D input precondition validation (import-light: numpy only).

Both the public pipeline (:func:`dvfopt.correct_dvf_25d`) and the publicly
exported mop (:func:`dvfopt.core.marching.mop_interior_3d`) validate their
input through :func:`require_25d_input`, so the two entry points enforce the
same contract.
"""

import numpy as np


def require_25d_input(phi, dz_tol=1e-12):
    """Validate the 2.5D preconditions: finite field, dz identically zero.

    NaN-aware: any non-finite value in phi raises (NaN would otherwise be
    invisible to every fold predicate and to the dz check itself).
    """
    if not np.isfinite(phi).all():
        raise ValueError(
            'phi contains non-finite values (NaN/Inf); 2.5D marching requires a finite field.'
        )
    if phi[0].size and float(np.abs(phi[0]).max()) > dz_tol:
        raise ValueError(
            'the 2.5D marching pipeline requires the through-plane channel '
            'dz (phi[0]) to be identically zero: the 2.5D inter-layer 6-tet '
            "math depends only on adjacent slices' in-plane displacement "
            '(dy/dx). '
            f'Found max|dz|={float(np.abs(phi[0]).max()):.3e} > dz_tol={dz_tol:.1e}. '
            'Run per-slice 2D correction first (which yields dz == 0) before '
            'calling this pipeline.'
        )
