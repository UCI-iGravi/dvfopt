"""DVF generation utilities."""

import numpy as np
from scipy.ndimage import zoom


def generate_random_dvf(shape, max_magnitude=5.0, seed=None):
    """Generate a random 2D deformation vector field (DVF).

    Parameters
    ----------
    shape : tuple
        ``(3, 1, H, W)`` — standard deformation field shape.
    max_magnitude : float
        Max displacement in pixels (uniform in ``[-mag, +mag]``).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(3, 1, H, W)``
    """
    rng = np.random.default_rng(seed)
    C, _, H, W = shape
    assert C == 3, "DVF must have 3 channels (dz, dy, dx)"
    return rng.uniform(-max_magnitude, max_magnitude, size=shape).astype(np.float32)


def generate_random_dvf_3d(shape, max_magnitude=5.0, seed=None):
    """Generate a random 3D deformation vector field (DVF).

    Parameters
    ----------
    shape : tuple
        ``(3, D, H, W)`` — standard 3D deformation field shape.
    max_magnitude : float
        Max displacement in voxels (uniform in ``[-mag, +mag]``).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(3, D, H, W)``
    """
    rng = np.random.default_rng(seed)
    C = shape[0]
    assert C == 3, "DVF must have 3 channels (dz, dy, dx)"
    return rng.uniform(-max_magnitude, max_magnitude, size=shape).astype(np.float64)
