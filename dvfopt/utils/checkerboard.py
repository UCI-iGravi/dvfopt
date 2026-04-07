"""Checkerboard image creation.

Used for visual validation of deformation field corrections -- a
checkerboard pattern makes folding artefacts easy to spot.
"""

import numpy as np


def create_checkerboard(num_squares=(8, 8), resolution=(400, 400)):
    """Create a binary checkerboard image.

    Parameters
    ----------
    num_squares : tuple of int
        ``(rows, cols)`` number of squares.
    resolution : tuple of int
        ``(height, width)`` in pixels.

    Returns
    -------
    ndarray, shape ``(height, width)``
        Values are 0 or 1.
    """
    rows, cols = num_squares
    height, width = resolution
    sq_h = height // rows
    sq_w = width // cols
    base = np.indices((rows, cols)).sum(axis=0) % 2
    board = np.kron(base, np.ones((sq_h, sq_w)))
    return board[:height, :width]
