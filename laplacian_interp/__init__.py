"""Laplacian interpolation for deformation field construction from correspondences."""

from laplacian_interp.matrix import (
    get_laplacian_index,
    get_adjacent_indices,
    laplacian_a_3d,
)
from laplacian_interp.solver import (
    slice_to_slice_3d_laplacian,
)

__all__ = [
    "get_laplacian_index",
    "get_adjacent_indices",
    "laplacian_a_3d",
    "slice_to_slice_3d_laplacian",
]
