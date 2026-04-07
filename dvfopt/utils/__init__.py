"""Small standalone utility modules."""

from dvfopt.utils.checkerboard import create_checkerboard
from dvfopt.utils.correspondences import (
    remove_duplicates,
    do_lines_intersect,
    swap_correspondences,
    detect_intersecting_segments,
    downsample_points,
)

__all__ = [
    "create_checkerboard",
    "remove_duplicates",
    "do_lines_intersect",
    "swap_correspondences",
    "detect_intersecting_segments",
    "downsample_points",
]
