"""2.5D marching sweep for inter-layer 6-tet fold repair.

This subpackage productizes the "marching full volume" experiment from
``research/strict_feasibility_3d``: each z-slice of a ``(3, D, H, W)``
displacement field is repaired against its already-repaired neighbour,
sweeping outward from the mildest layer so no slice is cold-started
against raw data.

The core primitives live in :mod:`dvfopt.core.marching._marching_25d` and
:mod:`dvfopt.core.marching._mop_interior_3d`; the public pipeline that wires
them together is :func:`dvfopt.correct_dvf_25d`.

Re-exporting the primitives here is safe for Windows spawn: ``ProcessPoolExecutor``
children import the worker module ``_marching_25d`` DIRECTLY (they never import
this package ``__init__``), so these re-exports add no import side effects to
spawned workers.
"""

from dvfopt.core.marching._marching_25d import layer_min_v, march_slice
from dvfopt.core.marching._mop_interior_3d import mop_interior_3d

__all__ = ['layer_min_v', 'march_slice', 'mop_interior_3d']
