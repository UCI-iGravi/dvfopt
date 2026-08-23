"""Windowed fold-correction engine — dvfopt's third shared engine.

Modules: ``_common`` (the engine: :func:`windowed_correct`, window finding,
round loop, giant tiling, mop, damage accounting), ``_locality``
(per-constraint :class:`WindowLocality` registry — ring widths, fold maps,
influenced rows), ``_inners`` (the :class:`WindowSub` reduced-problem
contract + inner-solver dispatch). See :mod:`._common` for the no-damage
invariant and the inner contract.
"""

from ._common import SliceReport, windowed_correct
from ._inners import WindowSub
from ._locality import LOCALITY, min_field, pixel_fold_mask

__all__ = [
    'LOCALITY',
    'SliceReport',
    'WindowSub',
    'min_field',
    'pixel_fold_mask',
    'windowed_correct',
]
