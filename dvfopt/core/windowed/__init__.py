"""Windowed fold-correction engine — dvfopt's third shared engine.

Modules: ``_common`` (the engine: :func:`windowed_correct`, window finding,
round loop, giant tiling, mop, damage accounting), ``_locality``
(per-constraint :class:`WindowLocality` registry — ring widths, fold maps,
influenced rows), ``_inners`` (the :class:`WindowSub` reduced-problem
contract + inner-solver dispatch). See :mod:`._common` for the no-damage
invariant and the inner contract.
"""

from ._banded import BandedReport, windowed_correct_banded
from ._common import (
    DEFAULTS_BY_DIM,
    SliceReport,
    build_subproblem,
    find_windows,
    resolve_dim_defaults,
    windowed_correct,
)
from ._inners import WindowSub
from ._locality import LOCALITY, min_field, pixel_fold_mask

__all__ = [
    'DEFAULTS_BY_DIM',
    'LOCALITY',
    'BandedReport',
    'SliceReport',
    'WindowSub',
    'build_subproblem',
    'find_windows',
    'min_field',
    'pixel_fold_mask',
    'resolve_dim_defaults',
    'windowed_correct',
    'windowed_correct_banded',
]
