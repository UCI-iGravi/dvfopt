"""Private utility modules for the windowed SLSQP solver loop.

These are pure-implementation helpers — accumulator scaffolding,
metric updates, window detection — that have a single consumer
(:mod:`dvfopt.core.solver`). They live under ``_internal/`` to
signal "this is not for external use" more strongly than a single
underscore prefix.

Nothing here is part of the public API; do not import directly from
user code.
"""

from dvfopt.core._internal.io import (
    _init_phi,
    _print_summary,
    _save_results,
    _setup_accumulators,
)
from dvfopt.core._internal.metrics import (
    _patch_jacobian_2d,
    _update_metrics,
)
from dvfopt.core._internal.window import (
    _apply_result,
    _full_grid_step,
    _optimize_single_window,
)

__all__ = [
    '_apply_result',
    '_full_grid_step',
    '_init_phi',
    '_optimize_single_window',
    '_patch_jacobian_2d',
    '_print_summary',
    '_save_results',
    '_setup_accumulators',
    '_update_metrics',
]
