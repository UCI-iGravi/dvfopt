"""Random 3D DVF cases with fixed seeds for reproducibility.

Uses dvfopt.dvf.generate_random_dvf_3d which builds a smooth random vector
field of the requested shape. We provide grid sizes 16, 24, 32 across two
fold-severity levels (controlled by max_magnitude).

Note: 32 cubed is the upper end here; larger grids (64, 128) push baseline
SLSQP runtime past the per-cell timeout. Once active-set or multigrid
variants are validated, larger sizes can be added.
"""

import numpy as np

from benchmarks.two_triangle.registry import register_case
from dvfopt.dvf import generate_random_dvf_3d


def _make(grid: int, magnitude: float, seed: int) -> np.ndarray:
    """generate_random_dvf_3d takes (3, D, H, W) shape; original_shape is the
    *underlying* random field that gets upsampled. We use a small original
    shape so the resulting field is smooth, then it's already at target size."""
    return generate_random_dvf_3d((3, grid, grid, grid), max_magnitude=magnitude, seed=seed)


_SEVERITIES = {"low": 1.5, "high": 4.0}
_GRIDS = (16, 24, 32)


def _register_grid(grid: int):
    for sev_name, sev_val in _SEVERITIES.items():
        case_name = f"rand3d_grid{grid}_{sev_name}"
        seed = 42 + grid + (0 if sev_name == "low" else 1)

        def _factory(g=grid, m=sev_val, s=seed, n=case_name):
            def case_fn():
                return _make(g, m, s), {
                    "title": f"Random 3D ({g}^3, {n.split('_')[-1]})",
                    "dim": 3,
                    "grid": g,
                    "magnitude": m,
                    "seed": s,
                }

            return case_fn

        register_case(case_name, category="random_3d", dim=3, grid=grid, severity=sev_name)(
            _factory()
        )


for _g in _GRIDS:
    _register_grid(_g)
