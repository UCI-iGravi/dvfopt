"""Multigrid wrapper variant.

Strategy: solve at 1/4 resolution first (cheap, smooths out adjacent folds),
upsample as warm-start to 1/2 resolution, solve, upsample to full, solve.
At each level uses the unmodified iterative_serial / iterative_3d.

Levels are skipped automatically when the downsampled grid would be smaller
than min_dim_per_axis (default 8).
"""

import time
import traceback

import numpy as np
import pandas as pd

from benchmarks.two_triangle.metrics import (
    fold_counts,
    l2_displacement,
    smoothness,
)
from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult
from dvfopt.core.slsqp_windowed.iterative import iterative_serial
from dvfopt.core.slsqp_windowed.iterative3d import iterative_3d
from dvfopt.dvf.scaling import scale_dvf, scale_dvf_3d


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _downsample(dvf: np.ndarray, factor: float) -> np.ndarray:
    if _is_3d(dvf):
        D, H, W = dvf.shape[1:]
        new = (max(2, int(D * factor)), max(2, int(H * factor)), max(2, int(W * factor)))
        return scale_dvf_3d(dvf, new)
    H, W = dvf.shape[2:]
    new = (max(2, int(H * factor)), max(2, int(W * factor)))
    return scale_dvf(dvf, new)


def _upsample_to(dvf: np.ndarray, target_shape: tuple) -> np.ndarray:
    if _is_3d(dvf):
        return scale_dvf_3d(dvf, target_shape)
    return scale_dvf(dvf, target_shape)


@register_variant("multigrid")
def multigrid(
    dvf: np.ndarray,
    *,
    threshold: float = 0.01,
    max_iterations: int = 100,
    min_dim_per_axis: int = 8,
    enforce_triangles: bool = True,
    timeout_s: float = 600.0,
    **_unused,
) -> SolverResult:
    is_3d = _is_3d(dvf)
    phi_initial = dvf.copy()
    t0 = time.perf_counter()
    err = None
    rows = []

    def canon(d):
        if is_3d or d.ndim != 4 or d.shape[1] != 1:
            return d
        return np.stack([d[1, 0], d[2, 0]])

    phi_can_init = canon(phi_initial)
    fc = fold_counts(phi_can_init, threshold=threshold)
    rows.append(
        {
            "outer_iter": 0,
            "time_s": 0.0,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": 0.0,
            "smoothness": smoothness(phi_can_init),
            "n_active_windows": 0,
            "inner_iters": 0,
        }
    )

    cur_dvf = dvf.copy()
    full_shape = cur_dvf.shape[1:] if is_3d else cur_dvf.shape[2:]

    levels = [0.25, 0.5, 1.0]
    for li, factor in enumerate(levels, start=1):
        if factor < 1.0:
            target_shape = tuple(max(2, int(d * factor)) for d in full_shape)
            if min(target_shape) < min_dim_per_axis:
                continue  # Skip levels too small to be meaningful
            level_dvf = _downsample(cur_dvf, factor)
        else:
            level_dvf = cur_dvf

        try:
            if is_3d:
                phi_solved = iterative_3d(
                    level_dvf, threshold=threshold, max_iterations=max_iterations, verbose=0
                )
            else:
                phi_solved = iterative_serial(
                    level_dvf,
                    threshold=threshold,
                    max_iterations=max_iterations,
                    verbose=0,
                    enforce_triangles=enforce_triangles,
                )
        except Exception:
            err = traceback.format_exc()
            break

        # Re-pack solver output back into (3, [1|D], H, W) form for upsampling
        if is_3d:
            cur_dvf = phi_solved
        else:
            cur_dvf = level_dvf.copy()
            cur_dvf[1, 0] = phi_solved[0]
            cur_dvf[2, 0] = phi_solved[1]

        if factor < 1.0:
            cur_dvf = _upsample_to(cur_dvf, full_shape)

        phi_can = canon(cur_dvf)
        fc = fold_counts(phi_can, threshold=threshold)
        rows.append(
            {
                "outer_iter": li,
                "time_s": time.perf_counter() - t0,
                "fold_count_jdet": fc["fold_count_jdet"],
                "fold_count_tri": fc["fold_count_tri"],
                "max_violation": fc["max_violation"],
                "l2_disp": l2_displacement(phi_can, phi_can_init),
                "smoothness": smoothness(phi_can),
                "n_active_windows": 0,
                "inner_iters": 0,
            }
        )

    phi_final = canon(cur_dvf)
    fc_final = fold_counts(phi_final, threshold=threshold)
    converged = err is None and fc_final["fold_count_jdet"] == 0 and fc_final["fold_count_tri"] == 0
    return SolverResult(
        phi_final=phi_final,
        trajectory=pd.DataFrame(rows),
        converged=converged,
        timed_out=False,
        error=err,
        meta={"variant": "multigrid", "is_3d": is_3d, "levels": levels, "threshold": threshold},
    )
