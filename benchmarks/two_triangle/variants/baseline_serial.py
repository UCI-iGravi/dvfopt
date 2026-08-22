"""Baseline variant: calls dvfopt iterative_serial / iterative_3d unchanged.

Produces a 1-row trajectory containing the final-state values only — no
per-iteration data is captured, since the existing solvers do not expose
hooks. This is the reference for all other variants.
"""

import time
import traceback

import numpy as np
import pandas as pd

from benchmarks.two_triangle.metrics import fold_counts, l2_displacement, smoothness
from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult
from dvfopt.core.slsqp_windowed.iterative import iterative_serial
from dvfopt.core.slsqp_windowed.iterative3d import iterative_3d


def _is_3d_input(dvf: np.ndarray) -> bool:
    """dvf shape (3, D, H, W) with D > 1 is 3D; (3, 1, H, W) is 2D."""
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


@register_variant("baseline_serial")
def baseline_serial(
    dvf: np.ndarray,
    *,
    threshold: float = 0.01,
    max_iterations: int = 100,
    enforce_triangles: bool = True,
    timeout_s: float = 600.0,
    **_unused,
) -> SolverResult:
    is_3d = _is_3d_input(dvf)
    phi_initial = dvf.copy()
    t0 = time.perf_counter()
    err = None
    converged = False
    try:
        if is_3d:
            phi_final = iterative_3d(
                dvf.copy(),
                threshold=threshold,
                max_iterations=max_iterations,
                verbose=0,
            )
        else:
            phi_final = iterative_serial(
                dvf.copy(),
                threshold=threshold,
                max_iterations=max_iterations,
                verbose=0,
                enforce_triangles=enforce_triangles,
            )
        elapsed = time.perf_counter() - t0
        # Determine convergence by checking remaining folds
        fc = fold_counts(phi_final, threshold=threshold)
        converged = fc["fold_count_jdet"] == 0 and fc["fold_count_tri"] == 0
    except Exception:  # pylint: disable=broad-except
        elapsed = time.perf_counter() - t0
        err = traceback.format_exc()
        # Return phi_initial as fallback so downstream metrics don't crash.
        phi_final = phi_initial[1:, 0] if not is_3d else phi_initial.copy()

    if not is_3d and phi_final.ndim == 4 and phi_final.shape[1] == 1:
        phi_final_canonical = np.stack([phi_final[1, 0], phi_final[2, 0]])
    else:
        phi_final_canonical = phi_final

    fc = fold_counts(phi_final_canonical, threshold=threshold)
    if not is_3d:
        phi_init_canonical = np.stack([phi_initial[1, 0], phi_initial[2, 0]])
    else:
        phi_init_canonical = phi_initial
    traj = pd.DataFrame(
        [
            {
                "outer_iter": 0,
                "time_s": elapsed,
                "fold_count_jdet": fc["fold_count_jdet"],
                "fold_count_tri": fc["fold_count_tri"],
                "max_violation": fc["max_violation"],
                "l2_disp": l2_displacement(phi_final_canonical, phi_init_canonical),
                "smoothness": smoothness(phi_final_canonical),
                "n_active_windows": 0,
                "inner_iters": 0,
            }
        ]
    )
    return SolverResult(
        phi_final=phi_final_canonical,
        trajectory=traj,
        converged=converged,
        timed_out=False,
        error=err,
        meta={
            "variant": "baseline_serial",
            "is_3d": is_3d,
            "threshold": threshold,
            "max_iterations": max_iterations,
            "enforce_triangles": enforce_triangles,
        },
    )
