"""Soft-margin variant: progressive threshold tightening.

Strategy:
- Start with threshold=0 (sign-only): "any positive area is feasible".
- Once a feasible point is found, tighten threshold to the user-supplied
  value (default 0.01) and continue iterating.
- Uses the minimal re-implemented loop with a constraint builder closure
  that closes over the current threshold.
"""

import numpy as np

from benchmarks.two_triangle._iterative_loop import run_minimal_iterative_2d
from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult
from dvfopt.core.slsqp.constraints import _build_constraints


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _builder_for_threshold(thr: float):
    def _build(
        phi_sub_flat,
        sub_size,
        is_at_edge,
        win_at_max,
        _thr_unused,
        enforce_shoelace=False,
        enforce_injectivity=False,
        enforce_triangles=True,
    ):
        return _build_constraints(
            phi_sub_flat,
            sub_size,
            is_at_edge,
            win_at_max,
            thr,
            enforce_shoelace=enforce_shoelace,
            enforce_injectivity=enforce_injectivity,
            enforce_triangles=enforce_triangles,
        )

    return _build


@register_variant("soft_margin")
def soft_margin(
    dvf: np.ndarray,
    *,
    threshold: float = 0.01,
    max_iterations: int = 100,
    enforce_triangles: bool = True,
    timeout_s: float = 600.0,
    **_unused,
) -> SolverResult:
    if _is_3d(dvf):
        # 3D not yet supported in the minimal loop; fall back to baseline.
        from benchmarks.two_triangle.variants.baseline_serial import baseline_serial

        r = baseline_serial(
            dvf,
            threshold=threshold,
            max_iterations=max_iterations,
            enforce_triangles=enforce_triangles,
            timeout_s=timeout_s,
        )
        r.meta["variant"] = "soft_margin"
        r.meta["fallback"] = "baseline_3d"
        return r

    phi_initial = np.stack([dvf[1, 0], dvf[2, 0]])

    # --- Stage 1: sign-only threshold ---
    half_iters = max(1, max_iterations // 2)
    r1 = run_minimal_iterative_2d(
        phi_initial,
        threshold=0.0,
        max_iterations=half_iters,
        enforce_triangles=enforce_triangles,
        timeout_s=timeout_s,
        constraint_builder=_builder_for_threshold(0.0),
        variant_name="soft_margin_stage1",
    )
    if r1.timed_out or r1.error is not None:
        return r1

    # --- Stage 2: tighten to target threshold ---
    r2 = run_minimal_iterative_2d(
        r1.phi_final,
        threshold=threshold,
        max_iterations=max_iterations - half_iters,
        enforce_triangles=enforce_triangles,
        timeout_s=timeout_s,
        constraint_builder=_builder_for_threshold(threshold),
        variant_name="soft_margin_stage2",
    )

    # Stitch trajectories: re-base stage2's outer_iter and time_s on top of
    # stage1's tail.
    if not r1.trajectory.empty:
        last = r1.trajectory.iloc[-1]
        r2.trajectory["outer_iter"] = r2.trajectory["outer_iter"] + last["outer_iter"]
        r2.trajectory["time_s"] = r2.trajectory["time_s"] + last["time_s"]
        # Drop the duplicate "row 0" of stage2 (it equals stage1's tail)
        if len(r2.trajectory) > 1:
            r2.trajectory = r2.trajectory.iloc[1:]
        import pandas as pd

        r2.trajectory = pd.concat([r1.trajectory, r2.trajectory], ignore_index=True)

    r2.meta["variant"] = "soft_margin"
    r2.meta["stage1_threshold"] = 0.0
    r2.meta["stage2_threshold"] = threshold
    return r2
