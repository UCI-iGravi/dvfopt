"""Minimal re-implementation of the iterative SLSQP outer loop for 2D.

Used by loop-owning variants (soft_margin, active_set, trust_constr). The
re-implementation supports per-iteration trajectory capture and pluggable
constraint builders, but deliberately omits the sophisticated escalation
and oscillation-livelock logic from `dvfopt.core.slsqp_windowed.iterative` — keeping
the harness loop small and easy to reason about. Loop-owning variants may
therefore converge slightly less aggressively than the baseline; the
baseline_serial variant uses the real solver and remains the convergence
reference.

Reuses (do NOT modify):
  - dvfopt.core.slsqp_windowed.spatial.{argmin_quality, neg_jdet_bounding_window,
                                 get_nearest_center, _edge_flags,
                                 get_phi_sub_flat_padded}
  - dvfopt.core.slsqp_windowed.constraints._build_constraints (default constraint builder)
  - dvfopt.objectives.L2Objective
  - dvfopt.jacobian.numpy_jdet.jacobian_det2D
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.optimize import minimize

from benchmarks.two_triangle.result import SolverResult
from benchmarks.two_triangle.trajectory import TrajectoryAccumulator
from dvfopt.core.slsqp_windowed.constraints import _build_constraints, _quality_map
from dvfopt.core.slsqp_windowed.spatial import (
    _edge_flags,
    argmin_quality,
    get_nearest_center,
    get_phi_sub_flat_padded,
    neg_jdet_bounding_window,
)
from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.objectives import L2Objective

# Same L2 anchor the package's own window solves use.
_L2 = L2Objective()


@dataclass
class LoopOptions:
    threshold: float = 0.01
    err_tol: float = 1e-5
    max_iterations: int = 100
    max_per_index_iter: int = 5
    max_minimize_iter: int = 100
    enforce_triangles: bool = True
    timeout_s: float = 600.0
    method: str = "SLSQP"
    constraint_builder: Optional[Callable] = None  # defaults to _build_constraints
    variant_name: str = "loop_owning"


def run_minimal_iterative_2d(
    phi_initial: np.ndarray,
    *,
    threshold: float = 0.01,
    err_tol: float = 1e-5,
    max_iterations: int = 100,
    max_per_index_iter: int = 5,
    max_minimize_iter: int = 100,
    enforce_triangles: bool = True,
    timeout_s: float = 600.0,
    method: str = "SLSQP",
    constraint_builder: Optional[Callable] = None,
    variant_name: str = "loop_owning",
) -> SolverResult:
    """Run a stripped-down iterative SLSQP loop on a (2, H, W) phi.

    Returns a SolverResult with a per-outer-iter trajectory DataFrame.
    """
    phi = phi_initial.copy()
    H, W = phi.shape[1:]
    slice_shape = (1, H, W)
    max_window = (H, W)

    if constraint_builder is None:
        constraint_builder = _build_constraints

    acc = TrajectoryAccumulator()
    t0 = time.perf_counter()
    timed_out = False
    err_msg = None

    jacobian_matrix = jacobian_det2D(phi)
    quality_matrix = _quality_map(
        phi,
        enforce_shoelace=False,
        enforce_injectivity=False,
        enforce_triangles=enforce_triangles,
        jacobian_matrix=jacobian_matrix,
    )
    inner_iters_total = 0
    n_windows_this_iter = 0

    acc.record(
        outer_iter=0,
        phi=phi,
        phi_initial=phi_initial,
        n_active_windows=0,
        inner_iters=0,
        t_elapsed=0.0,
        threshold=threshold,
    )

    iteration = 0
    try:
        while iteration < max_iterations and (quality_matrix[0] <= threshold - err_tol).any():
            if time.perf_counter() - t0 > timeout_s:
                timed_out = True
                break
            iteration += 1

            # Locate worst pixel and its CC bounding window
            neg_yx = argmin_quality(quality_matrix)
            neg_mask = quality_matrix[0] <= threshold - err_tol
            labeled, _ = scipy_label(neg_mask)
            sub_size, bbox_center = neg_jdet_bounding_window(
                quality_matrix, neg_yx, threshold, err_tol, labeled=labeled
            )
            sub_size = (min(sub_size[0], H), min(sub_size[1], W))
            cz, cy, cx = get_nearest_center(bbox_center, slice_shape, sub_size)
            is_at_edge, win_at_max = _edge_flags(cy, cx, sub_size, slice_shape, max_window)
            phi_sub_flat, actual_size = get_phi_sub_flat_padded(
                phi, cz, cy, cx, slice_shape, sub_size
            )
            phi_init_sub_flat, _ = get_phi_sub_flat_padded(
                phi_initial, cz, cy, cx, slice_shape, sub_size
            )

            constraints = constraint_builder(
                phi_sub_flat,
                actual_size,
                is_at_edge,
                win_at_max,
                threshold,
                enforce_shoelace=False,
                enforce_injectivity=False,
                enforce_triangles=enforce_triangles,
            )

            minimize_options = {"maxiter": max_minimize_iter}
            if method.upper() == "SLSQP":
                minimize_options["ftol"] = 1e-9

            res = minimize(
                lambda z, a: _L2(z - a),
                phi_sub_flat,
                args=(phi_init_sub_flat,),
                method=method,
                jac=True,
                constraints=constraints,
                options=minimize_options,
            )
            inner_iters_total += int(res.nit)
            n_windows_this_iter = 1

            # Splat optimised window back into phi.
            # get_phi_sub_flat_padded packs as [dx_flat, dy_flat] (phi[1] then phi[0]).
            sy, sx = actual_size
            new_phi_sub = res.x
            phix = new_phi_sub[: sy * sx].reshape(sy, sx)  # dx  -> phi[1]
            phiy = new_phi_sub[sy * sx :].reshape(sy, sx)  # dy  -> phi[0]
            hy, hx = sy // 2, sx // 2
            hy_hi, hx_hi = sy - hy, sx - hx
            phi[1, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi] = phix
            phi[0, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi] = phiy

            jacobian_matrix = jacobian_det2D(phi)
            quality_matrix = _quality_map(
                phi,
                enforce_shoelace=False,
                enforce_injectivity=False,
                enforce_triangles=enforce_triangles,
                jacobian_matrix=jacobian_matrix,
            )

            acc.record(
                outer_iter=iteration,
                phi=phi,
                phi_initial=phi_initial,
                n_active_windows=n_windows_this_iter,
                inner_iters=int(res.nit),
                t_elapsed=time.perf_counter() - t0,
                threshold=threshold,
            )
    except Exception:  # pylint: disable=broad-except
        import traceback as _tb

        err_msg = _tb.format_exc()

    traj = acc.to_frame()
    final_fc = traj.iloc[-1]
    converged = (
        not timed_out
        and err_msg is None
        and final_fc["fold_count_jdet"] == 0
        and final_fc["fold_count_tri"] == 0
    )

    return SolverResult(
        phi_final=phi,
        trajectory=traj,
        converged=bool(converged),
        timed_out=timed_out,
        error=err_msg,
        meta={
            "variant": variant_name,
            "iterations": iteration,
            "inner_iters_total": inner_iters_total,
            "threshold": threshold,
            "method": method,
        },
    )
