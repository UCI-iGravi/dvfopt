"""Combinatorial pre-fix wrapper variant.

Pre-pass: detect cells where exactly one of the four signed triangle areas
is negative (= 'isolated single-vertex flip'), find the offending vertex,
and damp its displacement towards zero by a halving search until the cell
becomes feasible. Then call the unmodified iterative_serial / iterative_3d
on the residual.

This is a deliberately simple heuristic — it won't fix all isolated flips
(some require coordinated multi-vertex moves), but it's cheap and never
makes things worse on its own (each damping step is rejected if it
introduces new folds).
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
from dvfopt.jacobian.shoelace import _all_triangle_areas_2d


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _prefix_pass_2d(phi: np.ndarray, threshold: float, max_halvings: int = 6) -> np.ndarray:
    """Damp displacements at single-vertex flip cells.

    For each cell with exactly one negative triangle area, identify the
    vertex shared by the most negative triangles and halve its displacement
    repeatedly until the cell is feasible. Reject the change and restore
    if any new fold is introduced anywhere.
    """
    phi = phi.copy()
    _H, _W = phi.shape[1:]
    for _ in range(max_halvings):
        tri = _all_triangle_areas_2d(phi[0], phi[1])  # (4, H-1, W-1)
        bad = tri < threshold
        bad_per_cell = bad.sum(axis=0)  # (H-1, W-1)
        # "Isolated" = exactly one bad triangle in the cell.
        isolated = np.argwhere(bad_per_cell == 1)
        if isolated.size == 0:
            break
        improved = False
        for cy, cx in isolated:
            # Damp each of the 4 corner vertices by 0.5 and accept the move
            # that maximises the cell-min triangle area without introducing
            # new folds.
            best_delta = None
            best_min = tri[:, cy, cx].min()
            for dy in (0, 1):
                for dx in (0, 1):
                    vy, vx = cy + dy, cx + dx
                    saved = phi[:, vy, vx].copy()
                    phi[:, vy, vx] *= 0.5
                    new_tri = _all_triangle_areas_2d(phi[0], phi[1])
                    new_cell_min = new_tri[:, cy, cx].min()
                    no_new_folds = (new_tri >= threshold).sum() >= (tri >= threshold).sum()
                    if new_cell_min > best_min and no_new_folds:
                        best_min = new_cell_min
                        best_delta = (vy, vx)
                    phi[:, vy, vx] = saved  # restore for next try
            if best_delta is not None:
                vy, vx = best_delta
                phi[:, vy, vx] *= 0.5
                improved = True
        if not improved:
            break
    return phi


def _prefix_pass_3d(dvf: np.ndarray, threshold: float) -> np.ndarray:
    """3D version: no-op for now (no equivalent helper in dvfopt yet).

    The 3D constraint coverage helper (notebook 12c) hasn't been factored
    into a reusable function, so the 3D prefix pass is identity. Once a
    `_all_tetrahedron_volumes_3d` helper exists, port `_prefix_pass_2d`.
    """
    return dvf.copy()


@register_variant("combinatorial_prefix")
def combinatorial_prefix(
    dvf: np.ndarray,
    *,
    threshold: float = 0.01,
    max_iterations: int = 100,
    enforce_triangles: bool = True,
    timeout_s: float = 600.0,
    **_unused,
) -> SolverResult:
    is_3d = _is_3d(dvf)
    phi_initial = dvf.copy()
    t0 = time.perf_counter()
    err = None
    rows = []

    # Initial state row
    if is_3d:
        phi_can = phi_initial.copy()
    else:
        phi_can = np.stack([phi_initial[1, 0], phi_initial[2, 0]])
    fc = fold_counts(phi_can, threshold=threshold)
    rows.append(
        {
            "outer_iter": 0,
            "time_s": 0.0,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": 0.0,
            "smoothness": smoothness(phi_can),
            "n_active_windows": 0,
            "inner_iters": 0,
        }
    )

    # --- Prefix pass ---
    if is_3d:
        dvf_pref = _prefix_pass_3d(dvf, threshold)
        phi_pref_can = dvf_pref.copy()
    else:
        phi_2d = np.stack([dvf[1, 0], dvf[2, 0]])
        phi_pref_2d = _prefix_pass_2d(phi_2d, threshold)
        dvf_pref = dvf.copy()
        dvf_pref[1, 0] = phi_pref_2d[0]
        dvf_pref[2, 0] = phi_pref_2d[1]
        phi_pref_can = phi_pref_2d

    fc = fold_counts(phi_pref_can, threshold=threshold)
    rows.append(
        {
            "outer_iter": 1,
            "time_s": time.perf_counter() - t0,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": l2_displacement(phi_pref_can, phi_can),
            "smoothness": smoothness(phi_pref_can),
            "n_active_windows": 0,
            "inner_iters": 0,
        }
    )

    # --- Baseline call on residual ---
    if max_iterations > 0:
        try:
            if is_3d:
                phi_post = iterative_3d(
                    dvf_pref, threshold=threshold, max_iterations=max_iterations, verbose=0
                )
            else:
                phi_post = iterative_serial(
                    dvf_pref,
                    threshold=threshold,
                    max_iterations=max_iterations,
                    verbose=0,
                    enforce_triangles=enforce_triangles,
                )
        except Exception:
            err = traceback.format_exc()
            phi_post = phi_pref_can
    else:
        phi_post = phi_pref_can

    if not is_3d and phi_post.ndim == 4 and phi_post.shape[1] == 1:
        phi_post_can = np.stack([phi_post[1, 0], phi_post[2, 0]])
    else:
        phi_post_can = phi_post

    fc = fold_counts(phi_post_can, threshold=threshold)
    rows.append(
        {
            "outer_iter": 2,
            "time_s": time.perf_counter() - t0,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": l2_displacement(phi_post_can, phi_can),
            "smoothness": smoothness(phi_post_can),
            "n_active_windows": 0,
            "inner_iters": 0,
        }
    )

    converged = err is None and fc["fold_count_jdet"] == 0 and fc["fold_count_tri"] == 0
    return SolverResult(
        phi_final=phi_post_can,
        trajectory=pd.DataFrame(rows),
        converged=converged,
        timed_out=False,
        error=err,
        meta={
            "variant": "combinatorial_prefix",
            "is_3d": is_3d,
            "threshold": threshold,
            "max_iterations": max_iterations,
        },
    )
