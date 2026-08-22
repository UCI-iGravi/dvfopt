"""SVF warm-start wrapper variant.

Pre-process: fit a stationary velocity field v to phi via L2 (a few
Gauss-Newton iterations on the residual `phi - exp(v)`), compute
phi_pre = exp(v) using scaling-and-squaring, then call iterative_serial
on the residual. SVFs are diffeomorphic when ||grad v|| is small enough,
so phi_pre often has many fewer folds than phi_initial.

This is a lightweight implementation — research-grade SVF libraries
(e.g. voxelmorph) do this much more carefully. The point of the harness
is to measure whether even a simple SVF projection helps.
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


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _exp_field_2d(v: np.ndarray, n_squarings: int = 4) -> np.ndarray:
    """Scaling-and-squaring exponential of a (2, H, W) velocity field."""
    phi = v / (2**n_squarings)
    H, W = phi.shape[1:]
    yy, xx = np.indices((H, W), dtype=np.float64)
    for _ in range(n_squarings):
        # phi_new(x) = phi(x) + phi(x + phi(x))
        # Sample phi at displaced locations via bilinear interp.
        sample_y = yy + phi[0]
        sample_x = xx + phi[1]
        sy0 = np.clip(np.floor(sample_y).astype(int), 0, H - 1)
        sx0 = np.clip(np.floor(sample_x).astype(int), 0, W - 1)
        sy1 = np.clip(sy0 + 1, 0, H - 1)
        sx1 = np.clip(sx0 + 1, 0, W - 1)
        wy = sample_y - sy0
        wx = sample_x - sx0
        sampled = np.zeros_like(phi)
        for c in (0, 1):
            sampled[c] = (
                phi[c, sy0, sx0] * (1 - wy) * (1 - wx)
                + phi[c, sy0, sx1] * (1 - wy) * wx
                + phi[c, sy1, sx0] * wy * (1 - wx)
                + phi[c, sy1, sx1] * wy * wx
            )
        phi = phi + sampled
    return phi


def _fit_svf_2d(phi_target: np.ndarray, n_iter: int = 3, step: float = 0.5) -> np.ndarray:
    """Tiny Gauss-Newton fitter: v += step * (phi_target - exp(v))."""
    v = phi_target.copy()  # start from phi_target as initial v
    for _ in range(n_iter):
        residual = phi_target - _exp_field_2d(v)
        v = v + step * residual
    return v


@register_variant("svf_warmstart")
def svf_warmstart(
    dvf: np.ndarray,
    *,
    threshold: float = 0.01,
    max_iterations: int = 100,
    enforce_triangles: bool = True,
    timeout_s: float = 600.0,
    svf_iter: int = 3,
    n_squarings: int = 4,
    **_unused,
) -> SolverResult:
    is_3d = _is_3d(dvf)
    if is_3d:
        # 3D SVF requires a separate exp implementation; fall back to baseline.
        from benchmarks.two_triangle.variants.baseline_serial import baseline_serial

        r = baseline_serial(
            dvf,
            threshold=threshold,
            max_iterations=max_iterations,
            enforce_triangles=enforce_triangles,
            timeout_s=timeout_s,
        )
        r.meta["variant"] = "svf_warmstart"
        r.meta["fallback"] = "baseline_3d"
        return r

    phi_initial = np.stack([dvf[1, 0], dvf[2, 0]])
    t0 = time.perf_counter()
    err = None
    rows = []

    fc = fold_counts(phi_initial, threshold=threshold)
    rows.append(
        {
            "outer_iter": 0,
            "time_s": 0.0,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": 0.0,
            "smoothness": smoothness(phi_initial),
            "n_active_windows": 0,
            "inner_iters": 0,
        }
    )

    # --- SVF projection ---
    try:
        v = _fit_svf_2d(phi_initial, n_iter=svf_iter)
        phi_svf = _exp_field_2d(v, n_squarings=n_squarings)
    except Exception:
        err = traceback.format_exc()
        phi_svf = phi_initial

    fc = fold_counts(phi_svf, threshold=threshold)
    rows.append(
        {
            "outer_iter": 1,
            "time_s": time.perf_counter() - t0,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": l2_displacement(phi_svf, phi_initial),
            "smoothness": smoothness(phi_svf),
            "n_active_windows": 0,
            "inner_iters": 0,
        }
    )

    # --- Baseline call on residual ---
    phi_post = phi_svf
    if err is None and max_iterations > 0:
        try:
            dvf_post = dvf.copy()
            dvf_post[1, 0] = phi_svf[0]
            dvf_post[2, 0] = phi_svf[1]
            phi_solved = iterative_serial(
                dvf_post,
                threshold=threshold,
                max_iterations=max_iterations,
                verbose=0,
                enforce_triangles=enforce_triangles,
            )
            phi_post = phi_solved
        except Exception:
            err = traceback.format_exc()

    fc = fold_counts(phi_post, threshold=threshold)
    rows.append(
        {
            "outer_iter": 2,
            "time_s": time.perf_counter() - t0,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": l2_displacement(phi_post, phi_initial),
            "smoothness": smoothness(phi_post),
            "n_active_windows": 0,
            "inner_iters": 0,
        }
    )

    converged = err is None and fc["fold_count_jdet"] == 0 and fc["fold_count_tri"] == 0
    return SolverResult(
        phi_final=phi_post,
        trajectory=pd.DataFrame(rows),
        converged=converged,
        timed_out=False,
        error=err,
        meta={
            "variant": "svf_warmstart",
            "is_3d": is_3d,
            "threshold": threshold,
            "svf_iter": svf_iter,
            "n_squarings": n_squarings,
        },
    )
