"""Two-phase penalty -> log-barrier solver for 3D Jdet correction.

Full-grid L-BFGS-B (no windowing). Phase 1 drives the iterate into the
feasible region with a smooth quadratic exterior penalty under λ-continuation;
Phase 2 polishes inside the feasible interior with a log barrier under
μ-continuation.
"""

import time

import numpy as np
from scipy.optimize import minimize

from dvfopt._defaults import _log, _resolve_params
from dvfopt.core.solver import _setup_accumulators, _print_summary, _save_results
from dvfopt.core.solver3d import _init_phi_3d, _update_metrics_3d
from dvfopt.core.barrier_objective import (
    penalty_objective_3d,
    barrier_objective_3d,
    jdet_full,
)


def _pack_phi(phi):
    """phi (3,D,H,W) -> flat [dx, dy, dz]."""
    return np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])


def _unpack_phi(phi_flat, grid_size, out=None):
    D, H, W = grid_size
    n = D * H * W
    if out is None:
        out = np.empty((3, D, H, W), dtype=phi_flat.dtype)
    out[2] = phi_flat[:n].reshape(D, H, W)
    out[1] = phi_flat[n:2 * n].reshape(D, H, W)
    out[0] = phi_flat[2 * n:].reshape(D, H, W)
    return out


def iterative_3d_barrier(
    deformation,
    verbose=1,
    save_path=None,
    threshold=None,
    err_tol=None,
    margin=1e-3,
    lam_schedule=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
    mu_schedule=(1e-1, 1e-2, 1e-3, 1e-4),
    max_minimize_iter=200,
    max_iterations=None,
):
    """Correct negative Jdet voxels in 3D via penalty -> log-barrier L-BFGS-B.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``
        Input field with channels ``[dz, dy, dx]``.
    threshold, err_tol : float or None
        Override default Jdet bounds.
    margin : float
        Phase-1 target slack: drive J ≥ threshold + margin before exiting Phase 1.
    lam_schedule, mu_schedule : sequences
        Continuation parameters for the two phases.
    max_minimize_iter : int
        L-BFGS-B ``maxiter`` per continuation step.
    max_iterations : int or None
        Reserved (unused; kept for API parity with ``iterative_3d``).

    Returns
    -------
    phi : ndarray, shape ``(3, D, H, W)``
    """
    p = _resolve_params(threshold=threshold, err_tol=err_tol)
    threshold = p["threshold"]
    err_tol = p["err_tol"]

    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

    error_list, num_neg_jac, iter_times, min_jdet_list, window_counts = _setup_accumulators()

    start_time = time.time()
    phi, phi_init, D, H, W = _init_phi_3d(deformation)
    grid_size = (D, H, W)

    _, init_neg, init_min = _update_metrics_3d(phi, phi_init, num_neg_jac, min_jdet_list)
    _log(verbose, 1, f"[init] Grid {D}x{H}x{W}  threshold={threshold}  margin={margin}")
    _log(verbose, 1, f"[init] Neg-Jdet voxels: {init_neg}  min Jdet: {init_min:.6f}")

    phi_flat = _pack_phi(phi)
    phi_init_flat = _pack_phi(phi_init)

    target = threshold + margin

    # ---------------- Phase 1: penalty continuation ----------------
    feasible = init_min >= target
    for k, lam in enumerate(lam_schedule):
        if feasible:
            break
        t0 = time.time()
        res = minimize(
            penalty_objective_3d,
            phi_flat,
            args=(phi_init_flat, grid_size, threshold, margin, lam),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_minimize_iter, "disp": verbose >= 2,
                     "gtol": 1e-6},
        )
        elapsed = time.time() - t0
        iter_times.append(elapsed)
        phi_flat = res.x
        j = jdet_full(phi_flat, grid_size)
        cur_neg = int((j <= 0).sum())
        cur_min = float(j.min())
        num_neg_jac.append(cur_neg)
        min_jdet_list.append(cur_min)
        l2 = float(np.linalg.norm(phi_flat - phi_init_flat))
        error_list.append(l2)
        _log(verbose, 1,
             f"[penalty {k+1}/{len(lam_schedule)}] lam={lam:g}  "
             f"neg={cur_neg:5d}  min_J={cur_min:+.6f}  "
             f"L2={l2:.4f}  iters={res.nit}  t={elapsed:.2f}s")
        if cur_min >= target:
            feasible = True
            break

    if not feasible:
        _log(verbose, 1,
             f"[penalty] did not reach feasibility (min_J={cur_min:+.6f} < {target}); "
             "skipping barrier phase")

    # ---------------- Phase 2: barrier continuation ----------------
    if feasible:
        for k, mu in enumerate(mu_schedule):
            t0 = time.time()
            res = minimize(
                barrier_objective_3d,
                phi_flat,
                args=(phi_init_flat, grid_size, threshold, mu),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": max_minimize_iter, "disp": verbose >= 2,
                         "gtol": 1e-6},
            )
            elapsed = time.time() - t0
            iter_times.append(elapsed)
            phi_flat = res.x
            j = jdet_full(phi_flat, grid_size)
            cur_neg = int((j <= 0).sum())
            cur_min = float(j.min())
            num_neg_jac.append(cur_neg)
            min_jdet_list.append(cur_min)
            l2 = float(np.linalg.norm(phi_flat - phi_init_flat))
            error_list.append(l2)
            _log(verbose, 1,
                 f"[barrier {k+1}/{len(mu_schedule)}] mu={mu:g}  "
                 f"neg={cur_neg:5d}  min_J={cur_min:+.6f}  "
                 f"L2={l2:.4f}  iters={res.nit}  t={elapsed:.2f}s")

    _unpack_phi(phi_flat, grid_size, out=phi)
    elapsed_total = time.time() - start_time

    j_final = jdet_full(phi_flat, grid_size)
    final_neg = int((j_final <= 0).sum())
    final_min = float(j_final.min())
    final_err = float(np.linalg.norm(phi_flat - phi_init_flat))

    _print_summary(verbose, "Penalty->Barrier L-BFGS-B - 3D",
                   (D, H, W), len(iter_times),
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed_total)

    if save_path is not None:
        _save_results(
            save_path, method="penalty_barrier_lbfgsb",
            threshold=threshold, err_tol=err_tol,
            max_iterations=len(iter_times),
            max_per_index_iter=0, max_minimize_iter=max_minimize_iter,
            grid_shape=(D, H, W), elapsed=elapsed_total,
            final_err=final_err, init_neg=init_neg, final_neg=final_neg,
            init_min=init_min, final_min=final_min,
            iteration=len(iter_times), phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi
