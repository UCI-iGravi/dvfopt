"""Two-phase penalty -> log-barrier solver for 3D Jdet correction.

Component-aware windowed L-BFGS-B (default) or full-grid. Phase 1 drives
the iterate into the feasible region with a smooth quadratic exterior
penalty under λ-continuation; Phase 2 polishes inside the feasible
interior with a log barrier under μ-continuation.

Windowed mode mirrors the SLSQP windowed approach: for each connected
component of negative-Jdet voxels, extract a bounding patch (bbox + pad
voxels on each side), freeze the outer boundary ring via bounds, and run
the penalty→barrier continuation on just that patch. Dramatically lower
memory (L-BFGS history vectors scale with patch size) and compute.
"""

import time

import numpy as np
from scipy.ndimage import label

from dvfopt._defaults import _log, _resolve_params
from dvfopt.core._barrier_core import run_penalty_barrier_lbfgs
from dvfopt.core.barrier_objective import _jdet_grad_T_v, jdet_full
from dvfopt.core.solver import _print_summary, _save_results, _setup_accumulators
from dvfopt.core.solver3d import _init_phi_3d, _update_metrics_3d


def _pack_phi(phi):
    """phi (3,D,H,W) -> flat [dx, dy, dz]."""
    return np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])


def _unpack_phi(phi_flat, grid_size, out=None):
    D, H, W = grid_size
    n = D * H * W
    if out is None:
        out = np.empty((3, D, H, W), dtype=phi_flat.dtype)
    out[2] = phi_flat[:n].reshape(D, H, W)
    out[1] = phi_flat[n : 2 * n].reshape(D, H, W)
    out[0] = phi_flat[2 * n :].reshape(D, H, W)
    return out


def _patch_bbox(comp_coords, pad, grid_shape):
    """Return (z0, z1, y0, y1, x0, x1) inclusive bbox of a component + pad, clamped."""
    zs, ys, xs = comp_coords
    D, H, W = grid_shape
    z0 = max(int(zs.min()) - pad, 0)
    z1 = min(int(zs.max()) + pad, D - 1)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, H - 1)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, W - 1)
    return z0, z1, y0, y1, x0, x1


def _patch_frozen_mask(z0, z1, y0, y1, x0, x1, grid_shape):
    """Boolean mask (Dp,Hp,Wp) marking patch-boundary voxels to freeze.

    Faces touching the full-grid boundary are NOT frozen (they have no
    exterior context anyway). All other faces of the patch ARE frozen.
    """
    D, H, W = grid_shape
    Dp, Hp, Wp = z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1
    mask = np.zeros((Dp, Hp, Wp), dtype=bool)
    if z0 > 0:
        mask[0, :, :] = True
    if z1 < D - 1:
        mask[-1, :, :] = True
    if y0 > 0:
        mask[:, 0, :] = True
    if y1 < H - 1:
        mask[:, -1, :] = True
    if x0 > 0:
        mask[:, :, 0] = True
    if x1 < W - 1:
        mask[:, :, -1] = True
    return mask


def _patch_bounds(phi_flat_patch, frozen_mask):
    """Build L-BFGS-B bounds list: frozen voxels pinned, others free."""
    Dp, Hp, Wp = frozen_mask.shape
    n = Dp * Hp * Wp
    frozen_flat = frozen_mask.ravel()
    bounds = [(None, None)] * (3 * n)
    for ch in range(3):
        base = ch * n
        for idx in np.nonzero(frozen_flat)[0]:
            v = phi_flat_patch[base + idx]
            bounds[base + idx] = (v, v)
    return bounds


def _optimize_patch(
    phi,
    phi_init,
    z0,
    z1,
    y0,
    y1,
    x0,
    x1,
    grid_shape,
    threshold,
    margin,
    lam_schedule,
    mu_schedule,
    max_minimize_iter,
    verbose,
):
    """Run penalty -> barrier continuation on a single patch.

    Anchors the data term to the *current* patch state (not the original
    phi_init), mirroring SLSQP's windowed sub-solve.  Mutates *phi* in
    place. Returns ``(patch_elapsed, lam_steps, mu_steps, patch_final_min)``.
    """
    phi_patch = phi[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1].copy()
    Dp, Hp, Wp = phi_patch.shape[1:]
    patch_size = (Dp, Hp, Wp)

    phi_flat = _pack_phi(phi_patch)
    phi_anchor = phi_flat.copy()

    frozen_mask = _patch_frozen_mask(z0, z1, y0, y1, x0, x1, grid_shape)
    bounds = _patch_bounds(phi_flat, frozen_mask)

    # The rim's Jdet uses one-sided differences that do not match the global
    # central-difference Jdet, so penalising it drives L-BFGS-B to chase a
    # phantom residual and introduces artefacts outside the patch. Restrict
    # the penalty/barrier sum to interior (non-frozen) voxels — their Jdet
    # agrees with the global field by construction.
    active_mask = (~frozen_mask).ravel()

    t_start = time.time()
    phi_flat, info = run_penalty_barrier_lbfgs(
        phi_flat,
        phi_anchor,
        constraint_values=lambda p: jdet_full(p, patch_size),
        constraint_adjoint=lambda p, v: _jdet_grad_T_v(p, patch_size, v),
        threshold=threshold,
        margin=margin,
        lam_schedule=lam_schedule,
        mu_schedule=mu_schedule,
        max_iter=max_minimize_iter,
        active_mask=active_mask,
        bounds=bounds,
        anchor='l2',
        verbose=max(0, verbose - 2),  # core's verbose=1 is per-step; suppress unless very chatty
    )
    lam_steps = info['lam_steps']
    mu_steps = info['mu_steps']

    # Write patch back into full phi.
    _unpack_phi(phi_flat, patch_size, out=phi_patch)
    phi[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = phi_patch

    j_final = jdet_full(phi_flat, patch_size)
    return (time.time() - t_start, lam_steps, mu_steps, float(j_final[active_mask].min()))


def _iterative_3d_barrier_windowed(
    phi,
    phi_init,
    phi_init_flat,
    grid_size,
    threshold,
    err_tol,
    margin,
    lam_schedule,
    mu_schedule,
    max_minimize_iter,
    max_iterations,
    pad,
    verbose,
    error_list,
    num_neg_jac,
    iter_times,
    min_jdet_list,
    window_counts,
):
    """Windowed penalty->barrier loop. Mutates phi & accumulators in place."""
    D, H, W = grid_size
    structure = np.ones((3, 3, 3))  # 26-connectivity
    converged = False
    cur_neg = -1
    cur_min = float("nan")

    for iteration in range(max_iterations):
        j = jdet_full(_pack_phi(phi), grid_size).reshape(D, H, W)
        neg_mask = j <= threshold - err_tol
        if not neg_mask.any():
            _log(verbose, 1, f"[iter {iteration + 1}] No neg-Jdet voxels remain — exiting")
            converged = True
            break

        labeled, n_components = label(neg_mask, structure=structure)

        t0 = time.time()
        total_lam = 0
        total_mu = 0
        for comp_id in range(1, n_components + 1):
            coords = np.where(labeled == comp_id)
            if coords[0].size == 0:
                continue
            z0, z1, y0, y1, x0, x1 = _patch_bbox(coords, pad, (D, H, W))
            window_counts[(z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)] += 1

            _, lam_steps, mu_steps, patch_min = _optimize_patch(
                phi,
                phi_init,
                z0,
                z1,
                y0,
                y1,
                x0,
                x1,
                (D, H, W),
                threshold,
                margin,
                lam_schedule,
                mu_schedule,
                max_minimize_iter,
                verbose,
            )
            total_lam += lam_steps
            total_mu += mu_steps

            _log(
                verbose,
                2,
                f"  [comp {comp_id}/{n_components}] "
                f"patch={z1 - z0 + 1}x{y1 - y0 + 1}x{x1 - x0 + 1}  "
                f"lam={lam_steps} mu={mu_steps}  patch_min={patch_min:+.4f}",
            )

        elapsed = time.time() - t0
        iter_times.append(elapsed)

        phi_flat = _pack_phi(phi)
        j = jdet_full(phi_flat, grid_size)
        cur_neg = int((j <= 0).sum())
        cur_min = float(j.min())
        l2 = float(np.linalg.norm(phi_flat - phi_init_flat))
        num_neg_jac.append(cur_neg)
        min_jdet_list.append(cur_min)
        error_list.append(l2)
        _log(
            verbose,
            1,
            f"[iter {iteration + 1}] comps={n_components:4d}  "
            f"lam_steps={total_lam:3d} mu_steps={total_mu:3d}  "
            f"neg={cur_neg:5d}  min_J={cur_min:+.6f}  "
            f"L2={l2:.4f}  t={elapsed:.2f}s",
        )

        if cur_neg == 0 and cur_min >= threshold - err_tol:
            converged = True
            break

    if not converged:
        _log(
            verbose,
            1,
            f"[done] max_iterations={max_iterations} reached without "
            f"clearing all folds (neg={cur_neg}, min_J={cur_min:+.6f})",
        )


def _iterative_3d_barrier_fullgrid(
    phi,
    phi_init,
    phi_init_flat,
    grid_size,
    threshold,
    margin,
    lam_schedule,
    mu_schedule,
    max_minimize_iter,
    verbose,
    error_list,
    num_neg_jac,
    iter_times,
    min_jdet_list,
):
    """Original full-grid penalty->barrier loop. Mutates phi & accumulators."""
    phi_flat = _pack_phi(phi)
    phi_flat, info = run_penalty_barrier_lbfgs(
        phi_flat,
        phi_init_flat,
        constraint_values=lambda p: jdet_full(p, grid_size),
        constraint_adjoint=lambda p, v: _jdet_grad_T_v(p, grid_size, v),
        threshold=threshold,
        margin=margin,
        lam_schedule=lam_schedule,
        mu_schedule=mu_schedule,
        max_iter=max_minimize_iter,
        anchor='l2',
        verbose=verbose,
        record_history=True,
    )
    if not info['feasible']:
        T_end = jdet_full(phi_flat, grid_size)
        _log(
            verbose,
            1,
            f"[penalty] did not reach feasibility "
            f"(min_J={float(T_end.min()):+.6f} < {threshold + margin}); "
            "skipping barrier phase",
        )
    for step in info['history']:
        num_neg_jac.append(step['n_neg'])
        min_jdet_list.append(step['min_T'])
        iter_times.append(step['wall_s'])
        if 'l2' in step:
            l2 = float(step['l2'])
        elif 'phi_flat' in step:
            l2 = float(np.linalg.norm(step['phi_flat'] - phi_init_flat))
        else:
            raise RuntimeError(
                "run_penalty_barrier_lbfgs(record_history=True) did not record "
                "per-step L2 data or iterates; cannot populate full-grid 3D "
                "barrier error_list without silently writing incorrect zeros."
            )
        error_list.append(l2)
    _unpack_phi(phi_flat, grid_size, out=phi)


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
    max_iterations=50,
    windowed=True,
    pad=2,
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
    max_iterations : int
        In windowed mode, max outer sweeps over connected components.
        Ignored in full-grid mode.
    windowed : bool
        When True (default), optimise each connected component of
        negative-Jdet voxels on its own padded patch with a frozen
        boundary ring. When False, optimise the full grid at once.
    pad : int
        Number of voxels to expand each component bbox on every side.
        pad=2 provides 1 voxel of optimisation room around the negative
        region plus 1 voxel for the frozen boundary ring.

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
    mode = "windowed" if windowed else "full-grid"
    _log(
        verbose, 1, f"[init] Grid {D}x{H}x{W}  threshold={threshold}  margin={margin}  mode={mode}"
    )
    _log(verbose, 1, f"[init] Neg-Jdet voxels: {init_neg}  min Jdet: {init_min:.6f}")

    phi_init_flat = _pack_phi(phi_init)

    if windowed:
        _iterative_3d_barrier_windowed(
            phi,
            phi_init,
            phi_init_flat,
            grid_size,
            threshold,
            err_tol,
            margin,
            lam_schedule,
            mu_schedule,
            max_minimize_iter,
            max_iterations,
            pad,
            verbose,
            error_list,
            num_neg_jac,
            iter_times,
            min_jdet_list,
            window_counts,
        )
    else:
        _iterative_3d_barrier_fullgrid(
            phi,
            phi_init,
            phi_init_flat,
            grid_size,
            threshold,
            margin,
            lam_schedule,
            mu_schedule,
            max_minimize_iter,
            verbose,
            error_list,
            num_neg_jac,
            iter_times,
            min_jdet_list,
        )

    elapsed_total = time.time() - start_time

    phi_final_flat = _pack_phi(phi)
    j_final = jdet_full(phi_final_flat, grid_size)
    final_neg = int((j_final <= 0).sum())
    final_min = float(j_final.min())
    final_err = float(np.linalg.norm(phi_final_flat - phi_init_flat))

    _print_summary(
        verbose,
        f"Penalty->Barrier L-BFGS-B - 3D ({mode})",
        (D, H, W),
        len(iter_times),
        init_neg,
        final_neg,
        init_min,
        final_min,
        final_err,
        elapsed_total,
    )

    if save_path is not None:
        _save_results(
            save_path,
            method=f"penalty_barrier_lbfgsb_{mode}",
            threshold=threshold,
            err_tol=err_tol,
            max_iterations=len(iter_times),
            max_per_index_iter=0,
            max_minimize_iter=max_minimize_iter,
            grid_shape=(D, H, W),
            elapsed=elapsed_total,
            final_err=final_err,
            init_neg=init_neg,
            final_neg=final_neg,
            init_min=init_min,
            final_min=final_min,
            iteration=len(iter_times),
            phi=phi,
            error_list=error_list,
            num_neg_jac=num_neg_jac,
            iter_times=iter_times,
            min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi
