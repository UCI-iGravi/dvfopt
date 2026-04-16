"""GPU-accelerated penalty -> log-barrier 3D solver via PyTorch autograd.

Same two-phase scheme as ``iterative_3d_barrier`` (numpy/scipy), but the
forward J(u), penalty/barrier objectives, and gradients all run on a
single (3, D, H, W) torch tensor with autograd. Optimisation uses
``torch.optim.LBFGS`` so iterates stay on-device.

Public entry: ``iterative_3d_barrier_torch(deformation, ...)``.
"""

import time

import numpy as np
import torch

from dvfopt._defaults import _log, _resolve_params
from dvfopt.core.solver import _setup_accumulators, _print_summary, _save_results


def _jdet_3d_torch(phi):
    """3D Jacobian determinant of phi shaped (3, D, H, W) on torch.

    Channel order: phi[0]=dz, phi[1]=dy, phi[2]=dx (matches numpy convention).
    Uses central differences (one-sided at endpoints) - matches np.gradient.
    """
    dz = phi[0]
    dy = phi[1]
    dx = phi[2]

    ddx_dx = torch.gradient(dx, dim=2)[0]
    ddx_dy = torch.gradient(dx, dim=1)[0]
    ddx_dz = torch.gradient(dx, dim=0)[0]
    ddy_dx = torch.gradient(dy, dim=2)[0]
    ddy_dy = torch.gradient(dy, dim=1)[0]
    ddy_dz = torch.gradient(dy, dim=0)[0]
    ddz_dx = torch.gradient(dz, dim=2)[0]
    ddz_dy = torch.gradient(dz, dim=1)[0]
    ddz_dz = torch.gradient(dz, dim=0)[0]

    a11 = 1.0 + ddx_dx;  a12 = ddx_dy;       a13 = ddx_dz
    a21 = ddy_dx;         a22 = 1.0 + ddy_dy;  a23 = ddy_dz
    a31 = ddz_dx;         a32 = ddz_dy;        a33 = 1.0 + ddz_dz

    return (a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31))


def _run_lbfgs(phi_var, closure, max_iter, tolerance_grad=1e-6,
               tolerance_change=1e-9, history_size=20, lr=1.0):
    """Run torch L-BFGS on ``phi_var`` with the given closure."""
    opt = torch.optim.LBFGS(
        [phi_var],
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    final_loss = opt.step(closure)
    return float(final_loss.detach())


def iterative_3d_barrier_torch(
    deformation,
    verbose=1,
    save_path=None,
    threshold=None,
    err_tol=None,
    margin=1e-3,
    lam_schedule=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
    mu_schedule=(1e-1, 1e-2, 1e-3, 1e-4),
    max_minimize_iter=200,
    device=None,
    dtype=torch.float64,
):
    """Penalty -> log-barrier 3D corrector on GPU/CPU via torch autograd.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``
        Channels ``[dz, dy, dx]``.
    threshold, err_tol : float or None
        Override default Jdet bounds.
    margin : float
        Phase-1 target slack.
    lam_schedule, mu_schedule : sequences
        Continuation parameters.
    max_minimize_iter : int
        L-BFGS ``max_iter`` per continuation step.
    device : str or torch.device or None
        ``"cuda"``/``"cpu"``. Defaults to ``"cuda"`` if available.
    dtype : torch.dtype
        Default ``float64`` to match the numpy solver. Use ``float32`` for
        more GPU throughput at the cost of barrier conditioning.

    Returns
    -------
    phi : ndarray, shape ``(3, D, H, W)`` on host.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    p = _resolve_params(threshold=threshold, err_tol=err_tol)
    threshold_f = float(p["threshold"])
    err_tol_f = float(p["err_tol"])

    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

    error_list, num_neg_jac, iter_times, min_jdet_list, window_counts = _setup_accumulators()

    start_time = time.time()
    deformation = np.asarray(deformation)
    _, D, H, W = deformation.shape
    phi_init = torch.tensor(deformation, dtype=dtype, device=device)
    phi_var = phi_init.detach().clone().requires_grad_(True)

    with torch.no_grad():
        j0 = _jdet_3d_torch(phi_init)
        init_neg = int((j0 <= 0).sum().item())
        init_min = float(j0.min().item())
    num_neg_jac.append(init_neg)
    min_jdet_list.append(init_min)
    _log(verbose, 1,
         f"[init] Grid {D}x{H}x{W}  threshold={threshold_f}  margin={margin}  "
         f"device={device}  dtype={dtype}")
    _log(verbose, 1, f"[init] Neg-Jdet voxels: {init_neg}  min Jdet: {init_min:.6f}")

    target = threshold_f + margin

    # ---------------- Phase 1: penalty ----------------
    feasible = init_min >= target
    cur_min = init_min
    for k, lam in enumerate(lam_schedule):
        if feasible:
            break

        def closure():
            if phi_var.grad is not None:
                phi_var.grad.zero_()
            diff = phi_var - phi_init
            data = 0.5 * (diff * diff).sum()
            j = _jdet_3d_torch(phi_var)
            viol = torch.clamp(target - j, min=0.0)
            pen = lam * (viol * viol).sum()
            loss = data + pen
            loss.backward()
            return loss

        opt = torch.optim.LBFGS(
            [phi_var],
            lr=1.0,
            max_iter=max_minimize_iter,
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
        t0 = time.time()
        opt.step(closure)
        elapsed = time.time() - t0
        iter_times.append(elapsed)

        with torch.no_grad():
            j = _jdet_3d_torch(phi_var)
            cur_neg = int((j <= 0).sum().item())
            cur_min = float(j.min().item())
            l2 = float(torch.linalg.norm(phi_var - phi_init).item())
        num_neg_jac.append(cur_neg)
        min_jdet_list.append(cur_min)
        error_list.append(l2)
        _log(verbose, 1,
             f"[penalty {k+1}/{len(lam_schedule)}] lam={lam:g}  "
             f"neg={cur_neg:5d}  min_J={cur_min:+.6f}  "
             f"L2={l2:.4f}  t={elapsed:.2f}s")
        if cur_min >= target:
            feasible = True
            break

    if not feasible:
        _log(verbose, 1,
             f"[penalty] did not reach feasibility (min_J={cur_min:+.6f} < {target}); "
             "skipping barrier phase")

    # ---------------- Phase 2: barrier ----------------
    if feasible:
        for k, mu in enumerate(mu_schedule):
            def closure():
                if phi_var.grad is not None:
                    phi_var.grad.zero_()
                diff = phi_var - phi_init
                data = 0.5 * (diff * diff).sum()
                j = _jdet_3d_torch(phi_var)
                slack = j - threshold_f
                # Guard infeasibility: penalize violation as quadratic so the
                # line search rejects the step rather than producing NaN.
                if (slack <= 0).any():
                    viol = torch.clamp(-slack + 1e-12, min=0.0)
                    bar = 1e8 * (viol * viol).sum()
                else:
                    bar = -mu * torch.log(slack).sum()
                loss = data + bar
                loss.backward()
                return loss

            opt = torch.optim.LBFGS(
                [phi_var],
                lr=1.0,
                max_iter=max_minimize_iter,
                tolerance_grad=1e-6,
                tolerance_change=1e-9,
                history_size=20,
                line_search_fn="strong_wolfe",
            )
            t0 = time.time()
            opt.step(closure)
            elapsed = time.time() - t0
            iter_times.append(elapsed)

            with torch.no_grad():
                j = _jdet_3d_torch(phi_var)
                cur_neg = int((j <= 0).sum().item())
                cur_min = float(j.min().item())
                l2 = float(torch.linalg.norm(phi_var - phi_init).item())
            num_neg_jac.append(cur_neg)
            min_jdet_list.append(cur_min)
            error_list.append(l2)
            _log(verbose, 1,
                 f"[barrier {k+1}/{len(mu_schedule)}] mu={mu:g}  "
                 f"neg={cur_neg:5d}  min_J={cur_min:+.6f}  "
                 f"L2={l2:.4f}  t={elapsed:.2f}s")

    elapsed_total = time.time() - start_time
    with torch.no_grad():
        j_final = _jdet_3d_torch(phi_var)
        final_neg = int((j_final <= 0).sum().item())
        final_min = float(j_final.min().item())
        final_err = float(torch.linalg.norm(phi_var - phi_init).item())

    phi_out = phi_var.detach().cpu().numpy()

    _print_summary(verbose, "Penalty->Barrier L-BFGS torch - 3D",
                   (D, H, W), len(iter_times),
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed_total)

    if save_path is not None:
        _save_results(
            save_path, method="penalty_barrier_lbfgs_torch",
            threshold=threshold_f, err_tol=err_tol_f,
            max_iterations=len(iter_times),
            max_per_index_iter=0, max_minimize_iter=max_minimize_iter,
            grid_shape=(D, H, W), elapsed=elapsed_total,
            final_err=final_err, init_neg=init_neg, final_neg=final_neg,
            init_min=init_min, final_min=final_min,
            iteration=len(iter_times), phi=phi_out,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi_out
