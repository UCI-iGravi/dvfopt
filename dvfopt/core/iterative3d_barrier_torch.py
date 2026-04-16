"""GPU-accelerated penalty -> log-barrier 3D solver via PyTorch autograd.

Same two-phase scheme as ``iterative_3d_barrier`` (numpy/scipy), but the
forward J(u), penalty/barrier objectives, and gradients all run on a
single (3, D, H, W) torch tensor with autograd. Optimisation uses
``torch.optim.LBFGS`` so iterates stay on-device.

Supports **windowed** mode (default): each connected component of
negative-Jdet voxels is optimised on its own padded patch with a frozen
boundary ring (via ``torch.where`` masking). Dramatically reduces memory
(L-BFGS history vectors scale with patch size) and compute.

Public entry: ``iterative_3d_barrier_torch(deformation, ...)``.
"""

import time

import numpy as np
import torch
from scipy.ndimage import label

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


def _patch_bbox_3d(comp_coords, pad, grid_shape):
    zs, ys, xs = comp_coords
    D, H, W = grid_shape
    z0 = max(int(zs.min()) - pad, 0)
    z1 = min(int(zs.max()) + pad, D - 1)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, H - 1)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, W - 1)
    return z0, z1, y0, y1, x0, x1


def _frozen_mask_3d_torch(z0, z1, y0, y1, x0, x1, grid_shape, device):
    D, H, W = grid_shape
    Dp, Hp, Wp = z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1
    mask = torch.zeros((Dp, Hp, Wp), dtype=torch.bool, device=device)
    if z0 > 0:     mask[0, :, :] = True
    if z1 < D - 1: mask[-1, :, :] = True
    if y0 > 0:     mask[:, 0, :] = True
    if y1 < H - 1: mask[:, -1, :] = True
    if x0 > 0:     mask[:, :, 0] = True
    if x1 < W - 1: mask[:, :, -1] = True
    return mask.unsqueeze(0)  # (1, Dp, Hp, Wp) broadcast to (3, ...)


def _optimize_patch_3d_torch(phi_full, z0, z1, y0, y1, x0, x1, grid_shape,
                             threshold_f, margin, lam_schedule, mu_schedule,
                             max_minimize_iter, device, dtype):
    """Run penalty -> barrier on a 3D patch. Mutates phi_full in place."""
    phi_patch_init = phi_full[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].detach().clone()
    phi_patch_var = phi_patch_init.detach().clone().requires_grad_(True)
    frozen_b = _frozen_mask_3d_torch(z0, z1, y0, y1, x0, x1, grid_shape, device)

    target = threshold_f + margin
    with torch.no_grad():
        j0 = _jdet_3d_torch(phi_patch_init)
        feasible = bool((j0.min() >= target).item())

    lam_steps = 0
    mu_steps = 0

    for lam in lam_schedule:
        if feasible:
            break

        def closure():
            if phi_patch_var.grad is not None:
                phi_patch_var.grad.zero_()
            phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
            diff = phi_eff - phi_patch_init
            data = 0.5 * (diff * diff).sum()
            j = _jdet_3d_torch(phi_eff)
            viol = torch.clamp(target - j, min=0.0)
            pen = lam * (viol * viol).sum()
            loss = data + pen
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([phi_patch_var], lr=1.0, max_iter=max_minimize_iter,
                                 tolerance_grad=1e-6, tolerance_change=1e-9,
                                 history_size=20, line_search_fn="strong_wolfe")
        opt.step(closure)
        lam_steps += 1
        with torch.no_grad():
            phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
            j = _jdet_3d_torch(phi_eff)
            if float(j.min().item()) >= target:
                feasible = True
                break

    if feasible:
        for mu in mu_schedule:
            def closure():
                if phi_patch_var.grad is not None:
                    phi_patch_var.grad.zero_()
                phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
                diff = phi_eff - phi_patch_init
                data = 0.5 * (diff * diff).sum()
                j = _jdet_3d_torch(phi_eff)
                slack = j - threshold_f
                if (slack <= 0).any():
                    viol = torch.clamp(-slack + 1e-12, min=0.0)
                    bar = 1e8 * (viol * viol).sum()
                else:
                    bar = -mu * torch.log(slack).sum()
                loss = data + bar
                loss.backward()
                return loss

            opt = torch.optim.LBFGS([phi_patch_var], lr=1.0, max_iter=max_minimize_iter,
                                     tolerance_grad=1e-6, tolerance_change=1e-9,
                                     history_size=20, line_search_fn="strong_wolfe")
            opt.step(closure)
            mu_steps += 1

    with torch.no_grad():
        phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
        phi_full[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = phi_eff
    return lam_steps, mu_steps


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
    max_iterations=50,
    windowed=True,
    pad=2,
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
    max_iterations : int
        Max outer sweeps in windowed mode.
    windowed : bool
        When True (default), optimise each connected component of
        negative-Jdet voxels on its own padded patch with a frozen
        boundary ring. When False, optimise the full grid at once.
    pad : int
        Voxels of expansion on each side of each component bbox.
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

    with torch.no_grad():
        j0 = _jdet_3d_torch(phi_init)
        init_neg = int((j0 <= 0).sum().item())
        init_min = float(j0.min().item())
    num_neg_jac.append(init_neg)
    min_jdet_list.append(init_min)
    mode = "windowed" if windowed else "full-grid"
    _log(verbose, 1,
         f"[init] Grid {D}x{H}x{W}  threshold={threshold_f}  margin={margin}  "
         f"device={device}  dtype={dtype}  mode={mode}")
    _log(verbose, 1, f"[init] Neg-Jdet voxels: {init_neg}  min Jdet: {init_min:.6f}")

    target = threshold_f + margin

    if windowed:
        phi_full = phi_init.detach().clone()
        structure = np.ones((3, 3, 3))

        for iteration in range(max_iterations):
            with torch.no_grad():
                j = _jdet_3d_torch(phi_full)
                j_np = j.cpu().numpy()
            neg_mask = j_np <= threshold_f - err_tol_f
            if not neg_mask.any():
                _log(verbose, 1, f"[iter {iteration+1}] No neg-Jdet remain — exiting")
                break
            labeled, n_components = label(neg_mask, structure=structure)

            t0 = time.time()
            total_lam = 0
            total_mu = 0
            for comp_id in range(1, n_components + 1):
                coords = np.where(labeled == comp_id)
                if coords[0].size == 0:
                    continue
                z0, z1, y0, y1, x0, x1 = _patch_bbox_3d(coords, pad, (D, H, W))
                window_counts[(z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)] += 1
                lam_steps, mu_steps = _optimize_patch_3d_torch(
                    phi_full, z0, z1, y0, y1, x0, x1, (D, H, W),
                    threshold_f, margin, lam_schedule, mu_schedule,
                    max_minimize_iter, device, dtype)
                total_lam += lam_steps
                total_mu += mu_steps

            elapsed = time.time() - t0
            iter_times.append(elapsed)

            with torch.no_grad():
                j = _jdet_3d_torch(phi_full)
                cur_neg = int((j <= 0).sum().item())
                cur_min = float(j.min().item())
                l2 = float(torch.linalg.norm(phi_full - phi_init).item())
            num_neg_jac.append(cur_neg); min_jdet_list.append(cur_min); error_list.append(l2)
            _log(verbose, 1,
                 f"[iter {iteration+1}] comps={n_components:4d}  "
                 f"lam_steps={total_lam:3d} mu_steps={total_mu:3d}  "
                 f"neg={cur_neg:5d}  min_J={cur_min:+.6f}  "
                 f"L2={l2:.4f}  t={elapsed:.2f}s")
            if cur_neg == 0 and cur_min >= threshold_f - err_tol_f:
                break

        phi_var_final = phi_full
    else:
        phi_var = phi_init.detach().clone().requires_grad_(True)
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

        if feasible:
            for k, mu in enumerate(mu_schedule):
                def closure():
                    if phi_var.grad is not None:
                        phi_var.grad.zero_()
                    diff = phi_var - phi_init
                    data = 0.5 * (diff * diff).sum()
                    j = _jdet_3d_torch(phi_var)
                    slack = j - threshold_f
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

        phi_var_final = phi_var.detach()

    elapsed_total = time.time() - start_time
    with torch.no_grad():
        j_final = _jdet_3d_torch(phi_var_final)
        final_neg = int((j_final <= 0).sum().item())
        final_min = float(j_final.min().item())
        final_err = float(torch.linalg.norm(phi_var_final - phi_init).item())

    phi_out = phi_var_final.detach().cpu().numpy()

    _print_summary(verbose, f"Penalty->Barrier L-BFGS torch - 3D ({mode})",
                   (D, H, W), len(iter_times),
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed_total)

    if save_path is not None:
        _save_results(
            save_path, method=f"penalty_barrier_lbfgs_torch_{mode}",
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
