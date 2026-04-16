"""2D penalty -> log-barrier solvers (numpy/scipy CPU + torch GPU).

Mirrors the 3D versions but for ``(2, H, W)`` displacement fields where
``phi[0]=dy``, ``phi[1]=dx``. Accepts the standard 2D input shape
``(3, 1, H, W)`` (channels [dz, dy, dx] with dz ignored) for parity with
``iterative_serial``.
"""

import time

import numpy as np
import torch
from scipy.optimize import minimize

from dvfopt._defaults import _log, _resolve_params
from dvfopt.core.solver import _setup_accumulators, _print_summary
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d


# ---------- shared helpers ----------
def _coerce_2d(deformation):
    """Accept ``(3,1,H,W)`` or ``(2,H,W)``; return phi_init (2,H,W) and (H,W)."""
    arr = np.asarray(deformation, dtype=np.float64)
    if arr.ndim == 4:
        H, W = arr.shape[-2:]
        phi = np.empty((2, H, W), dtype=np.float64)
        phi[0] = arr[1, 0]   # dy
        phi[1] = arr[2, 0]   # dx
    elif arr.ndim == 3 and arr.shape[0] == 2:
        H, W = arr.shape[-2:]
        phi = arr.copy()
    else:
        raise ValueError(f"unexpected deformation shape {arr.shape}")
    return phi, H, W


# ============================================================
# Numpy / scipy backend
# ============================================================
def _adjoint_central_diff(w, axis):
    n = w.shape[axis]
    if n == 1:
        return np.zeros_like(w)
    w_m = np.moveaxis(w, axis, 0)
    out_m = np.zeros_like(w_m)
    c_next = np.full(n, 0.5);  c_next[0] = 1.0
    c_prev = np.full(n, 0.5);  c_prev[n - 1] = 1.0
    bshape = [1] * w_m.ndim;  bshape[0] = n - 1
    out_m[1:n] += c_next[:n - 1].reshape(bshape) * w_m[:n - 1]
    out_m[0:n - 1] -= c_prev[1:n].reshape(bshape) * w_m[1:n]
    out_m[0] -= w_m[0]
    out_m[n - 1] += w_m[n - 1]
    return np.moveaxis(out_m, 0, axis)


def _split_phi_2d(phi_flat, grid_size):
    H, W = grid_size
    n = H * W
    dx = phi_flat[:n].reshape(H, W)
    dy = phi_flat[n:].reshape(H, W)
    return dx, dy, n


def _jdet_2d_flat(phi_flat, grid_size):
    dx, dy, _ = _split_phi_2d(phi_flat, grid_size)
    return _numpy_jdet_2d(dy, dx).flatten()


def _jdet_grad_T_v_2d(phi_flat, grid_size, v):
    H, W = grid_size
    n = H * W
    dx, dy, _ = _split_phi_2d(phi_flat, grid_size)
    v2 = v.reshape(H, W)

    ddx_dx = np.gradient(dx, axis=1)
    ddy_dy = np.gradient(dy, axis=0)
    ddx_dy = np.gradient(dx, axis=0)
    ddy_dx = np.gradient(dy, axis=1)
    a11 = 1 + ddx_dx;  a22 = 1 + ddy_dy
    # J = a11*a22 - ddx_dy*ddy_dx
    # Cofactors: dJ/da11 = a22; dJ/da22 = a11; dJ/d(ddx_dy) = -ddy_dx; dJ/d(ddy_dx) = -ddx_dy
    # dx column: contributes via a11 (∂/∂x) and ddx_dy (∂/∂y).
    g_dx = (_adjoint_central_diff(a22 * v2, axis=1)
            + _adjoint_central_diff(-ddy_dx * v2, axis=0))
    g_dy = (_adjoint_central_diff(a11 * v2, axis=0)
            + _adjoint_central_diff(-ddx_dy * v2, axis=1))
    out = np.empty(2 * n)
    out[:n] = g_dx.ravel()
    out[n:] = g_dy.ravel()
    return out


def _penalty_2d(phi_flat, phi_init_flat, grid_size, threshold, margin, lam):
    diff = phi_flat - phi_init_flat
    data = 0.5 * float(np.dot(diff, diff))
    j = _jdet_2d_flat(phi_flat, grid_size)
    target = threshold + margin
    viol = np.maximum(0.0, target - j)
    pen = lam * float(np.dot(viol, viol))
    grad = diff.copy()
    if np.any(viol > 0):
        grad += _jdet_grad_T_v_2d(phi_flat, grid_size, -2.0 * lam * viol)
    return data + pen, grad


def _barrier_2d(phi_flat, phi_init_flat, grid_size, threshold, mu):
    diff = phi_flat - phi_init_flat
    data = 0.5 * float(np.dot(diff, diff))
    j = _jdet_2d_flat(phi_flat, grid_size)
    slack = j - threshold
    if np.any(slack <= 0.0):
        return np.inf, np.zeros_like(phi_flat)
    bar = -mu * float(np.log(slack).sum())
    grad = diff + _jdet_grad_T_v_2d(phi_flat, grid_size, -mu / slack)
    return data + bar, grad


def iterative_2d_barrier(
    deformation,
    verbose=1,
    threshold=None,
    err_tol=None,
    margin=1e-3,
    lam_schedule=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
    mu_schedule=(1e-1, 1e-2, 1e-3, 1e-4),
    max_minimize_iter=200,
):
    """CPU penalty->barrier solver for 2D fields."""
    p = _resolve_params(threshold=threshold, err_tol=err_tol)
    threshold = p["threshold"]
    if verbose is True: verbose = 1
    elif verbose is False: verbose = 0

    error_list, num_neg_jac, iter_times, min_jdet_list, _ = _setup_accumulators()

    phi_init_3d, H, W = _coerce_2d(deformation)
    grid_size = (H, W)
    phi_init_flat = np.concatenate([phi_init_3d[1].ravel(), phi_init_3d[0].ravel()])
    phi_flat = phi_init_flat.copy()

    j0 = _jdet_2d_flat(phi_flat, grid_size)
    init_neg = int((j0 <= 0).sum());  init_min = float(j0.min())
    num_neg_jac.append(init_neg);  min_jdet_list.append(init_min)
    _log(verbose, 1, f"[init] Grid {H}x{W}  threshold={threshold}  margin={margin}")
    _log(verbose, 1, f"[init] Neg-Jdet: {init_neg}  min Jdet: {init_min:.6f}")

    target = threshold + margin
    start = time.time()
    feasible = init_min >= target
    cur_min = init_min
    for k, lam in enumerate(lam_schedule):
        if feasible: break
        t0 = time.time()
        res = minimize(_penalty_2d, phi_flat,
                       args=(phi_init_flat, grid_size, threshold, margin, lam),
                       jac=True, method="L-BFGS-B",
                       options={"maxiter": max_minimize_iter, "gtol": 1e-6})
        elapsed = time.time() - t0
        iter_times.append(elapsed)
        phi_flat = res.x
        j = _jdet_2d_flat(phi_flat, grid_size)
        cur_neg = int((j <= 0).sum()); cur_min = float(j.min())
        l2 = float(np.linalg.norm(phi_flat - phi_init_flat))
        num_neg_jac.append(cur_neg); min_jdet_list.append(cur_min); error_list.append(l2)
        _log(verbose, 1, f"[penalty {k+1}] lam={lam:g}  neg={cur_neg}  min_J={cur_min:+.6f}  L2={l2:.4f}  t={elapsed:.2f}s")
        if cur_min >= target: feasible = True; break

    if feasible:
        for k, mu in enumerate(mu_schedule):
            t0 = time.time()
            res = minimize(_barrier_2d, phi_flat,
                           args=(phi_init_flat, grid_size, threshold, mu),
                           jac=True, method="L-BFGS-B",
                           options={"maxiter": max_minimize_iter, "gtol": 1e-6})
            elapsed = time.time() - t0
            iter_times.append(elapsed)
            phi_flat = res.x
            j = _jdet_2d_flat(phi_flat, grid_size)
            cur_neg = int((j <= 0).sum()); cur_min = float(j.min())
            l2 = float(np.linalg.norm(phi_flat - phi_init_flat))
            num_neg_jac.append(cur_neg); min_jdet_list.append(cur_min); error_list.append(l2)
            _log(verbose, 1, f"[barrier {k+1}] mu={mu:g}  neg={cur_neg}  min_J={cur_min:+.6f}  L2={l2:.4f}  t={elapsed:.2f}s")

    elapsed = time.time() - start
    n = H * W
    phi_out = np.empty((2, H, W))
    phi_out[1] = phi_flat[:n].reshape(H, W)   # dx
    phi_out[0] = phi_flat[n:].reshape(H, W)   # dy
    final_neg = int((_jdet_2d_flat(phi_flat, grid_size) <= 0).sum())
    final_min = float(_jdet_2d_flat(phi_flat, grid_size).min())
    final_err = float(np.linalg.norm(phi_flat - phi_init_flat))
    _print_summary(verbose, "Penalty->Barrier L-BFGS-B - 2D", (H, W),
                   len(iter_times), init_neg, final_neg, init_min, final_min,
                   final_err, elapsed)
    return phi_out


# ============================================================
# Torch backend
# ============================================================
def _jdet_2d_torch(phi):
    """phi shape (2, H, W) with phi[0]=dy, phi[1]=dx."""
    dy = phi[0];  dx = phi[1]
    ddx_dx = torch.gradient(dx, dim=1)[0]
    ddy_dy = torch.gradient(dy, dim=0)[0]
    ddx_dy = torch.gradient(dx, dim=0)[0]
    ddy_dx = torch.gradient(dy, dim=1)[0]
    return (1.0 + ddx_dx) * (1.0 + ddy_dy) - ddx_dy * ddy_dx


def iterative_2d_barrier_torch(
    deformation,
    verbose=1,
    threshold=None,
    err_tol=None,
    margin=1e-3,
    lam_schedule=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
    mu_schedule=(1e-1, 1e-2, 1e-3, 1e-4),
    max_minimize_iter=200,
    device=None,
    dtype=torch.float64,
):
    """GPU/CPU penalty->barrier solver for 2D fields via torch autograd."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    p = _resolve_params(threshold=threshold, err_tol=err_tol)
    threshold_f = float(p["threshold"])
    if verbose is True: verbose = 1
    elif verbose is False: verbose = 0

    error_list, num_neg_jac, iter_times, min_jdet_list, _ = _setup_accumulators()

    phi_init_np, H, W = _coerce_2d(deformation)
    phi_init = torch.tensor(phi_init_np, dtype=dtype, device=device)
    phi_var = phi_init.detach().clone().requires_grad_(True)

    with torch.no_grad():
        j0 = _jdet_2d_torch(phi_init)
        init_neg = int((j0 <= 0).sum().item())
        init_min = float(j0.min().item())
    num_neg_jac.append(init_neg);  min_jdet_list.append(init_min)
    _log(verbose, 1, f"[init] Grid {H}x{W}  threshold={threshold_f}  device={device}")
    _log(verbose, 1, f"[init] Neg-Jdet: {init_neg}  min Jdet: {init_min:.6f}")

    target = threshold_f + margin
    start = time.time()
    feasible = init_min >= target
    cur_min = init_min
    for k, lam in enumerate(lam_schedule):
        if feasible: break

        def closure():
            if phi_var.grad is not None: phi_var.grad.zero_()
            diff = phi_var - phi_init
            data = 0.5 * (diff * diff).sum()
            j = _jdet_2d_torch(phi_var)
            viol = torch.clamp(target - j, min=0.0)
            pen = lam * (viol * viol).sum()
            loss = data + pen
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([phi_var], lr=1.0, max_iter=max_minimize_iter,
                                 tolerance_grad=1e-6, tolerance_change=1e-9,
                                 history_size=20, line_search_fn="strong_wolfe")
        t0 = time.time()
        opt.step(closure)
        elapsed = time.time() - t0
        iter_times.append(elapsed)
        with torch.no_grad():
            j = _jdet_2d_torch(phi_var)
            cur_neg = int((j <= 0).sum().item());  cur_min = float(j.min().item())
            l2 = float(torch.linalg.norm(phi_var - phi_init).item())
        num_neg_jac.append(cur_neg); min_jdet_list.append(cur_min); error_list.append(l2)
        _log(verbose, 1, f"[penalty {k+1}] lam={lam:g}  neg={cur_neg}  min_J={cur_min:+.6f}  L2={l2:.4f}  t={elapsed:.2f}s")
        if cur_min >= target: feasible = True; break

    if feasible:
        for k, mu in enumerate(mu_schedule):
            def closure():
                if phi_var.grad is not None: phi_var.grad.zero_()
                diff = phi_var - phi_init
                data = 0.5 * (diff * diff).sum()
                j = _jdet_2d_torch(phi_var)
                slack = j - threshold_f
                if (slack <= 0).any():
                    viol = torch.clamp(-slack + 1e-12, min=0.0)
                    bar = 1e8 * (viol * viol).sum()
                else:
                    bar = -mu * torch.log(slack).sum()
                loss = data + bar
                loss.backward()
                return loss

            opt = torch.optim.LBFGS([phi_var], lr=1.0, max_iter=max_minimize_iter,
                                     tolerance_grad=1e-6, tolerance_change=1e-9,
                                     history_size=20, line_search_fn="strong_wolfe")
            t0 = time.time()
            opt.step(closure)
            elapsed = time.time() - t0
            iter_times.append(elapsed)
            with torch.no_grad():
                j = _jdet_2d_torch(phi_var)
                cur_neg = int((j <= 0).sum().item());  cur_min = float(j.min().item())
                l2 = float(torch.linalg.norm(phi_var - phi_init).item())
            num_neg_jac.append(cur_neg); min_jdet_list.append(cur_min); error_list.append(l2)
            _log(verbose, 1, f"[barrier {k+1}] mu={mu:g}  neg={cur_neg}  min_J={cur_min:+.6f}  L2={l2:.4f}  t={elapsed:.2f}s")

    elapsed = time.time() - start
    with torch.no_grad():
        j_final = _jdet_2d_torch(phi_var)
        final_neg = int((j_final <= 0).sum().item())
        final_min = float(j_final.min().item())
        final_err = float(torch.linalg.norm(phi_var - phi_init).item())
    phi_out = phi_var.detach().cpu().numpy()
    _print_summary(verbose, "Penalty->Barrier L-BFGS torch - 2D", (H, W),
                   len(iter_times), init_neg, final_neg, init_min, final_min,
                   final_err, elapsed)
    return phi_out
