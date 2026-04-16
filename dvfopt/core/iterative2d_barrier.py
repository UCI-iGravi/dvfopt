"""2D penalty -> log-barrier solvers (numpy/scipy CPU + torch GPU).

Mirrors the 3D versions but for ``(2, H, W)`` displacement fields where
``phi[0]=dy``, ``phi[1]=dx``. Accepts the standard 2D input shape
``(3, 1, H, W)`` (channels [dz, dy, dx] with dz ignored) for parity with
``iterative_serial``.
"""

import time

import numpy as np
import torch
from scipy.ndimage import label
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


def _pack_phi_2d(phi2):
    """phi (2,H,W) [dy,dx] -> flat [dx, dy]."""
    return np.concatenate([phi2[1].ravel(), phi2[0].ravel()])


def _unpack_phi_2d(phi_flat, grid_size):
    H, W = grid_size
    n = H * W
    phi2 = np.empty((2, H, W), dtype=phi_flat.dtype)
    phi2[1] = phi_flat[:n].reshape(H, W)   # dx
    phi2[0] = phi_flat[n:].reshape(H, W)   # dy
    return phi2


def _patch_bbox_2d(comp_coords, pad, grid_shape):
    ys, xs = comp_coords
    H, W = grid_shape
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, H - 1)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, W - 1)
    return y0, y1, x0, x1


def _patch_frozen_mask_2d(y0, y1, x0, x1, grid_shape):
    H, W = grid_shape
    Hp, Wp = y1 - y0 + 1, x1 - x0 + 1
    mask = np.zeros((Hp, Wp), dtype=bool)
    if y0 > 0:     mask[0, :] = True
    if y1 < H - 1: mask[-1, :] = True
    if x0 > 0:     mask[:, 0] = True
    if x1 < W - 1: mask[:, -1] = True
    return mask


def _patch_bounds_2d(phi_flat_patch, frozen_mask):
    Hp, Wp = frozen_mask.shape
    n = Hp * Wp
    frozen_flat = frozen_mask.ravel()
    bounds = [(None, None)] * (2 * n)
    for ch in range(2):
        base = ch * n
        for idx in np.nonzero(frozen_flat)[0]:
            v = phi_flat_patch[base + idx]
            bounds[base + idx] = (v, v)
    return bounds


def _optimize_patch_2d(phi, y0, y1, x0, x1, grid_shape,
                      threshold, margin, lam_schedule, mu_schedule,
                      max_minimize_iter, verbose):
    """Run penalty->barrier on a 2D patch. Mutates phi (2,H,W) in place."""
    phi_patch = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
    Hp, Wp = phi_patch.shape[1:]
    patch_size = (Hp, Wp)

    phi_flat = _pack_phi_2d(phi_patch)
    phi_anchor = phi_flat.copy()

    frozen_mask = _patch_frozen_mask_2d(y0, y1, x0, x1, grid_shape)
    bounds = _patch_bounds_2d(phi_flat, frozen_mask)

    target = threshold + margin
    j0 = _jdet_2d_flat(phi_flat, patch_size)
    feasible = float(j0.min()) >= target

    lam_steps = 0
    mu_steps = 0

    for lam in lam_schedule:
        if feasible:
            break
        res = minimize(_penalty_2d, phi_flat,
                       args=(phi_anchor, patch_size, threshold, margin, lam),
                       jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_minimize_iter, "gtol": 1e-6,
                                "disp": verbose >= 3})
        phi_flat = res.x
        lam_steps += 1
        j = _jdet_2d_flat(phi_flat, patch_size)
        if float(j.min()) >= target:
            feasible = True
            break

    if feasible:
        for mu in mu_schedule:
            res = minimize(_barrier_2d, phi_flat,
                           args=(phi_anchor, patch_size, threshold, mu),
                           jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": max_minimize_iter, "gtol": 1e-6,
                                    "disp": verbose >= 3})
            phi_flat = res.x
            mu_steps += 1

    phi_patch = _unpack_phi_2d(phi_flat, patch_size)
    phi[:, y0:y1 + 1, x0:x1 + 1] = phi_patch

    return lam_steps, mu_steps


def iterative_2d_barrier(
    deformation,
    verbose=1,
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
    """CPU penalty->barrier solver for 2D fields.

    Parameters
    ----------
    windowed : bool
        When True (default), optimise each connected component of
        negative-Jdet pixels on its own padded patch with a frozen
        boundary ring. When False, optimise the full grid at once.
    pad : int
        Voxels of expansion on each side of the component bbox.
    max_iterations : int
        Max outer sweeps in windowed mode.
    """
    p = _resolve_params(threshold=threshold, err_tol=err_tol)
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    if verbose is True: verbose = 1
    elif verbose is False: verbose = 0

    error_list, num_neg_jac, iter_times, min_jdet_list, window_counts = _setup_accumulators()

    phi_init_2d, H, W = _coerce_2d(deformation)
    phi = phi_init_2d.copy()
    grid_size = (H, W)
    phi_init_flat = _pack_phi_2d(phi_init_2d)

    j0 = _jdet_2d_flat(_pack_phi_2d(phi), grid_size)
    init_neg = int((j0 <= 0).sum());  init_min = float(j0.min())
    num_neg_jac.append(init_neg);  min_jdet_list.append(init_min)
    mode = "windowed" if windowed else "full-grid"
    _log(verbose, 1, f"[init] Grid {H}x{W}  threshold={threshold}  margin={margin}  mode={mode}")
    _log(verbose, 1, f"[init] Neg-Jdet: {init_neg}  min Jdet: {init_min:.6f}")

    target = threshold + margin
    start = time.time()

    if windowed:
        structure = np.ones((3, 3))  # 8-connectivity
        for iteration in range(max_iterations):
            phi_flat = _pack_phi_2d(phi)
            j = _jdet_2d_flat(phi_flat, grid_size).reshape(H, W)
            neg_mask = j <= threshold - err_tol
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
                y0, y1, x0, x1 = _patch_bbox_2d(coords, pad, (H, W))
                window_counts[(y1 - y0 + 1, x1 - x0 + 1)] += 1
                lam_steps, mu_steps = _optimize_patch_2d(
                    phi, y0, y1, x0, x1, (H, W),
                    threshold, margin, lam_schedule, mu_schedule,
                    max_minimize_iter, verbose)
                total_lam += lam_steps
                total_mu += mu_steps

            elapsed = time.time() - t0
            iter_times.append(elapsed)

            phi_flat = _pack_phi_2d(phi)
            j = _jdet_2d_flat(phi_flat, grid_size)
            cur_neg = int((j <= 0).sum()); cur_min = float(j.min())
            l2 = float(np.linalg.norm(phi_flat - phi_init_flat))
            num_neg_jac.append(cur_neg); min_jdet_list.append(cur_min); error_list.append(l2)
            _log(verbose, 1,
                 f"[iter {iteration+1}] comps={n_components:4d}  "
                 f"lam_steps={total_lam:3d} mu_steps={total_mu:3d}  "
                 f"neg={cur_neg:5d}  min_J={cur_min:+.6f}  "
                 f"L2={l2:.4f}  t={elapsed:.2f}s")
            if cur_neg == 0 and cur_min >= threshold - err_tol:
                break
    else:
        # Full-grid mode
        phi_flat = _pack_phi_2d(phi)
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

        phi = _unpack_phi_2d(phi_flat, grid_size)

    elapsed = time.time() - start
    phi_flat = _pack_phi_2d(phi)
    final_neg = int((_jdet_2d_flat(phi_flat, grid_size) <= 0).sum())
    final_min = float(_jdet_2d_flat(phi_flat, grid_size).min())
    final_err = float(np.linalg.norm(phi_flat - phi_init_flat))
    _print_summary(verbose, f"Penalty->Barrier L-BFGS-B - 2D ({mode})", (H, W),
                   len(iter_times), init_neg, final_neg, init_min, final_min,
                   final_err, elapsed)
    return phi


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


def _jdet_2d_torch_batched(phi_b):
    """Batched Jdet. phi_b shape (K, 2, H, W) with phi_b[:,0]=dy, phi_b[:,1]=dx.
    Returns (K, H, W). torch.gradient along dim=2/3 treats dim=0 (K) as batch."""
    dy = phi_b[:, 0];  dx = phi_b[:, 1]   # (K, H, W)
    ddx_dx = torch.gradient(dx, dim=2)[0]
    ddy_dy = torch.gradient(dy, dim=1)[0]
    ddx_dy = torch.gradient(dx, dim=1)[0]
    ddy_dx = torch.gradient(dy, dim=2)[0]
    return (1.0 + ddx_dx) * (1.0 + ddy_dy) - ddx_dy * ddy_dx


def _bbox_overlap_2d(b1, b2):
    y0a, y1a, x0a, x1a = b1
    y0b, y1b, x0b, x1b = b2
    return not (y1a < y0b or y1b < y0a or x1a < x0b or x1b < x0a)


def _group_nonoverlapping_2d(bboxes, max_batch=32):
    """Greedy-pack bboxes into batches of non-overlapping boxes (each ≤ max_batch)."""
    batches = []
    for bb in bboxes:
        placed = False
        for batch in batches:
            if len(batch) >= max_batch:
                continue
            if all(not _bbox_overlap_2d(bb, o) for o in batch):
                batch.append(bb)
                placed = True
                break
        if not placed:
            batches.append([bb])
    return batches


def _optimize_batch_2d_torch(phi_full, bboxes, grid_shape,
                              threshold_f, margin, lam_schedule, mu_schedule,
                              max_minimize_iter, device, dtype):
    """Batched penalty->barrier on K non-overlapping interior 2D patches.
    Shares a single LBFGS optimizer across K patches via (K, 2, Hmax, Wmax)
    tensor with per-patch frozen/active masks."""
    H, W = grid_shape
    K = len(bboxes)
    Hmax = max(y1 - y0 + 1 for (y0, y1, x0, x1) in bboxes)
    Wmax = max(x1 - x0 + 1 for (y0, y1, x0, x1) in bboxes)

    phi_batch_init = torch.zeros((K, 2, Hmax, Wmax), dtype=dtype, device=device)
    cell_frozen = torch.ones((K, Hmax, Wmax), dtype=torch.bool, device=device)
    active_cell = torch.zeros((K, Hmax, Wmax), dtype=torch.bool, device=device)

    for k, (y0, y1, x0, x1) in enumerate(bboxes):
        Hp, Wp = y1 - y0 + 1, x1 - x0 + 1
        phi_batch_init[k, :, :Hp, :Wp] = phi_full[:, y0:y1 + 1, x0:x1 + 1]
        pf = torch.zeros((Hp, Wp), dtype=torch.bool, device=device)
        pf[0, :] = True;  pf[-1, :] = True
        pf[:, 0] = True;  pf[:, -1] = True
        cell_frozen[k, :Hp, :Wp] = pf
        active_cell[k, :Hp, :Wp] = ~pf

    cell_frozen_b = cell_frozen.unsqueeze(1)
    active_f = active_cell.to(dtype=dtype)
    phi_var = phi_batch_init.detach().clone().requires_grad_(True)

    tol_change = 1e-9 if dtype == torch.float64 else 1e-7
    hs = 20
    mi = max_minimize_iter
    target = threshold_f + margin
    n_lam = len(lam_schedule)
    lam_steps = 0
    mu_steps = 0

    def _per_patch_min(phi_eff):
        j = _jdet_2d_torch_batched(phi_eff)
        j_masked = torch.where(active_cell, j, torch.full_like(j, float("inf")))
        return j_masked.reshape(K, -1).min(dim=1).values

    with torch.no_grad():
        phi_eff = torch.where(cell_frozen_b, phi_batch_init, phi_var)
        per_patch_min = _per_patch_min(phi_eff)
        feasible_k = per_patch_min >= target

    for lam_idx, lam in enumerate(lam_schedule):
        if bool(feasible_k.all().item()):
            break

        def closure():
            if phi_var.grad is not None:
                phi_var.grad.zero_()
            phi_eff = torch.where(cell_frozen_b, phi_batch_init, phi_var)
            diff = phi_eff - phi_batch_init
            data = 0.5 * (diff * diff).sum()
            j = _jdet_2d_torch_batched(phi_eff)
            viol = torch.clamp(target - j, min=0.0)
            pen = lam * (viol * viol * active_f).sum()
            loss = data + pen
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([phi_var], lr=1.0, max_iter=mi,
                                 tolerance_grad=1e-6, tolerance_change=tol_change,
                                 history_size=hs, line_search_fn="strong_wolfe")
        opt.step(closure)
        lam_steps += 1
        if lam_idx % 2 == 1 or lam_idx == n_lam - 1:
            with torch.no_grad():
                phi_eff = torch.where(cell_frozen_b, phi_batch_init, phi_var)
                per_patch_min = _per_patch_min(phi_eff)
                feasible_k = per_patch_min >= target

    if bool(feasible_k.any().item()):
        active_bar = active_cell & feasible_k.view(K, 1, 1)
        active_bar_f = active_bar.to(dtype=dtype)
        prev_l2 = None
        for mu in mu_schedule:
            def closure_bar():
                if phi_var.grad is not None:
                    phi_var.grad.zero_()
                phi_eff = torch.where(cell_frozen_b, phi_batch_init, phi_var)
                diff = phi_eff - phi_batch_init
                data = 0.5 * (diff * diff).sum()
                j = _jdet_2d_torch_batched(phi_eff)
                slack = j - threshold_f
                safe_slack = torch.where(active_bar, slack, torch.ones_like(slack))
                if (slack * active_bar_f).min() <= 0:
                    viol = torch.clamp(-slack + 1e-12, min=0.0) * active_bar_f
                    bar = 1e8 * (viol * viol).sum()
                else:
                    bar = -mu * (torch.log(safe_slack) * active_bar_f).sum()
                loss = data + bar
                loss.backward()
                return loss

            opt = torch.optim.LBFGS([phi_var], lr=1.0, max_iter=max_minimize_iter,
                                     tolerance_grad=1e-6, tolerance_change=tol_change,
                                     history_size=20, line_search_fn="strong_wolfe")
            opt.step(closure_bar)
            mu_steps += 1
            with torch.no_grad():
                phi_eff = torch.where(cell_frozen_b, phi_batch_init, phi_var)
                cur_l2 = float(torch.linalg.norm(phi_eff - phi_batch_init).item())
            if prev_l2 is not None and abs(cur_l2 - prev_l2) / max(prev_l2, 1e-9) < 1e-5:
                break
            prev_l2 = cur_l2

    with torch.no_grad():
        phi_eff = torch.where(cell_frozen_b, phi_batch_init, phi_var)
        for k, (y0, y1, x0, x1) in enumerate(bboxes):
            Hp, Wp = y1 - y0 + 1, x1 - x0 + 1
            phi_full[:, y0:y1 + 1, x0:x1 + 1] = phi_eff[k, :, :Hp, :Wp]

    return lam_steps, mu_steps


def _optimize_patch_2d_torch(phi_full, y0, y1, x0, x1, grid_shape,
                              threshold_f, margin, lam_schedule, mu_schedule,
                              max_minimize_iter, device, dtype):
    """Run penalty->barrier on a 2D patch via torch LBFGS.

    *phi_full* is a CPU/GPU tensor shape (2, H, W). The patch region is
    mutated in place. Returns (lam_steps, mu_steps).
    """
    H, W = grid_shape
    Hp, Wp = y1 - y0 + 1, x1 - x0 + 1

    phi_patch_init = phi_full[:, y0:y1 + 1, x0:x1 + 1].detach().clone()
    phi_patch_var = phi_patch_init.detach().clone().requires_grad_(True)

    # Frozen mask: patch boundary voxels not touching grid boundary.
    frozen = torch.zeros((Hp, Wp), dtype=torch.bool, device=device)
    if y0 > 0:     frozen[0, :] = True
    if y1 < H - 1: frozen[-1, :] = True
    if x0 > 0:     frozen[:, 0] = True
    if x1 < W - 1: frozen[:, -1] = True
    frozen_b = frozen.unsqueeze(0)  # broadcast to (2, Hp, Wp)
    # Interior mask for Jdet-based sums (see 3D solver for rationale).
    active = ~frozen  # (Hp, Wp)

    tol_change = 1e-9 if dtype == torch.float64 else 1e-7

    # Adaptive LBFGS sizing: tiny patches don't benefit from history=20
    # (Hessian approx saturates) and per-iter overhead dominates.
    n_active = int(active.sum().item())
    if n_active < 500:
        hs = 10
        mi = min(max_minimize_iter, 100)
    else:
        hs = 20
        mi = max_minimize_iter

    target = threshold_f + margin
    with torch.no_grad():
        j0 = _jdet_2d_torch(phi_patch_init)
        feasible = bool((j0[active].min() >= target).item())

    lam_steps = 0
    mu_steps = 0
    n_lam = len(lam_schedule)

    for lam_idx, lam in enumerate(lam_schedule):
        if feasible:
            break

        def closure():
            if phi_patch_var.grad is not None:
                phi_patch_var.grad.zero_()
            phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
            diff = phi_eff - phi_patch_init
            data = 0.5 * (diff * diff).sum()
            j = _jdet_2d_torch(phi_eff)
            viol = torch.clamp(target - j, min=0.0)
            pen = lam * (viol * viol * active).sum()
            loss = data + pen
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([phi_patch_var], lr=1.0, max_iter=mi,
                                 tolerance_grad=1e-6, tolerance_change=tol_change,
                                 history_size=hs, line_search_fn="strong_wolfe")
        opt.step(closure)
        lam_steps += 1
        # Skip GPU->CPU .item() sync on even lam indices (check odd + last).
        if lam_idx % 2 == 1 or lam_idx == n_lam - 1:
            with torch.no_grad():
                phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
                j = _jdet_2d_torch(phi_eff)
                if float(j[active].min().item()) >= target:
                    feasible = True
                    break

    if feasible:
        active_f = active.to(dtype=phi_patch_var.dtype)
        prev_l2 = None
        for mu in mu_schedule:
            def closure():
                if phi_patch_var.grad is not None:
                    phi_patch_var.grad.zero_()
                phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
                diff = phi_eff - phi_patch_init
                data = 0.5 * (diff * diff).sum()
                j = _jdet_2d_torch(phi_eff)
                slack = j - threshold_f
                safe_slack = torch.where(active, slack, torch.ones_like(slack))
                if (slack * active_f).min() <= 0:
                    viol = torch.clamp(-slack + 1e-12, min=0.0) * active_f
                    bar = 1e8 * (viol * viol).sum()
                else:
                    bar = -mu * (torch.log(safe_slack) * active_f).sum()
                loss = data + bar
                loss.backward()
                return loss

            opt = torch.optim.LBFGS([phi_patch_var], lr=1.0, max_iter=max_minimize_iter,
                                     tolerance_grad=1e-6, tolerance_change=tol_change,
                                     history_size=20, line_search_fn="strong_wolfe")
            opt.step(closure)
            mu_steps += 1
            with torch.no_grad():
                phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
                cur_l2 = float(torch.linalg.norm(phi_eff - phi_patch_init).item())
            if prev_l2 is not None and abs(cur_l2 - prev_l2) / max(prev_l2, 1e-9) < 1e-5:
                break
            prev_l2 = cur_l2

    # Write back using the masked combination (so frozen voxels stay exact).
    with torch.no_grad():
        phi_eff = torch.where(frozen_b, phi_patch_init, phi_patch_var)
        phi_full[:, y0:y1 + 1, x0:x1 + 1] = phi_eff
    return lam_steps, mu_steps


def iterative_2d_barrier_torch(
    deformation,
    verbose=1,
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
    dtype=torch.float32,
):
    """GPU/CPU penalty->barrier solver for 2D fields via torch autograd.

    Parameters
    ----------
    windowed : bool
        When True (default), component-wise patched optimisation with
        frozen boundary ring; when False, full-grid at once.
    pad : int
        Voxels of expansion on each side of each component bbox.
    max_iterations : int
        Max outer sweeps in windowed mode.
    dtype : torch.dtype
        Default ``float32`` for GPU throughput. Pass ``torch.float64`` for
        parity with the numpy solver if a case stalls near the barrier.
        ``float16``/``bfloat16`` are not recommended — ~3-digit mantissa
        loses precision in the ``log(J - threshold)`` term.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    p = _resolve_params(threshold=threshold, err_tol=err_tol)
    threshold_f = float(p["threshold"])
    err_tol_f = float(p["err_tol"])
    if verbose is True: verbose = 1
    elif verbose is False: verbose = 0

    error_list, num_neg_jac, iter_times, min_jdet_list, window_counts = _setup_accumulators()

    phi_init_np, H, W = _coerce_2d(deformation)
    phi_init = torch.tensor(phi_init_np, dtype=dtype, device=device)

    with torch.no_grad():
        j0 = _jdet_2d_torch(phi_init)
        init_neg = int((j0 <= 0).sum().item())
        init_min = float(j0.min().item())
    num_neg_jac.append(init_neg);  min_jdet_list.append(init_min)
    mode = "windowed" if windowed else "full-grid"
    _log(verbose, 1, f"[init] Grid {H}x{W}  threshold={threshold_f}  margin={margin}  "
                     f"device={device}  mode={mode}")
    _log(verbose, 1, f"[init] Neg-Jdet: {init_neg}  min Jdet: {init_min:.6f}")

    target = threshold_f + margin
    start = time.time()

    if windowed:
        phi_full = phi_init.detach().clone()
        structure = np.ones((3, 3))

        for iteration in range(max_iterations):
            with torch.no_grad():
                j = _jdet_2d_torch(phi_full)
                neg_mask_t = j <= threshold_f - err_tol_f
                any_neg = bool(neg_mask_t.any().item())
            if not any_neg:
                _log(verbose, 1, f"[iter {iteration+1}] No neg-Jdet remain — exiting")
                break
            neg_mask = neg_mask_t.cpu().numpy()
            labeled, n_components = label(neg_mask, structure=structure)

            t0 = time.time()
            total_lam = 0
            total_mu = 0
            # Partition components: small+interior → batchable; rest → individual.
            # Interior = bbox doesn't touch grid edge (required because torch.gradient
            # at tensor endpoints uses one-sided diff, which won't match the batched
            # central-diff at within-tensor patch boundaries).
            small_bboxes = []
            large_bboxes = []
            for comp_id in range(1, n_components + 1):
                coords = np.where(labeled == comp_id)
                if coords[0].size == 0:
                    continue
                y0, y1, x0, x1 = _patch_bbox_2d(coords, pad, (H, W))
                window_counts[(y1 - y0 + 1, x1 - x0 + 1)] += 1
                Hp, Wp = y1 - y0 + 1, x1 - x0 + 1
                is_interior = (y0 > 0 and y1 < H - 1 and x0 > 0 and x1 < W - 1)
                # Active voxel count = (Hp-2)*(Wp-2) for interior patches (rim-frozen).
                n_active_est = max(0, (Hp - 2) * (Wp - 2))
                if is_interior and n_active_est < 3000:
                    small_bboxes.append((y0, y1, x0, x1))
                else:
                    large_bboxes.append((y0, y1, x0, x1))

            for batch in _group_nonoverlapping_2d(small_bboxes, max_batch=32):
                if len(batch) >= 2:
                    lam_steps, mu_steps = _optimize_batch_2d_torch(
                        phi_full, batch, (H, W),
                        threshold_f, margin, lam_schedule, mu_schedule,
                        max_minimize_iter, device, dtype)
                    total_lam += lam_steps
                    total_mu += mu_steps
                else:
                    (y0, y1, x0, x1) = batch[0]
                    lam_steps, mu_steps = _optimize_patch_2d_torch(
                        phi_full, y0, y1, x0, x1, (H, W),
                        threshold_f, margin, lam_schedule, mu_schedule,
                        max_minimize_iter, device, dtype)
                    total_lam += lam_steps
                    total_mu += mu_steps

            for (y0, y1, x0, x1) in large_bboxes:
                lam_steps, mu_steps = _optimize_patch_2d_torch(
                    phi_full, y0, y1, x0, x1, (H, W),
                    threshold_f, margin, lam_schedule, mu_schedule,
                    max_minimize_iter, device, dtype)
                total_lam += lam_steps
                total_mu += mu_steps

            elapsed = time.time() - t0
            iter_times.append(elapsed)

            with torch.no_grad():
                j = _jdet_2d_torch(phi_full)
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
        tol_change = 1e-9 if dtype == torch.float64 else 1e-7
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
                                     tolerance_grad=1e-6, tolerance_change=tol_change,
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
            prev_l2 = None
            for k, mu in enumerate(mu_schedule):
                def closure():
                    if phi_var.grad is not None: phi_var.grad.zero_()
                    diff = phi_var - phi_init
                    data = 0.5 * (diff * diff).sum()
                    j = _jdet_2d_torch(phi_var)
                    slack = j - threshold_f
                    if slack.min() <= 0:
                        viol = torch.clamp(-slack + 1e-12, min=0.0)
                        bar = 1e8 * (viol * viol).sum()
                    else:
                        bar = -mu * torch.log(slack).sum()
                    loss = data + bar
                    loss.backward()
                    return loss

                opt = torch.optim.LBFGS([phi_var], lr=1.0, max_iter=max_minimize_iter,
                                         tolerance_grad=1e-6, tolerance_change=tol_change,
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
                if prev_l2 is not None and abs(l2 - prev_l2) / max(prev_l2, 1e-9) < 1e-5:
                    _log(verbose, 1, f"[barrier early-exit] dL2/L2 < 1e-5, stopping mu schedule")
                    break
                prev_l2 = l2

        phi_var_final = phi_var.detach()

    elapsed = time.time() - start
    with torch.no_grad():
        j_final = _jdet_2d_torch(phi_var_final)
        final_neg = int((j_final <= 0).sum().item())
        final_min = float(j_final.min().item())
        final_err = float(torch.linalg.norm(phi_var_final - phi_init).item())
    phi_out = phi_var_final.detach().cpu().numpy()
    _print_summary(verbose, f"Penalty->Barrier L-BFGS torch - 2D ({mode})", (H, W),
                   len(iter_times), init_neg, final_neg, init_min, final_min,
                   final_err, elapsed)
    return phi_out
