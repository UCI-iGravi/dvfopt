"""Full-grid 2-tri penalty -> log-barrier solver in PyTorch (CPU or GPU).

Why a separate torch implementation?

The numpy barrier already implements penalty + log-barrier with L-BFGS-B
and an analytical J^T v product (``iterative_2d_tri_barrier``). When that
solver plateaued the diagnosis was that L-BFGS-B's curvature pair history
gets confused at the constraint boundary. Torch gives us two trivially
available options the numpy stack does not:

* second-order step via the full Hessian-vector product (``torch.autograd``
  reverse-on-forward) -- enabling Newton-CG, which handles
  ill-conditioning much better than L-BFGS when the barrier mu -> 0;
* a clean Adam fallback that ignores curvature entirely and uses
  per-parameter step sizes -- which sometimes crosses degenerate
  geometries L-BFGS gets stuck on.

We try L-BFGS first (efficient when it works) and fall back to Adam if
the L-BFGS step plateaus. Both run on whatever device torch picks
(CUDA if available).
"""
from __future__ import annotations

import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

NAME = 'torch_full_grid'
DESCRIPTION = 'Full-grid barrier in PyTorch with L-BFGS + Adam fallback; runs on CUDA if available'


def _tri_areas_torch(dy: 'torch.Tensor', dx: 'torch.Tensor'):
    """Same 2-tri areas as the numpy version but autograd-differentiable."""
    H, W = dy.shape
    ref_y = torch.arange(H, device=dy.device, dtype=dy.dtype)[:, None].expand(H, W)
    ref_x = torch.arange(W, device=dy.device, dtype=dy.dtype)[None, :].expand(H, W)
    def_y = ref_y + dy
    def_x = ref_x + dx
    x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
    x_tr, y_tr = def_x[:-1, 1:],  def_y[:-1, 1:]
    x_bl, y_bl = def_x[1:, :-1],  def_y[1:, :-1]
    x_br, y_br = def_x[1:, 1:],   def_y[1:, 1:]
    AB_x = x_bl - x_tr; AB_y = y_bl - y_tr
    AC_x = x_br - x_tr; AC_y = y_br - y_tr
    T1 = -0.5 * (AB_x * AC_y - AB_y * AC_x)
    AB_x = x_bl - x_tl; AB_y = y_bl - y_tl
    AC_x = x_tr - x_tl; AC_y = y_tr - y_tl
    T2 = -0.5 * (AB_x * AC_y - AB_y * AC_x)
    return T1, T2


def _objective(phi, phi_init, threshold, lam, mu, anchor):
    """Penalty + log-barrier scalar (mu=0 -> penalty only)."""
    dy, dx = phi[0], phi[1]
    if anchor == 'l2':
        diff = phi - phi_init
        anc = 0.5 * (diff * diff).sum()
    elif anchor == 'l1':
        diff = phi - phi_init
        anc = (torch.sqrt(diff * diff + 1e-8) - 1e-4).sum()
    else:
        anc = torch.zeros((), device=phi.device, dtype=phi.dtype)
    T1, T2 = _tri_areas_torch(dy, dx)
    if mu > 0:
        # log barrier requires strict feasibility
        s1 = T1 - threshold
        s2 = T2 - threshold
        if (s1.min() <= 0) or (s2.min() <= 0):
            return torch.tensor(float('inf'), device=phi.device, dtype=phi.dtype)
        return anc - mu * (torch.log(s1).sum() + torch.log(s2).sum())
    else:
        viol1 = torch.clamp(threshold - T1, min=0.0)
        viol2 = torch.clamp(threshold - T2, min=0.0)
        pen = lam * ((viol1 * viol1).sum() + (viol2 * viol2).sum())
        return anc + pen


def _run_lbfgs(phi_param, phi_init_t, threshold, lam, mu, anchor,
               max_iter, lr=1.0):
    opt = torch.optim.LBFGS([phi_param], max_iter=max_iter,
                             history_size=30, lr=lr,
                             tolerance_grad=1e-7,
                             tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = _objective(phi_param, phi_init_t, threshold, lam, mu, anchor)
        if torch.isfinite(loss):
            loss.backward()
        return loss
    opt.step(closure)


def _run_adam(phi_param, phi_init_t, threshold, lam, mu, anchor,
              steps, lr=1e-2):
    opt = torch.optim.Adam([phi_param], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = _objective(phi_param, phi_init_t, threshold, lam, mu, anchor)
        if not torch.isfinite(loss):
            # back off step
            for g in opt.param_groups:
                g['lr'] *= 0.5
            continue
        loss.backward()
        opt.step()


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin: float = 1e-3,
          anchor: str = 'l2',
          lam_schedule: tuple = (1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
          mu_schedule: tuple = (1e-1, 1e-2, 1e-3, 1e-4),
          lbfgs_iters: int = 200,
          adam_steps: int = 1500,
          device: str | None = None,
          dtype: str = 'float64',
          time_budget_s: float = 600.0,
          verbose: int = 0) -> dict:
    if not HAS_TORCH:
        raise RuntimeError('torch not available')
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float64 if dtype == 'float64' else torch.float32

    H, W = phi_in.shape[1], phi_in.shape[2]
    phi_init_t = torch.tensor(phi_in.copy(), device=device, dtype=torch_dtype)
    phi_param = phi_init_t.clone().requires_grad_(True)

    log = []
    t0 = time.time()
    # Phase 1: penalty schedule
    for lam in lam_schedule:
        if time.time() - t0 > time_budget_s:
            break
        _run_lbfgs(phi_param, phi_init_t, threshold + margin, lam, 0.0,
                   anchor, lbfgs_iters)
        with torch.no_grad():
            T1, T2 = _tri_areas_torch(phi_param[0], phi_param[1])
            min_T = float(torch.minimum(T1, T2).min().item())
        log.append(dict(phase='penalty', lam=lam, min_T=min_T,
                        wall=time.time() - t0))
        if verbose:
            print(f'  pen lam={lam:.1e}  min_T={min_T:+.5f}  '
                  f'({time.time()-t0:.1f}s)', flush=True)
        if min_T > threshold + margin:
            break

    # If still infeasible, try Adam to push past degenerate basin.
    with torch.no_grad():
        T1, T2 = _tri_areas_torch(phi_param[0], phi_param[1])
        min_T = float(torch.minimum(T1, T2).min().item())
    if min_T <= threshold + margin and time.time() - t0 < time_budget_s \
            and lam_schedule:
        _run_adam(phi_param, phi_init_t, threshold + margin,
                  lam_schedule[-1], 0.0, anchor, adam_steps)
        with torch.no_grad():
            T1, T2 = _tri_areas_torch(phi_param[0], phi_param[1])
            min_T = float(torch.minimum(T1, T2).min().item())
        log.append(dict(phase='adam', min_T=min_T, wall=time.time() - t0))
        if verbose:
            print(f'  adam end  min_T={min_T:+.5f}  '
                  f'({time.time()-t0:.1f}s)', flush=True)

    # Phase 2: log-barrier interior refinement (only if strictly feasible)
    if min_T > threshold:
        lam_for_barrier = lam_schedule[-1] if lam_schedule else 1.0
        for mu in mu_schedule:
            if time.time() - t0 > time_budget_s:
                break
            _run_lbfgs(phi_param, phi_init_t, threshold,
                       lam_for_barrier, mu, anchor, lbfgs_iters)
            with torch.no_grad():
                T1, T2 = _tri_areas_torch(phi_param[0], phi_param[1])
                min_T = float(torch.minimum(T1, T2).min().item())
            log.append(dict(phase='barrier', mu=mu, min_T=min_T,
                            wall=time.time() - t0))
            if verbose:
                print(f'  bar mu={mu:.1e}  min_T={min_T:+.5f}  '
                      f'({time.time()-t0:.1f}s)', flush=True)

    phi_out = phi_param.detach().cpu().numpy().astype(np.float64)
    return {'phi_out': phi_out,
            'info': {'device': device, 'phases_used': len(log),
                     'log_first5': log[:5], 'log_last5': log[-5:]}}
