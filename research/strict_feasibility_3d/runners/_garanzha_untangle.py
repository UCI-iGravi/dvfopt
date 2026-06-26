"""Garanzha-style chi_eps regularized-barrier untangler (comparison baseline).

Faithful port of the regularized-determinant continuation from Garanzha
et al. 2021 ("Foldover-free maps in 50 lines of code") and Escobar et al.
2003, adapted to our 6-tet-per-cube discrete setting and our data-fidelity
objective.

Per tet k with signed volume V_k:
    chi_eps(V) = 0.5 * (V + sqrt(V^2 + eps^2))           # smooth ~max(V,0)
    f_vol(V)   = (1 + V^2) / chi_eps(V)                  # +inf as V->-inf
Energy:
    E(phi) = sum_k f_vol(V_k) + lam_anchor * ||phi - phi0||^2
with an eps-continuation eps: large -> small (anneal toward the true
barrier). f_vol is finite on inverted tets, so it ACCEPTS the tangled
input directly (the whole point of a regularized-barrier untangler).

Gradient (analytic, verified against finite differences):
    dE/dphi = sum_k f_vol'(V_k) * dV_k/dphi + 2*lam*(phi - phi0)
    f_vol'(V) = (2V*chi - (1+V^2)*chi') / chi^2,  chi' = 0.5*(1 + V/sqrt(V^2+eps^2))
We reuse tet_grad_T_v (the tet-volume adjoint J^T v) with v_k = f_vol'(V_k).

This tests the literature workflow's adversarial prediction: that a
standard global node-movement untangler converges to the SAME
shared-corner local minimum our M10Tet/coupled-kring stack reaches.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import minimize

from dvfopt.jacobian.tetrahedron_sign import (
    six_tet_volumes_3d,
    tet_grad_T_v,
    tet_volumes_flat,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def _phi_to_flat(phi):
    """(3, D, H, W) [dz,dy,dx] -> flat [dx, dy, dz] (DX_FIRST pack)."""
    dz, dy, dx = phi[0], phi[1], phi[2]
    return np.concatenate([dx.ravel(), dy.ravel(), dz.ravel()])


def _flat_to_phi(flat, D, H, W):
    n = D * H * W
    dx = flat[:n].reshape(D, H, W)
    dy = flat[n : 2 * n].reshape(D, H, W)
    dz = flat[2 * n :].reshape(D, H, W)
    return np.stack([dz, dy, dx])


def energy_and_grad(flat, flat0, D, H, W, eps, lam, target):
    """Garanzha chi_eps energy + analytic gradient over the flat field."""
    V = tet_volumes_flat(flat, D, H, W)  # (6*ncells,)
    Vt = V - target  # shift so the barrier targets V > target
    root = np.sqrt(Vt * Vt + eps * eps)
    chi = 0.5 * (Vt + root)
    chi = np.maximum(chi, 1e-12)
    f_vol = (1.0 + Vt * Vt) / chi
    E_barrier = float(f_vol.sum())
    # df/dV.
    chip = 0.5 * (1.0 + Vt / root)
    dfdV = (2.0 * Vt * chi - (1.0 + Vt * Vt) * chip) / (chi * chi)
    g_barrier = tet_grad_T_v(flat, D, H, W, dfdV)
    # Anchor.
    diff = flat - flat0
    E_anchor = lam * float(diff @ diff)
    g_anchor = 2.0 * lam * diff
    return E_barrier + E_anchor, g_barrier + g_anchor


def _verify_gradient():
    """FD-check the analytic gradient on a tiny random field."""
    rng = np.random.default_rng(0)
    D, H, W = 4, 4, 4
    phi = rng.normal(0, 0.3, (3, D, H, W))
    flat = _phi_to_flat(phi)
    flat0 = flat.copy()
    eps, lam, target = 0.1, 0.01, 0.0
    E0, g = energy_and_grad(flat, flat0, D, H, W, eps, lam, target)
    fd = np.zeros_like(g)
    h = 1e-6
    for i in range(len(flat)):
        fp = flat.copy()
        fp[i] += h
        fm = flat.copy()
        fm[i] -= h
        Ep, _ = energy_and_grad(fp, flat0, D, H, W, eps, lam, target)
        Em, _ = energy_and_grad(fm, flat0, D, H, W, eps, lam, target)
        fd[i] = (Ep - Em) / (2 * h)
    err = np.max(np.abs(g - fd))
    print(f'[gradient check] max|analytic - FD| = {err:.3e}', flush=True)
    return err


def run_garanzha(
    phi,
    *,
    eps_schedule=(1.0, 0.3, 0.1, 0.03, 0.01, 0.003),
    lam=1e-3,
    target=0.0,
    maxiter=200,
    verbose=1,
):
    D, H, W = phi.shape[1:]
    flat0 = _phi_to_flat(phi)
    flat = flat0.copy()
    V0 = six_tet_volumes_3d(phi)
    if verbose:
        print(
            f'Start: n_neg={int((V0 <= 0).sum())} '
            f'n<0.01={int((V0 < THRESHOLD - 1e-5).sum())} min_T={V0.min():+.6f}',
            flush=True,
        )
    t0 = time.time()
    for ei, eps in enumerate(eps_schedule):
        res = minimize(
            energy_and_grad,
            flat,
            args=(flat0, D, H, W, eps, lam, target),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        flat = res.x
        phi_cur = _flat_to_phi(flat, D, H, W)
        V = six_tet_volumes_3d(phi_cur)
        if verbose:
            print(
                f'  eps={eps:.3f}: n_neg={int((V <= 0).sum())} '
                f'n<0.01={int((V < THRESHOLD - 1e-5).sum())} '
                f'min_T={V.min():+.6f} '
                f'L1={float(np.abs(phi_cur - phi).sum()):.1f} '
                f'iters={res.nit} wall={time.time() - t0:.1f}s',
                flush=True,
            )
    phi_out = _flat_to_phi(flat, D, H, W)
    return phi_out


def main():
    print('=== Gradient verification ===', flush=True)
    err = _verify_gradient()
    assert err < 1e-5, f'gradient check failed: {err}'

    # Run on the 1-fold attractor case first (fast, diagnostic).
    print('\n=== Garanzha chi_eps on CHAIN_BEST (1 fold) ===', flush=True)
    phi = np.load(OUTPUT / 'b0039_z0_15_chain_best.npy').astype(np.float64)
    t0 = time.time()
    out = run_garanzha(phi, lam=1e-3, target=THRESHOLD)
    V = six_tet_volumes_3d(out)
    print(
        f'  FINAL: n_neg={int((V <= 0).sum())} n<0.01={int((V < THRESHOLD - 1e-5).sum())} '
        f'min_T={V.min():+.6f} L1={float(np.abs(out - phi).sum()):.1f} '
        f'wall={time.time() - t0:.1f}s',
        flush=True,
    )
    np.save(OUTPUT / 'b0039_z0_15_garanzha_from_chainbest.npy', out)

    # Run on the RAW dense band (the real comparison vs M10Tet's 173->19).
    print('\n=== Garanzha chi_eps on RAW dense band (z0-15, 173 folds) ===', flush=True)
    phi_raw = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    t0 = time.time()
    out_raw = run_garanzha(
        phi_raw,
        lam=1e-4,
        target=THRESHOLD,
        eps_schedule=(2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01),
    )
    V = six_tet_volumes_3d(out_raw)
    print(
        f'  FINAL: n_neg={int((V <= 0).sum())} n<0.01={int((V < THRESHOLD - 1e-5).sum())} '
        f'min_T={V.min():+.6f} L1={float(np.abs(out_raw - phi_raw).sum()):.1f} '
        f'wall={time.time() - t0:.1f}s',
        flush=True,
    )
    np.save(OUTPUT / 'b0039_z0_15_garanzha_from_raw.npy', out_raw)


if __name__ == '__main__':
    main()
