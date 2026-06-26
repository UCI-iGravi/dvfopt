"""TLC-style (Total Lifted Content) untangler — comparison baseline.

Du et al. 2020, "Lifting Simplices to Find Injectivity" (SIGGRAPH). The
idea: replace each element's signed content (volume) by a "lifted
content" that is smooth and bounded-below through inverted states, so a
gradient descent from a TANGLED start flows toward an injective map. For
a tet with signed volume V, the lifted content is

    L_eps(V) = 0.5 * ( sqrt(V^2 + eps^2) + V )            ~ max(V, 0)

(the positive part, smoothed) — i.e. TLC sums the *unsigned positive
content*; a tangled element (V<0) contributes ~0 lifted content but a
nonzero true |content|, so minimizing the total lifted content while the
rest-content is fixed drives V positive. In practice TLC minimizes the
ratio total-lifted-content / total-signed-content; here we use the
widely-used surrogate: minimize sum_k [ L_eps(V_k) ] is degenerate
(wants V->0), so we use the TLC *energy* form

    E(phi) = sum_k ( sqrt(V_k^2 + eps^2) - V_k ) / 2      = sum_k max(-V_k, 0) smoothed

which is exactly the total "negative content" (lifted), zero for feasible
tets and positive (with magnitude |V|) for inverted ones — the cleanest
TLC-spirit untangling objective. Add a small data anchor + an eps
continuation.

This is a faithful-spirit, tractable port for COMPARISON, not a
verbatim reimplementation of the paper's rest-shape-normalized form.
Gradient is analytic and FD-verified.
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
    dz, dy, dx = phi[0], phi[1], phi[2]
    return np.concatenate([dx.ravel(), dy.ravel(), dz.ravel()])


def _flat_to_phi(flat, D, H, W):
    n = D * H * W
    dx = flat[:n].reshape(D, H, W)
    dy = flat[n : 2 * n].reshape(D, H, W)
    dz = flat[2 * n :].reshape(D, H, W)
    return np.stack([dz, dy, dx])


def energy_and_grad(flat, flat0, D, H, W, eps, lam, target):
    """TLC negative-content energy + analytic gradient.

    Per tet: penalize (target - V) smoothly:
        u = target - V
        L = 0.5 * (sqrt(u^2 + eps^2) + u)     ~ max(u, 0) = max(target - V, 0)
    Minimizing sum_k L drives V_k above target. dL/dV = -0.5*(1 + u/root).
    """
    V = tet_volumes_flat(flat, D, H, W)
    u = target - V
    root = np.sqrt(u * u + eps * eps)
    L = 0.5 * (root + u)
    E_barrier = float(L.sum())
    dLdV = -0.5 * (1.0 + u / root)
    g_barrier = tet_grad_T_v(flat, D, H, W, dLdV)
    diff = flat - flat0
    E_anchor = lam * float(diff @ diff)
    g_anchor = 2.0 * lam * diff
    return E_barrier + E_anchor, g_barrier + g_anchor


def _verify_gradient():
    rng = np.random.default_rng(1)
    D, H, W = 4, 4, 4
    phi = rng.normal(0, 0.3, (3, D, H, W))
    flat = _phi_to_flat(phi)
    flat0 = flat.copy()
    eps, lam, target = 0.1, 0.01, 0.0
    _, g = energy_and_grad(flat, flat0, D, H, W, eps, lam, target)
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


def run_tlc(
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
    for eps in eps_schedule:
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
                f'n<0.01={int((V < THRESHOLD - 1e-5).sum())} min_T={V.min():+.6f} '
                f'L1={float(np.abs(phi_cur - phi).sum()):.1f} '
                f'iters={res.nit} wall={time.time() - t0:.1f}s',
                flush=True,
            )
    return _flat_to_phi(flat, D, H, W)


def main():
    print('=== Gradient verification ===', flush=True)
    err = _verify_gradient()
    assert err < 1e-5, f'gradient check failed: {err}'

    print('\n=== TLC on CHAIN_BEST (1 fold) ===', flush=True)
    phi = np.load(OUTPUT / 'b0039_z0_15_chain_best.npy').astype(np.float64)
    t0 = time.time()
    out = run_tlc(phi, lam=1e-3, target=THRESHOLD)
    V = six_tet_volumes_3d(out)
    print(
        f'  FINAL: n_neg={int((V <= 0).sum())} n<0.01={int((V < THRESHOLD - 1e-5).sum())} '
        f'min_T={V.min():+.6f} L1={float(np.abs(out - phi).sum()):.1f} '
        f'wall={time.time() - t0:.1f}s',
        flush=True,
    )
    np.save(OUTPUT / 'b0039_z0_15_tlc_from_chainbest.npy', out)

    print('\n=== TLC on RAW dense band (z0-15, 173 folds) ===', flush=True)
    phi_raw = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    t0 = time.time()
    out_raw = run_tlc(
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
    np.save(OUTPUT / 'b0039_z0_15_tlc_from_raw.npy', out_raw)


if __name__ == '__main__':
    main()
