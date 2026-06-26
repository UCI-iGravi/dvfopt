"""Ipopt global NLP for full 3D strict feasibility.

Uses cyipopt to solve the constrained NLP at full chunk scale:
  minimize  0.5 * ||phi - phi_in||^2
  subject to  V_k(phi) >= threshold for all tets k

Variables: 3 * D * H * W phi values (~7M for z=0..15 chunk)
Constraints: 6 * (D-1) * (H-1) * (W-1) tet volumes (~13M)

Uses sparse Jacobian via build_tet_sparse_jac (already exists).

Run with: C:/Users/Andy/anaconda3/python.exe (cyipopt only in
anaconda base, not in .venv).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import scipy.sparse as sp
import cyipopt

from dvfopt.jacobian.tetrahedron_sign import (
    build_tet_sparse_jac,
    tet_volumes_flat,
    six_tet_volumes_3d,
)


OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


class IpoptTet6NLP:
    """Cyipopt NLP wrapper for global 3D 6-tet feasibility."""

    def __init__(self, phi_in_3dhw, threshold):
        self.D, self.H, self.W = phi_in_3dhw.shape[1:]
        self.threshold = threshold
        # Pack phi_in into the variable order [dx, dy, dz] x ravel.
        n = self.D * self.H * self.W
        dz = phi_in_3dhw[0]
        dy = phi_in_3dhw[1]
        dx = phi_in_3dhw[2]
        self.x0 = np.concatenate([dx.ravel(), dy.ravel(), dz.ravel()])
        self.n_vars = 3 * n
        self.n_constr = 6 * (self.D - 1) * (self.H - 1) * (self.W - 1)
        # Build sparse Jacobian builder.
        self.jac_builder = build_tet_sparse_jac(self.D, self.H, self.W)
        # Precompute Jacobian sparsity (rows, cols).
        J0 = self.jac_builder(self.x0)
        self.J_coo = J0.tocoo()
        self.jac_rows = self.J_coo.row.copy()
        self.jac_cols = self.J_coo.col.copy()
        # Cache.
        self._cached_J_csr = None
        self._cached_x_id = -1

    def objective(self, x):
        d = x - self.x0
        return 0.5 * float(d @ d)

    def gradient(self, x):
        return x - self.x0

    def constraints(self, x):
        return tet_volumes_flat(x, self.D, self.H, self.W)

    def jacobianstructure(self):
        return self.jac_rows, self.jac_cols

    def jacobian(self, x):
        J = self.jac_builder(x).tocoo()
        # Reorder values by the (row, col) ordering returned in jacobianstructure.
        # Since jac_builder returns the same structure each call, the data array
        # is in the same order as our cached (rows, cols).
        return J.data


def main():
    print('Loading checkpoint...', flush=True)
    phi_in = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    print(f'  shape: {phi_in.shape}', flush=True)
    V0 = six_tet_volumes_3d(phi_in)
    n_neg0 = int((V0 <= 0).sum())
    n_below0 = int((V0 < THRESHOLD - 1e-5).sum())
    print(f'  start: n_neg={n_neg0}  n<0.01={n_below0}  min_T={float(V0.min()):+.6f}', flush=True)

    nlp = IpoptTet6NLP(phi_in, THRESHOLD)
    print(f'  problem: {nlp.n_vars} vars, {nlp.n_constr} constraints', flush=True)

    # Setup cyipopt problem.
    cl = np.full(nlp.n_constr, THRESHOLD, dtype=np.float64)  # lower bound = threshold
    cu = np.full(nlp.n_constr, 1e20, dtype=np.float64)  # upper bound = infinity

    problem = cyipopt.Problem(
        n=nlp.n_vars,
        m=nlp.n_constr,
        problem_obj=nlp,
        lb=np.full(nlp.n_vars, -1e6, dtype=np.float64),
        ub=np.full(nlp.n_vars, +1e6, dtype=np.float64),
        cl=cl,
        cu=cu,
    )
    problem.add_option('max_iter', 100)
    problem.add_option('tol', 1e-6)
    problem.add_option('hessian_approximation', 'limited-memory')
    problem.add_option('print_level', 5)
    problem.add_option('linear_solver', 'mumps')
    problem.add_option('mu_strategy', 'adaptive')

    print('\nStarting Ipopt solve...', flush=True)
    t0 = time.time()
    x_opt, info = problem.solve(nlp.x0)
    wall = time.time() - t0
    print(f'\nIpopt finished in {wall:.1f}s (status: {info["status_msg"]})', flush=True)

    # Unpack and check feasibility.
    n = nlp.D * nlp.H * nlp.W
    dx_out = x_opt[:n].reshape(nlp.D, nlp.H, nlp.W)
    dy_out = x_opt[n:2 * n].reshape(nlp.D, nlp.H, nlp.W)
    dz_out = x_opt[2 * n:].reshape(nlp.D, nlp.H, nlp.W)
    phi_out = np.stack([dz_out, dy_out, dx_out])
    V_final = six_tet_volumes_3d(phi_out)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(phi_out - phi_in).sum())
    print(
        f'\nFinal:  n_neg={n_neg}  n<0.01={n_below}  '
        f'min_T={float(V_final.min()):+.6f}  L1_from_input={L1:.1f}\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_ipopt.npy', phi_out)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
