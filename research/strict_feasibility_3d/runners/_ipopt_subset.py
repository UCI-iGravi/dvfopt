"""Ipopt global NLP on the actual fold region only.

The 94 unfixable cells live in z=0..7, y=136..221, x=191..283.
Cropping to this bounding box (with some padding) gives a problem
~5-10x smaller than the full chunk. Then MUMPS may fit.

Crop the chunk to its fold bbox + ring of 4 cells. Solve. Splice
back. Verify global feasibility.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import cyipopt
import numpy as np

from dvfopt.jacobian.tetrahedron_sign import (
    build_tet_sparse_jac,
    six_tet_volumes_3d,
    tet_volumes_flat,
)
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
PAD = 4  # cell-units of padding around the fold bbox


class IpoptTet6NLP:
    def __init__(self, phi_in_3dhw, threshold):
        self.D, self.H, self.W = phi_in_3dhw.shape[1:]
        self.threshold = threshold
        n = self.D * self.H * self.W
        self.x0 = np.concatenate(
            [
                phi_in_3dhw[2].ravel(),  # dx
                phi_in_3dhw[1].ravel(),  # dy
                phi_in_3dhw[0].ravel(),  # dz
            ]
        )
        self.n_vars = 3 * n
        self.n_constr = 6 * (self.D - 1) * (self.H - 1) * (self.W - 1)
        self.jac_builder = build_tet_sparse_jac(self.D, self.H, self.W)
        J0 = self.jac_builder(self.x0)
        self.J_coo = J0.tocoo()
        self.jac_rows = self.J_coo.row.copy()
        self.jac_cols = self.J_coo.col.copy()

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
        return J.data


def main():
    print('Loading checkpoint...', flush=True)
    phi_full = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    D_full, H_full, W_full = phi_full.shape[1:]
    print(f'  full shape: {phi_full.shape}', flush=True)

    # Find unfixable cells.
    best_min = _best_min_per_cell(phi_full)
    unfix_mask = best_min <= 0
    nz, ny, nx = np.where(unfix_mask)
    z_min, z_max = int(nz.min()), int(nz.max())
    y_min, y_max = int(ny.min()), int(ny.max())
    x_min, x_max = int(nx.min()), int(nx.max())
    # Crop with padding (in CORNER grid space).
    z0 = max(0, z_min - PAD)
    z1 = min(D_full, z_max + 1 + PAD + 1)  # +1 for corner-grid
    y0 = max(0, y_min - PAD)
    y1 = min(H_full, y_max + 1 + PAD + 1)
    x0 = max(0, x_min - PAD)
    x1 = min(W_full, x_max + 1 + PAD + 1)
    crop = phi_full[:, z0:z1, y0:y1, x0:x1].copy()
    print(f'  fold bbox: z[{z_min},{z_max}] y[{y_min},{y_max}] x[{x_min},{x_max}]', flush=True)
    print(f'  crop (with pad={PAD}): shape={crop.shape}', flush=True)

    V0 = six_tet_volumes_3d(crop)
    n_neg0 = int((V0 <= 0).sum())
    n_below0 = int((V0 < THRESHOLD - 1e-5).sum())
    print(
        f'  crop start: n_neg={n_neg0}  n<0.01={n_below0}  min_T={float(V0.min()):+.6f}', flush=True
    )

    nlp = IpoptTet6NLP(crop, THRESHOLD)
    print(f'  problem: {nlp.n_vars} vars, {nlp.n_constr} constraints', flush=True)

    cl = np.full(nlp.n_constr, THRESHOLD, dtype=np.float64)
    cu = np.full(nlp.n_constr, 1e20, dtype=np.float64)

    problem = cyipopt.Problem(
        n=nlp.n_vars,
        m=nlp.n_constr,
        problem_obj=nlp,
        lb=np.full(nlp.n_vars, -1e6, dtype=np.float64),
        ub=np.full(nlp.n_vars, +1e6, dtype=np.float64),
        cl=cl,
        cu=cu,
    )
    problem.add_option('max_iter', 200)
    problem.add_option('tol', 1e-6)
    problem.add_option('hessian_approximation', 'limited-memory')
    problem.add_option('print_level', 5)
    problem.add_option('linear_solver', 'mumps')
    problem.add_option('mu_strategy', 'adaptive')

    print('\nStarting Ipopt solve...', flush=True)
    t0 = time.time()
    try:
        x_opt, info = problem.solve(nlp.x0)
    except Exception as e:
        print(f'EXCEPTION: {type(e).__name__}: {e}', flush=True)
        return
    wall = time.time() - t0
    print(f'\nIpopt finished in {wall:.1f}s (status: {info["status_msg"]})', flush=True)

    n = nlp.D * nlp.H * nlp.W
    dx_out = x_opt[:n].reshape(nlp.D, nlp.H, nlp.W)
    dy_out = x_opt[n : 2 * n].reshape(nlp.D, nlp.H, nlp.W)
    dz_out = x_opt[2 * n :].reshape(nlp.D, nlp.H, nlp.W)
    crop_out = np.stack([dz_out, dy_out, dx_out])
    V_final = six_tet_volumes_3d(crop_out)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1 = float(np.abs(crop_out - crop).sum())
    print(
        f'\nCrop final:  n_neg={n_neg}  n<0.01={n_below}  '
        f'min_T={float(V_final.min()):+.6f}  L1_from_input={L1:.1f}\n'
        f'  STRICT crop feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )

    # Splice back into full chunk.
    phi_global = phi_full.copy()
    # Splice ALL crop corners (since we're not worried about boundary
    # interaction with outside the crop within the inner region).
    phi_global[:, z0:z1, y0:y1, x0:x1] = crop_out
    V_global = six_tet_volumes_3d(phi_global)
    n_neg_g = int((V_global <= 0).sum())
    n_below_g = int((V_global < THRESHOLD - 1e-5).sum())
    L1_g = float(np.abs(phi_global - phi_full).sum())
    print(
        f'\nGlobal post-splice:  n_neg={n_neg_g}  n<0.01={n_below_g}  '
        f'min_T={float(V_global.min()):+.6f}  L1_from_input={L1_g:.1f}\n'
        f'  STRICT global feas: {n_neg_g == 0 and n_below_g == 0}',
        flush=True,
    )
    if n_neg_g == 0 and n_below_g == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_ipopt_subset.npy', phi_global)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
