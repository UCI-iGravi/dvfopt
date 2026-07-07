"""Part XXI option C (3D probe): elastic 6-tet LP at the escape plateau.

The hard-constrained focused LP was INFEASIBLE at every trust radius at
the 173-fold stage-3 plateau (`_focused_polish` v2/v3). Elastic mode is
feasible by construction — the question is whether its steps make real
progress (violation decreasing toward 0) or stall (LP-invisible
nonconvexity), i.e. whether an LP path exists in 3D at all.

Probe: crop around the worst fold of the stage-3 band, free ALL interior
corner displacements (dx, dy, dz), run elastic SLP for a bounded number
of iterations; report exact min_T / n_neg / violation trajectory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parents[3]))

from dvfopt.jacobian.tetrahedron_sign import (  # noqa: E402
    build_tet_sparse_jac,
    six_tet_min_volume_3d,
    tet_volumes_flat,
)

THR = 0.01 + 1e-4
MU = 1000.0


def elastic_6tet(crop, max_iter=25, trust0=0.25, verbose=1):
    C, D, H, W = crop.shape
    n_pix = D * H * W
    jac = build_tet_sparse_jac(D, H, W)

    inner = np.zeros((D, H, W), dtype=bool)
    inner[1:-1, 1:-1, 1:-1] = True
    ii = np.where(inner.ravel())[0]
    free = np.concatenate([ii, n_pix + ii, 2 * n_pix + ii])  # dx, dy, dz blocks

    def _flat(c):
        return np.concatenate([c[2].ravel(), c[1].ravel(), c[0].ravel()])

    phi = _flat(crop)
    anchor_f = phi[free].copy()
    nf = free.size

    def _viol(pf):
        V = tet_volumes_flat(pf, D, H, W)
        return float(np.maximum(0, THR - V).sum()), float(V.min()), int((V <= 0).sum())

    viol, mn, nn = _viol(phi)
    print(f'  start: viol={viol:.4e} min_T={mn:+.5f} n_neg={nn} nf={nf}', flush=True)
    trust = trust0
    t0 = time.time()
    for it in range(max_iter):
        if viol <= 1e-12:
            break
        V = tet_volumes_flat(phi, D, H, W)
        J = jac(phi).tocsc()[:, free].tocsr()
        act = np.where(V < THR + 0.5)[0]
        Ja = J[act]
        Ka = act.size
        xf = phi[free]
        c_obj = np.concatenate([np.zeros(nf), np.ones(nf), MU * np.ones(Ka)])
        A1 = sp.hstack([sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        A2 = sp.hstack([-sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        E = sp.hstack([-Ja, sp.csr_matrix((Ka, nf)), -sp.eye(Ka)])
        A_ub = sp.vstack([A1, A2, E]).tocsr()
        b_ub = np.concatenate([anchor_f, -anchor_f, -THR + V[act] - Ja @ xf])
        bounds = ([(float(xf[i] - trust), float(xf[i] + trust)) for i in range(nf)]
                  + [(0.0, None)] * nf + [(0.0, None)] * Ka)
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            trust *= 0.5
            if trust < 1e-4:
                break
            continue
        cand = phi.copy()
        cand[free] = res.x[:nf]
        v2, m2, n2 = _viol(cand)
        ok = v2 < viol * (1 - 1e-9)
        if verbose:
            print(f'  [it {it}] viol {viol:.4e}->{v2:.4e} minT {m2:+.5f} '
                  f'n_neg {n2}  trust={trust:.4f} {"ACC" if ok else "rej"}',
                  flush=True)
        if ok:
            phi, viol, mn, nn = cand, v2, m2, n2
            trust = min(trust * 2, trust0)
        else:
            trust *= 0.5
            if trust < 1e-4:
                break
    print(f'  end: viol={viol:.4e} min_T={mn:+.5f} n_neg={nn} '
          f'({time.time() - t0:.1f}s)', flush=True)


def main():
    OUT = Path(__file__).parent / 'output'
    phi = np.load(OUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    mv = six_tet_min_volume_3d(phi)
    cz, cy, cx = np.unravel_index(int(mv.argmin()), mv.shape)
    print(f'worst cell ({cz},{cy},{cx}) min_T={float(mv.min()):+.5f}', flush=True)
    for half in (5, 8):
        z0, z1 = max(0, cz - half), min(phi.shape[1], cz + half)
        y0, y1 = max(0, cy - half), min(phi.shape[2], cy + half)
        x0, x1 = max(0, cx - half), min(phi.shape[3], cx + half)
        crop = phi[:, z0:z1, y0:y1, x0:x1].copy()
        print(f'\n=== crop half={half} {crop.shape[1:]} ===', flush=True)
        elastic_6tet(crop)


if __name__ == '__main__':
    main()
