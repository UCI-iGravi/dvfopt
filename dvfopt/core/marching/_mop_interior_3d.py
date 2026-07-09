"""Frozen-rim 3D-interior elastic-SLP mop of residual 6-tet folds.

The 2.5D marching sweep freezes one WHOLE slice and moves the other's
interior, so it cannot fix folds that need BOTH slices of a folded pair to
move. This mop crops a small box around each residual fold cluster, freezes
the ENTIRE rim (all six faces -> safe paste, no boundary-seam folds), and
frees the true 3D interior ``(1:D-1, 1:H-1, 1:W-1)`` of the box so both
slices move together. On a real volume it reduced the residual fold count
from 97 -> 33; the packaged ``active_band_alm_recovery_3d`` was tried for
this and FAILED on the same residuals -- this module is the working tool.

Field layout is ``(3, D, H, W)`` = ``[dz, dy, dx]``. This module operates on
``phi[1:3]`` -> a ``(2, D, H, W)`` box ``[dy, dx]``. **dz stays exactly 0
throughout** (the 2.5D precondition) -- ``phi[0]`` is never written.

Ported from ``research/strict_feasibility_3d/runners/_marching_mopup_3d_interior.py``;
all B0039 file-path / checkpoint / resume specifics removed and module
constants turned into parameters.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label
from scipy.optimize import linprog

from dvfopt.core.tri_primitives import tri_areas_flat
from dvfopt.jacobian.tetrahedron_sign import (
    build_tet_sparse_jac,
    six_tet_volumes_3d,
    tet_volumes_flat,
)


def _stack(box):
    """box (2,D,H,W)=[dy,dx] -> [dx_flat, dy_flat, dz=0] slice-major (DX_FIRST)."""
    dx = box[1].reshape(-1)
    dy = box[0].reshape(-1)
    return np.concatenate([dx, dy, np.zeros_like(dx)])


def _viol(box, D, H, W, thr3, thr2):
    """Exact hinge violation: inter-layer 6-tet term + intra-slice 2-tri term."""
    v3 = tet_volumes_flat(_stack(box), D, H, W)
    tot = float(np.maximum(0.0, thr3 - v3).sum())
    for s in range(D):
        # tri_areas_flat takes the DY_FIRST pack ([dy, dx]) -> concat order correct.
        t2 = tri_areas_flat(
            np.concatenate([box[0, s].ravel(), box[1, s].ravel()]), H, W
        )
        tot += float(np.maximum(0.0, thr2 - t2).sum())
    return tot


def _repair_box(box, thr3, thr2, mu, max_iters):
    """Elastic SLP over interior dy/dx of a (2,D,H,W) box; rim frozen."""
    _, D, H, W = box.shape
    N = D * H * W
    jac3 = build_tet_sparse_jac(D, H, W)

    # interior node ids (not on any of the 6 faces)
    ss, ii, jj = np.meshgrid(
        np.arange(1, D - 1),
        np.arange(1, H - 1),
        np.arange(1, W - 1),
        indexing="ij",
    )
    nodes = (ss * H * W + ii * W + jj).ravel()
    if nodes.size == 0:
        return box, _viol(box, D, H, W, thr3, thr2)
    free3 = np.concatenate([nodes, N + nodes])  # dx cols, dy cols
    nf = free3.size

    def anchor_vec(b):
        return np.concatenate(
            [b[1].reshape(-1)[nodes], b[0].reshape(-1)[nodes]]
        )

    def apply(b, v):
        out = b.copy()
        dx = out[1].reshape(-1)
        dx[nodes] = v[: nodes.size]
        dy = out[0].reshape(-1)
        dy[nodes] = v[nodes.size:]
        out[1] = dx.reshape(D, H, W)
        out[0] = dy.reshape(D, H, W)
        return out

    anchor = anchor_vec(box)
    cur = box.copy()
    viol = _viol(cur, D, H, W, thr3, thr2)
    trust = 0.5
    for _ in range(max_iters):
        if viol <= 1e-12:
            break
        pf = _stack(cur)
        T3 = tet_volumes_flat(pf, D, H, W)
        J3 = jac3(pf).tocsc()[:, free3].tocsr()
        a3 = np.where(T3 < thr3 + 0.5)[0]
        J3a = J3[a3]
        xf = anchor_vec(cur)

        # 6-tet-only elastic LP; the exact-violation acceptance (which
        # includes intra-slice 2-tri) rejects any step that breaks a 2D area.
        Ka = a3.size
        c_obj = np.concatenate([np.zeros(nf), np.ones(nf), mu * np.ones(Ka)])
        A1 = sp.hstack([sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        A2 = sp.hstack([-sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        E3 = sp.hstack([-J3a, sp.csr_matrix((Ka, nf)), -sp.eye(Ka)])
        A_ub = sp.vstack([A1, A2, E3]).tocsr()
        b_ub = np.concatenate([anchor, -anchor, -thr3 + T3[a3] - J3a @ xf])
        bounds = (
            [(float(xf[i] - trust), float(xf[i] + trust)) for i in range(nf)]
            + [(0.0, None)] * nf
            + [(0.0, None)] * Ka
        )
        res = linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )
        if not res.success:
            trust *= 0.5
            if trust < 1e-3:
                break
            continue
        cand = apply(cur, res.x[:nf])
        vnew = _viol(cand, D, H, W, thr3, thr2)
        if vnew < viol * (1 - 1e-9):
            cur, viol = cand, vnew
            trust = min(trust * 2, 0.5)
        else:
            trust *= 0.5
            if trust < 1e-3:
                break
    return cur, viol


def _pass(full, zpad, pad, thr3, thr2, mu, dil, max_iters):
    """One repair pass over all current fold clusters. Returns n_fixed."""
    V0 = six_tet_volumes_3d(full)
    bad = V0.min(axis=0) <= 0
    lab, _ = cc_label(binary_dilation(bad, iterations=dil))
    boxes = find_objects(lab)
    dyx = full[1:3]
    fixed = 0
    for bb in boxes:
        if bb is None:
            continue
        z0 = max(0, bb[0].start - zpad)
        z1 = min(full.shape[1], bb[0].stop + zpad)
        y0 = max(0, bb[1].start - pad)
        y1 = min(full.shape[2], bb[1].stop + pad + 1)
        x0 = max(0, bb[2].start - pad)
        x1 = min(full.shape[3], bb[2].stop + pad + 1)
        box = dyx[:, z0:z1 + 1, y0:y1, x0:x1].copy()
        D, H, W = box.shape[1:]
        if D < 3 or H < 3 or W < 3:
            continue
        v_before = _viol(box, D, H, W, thr3, thr2)
        if v_before <= 1e-12:
            continue
        box2, v_after = _repair_box(box, thr3, thr2, mu, max_iters)
        if v_after < v_before:
            dyx[:, z0:z1 + 1, y0:y1, x0:x1] = box2
            fixed += 1
    full[1:3] = dyx
    return fixed


def mop_interior_3d(phi, *, threshold=0.01, thr3=None, thr2=0.01, mu=1000.0,
                    pass_pads=((2, 4), (3, 6)), dil=1, max_iters=40, verbose=0):
    """Frozen-rim 3D-interior elastic-SLP mop of residual 6-tet folds.

    Crops a small box around each residual fold cluster, freezes the entire
    rim (all six faces, giving a seam-safe paste), and frees the true 3D
    interior ``(1:D-1, 1:H-1, 1:W-1)`` so both slices of a folded pair move
    together. Each cropped box is repaired with an elastic sequential-LP
    (inter-layer 6-tet linearized rows + intra-slice 2-tri exact-violation
    acceptance, trust-region). The 2.5D precondition ``dz == 0`` is preserved:
    only ``phi[1:3]`` (``[dy, dx]``) is ever written.

    Parameters
    ----------
    phi : ndarray, shape (3, D, H, W)
        Field ``[dz, dy, dx]``. ``phi[0]`` must be (and remains) zero.
        Operated on via a copy -- the caller's array is never mutated.
    threshold : float, default 0.01
        The strict feasibility target for 6-tet volumes.
    thr3 : float or None, default None
        6-tet feasibility target with margin. Defaults to
        ``threshold + 1e-4`` when None (strict target with a small margin).
    thr2 : float, default 0.01
        Intra-slice 2-tri area feasibility target.
    mu : float, default 1000.0
        Elastic-slack penalty weight in the per-box LP objective.
    pass_pads : sequence of (zpad, pad), default ((2, 4), (3, 6))
        Escalating ``(zpad, pad)`` schedule -- one outer pass per entry, each
        expanding cropped boxes by ``zpad`` on z and ``pad`` on y/x.
        **Kept deliberately short.** Measured on a real volume: pad=4 fixed
        most clusters, pad=6 fixed a couple more, and pads >= 9 cost
        exponentially more wall time for ZERO additional fixes (the remaining
        folds are at the geometric floor). Do NOT "helpfully" escalate this.
    dil : int, default 1
        Binary-dilation iterations used to merge nearby fold voxels into
        clusters before bounding-box extraction.
    max_iters : int, default 40
        Trust-region SLP iterations per box repair.
    verbose : int, default 0
        ``>= 1`` prints one bracketed line per pass.

    Returns
    -------
    phi_out : ndarray
        Corrected copy of ``phi``.
    info : dict
        Keys: ``n_neg_before``, ``n_neg_after``, ``min_T_after``,
        ``n_fixed``, ``passes`` (list of per-pass dicts), ``wall_s``.
    """
    if thr3 is None:
        thr3 = threshold + 1e-4

    phi_out = np.array(phi, dtype=np.float64, copy=True)

    t0 = time.time()
    V = six_tet_volumes_3d(phi_out)
    n_neg_before = int((V <= 0).sum())

    passes = []
    total_fixed = 0
    for i, (zpad, pad) in enumerate(pass_pads, start=1):
        before = int((six_tet_volumes_3d(phi_out) <= 0).sum())
        if before == 0:
            break
        fixed = _pass(phi_out, zpad, pad, thr3, thr2, mu, dil, max_iters)
        total_fixed += fixed
        V = six_tet_volumes_3d(phi_out)
        after = int((V <= 0).sum())
        mn = float(V.min())
        passes.append({
            "pass": i,
            "zpad": zpad,
            "pad": pad,
            "n_neg_before": before,
            "n_neg_after": after,
            "min_T": mn,
            "n_fixed": fixed,
        })
        if verbose >= 1:
            print(
                f"  [mop pass {i} zpad={zpad} pad={pad}] "
                f"n_neg {before}->{after} min_T={mn:.5f}",
                flush=True,
            )

    V = six_tet_volumes_3d(phi_out)
    info = {
        "n_neg_before": n_neg_before,
        "n_neg_after": int((V <= 0).sum()),
        "min_T_after": float(V.min()),
        "n_fixed": total_fixed,
        "passes": passes,
        "wall_s": time.time() - t0,
    }
    return phi_out, info
