"""Single-objective 2-tri untangler: distortion-over-chi energy + eps-continuation.

ONE smooth energy whose minimization from any (however tangled) start drives every
triangle to A >= t — the meshing literature's mature answer (Escobar et al. 2003
simultaneous untangling+smoothing; Garanzha et al. 2021 foldover-free maps):

    E_eps(phi) = sum_tau  ||J_tau||_F^2 / chi_eps(det J_tau - 2t)
                 + (w/2) ||phi - phi_anchor||^2,
    chi_eps(s) = (s + sqrt(s^2 + eps^2)) / 2   (evaluated stably),

with continuation on eps (adapted to the current worst inversion, annealed to 0).

Why this exact shape (both parts are load-bearing, empirically verified here):
- chi is finite/smooth for INVERTED triangles, so the optimizer can pass THROUGH
  folds — required: the fold-free set is disconnected around twisted cells.
- The ||J||^2 numerator is essential: the bare barrier sum(-log chi) is
  structurally degenerate — inverting one triangle costs only log|s| while
  inflating its neighbours gains -log(s_j) unboundedly, so pure-log runs collapse
  the mesh (observed: min(A-t) -0.04 -> -380 oscillations). Distortion-over-chi
  charges stretching quadratically and inversion ~||J||^2|s|/eps^2.

Per-triangle Jacobians are built so the identity field gives J = I exactly:
T1 = (TR, BL, BR): J1 = [BR-BL, BR-TR];  T2 = (TL, BL, TR): J2 = [TR-TL, BL-TL].
det J = 2 * (triangle area), consistent with dvfopt's tri primitives (asserted).

Usage:
    python benchmarks/chi_untangle.py --input <field.npy> [--w 1e-2] [--out out.npy]
    python benchmarks/chi_untangle.py --gradcheck
"""

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


def chi_stable(s, eps):
    """(chi, chi') evaluated without cancellation. chi'(s) = (r+s)/(2r), and for
    s < 0 use the conjugate forms chi = eps^2/(2(r-s)), chi' = eps^2/(2 r (r-s))."""
    r = np.sqrt(s * s + eps * eps)
    neg = s < 0
    val = np.where(neg, (eps * eps) / (2.0 * (r - s)), 0.5 * (s + r))
    dval = np.where(neg, (eps * eps) / (2.0 * r * (r - s)), 0.5 * (1.0 + s / r))
    return val, dval


def _positions(phi):
    H, W = phi.shape[1:]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return xx + phi[1], yy + phi[0]  # X, Y


def ref_inverses(phi_ref, det_floor=0.05):
    """Per-triangle R = J_ref^{-1} (2x2 entries as arrays) from the INPUT field;
    identity where the input triangle is degenerate/inverted (det_ref < det_floor).
    Referencing the input makes healthy regions sit at the energy minimum already
    (no drive to move) — the fidelity fix for the global distortion term."""
    H, W = phi_ref.shape[1:]
    X, Y = _positions(phi_ref)
    P = {
        "TL": (X[:-1, :-1], Y[:-1, :-1]),
        "TR": (X[:-1, 1:], Y[:-1, 1:]),
        "BL": (X[1:, :-1], Y[1:, :-1]),
        "BR": (X[1:, 1:], Y[1:, 1:]),
    }
    out = []
    for (a1, b1), (a2, b2) in ((("BR", "BL"), ("BR", "TR")), (("TR", "TL"), ("BL", "TL"))):
        c1x = P[a1][0] - P[b1][0]
        c1y = P[a1][1] - P[b1][1]
        c2x = P[a2][0] - P[b2][0]
        c2y = P[a2][1] - P[b2][1]
        det = c1x * c2y - c1y * c2x
        ok = det > det_floor
        d = np.where(ok, det, 1.0)
        # J = [c1 c2]; J^{-1} = (1/det) [[c2y, -c2x], [-c1y, c1x]]
        r11 = np.where(ok, c2y / d, 1.0)
        r12 = np.where(ok, -c2x / d, 0.0)
        r21 = np.where(ok, -c1y / d, 0.0)
        r22 = np.where(ok, c1x / d, 1.0)
        out.append((r11, r12, r21, r22))
    return out


def energy_grad(phi, threshold, eps, anchor, w, refs=None, beta=0.0):
    """E_eps and dE/dphi (same (2,H,W) layout). Vectorized over the cell grid.
    refs: per-triangle R = J_ref^{-1} from ref_inverses (None = identity reference)."""
    H, W = phi.shape[1:]
    X, Y = _positions(phi)
    TLx, TLy = X[:-1, :-1], Y[:-1, :-1]
    TRx, TRy = X[:-1, 1:], Y[:-1, 1:]
    BLx, BLy = X[1:, :-1], Y[1:, :-1]
    BRx, BRy = X[1:, 1:], Y[1:, 1:]
    s0 = 2.0 * threshold
    gX = np.zeros_like(X)
    gY = np.zeros_like(Y)
    E = 0.0
    # triangle -> (corner names, J columns c1, c2 with identity = e1, e2)
    tris = (
        # T1 = (TR, BL, BR): c1 = BR-BL, c2 = BR-TR
        (("BR", "BL"), ("BR", "TR")),
        # T2 = (TL, BL, TR): c1 = TR-TL, c2 = BL-TL
        (("TR", "TL"), ("BL", "TL")),
    )
    P = {
        "TL": (TLx, TLy),
        "TR": (TRx, TRy),
        "BL": (BLx, BLy),
        "BR": (BRx, BRy),
    }
    SL = {  # scatter slices into the (H, W) grids
        "TL": (slice(0, H - 1), slice(0, W - 1)),
        "TR": (slice(0, H - 1), slice(1, W)),
        "BL": (slice(1, H), slice(0, W - 1)),
        "BR": (slice(1, H), slice(1, W)),
    }
    for ti, ((a1, b1), (a2, b2)) in enumerate(tris):
        c1x = P[a1][0] - P[b1][0]
        c1y = P[a1][1] - P[b1][1]
        c2x = P[a2][0] - P[b2][0]
        c2y = P[a2][1] - P[b2][1]
        det = c1x * c2y - c1y * c2x
        if refs is None:
            m11, m21, m12, m22 = c1x, c1y, c2x, c2y  # J R with R = I
            detR = 1.0
        else:
            r11, r12, r21, r22 = refs[ti]
            # M = J R with J = [c1 c2] (columns): M = [[c1x r11 + c2x r21, c1x r12 + c2x r22],
            #                                          [c1y r11 + c2y r21, c1y r12 + c2y r22]]
            m11 = c1x * r11 + c2x * r21
            m21 = c1y * r11 + c2y * r21
            m12 = c1x * r12 + c2x * r22
            m22 = c1y * r12 + c2y * r22
            detR = r11 * r22 - r12 * r21
        fro = m11 * m11 + m21 * m21 + m12 * m12 + m22 * m22
        fro = fro + beta  # +beta closes the point-collapse escape (fro->0 at A=0 while infeasible)
        val, dval = chi_stable(det - s0, eps)
        E += float((fro / val).sum())
        # d(fro)/dJ = 2 M R^T (chain through M = J R); identity R keeps old formulas
        if refs is None:
            f1x, f1y, f2x, f2y = 2.0 * c1x, 2.0 * c1y, 2.0 * c2x, 2.0 * c2y
        else:
            f1x = 2.0 * (m11 * r11 + m12 * r12)
            f1y = 2.0 * (m21 * r11 + m22 * r12)
            f2x = 2.0 * (m11 * r21 + m12 * r22)
            f2y = 2.0 * (m21 * r21 + m22 * r22)
        del detR
        k = -(fro * dval) / (val * val)
        # ddet/dc1 = ( c2y, -c2x); ddet/dc2 = (-c1y,  c1x)  (det of J, not M)
        d1x = f1x / val + k * c2y
        d1y = f1y / val - k * c2x
        d2x = f2x / val - k * c1y
        d2y = f2y / val + k * c1x
        gX[SL[a1]] += d1x
        gY[SL[a1]] += d1y
        gX[SL[b1]] -= d1x
        gY[SL[b1]] -= d1y
        gX[SL[a2]] += d2x
        gY[SL[a2]] += d2y
        gX[SL[b2]] -= d2x
        gY[SL[b2]] -= d2y
    d = phi - anchor
    E += 0.5 * w * float((d * d).sum())
    grad = np.stack([gY, gX]) + w * d  # phi layout is [dy, dx]
    return E, grad


def min_area_margin(phi, threshold):
    """min over triangles of (det - 2t)/2 = A - t, plus fold counts."""
    X, Y = _positions(phi)
    TL = (X[:-1, :-1], Y[:-1, :-1])
    TR = (X[:-1, 1:], Y[:-1, 1:])
    BL = (X[1:, :-1], Y[1:, :-1])
    BR = (X[1:, 1:], Y[1:, 1:])
    d1 = (BR[0] - BL[0]) * (BR[1] - TR[1]) - (BR[1] - BL[1]) * (BR[0] - TR[0])
    d2 = (TR[0] - TL[0]) * (BL[1] - TL[1]) - (TR[1] - TL[1]) * (BL[0] - TL[0])
    m = np.minimum(d1, d2) / 2.0
    return float(m.min() - threshold), int((m < threshold).sum()), int((m < 0).sum())


def untangle(
    phi_in,
    threshold=0.01,
    w=1e-2,
    eps_min=1e-4,
    max_stages=30,
    maxiter=400,
    ref="identity",
    beta=0.0,
    w_final=None,
    log=print,
):
    phi0 = np.asarray(phi_in, dtype=np.float64).copy()
    anchor = phi0.copy()
    refs = ref_inverses(phi0) if ref == "input" else None
    shape = phi0.shape
    x = phi0.ravel().copy()
    t0 = time.perf_counter()
    eps = 1.0
    x_safe = None
    smin, nfold, nneg = min_area_margin(phi0, threshold)
    for stage in range(max_stages):
        if smin > 0:
            x_safe = x.copy()
            if eps <= eps_min * 1.01:
                if w_final is None or w >= w_final * 0.999:
                    break
                # feasible at eps_min: continuation on the anchor weight — walk toward
                # the L2 projection onto the feasible set with chi as the barrier
                w = min(w * 4.0, w_final)
            else:
                eps = max(eps_min, eps / 4.0)
        else:
            eps = min(max(eps_min, abs(smin) + 1e-3), 10.0)

        def fg(z, _eps=eps, _w=w):
            p = z.reshape(shape)
            e, g = energy_grad(p, threshold, _eps, anchor, _w, refs=refs, beta=beta)
            return e, g.ravel()

        r = minimize(fg, x, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20})
        x = r.x
        smin, nfold, nneg = min_area_margin(x.reshape(shape), threshold)
        if smin <= 0 and w_final is not None and x_safe is not None:
            x = x_safe
            smin, nfold, nneg = min_area_margin(x.reshape(shape), threshold)
            log(f"  w-anneal lost feasibility at w={w:.3g}; reverted to last feasible state")
            break
        log(
            f"  stage {stage:2d}: eps={eps:.4g} w={w:.3g} nit={r.nit:4d} E={r.fun:.4e} -> "
            f"min(A-t)={smin:+.5f} folds={nfold} neg={nneg} wall={time.perf_counter() - t0:.0f}s"
        )
    out = x.reshape(shape)
    d = out - anchor
    info = {
        "min_margin": smin,
        "folds": nfold,
        "neg": nneg,
        "l2_move": float(np.linalg.norm(d)),
        "l1_move": float(np.abs(d).sum()),
        "wall_s": time.perf_counter() - t0,
    }
    return out, info


def _gradcheck():
    from dvfopt.constraints import TriConstraint2D

    rng = np.random.default_rng(0)
    phi = np.stack([rng.normal(0, 0.6, (7, 8)), rng.normal(0, 0.6, (7, 8))])
    # sign/scale consistency: our det/2 must equal dvfopt's triangle areas
    c = TriConstraint2D(shape=(7, 8))
    vals = np.asarray(c.values(c.flatten(phi)))
    m_ref = np.minimum(vals[: 6 * 7], vals[6 * 7 :]).min()
    m_ours = min_area_margin(phi, 0.0)[0]
    assert abs(m_ref - m_ours) < 1e-12, (m_ref, m_ours)
    # identity field: J = I, det = 1, fro = 2 -> E = n_tri * 2/chi(1-2t)
    ident = np.zeros((2, 7, 8))
    e, g = energy_grad(ident, 0.01, 0.05, ident, 0.0)
    ntri = 2 * 6 * 7
    val, _ = chi_stable(np.array([1.0 - 0.02]), 0.05)
    assert abs(e - ntri * 2.0 / val[0]) < 1e-9
    assert np.abs(g[:, 1:-1, 1:-1]).max() < 1e-9  # interior gradient zero at identity
    # finite-difference gradient
    anchor = phi + rng.normal(0, 0.1, phi.shape)
    e0, g = energy_grad(phi, 0.01, 0.05, anchor, 1e-2)
    num = np.zeros_like(g)
    h = 1e-6
    for idx in np.ndindex(phi.shape):
        pp = phi.copy()
        pp[idx] += h
        num[idx] = (energy_grad(pp, 0.01, 0.05, anchor, 1e-2)[0] - e0) / h
    err = np.abs(g - num).max() / max(1.0, np.abs(g).max())
    assert err < 1e-4, f"gradcheck rel err {err:.2e}"
    refs = ref_inverses(anchor)
    e0, g = energy_grad(phi, 0.01, 0.05, anchor, 1e-2, refs=refs)
    num = np.zeros_like(g)
    for idx in np.ndindex(phi.shape):
        pp = phi.copy()
        pp[idx] += h
        num[idx] = (energy_grad(pp, 0.01, 0.05, anchor, 1e-2, refs=refs)[0] - e0) / h
    err2 = np.abs(g - num).max() / max(1.0, np.abs(g).max())
    assert err2 < 1e-4, f"input-ref gradcheck rel err {err2:.2e}"
    e0, g = energy_grad(phi, 0.01, 0.05, anchor, 1e-2, beta=0.7)
    num = np.zeros_like(g)
    for idx in np.ndindex(phi.shape):
        pp = phi.copy()
        pp[idx] += h
        num[idx] = (energy_grad(pp, 0.01, 0.05, anchor, 1e-2, beta=0.7)[0] - e0) / h
    err3 = np.abs(g - num).max() / max(1.0, np.abs(g).max())
    assert err3 < 1e-4, f"beta gradcheck rel err {err3:.2e}"
    print(f"beta gradcheck OK ({err3:.2e})")
    print(f"gradcheck OK (identity {err:.2e}, input-ref {err2:.2e}); sign/identity checks OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--w", type=float, default=1e-2, help="fidelity (anchor) weight")
    ap.add_argument("--maxiter", type=int, default=400, help="L-BFGS iterations per stage")
    ap.add_argument("--ref", default="identity", choices=["identity", "input"])
    ap.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="numerator = ||J||^2 + beta (beta>0 closes the collapse valley)",
    )
    ap.add_argument(
        "--w-final",
        type=float,
        default=None,
        help="anneal w up to this after feasibility (fidelity recovery)",
    )
    ap.add_argument("--gradcheck", action="store_true")
    a = ap.parse_args()
    if a.gradcheck:
        _gradcheck()
        return
    phi = np.load(a.input).astype(np.float64)
    if phi.ndim == 4:
        phi = phi[-2:, 0]
    print(f"input {a.input} shape {phi.shape} w={a.w}")
    out, info = untangle(
        phi,
        threshold=a.threshold,
        w=a.w,
        maxiter=a.maxiter,
        ref=a.ref,
        beta=a.beta,
        w_final=a.w_final,
    )
    print(
        f"RESULT: folds={info['folds']} neg={info['neg']} min_margin={info['min_margin']:+.5f} "
        f"L2move={info['l2_move']:.1f} L1={info['l1_move']:.0f} {info['wall_s']:.0f}s"
    )
    if a.out:
        np.save(a.out, out)


if __name__ == "__main__":
    main()
