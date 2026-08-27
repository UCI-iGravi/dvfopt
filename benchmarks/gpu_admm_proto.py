#!/usr/bin/env python
"""Batched GPU ADMM (OSQP-style) for the windowed-isqp engine's window QPs.

RESEARCH PROTOTYPE.  Lives in benchmarks/, imports nothing from dvfopt, ships
nothing to the library.  Question it answers: can K independent window QPs
solved *simultaneously* on one GPU beat the CPU process pool's measured
memory-bandwidth ceiling (2.6x single-core at 4 workers)?

Problem form (OSQP convention, P upper-triangular CSC on disk):

    min 1/2 x'Px + q'x    s.t.   l <= A x <= u

Two linear-solve strategies for the ADMM's reduced KKT system
M = P + sigma I + rho A'A:

  indirect  Jacobi-preconditioned CG, warm-started across ADMM iterations,
            batched over K QPs (cuOSQP's approach).  Matvec = block-diagonal
            cusparse SpMV, so a whole batch is one kernel launch per operator.
  direct    per-QP scipy splu of the same reduced matrix on the CPU, then the
            per-iteration triangular solves batched on the GPU via
            cupyx.scipy.sparse.linalg.spsolve_triangular.

Data-specific simplification (asserted at load, not assumed silently):
P is DIAGONAL for every captured window QP, so it is carried as a length-n
vector.  A non-diagonal P would need a batched SpMV for P too; nothing else
changes.

Usage:
    python benchmarks/gpu_admm_proto.py --qp-dir <dir> [--batches 8,32,64]
    python benchmarks/gpu_admm_proto.py --self-check     # CPU-only correctness test
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# algorithm constants (OSQP defaults)
# ---------------------------------------------------------------------------
SIGMA = 1e-6
RHO0 = 0.1
ALPHA = 1.6
RHO_EQ_FACTOR = 1e3
EPS_ABS = 1e-3
EPS_REL = 1e-3
MAX_ITER = 4000
CHECK_EVERY = 25
RUIZ_ITERS = 10
PCG_MAX = 50
PCG_TOL_FRAC = 0.15  # cuOSQP's cg_tol_fraction
PCG_TOL_REDUCTION = 5.0  # cuOSQP's cg_tol_reduction, applied at every termination check
PCG_TOL_MIN = 1e-7  # cuOSQP's cg_tol_min (absolute, on ||r||_2 of the reduced system)


# ---------------------------------------------------------------------------
# loading + scaling  (CPU, scipy)
# ---------------------------------------------------------------------------
@dataclass
class QP:
    pdiag: np.ndarray  # (n,) -- P is diagonal for this data
    q: np.ndarray
    A: sp.csr_matrix
    lo: np.ndarray
    up: np.ndarray
    x_ref: np.ndarray
    y_ref: np.ndarray
    osqp_time: float = 0.0
    osqp_iter: int = 0
    # Ruiz factors, filled in by scale()
    D: np.ndarray = field(default_factory=lambda: np.empty(0))
    E: np.ndarray = field(default_factory=lambda: np.empty(0))
    c: float = 1.0


def load_qps(qp_dir: Path, count: int | None = None) -> list[QP]:
    idx = sorted(int(p.stem[1:]) for p in qp_dir.glob("P*.npz"))
    if count is not None:
        idx = idx[:count]
    out = []
    for i in idx:
        P = sp.load_npz(qp_dir / f"P{i}.npz").tocoo()
        assert (P.row == P.col).all(), f"P{i} is not diagonal; see module docstring"
        A = sp.load_npz(qp_dir / f"A{i}.npz").tocsr()
        # allow_pickle: these captures are written by our own qp-capture hook
        # (the 0-d `status`/`settings` string fields need it); not untrusted input.
        v = np.load(qp_dir / f"v{i}.npz", allow_pickle=True)
        out.append(
            QP(
                pdiag=np.asarray(P.tocsr().diagonal(), dtype=np.float64),
                q=v["q"].astype(np.float64),
                A=A,
                lo=v["l"].astype(np.float64),
                up=v["u"].astype(np.float64),
                x_ref=v["x"].astype(np.float64),
                y_ref=v["y"].astype(np.float64),
                osqp_time=float(v["osqp_time"]),
                osqp_iter=int(v["osqp_iter"]),
            )
        )
    return out


def _col_inf(A: sp.csr_matrix) -> np.ndarray:
    return np.asarray(abs(A).max(axis=0).todense()).ravel()


def _row_inf(A: sp.csr_matrix) -> np.ndarray:
    return np.asarray(abs(A).max(axis=1).todense()).ravel()


def scale(qp: QP, iters: int = RUIZ_ITERS) -> QP:
    """OSQP's modified Ruiz equilibration.  Returns a NEW, scaled QP.

    x = D x~,  y = E y~ / c.  Scaled data: P~ = c D P D, q~ = c D q,
    A~ = E A D, l~ = E l, u~ = E u.
    """
    n, m = qp.pdiag.size, qp.A.shape[0]
    D, E, c = np.ones(n), np.ones(m), 1.0
    pd, q, A = qp.pdiag.copy(), qp.q.copy(), qp.A.copy()
    for _ in range(iters):
        delta = 1.0 / np.sqrt(np.maximum(np.maximum(np.abs(pd), _col_inf(A)), 1e-10))
        eps = 1.0 / np.sqrt(np.maximum(_row_inf(A), 1e-10))
        pd = delta * pd * delta
        q = q * delta
        A = (sp.diags(eps) @ A @ sp.diags(delta)).tocsr()
        D *= delta
        E *= eps
        gamma = 1.0 / max(max(float(np.mean(np.abs(pd))), float(np.max(np.abs(q)))), 1e-10)
        gamma = float(np.clip(gamma, 1e-6, 1e6))
        pd *= gamma
        q *= gamma
        c *= gamma
    return QP(
        pdiag=pd,
        q=q,
        A=A.tocsr(),
        lo=E * qp.lo,
        up=E * qp.up,
        x_ref=qp.x_ref,
        y_ref=qp.y_ref,
        osqp_time=qp.osqp_time,
        osqp_iter=qp.osqp_iter,
        D=D,
        E=E,
        c=c,
    )


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------
def _block_diag_csr(mats: list[sp.csr_matrix]) -> sp.csr_matrix:
    """block_diag for equal-*shape* CSR blocks (nnz may differ), by index arithmetic."""
    r, c = mats[0].shape
    assert all(m.shape == (r, c) for m in mats), "ragged block shapes"
    offs = np.cumsum([0] + [m.nnz for m in mats])
    indices = np.concatenate([m.indices + i * c for i, m in enumerate(mats)])
    data = np.concatenate([m.data for m in mats])
    indptr = np.concatenate([[0]] + [m.indptr[1:] + offs[i] for i, m in enumerate(mats)])
    return sp.csr_matrix(
        (data, indices.astype(np.int32), indptr.astype(np.int32)),
        shape=(len(mats) * r, len(mats) * c),
    )


class Batch:
    """K padded, Ruiz-scaled QPs as block-diagonal operators on the target device."""

    def __init__(self, qps: list[QP], xp, cusp, m_force: int | None = None):
        self.xp = xp
        self.K = K = len(qps)
        self.n = n = qps[0].pdiag.size
        # m_force keeps the padded row count constant across compaction rebuilds
        self.m = m = m_force or max(q.A.shape[0] for q in qps)

        As, ATs = [], []
        pd = np.zeros((K, n))
        qv = np.zeros((K, n))
        lo = np.full((K, m), -np.inf)
        up = np.full((K, m), np.inf)
        Dv = np.zeros((K, n))
        Ev = np.ones((K, m))
        cv = np.zeros((K, 1))
        for i, qp in enumerate(qps):
            mi = qp.A.shape[0]
            # pad with all-zero rows, l=-inf/u=+inf: inert in the projection and in A'A
            Ai = sp.vstack([qp.A, sp.csr_matrix((m - mi, n))], format="csr") if mi < m else qp.A
            As.append(Ai)
            ATs.append(Ai.T.tocsr())
            pd[i] = qp.pdiag
            qv[i] = qp.q
            lo[i, :mi] = qp.lo
            up[i, :mi] = qp.up
            Dv[i] = qp.D
            Ev[i, :mi] = qp.E
            cv[i, 0] = qp.c

        self.A = cusp.csr_matrix(_block_diag_csr(As))
        self.AT = cusp.csr_matrix(_block_diag_csr(ATs))
        self.pd = xp.asarray(pd)
        self.q = xp.asarray(qv)
        self.lo = xp.asarray(lo)
        self.up = xp.asarray(up)
        self.Dsc = xp.asarray(Dv)
        self.Esc = xp.asarray(Ev)
        self.Dinv = 1.0 / self.Dsc
        self.Einv = 1.0 / self.Esc
        self.c = xp.asarray(cv)
        # rho row multiplier: OSQP bumps equality rows (l == u) by 1e3
        self.eqfac = xp.where(xp.asarray(lo == up), RHO_EQ_FACTOR, 1.0)
        # column sums of A^2 weighted by eqfac -> Jacobi preconditioner
        AT2 = self.AT.copy()
        AT2.data = AT2.data**2
        self.asq_eq = (AT2 @ self.eqfac.ravel()).reshape(K, n)
        self.op_bytes = (
            self.A.data.nbytes
            + self.A.indices.nbytes
            + self.A.indptr.nbytes
            + self.AT.data.nbytes
            + self.AT.indices.nbytes
            + self.AT.indptr.nbytes
        )

    def mv(self, X):  # (K,n) -> (K,m)
        return (self.A @ X.ravel()).reshape(self.K, self.m)

    def rmv(self, Y):  # (K,m) -> (K,n)
        return (self.AT @ Y.ravel()).reshape(self.K, self.n)


# ---------------------------------------------------------------------------
# indirect solve: batched Jacobi-PCG on P + sigma I + rho A'A
# ---------------------------------------------------------------------------
def _pcg(b, x, batch, rho_vec, minv, tol, xp, maxit=PCG_MAX):
    """Batched PCG.  tol is (K,1), absolute on ||r||_2.  Runs until *every* QP meets it."""

    def op(v):
        return (batch.pd + SIGMA) * v + batch.rmv(rho_vec * batch.mv(v))

    r = b - op(x)
    z = minv * r
    p = z.copy()
    rz = (r * z).sum(axis=1, keepdims=True)
    for it in range(1, maxit + 1):
        if bool((xp.linalg.norm(r, axis=1, keepdims=True) <= tol).all()):
            return x, it - 1
        Ap = op(p)
        pAp = (p * Ap).sum(axis=1, keepdims=True)
        alpha = rz / xp.maximum(pAp, 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        z = minv * r
        rz_new = (r * z).sum(axis=1, keepdims=True)
        p = z + (rz_new / xp.maximum(rz, 1e-300)) * p
        rz = rz_new
    return x, maxit


def admm(
    make_batch,
    ids,
    xp,
    x0=None,
    y0=None,
    max_iter=MAX_ITER,
    eps_abs=EPS_ABS,
    eps_rel=EPS_REL,
    adaptive_rho=True,
    compact=True,
):
    """Batched OSQP-style ADMM over the QPs named by `ids`.

    `make_batch(ids) -> Batch` is called once up front and again on every
    compaction.  Compaction matters: without it the whole batch runs until the
    *slowest* QP converges, so per-QP cost grows with K on a heterogeneous set.
    Converged QPs are retired (their solution frozen) and the block-diagonal
    operator is rebuilt from the survivors.

    Returns (x, y) UNSCALED at full batch width, plus iteration stats.
    """
    ids = np.asarray(ids)
    K0 = ids.size
    batch = make_batch(ids)
    n, m = batch.n, batch.m
    K = K0
    active = np.arange(K0)  # positions in the original batch still being solved
    X_out = np.zeros((K0, n))
    Y_out = np.zeros((K0, m))
    x = xp.zeros((K, n)) if x0 is None else xp.asarray(x0).copy()
    y = xp.zeros((K, m)) if y0 is None else xp.asarray(y0).copy()
    z = xp.clip(batch.mv(x), batch.lo, batch.up)
    rho = xp.full((K, 1), RHO0)
    rho_vec = rho * batch.eqfac
    minv = 1.0 / (batch.pd + SIGMA + rho * batch.asq_eq)
    xt = x.copy()
    # PCG stop: absolute (cuOSQP's rule, set at each termination check) capped by a
    # relative one so the first CHECK_EVERY iterations have a criterion at all.
    tol_abs = xp.full((K, 1), np.inf)
    conv_iter = np.full(K0, -1)
    pcg_total = 0
    n_rebuild = 0
    it = 0
    for it in range(1, max_iter + 1):
        rhs = SIGMA * x - batch.q + batch.rmv(rho_vec * z - y)
        tol = xp.minimum(PCG_TOL_FRAC * xp.linalg.norm(rhs, axis=1, keepdims=True), tol_abs)
        xt, npcg = _pcg(rhs, xt, batch, rho_vec, minv, tol, xp)
        pcg_total += npcg
        zt = batch.mv(xt)
        x = ALPHA * xt + (1.0 - ALPHA) * x
        zt_a = ALPHA * zt + (1.0 - ALPHA) * z
        z_new = xp.clip(zt_a + y / rho_vec, batch.lo, batch.up)
        y = y + rho_vec * (zt_a - z_new)
        z = z_new

        if it % CHECK_EVERY == 0 or it == max_iter:
            Ax = batch.mv(x)
            ATy = batch.rmv(y)
            Px = batch.pd * x
            eAx = xp.abs(batch.Einv * Ax).max(axis=1, keepdims=True)
            ez = xp.abs(batch.Einv * z).max(axis=1, keepdims=True)
            r_prim = xp.abs(batch.Einv * (Ax - z)).max(axis=1, keepdims=True)
            dPx = xp.abs(batch.Dinv * Px).max(axis=1, keepdims=True) / batch.c
            dq = xp.abs(batch.Dinv * batch.q).max(axis=1, keepdims=True) / batch.c
            dATy = xp.abs(batch.Dinv * ATy).max(axis=1, keepdims=True) / batch.c
            r_dual = xp.abs(batch.Dinv * (Px + batch.q + ATy)).max(axis=1, keepdims=True) / batch.c
            s_prim = xp.maximum(eAx, ez)
            s_dual = xp.maximum(xp.maximum(dPx, dq), dATy)
            e_prim = eps_abs + eps_rel * s_prim
            e_dual = eps_abs + eps_rel * s_dual
            ok = (r_prim <= e_prim) & (r_dual <= e_dual)
            ok_h = np.asarray(xp.asnumpy(ok) if hasattr(xp, "asnumpy") else ok).ravel()
            if ok_h.any():
                xu = x * batch.Dsc
                yu = y * batch.Esc / batch.c
                xu = xp.asnumpy(xu) if hasattr(xp, "asnumpy") else np.asarray(xu)
                yu = xp.asnumpy(yu) if hasattr(xp, "asnumpy") else np.asarray(yu)
                done = active[ok_h]
                X_out[done] = xu[ok_h]
                Y_out[done] = yu[ok_h]
                conv_iter[done] = np.where(conv_iter[done] < 0, it, conv_iter[done])
            if ok_h.all():
                break
            if compact and ok_h.any():
                sel = ~ok_h
                active = active[sel]
                K = active.size
                batch = make_batch(ids[active])
                n_rebuild += 1
                x, xt = x[sel], xt[sel]
                y, z = y[sel], z[sel]
                rho, tol_abs = rho[sel], tol_abs[sel]
                r_prim, r_dual = r_prim[sel], r_dual[sel]
                e_prim, e_dual = e_prim[sel], e_dual[sel]
                s_prim, s_dual = s_prim[sel], s_dual[sel]
                rho_vec = rho * batch.eqfac
                minv = 1.0 / (batch.pd + SIGMA + rho * batch.asq_eq)
            # PCG tolerance: cuOSQP's rule, monotone.  A *fixed* loose tolerance
            # stalls ADMM outright (verified by --self-check); a fixed tight one
            # burns ~25 CG iterations per ADMM iteration.
            tol_abs = xp.maximum(
                xp.minimum(PCG_TOL_FRAC * xp.sqrt(r_prim * r_dual), tol_abs / PCG_TOL_REDUCTION),
                PCG_TOL_MIN,
            )
            if adaptive_rho:
                ratio = xp.sqrt(
                    (r_prim / xp.maximum(s_prim, 1e-300))
                    / xp.maximum(r_dual / xp.maximum(s_dual, 1e-300), 1e-300)
                )
                upd = (ratio > 5.0) | (ratio < 0.2)
                rho = xp.where(upd, xp.clip(rho * ratio, 1e-6, 1e6), rho)
                rho_vec = rho * batch.eqfac
                minv = 1.0 / (batch.pd + SIGMA + rho * batch.asq_eq)

    # whatever is still active at max_iter: report its last iterate
    if active.size:
        xu = x * batch.Dsc
        yu = y * batch.Esc / batch.c
        X_out[active] = xp.asnumpy(xu) if hasattr(xp, "asnumpy") else np.asarray(xu)
        Y_out[active] = xp.asnumpy(yu) if hasattr(xp, "asnumpy") else np.asarray(yu)
    return X_out, Y_out, it, conv_iter, pcg_total, n_rebuild


# ---------------------------------------------------------------------------
# direct-hybrid probe: CPU splu + batched GPU triangular solves
# ---------------------------------------------------------------------------
def direct_probe(qps: list[QP], K: int, xp, cusp, cuspl):
    import scipy.sparse.linalg as spl

    t0 = time.perf_counter()
    Ls, Us, nnz = [], [], 0
    for qp in [qps[i % len(qps)] for i in range(K)]:  # tile with copies, like the indirect path
        M = (sp.diags(qp.pdiag + SIGMA) + RHO0 * (qp.A.T @ qp.A)).tocsc()
        lu = spl.splu(M, permc_spec="COLAMD")
        Ls.append(lu.L.tocsr())
        Us.append(lu.U.tocsr())
        nnz += lu.L.nnz + lu.U.nnz
    t_fact = time.perf_counter() - t0

    gL = cusp.csr_matrix(sp.block_diag(Ls, format="csr"))
    gU = cusp.csr_matrix(sp.block_diag(Us, format="csr"))
    b = xp.random.rand(gL.shape[0])
    for _ in range(2):  # warm up JIT / cusparse analysis
        cuspl.spsolve_triangular(gU, cuspl.spsolve_triangular(gL, b, lower=True), lower=False)
    xp.cuda.Stream.null.synchronize()
    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        cuspl.spsolve_triangular(gU, cuspl.spsolve_triangular(gL, b, lower=True), lower=False)
    xp.cuda.Stream.null.synchronize()
    t_solve = (time.perf_counter() - t0) / reps
    return {
        "K": K,
        "fact_s_per_qp": t_fact / K,
        "lu_nnz_per_qp": nnz / K,
        "tri_solve_s_per_qp": t_solve / K,
        "tri_solve_s_batch": t_solve,
        "gpu_bytes_per_qp": (
            gL.data.nbytes + gL.indices.nbytes + gU.data.nbytes + gU.indices.nbytes
        )
        / K,
    }


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
def check(qps_raw: list[QP], X):
    rows = []
    for i, qp in enumerate(qps_raw):
        x = np.asarray(X[i])[: qp.pdiag.size]
        Ax = qp.A @ x
        viol = float(np.maximum(np.maximum(qp.lo - Ax, Ax - qp.up), 0.0).max())
        obj = 0.5 * float(x @ (qp.pdiag * x)) + float(qp.q @ x)
        xr = qp.x_ref
        Axr = qp.A @ xr
        obj_ref = 0.5 * float(xr @ (qp.pdiag * xr)) + float(qp.q @ xr)
        viol_ref = float(np.maximum(np.maximum(qp.lo - Axr, Axr - qp.up), 0.0).max())
        rows.append(
            {
                "viol": viol,
                "viol_osqp": viol_ref,
                "obj": obj,
                "obj_osqp": obj_ref,
                "obj_rel": (obj - obj_ref) / max(abs(obj_ref), 1.0),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CPU baseline
# ---------------------------------------------------------------------------
def cpu_osqp(qps: list[QP], mode: str):
    """mode: 'cold' | 'warm_self' (from the captured solution) | 'warm_prev' (previous QP)."""
    import osqp

    times, iters, sols = [], [], []
    for i, qp in enumerate(qps):
        P = sp.triu(sp.diags(qp.pdiag)).tocsc()
        A = qp.A.tocsc()
        t0 = time.perf_counter()
        s = osqp.OSQP()
        s.setup(
            P=P,
            q=qp.q,
            A=A,
            l=qp.lo,
            u=qp.up,
            verbose=False,
            eps_abs=EPS_ABS,
            eps_rel=EPS_REL,
            max_iter=MAX_ITER,
            polishing=False,
            warm_starting=(mode != "cold"),
        )
        if mode == "warm_self":
            s.warm_start(x=qp.x_ref, y=qp.y_ref)
        elif mode == "warm_prev":
            src = qps[i - 1] if i else qps[-1]
            if src.A.shape[0] == qp.A.shape[0]:
                s.warm_start(x=src.x_ref, y=src.y_ref)
            else:
                s.warm_start(x=src.x_ref)
        r = s.solve()
        times.append(time.perf_counter() - t0)
        iters.append(int(r.info.iter))
        sols.append(np.asarray(r.x))
    return times, iters, sols


# ---------------------------------------------------------------------------
class _NumpySparse:
    """Minimal cupyx.scipy.sparse stand-in so the CPU self-check needs no GPU."""

    @staticmethod
    def csr_matrix(m):
        return m


def self_check():
    """Tiny QP with a known closed-form solution: min 1/2||x||^2 - c'x  s.t.  x >= 1."""
    n = 6
    c = np.array([2.0, 3.0, -1.0, 0.5, 4.0, 0.0])
    qp = QP(
        pdiag=np.ones(n),
        q=-c,
        A=sp.eye(n, format="csr"),
        lo=np.ones(n),
        up=np.full(n, np.inf),
        x_ref=np.maximum(c, 1.0),
        y_ref=np.zeros(n),
    )
    pool = [scale(qp), scale(qp), scale(qp)]
    X, Y, it, conv, _, _ = admm(
        lambda ids: Batch([pool[i] for i in ids], np, _NumpySparse(), m_force=n),
        np.arange(len(pool)),
        np,
        max_iter=2000,
    )
    want = np.maximum(c, 1.0)
    err = float(np.abs(np.asarray(X)[:, :n] - want).max())
    assert err < 5e-3, f"self-check FAILED: max|x-x*|={err:.3e} after {it} iters (conv={conv})"
    assert (conv > 0).all(), f"self-check FAILED: not all QPs converged ({conv})"
    print(f"self-check OK: max|x - x*| = {err:.2e}, converged at iters {conv.tolist()}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qp-dir", type=Path, default=Path("benchmarks/output/qp_capture"))
    ap.add_argument("--batches", default="8,32,64")
    ap.add_argument("--strategy", default="both", choices=["indirect", "direct", "both"])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true", help="skip the CPU OSQP baselines")
    ap.add_argument(
        "--compaction-sweep",
        action="store_true",
        help="also time each batch WITHOUT retiring converged QPs",
    )
    ap.add_argument("--regimes", default="cold,warm_self,warm_prev")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/output/gpu_admm_results.json"))
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    import cupy as cp
    import cupyx.scipy.sparse as cusp
    import cupyx.scipy.sparse.linalg as cuspl

    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    raw = load_qps(args.qp_dir)
    print(
        f"loaded {len(raw)} QPs  n={raw[0].pdiag.size}  m in {sorted({q.A.shape[0] for q in raw})}",
        flush=True,
    )

    results: dict = {"n": int(raw[0].pdiag.size), "n_qps": len(raw)}

    # ---- CPU baselines -----------------------------------------------------
    for mode in () if args.skip_cpu else ("cold", "warm_self", "warm_prev"):
        t, itc, sols = cpu_osqp(raw, mode)
        rows = check(raw, sols)
        results[f"cpu_{mode}"] = {
            "s_per_qp_mean": float(np.mean(t)),
            "s_per_qp_median": float(np.median(t)),
            "iters_median": float(np.median(itc)),
            "iters_total": int(np.sum(itc)),
            "max_viol": float(max(r["viol"] for r in rows)),
            "median_viol": float(np.median([r["viol"] for r in rows])),
            # control: this is OSQP-without-polish vs the captured OSQP-with-polish
            # solution, i.e. the objective gap attributable to eps=1e-3 alone
            "obj_rel_median": float(np.median([r["obj_rel"] for r in rows])),
            "obj_rel_max": float(max(abs(r["obj_rel"]) for r in rows)),
        }
        print(
            f"CPU osqp {mode:10s}: {np.mean(t) * 1000:8.1f} ms/QP "
            f"(median {np.median(t) * 1000:8.1f})  iters median {np.median(itc):6.0f}  "
            f"viol {max(r['viol'] for r in rows):.2e}  "
            f"dobj med {np.median([r['obj_rel'] for r in rows]):+.2e} "
            f"max {max(abs(r['obj_rel']) for r in rows):.2e}",
            flush=True,
        )

    scaled_all = [scale(q) for q in raw]

    # ---- GPU indirect ------------------------------------------------------
    m_max = max(q.A.shape[0] for q in raw)

    def make_batch(ids):
        return Batch([scaled_all[i] for i in ids], cp, cusp, m_force=m_max)

    def warm_start(pick, mode):
        """Scaled (x0, y0) for the batch.  'self' = own captured solution (an
        overhead floor); 'prev' = the previous QP's solution, what the engine does."""
        K = len(pick)
        x0 = np.zeros((K, raw[0].pdiag.size))
        y0 = np.zeros((K, m_max))
        for j, i in enumerate(pick):
            src = raw[i if mode == "self" else (i - 1) % len(raw)]
            x0[j] = src.x_ref / scaled_all[i].D
            if src.A.shape[0] == raw[i].A.shape[0]:
                mi = raw[i].A.shape[0]
                y0[j, :mi] = scaled_all[i].c * src.y_ref / scaled_all[i].E
        return x0, y0

    if args.strategy in ("indirect", "both"):
        for K in [int(s) for s in args.batches.split(",")]:
            pick = np.array([i % len(raw) for i in range(K)])
            raws = [raw[i] for i in pick]
            for compact in (False, True) if args.compaction_sweep else (True,):
                tag = "cmpct" if compact else "flat"
                for label in args.regimes.split(","):
                    if label == "cold":
                        x0 = y0 = None
                    else:
                        x0, y0 = warm_start(pick, label.split("_")[1])
                    cp.get_default_memory_pool().free_all_blocks()
                    # warm up kernels, and measure the resident footprint of one batch
                    free_before = cp.cuda.runtime.memGetInfo()[0]
                    admm(make_batch, pick, cp, x0=x0, y0=y0, max_iter=CHECK_EVERY, compact=False)
                    cp.cuda.Stream.null.synchronize()
                    resident = free_before - cp.cuda.runtime.memGetInfo()[0]
                    t0 = time.perf_counter()
                    X, Y, it, conv, pcg_tot, nreb = admm(
                        make_batch, pick, cp, x0=x0, y0=y0, compact=compact
                    )
                    cp.cuda.Stream.null.synchronize()
                    dt = time.perf_counter() - t0
                    rows = check(raws, X)
                    nconv = int((conv > 0).sum())
                    results[f"gpu_indirect_K{K}_{label}_{tag}"] = {
                        "batch_s": dt,
                        "s_per_qp": dt / K,
                        "admm_iters": it,
                        "rebuilds": nreb,
                        "converged": nconv,
                        "conv_iter_median": float(np.median(conv[conv > 0])) if nconv else None,
                        "conv_iter_max": int(conv.max()),
                        "pcg_per_admm_iter": pcg_tot / max(it, 1),
                        "max_viol": float(max(r["viol"] for r in rows)),
                        "median_viol": float(np.median([r["viol"] for r in rows])),
                        "max_viol_osqp": float(max(r["viol_osqp"] for r in rows)),
                        "obj_rel_max": float(max(abs(r["obj_rel"]) for r in rows)),
                        "obj_rel_median": float(np.median([r["obj_rel"] for r in rows])),
                        "obj_rel_signed_min": float(min(r["obj_rel"] for r in rows)),
                        "obj_rel_signed_max": float(max(r["obj_rel"] for r in rows)),
                        "resident_bytes_per_qp": resident / K,
                    }
                    print(
                        f"GPU indirect K={K:3d} {label:9s} {tag:5s}: "
                        f"{dt / K * 1000:8.1f} ms/QP  batch {dt:7.2f} s  admm {it:5d}  "
                        f"reb {nreb:2d}  pcg/iter {pcg_tot / max(it, 1):5.2f}  "
                        f"conv {nconv}/{K}  viol {max(r['viol'] for r in rows):.2e}  "
                        f"dobj med {np.median([r['obj_rel'] for r in rows]):+.2e} "
                        f"max {max(abs(r['obj_rel']) for r in rows):.2e}  "
                        f"{resident / K / 2**20:.1f} MiB/QP",
                        flush=True,
                    )
                    del X, Y
                    cp.get_default_memory_pool().free_all_blocks()

    # ---- GPU direct-hybrid probe ------------------------------------------
    if args.strategy in ("direct", "both"):
        for K in [int(s) for s in args.batches.split(",")]:
            try:
                r = direct_probe(scaled_all, K, cp, cusp, cuspl)
                results[f"gpu_direct_K{K}"] = r
                print(
                    f"GPU direct   K={K:3d}     : tri-solve {r['tri_solve_s_per_qp'] * 1000:8.2f} "
                    f"ms/QP/ADMM-iter  splu {r['fact_s_per_qp'] * 1000:6.1f} ms/QP  "
                    f"LU nnz {r['lu_nnz_per_qp'] / 1e6:.2f}M  "
                    f"{r['gpu_bytes_per_qp'] / 2**20:.0f} MiB/QP",
                    flush=True,
                )
            except Exception as exc:
                results[f"gpu_direct_K{K}"] = {"error": repr(exc)}
                print(f"GPU direct K={K}: FAILED {exc!r}", flush=True)
            cp.get_default_memory_pool().free_all_blocks()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
