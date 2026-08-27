"""Newton-type SQP prototype: Lagrangian Hessian inside the elastic-QP subproblem.

RESEARCH PROTOTYPE (benchmarks only, no library change). A patched copy of
:func:`dvfopt.core.primitives.isqp.isqp_solve` whose QP Hessian is

    P_dd = H_obj + sum_i lam_i * Hc_i        (+ convexification)

instead of the objective's diagonal curvature alone. ``Hc_i`` is the CONSTANT
Hessian of enforced constraint row ``i``; ``lam_i`` are that row's multipliers
from the PREVIOUS QP solve.

Why the row Hessians are constant
---------------------------------
Every 2-tri / bilinear row is a signed triangle area

    c = -1/2 [ (x_B-x_A)(y_C-y_A) - (y_B-y_A)(x_C-x_A) ]

with ``x_P = ref_x_P + dx_P``, ``y_P = ref_y_P + dy_P`` -- a bilinear form in the
vertex coordinates, so ``d2c/ddx ddy`` is a constant sparse matrix and every
other second derivative vanishes. In the driver's flat DY_FIRST packing
(``phi[:N] = dy``, ``phi[N:] = dx``; y-var of pixel ``p`` is ``p``, x-var is
``p + N``) the six nonzero entries are

    H[y_Q, x_P] = -1/2 * cyc(P, Q),   cyc(A,B) = cyc(B,C) = cyc(C,A) = +1

i.e. +-1/2 coupling the x-coordinate of one vertex with the y-coordinate of
another, and NOTHING on the diagonal. Verified exhaustively against second
differences of ``constraint.values`` by ``benchmarks/newton_sqp_proto.py --check``.

Row -> triangle map (measured earlier, re-confirmed here): rows are BLOCK-MAJOR,
``row = b*(H-1)*(W-1) + i*(W-1) + j``, with ordered vertices per block ``b``

    b=0 T1 = (TR, BL, BR)   b=1 T2 = (TL, BL, TR)
    b=2 U1 = (TL, BL, BR)   b=3 U2 = (TR, TL, BR)

(the U pair is the x-mirror of the T pair; mirroring swaps TL<->TR and BL<->BR
and flips the area sign, which is where the orderings come from).

Dual sign convention
--------------------
OSQP solves ``min 1/2 z'Pz + q'z  s.t. l <= Az <= u`` with stationarity
``Pz + q + A'y = 0`` and ``y <= 0`` on a row held at its LOWER bound. The
linearized rows are ``c + J d + s >= 0`` -> ``l = -c, u = +inf``, so only the
lower bound can be active and ``y <= 0`` there. The NLP Lagrangian for
``c(x) >= 0`` is ``L = f - mu'c`` with ``mu >= 0``, and matching the two KKT
systems gives ``mu = -y``. Hence

    Hess(L) = H_obj - sum_i mu_i Hc_i = H_obj + sum_i y_i Hc_i

so OSQP's raw ``y`` is used with a PLUS sign. Wrong-sign noise is clipped
(``lam = min(y, 0)``) since positive ``y`` on these rows is impossible at a true
optimum.

Convexification (OSQP needs P PSD)
----------------------------------
Each ``Hc_i`` is indefinite with spectrum ``{+s, +s, 0, 0, -s, -s}``,
``s = sqrt(3)/2``. Two modes:

``'gershgorin'`` (default) -- global diagonal shift. The assembled
``S = sum lam_i Hc_i`` has a ZERO diagonal and couples only y-vars to x-vars, so
under the (y | x) split it is ``[[0, B], [B', 0]]`` with
``lambda_min(S) = -sigma_max(B) >= -sqrt(||B||_1 ||B||_inf)``. Taking
``tau = sqrt(||B||_1 ||B||_inf)`` makes ``S + tau I`` PSD. Tighter than plain
Gershgorin on the full matrix (which gives ``max(||B||_1, ||B||_inf)``) and
still O(nnz).

``'psd_row'`` -- per-row PSD projection, no global shift. With
``Hc = [[0, M], [M', 0]]``, ``M = -M'`` the cross-product matrix of
``w = (1/2, 1/2, 1/2)``, the positive part of ``lam*Hc`` is

    P_+ = 1/2 [[ |lam| s (I - w^ w^'),  lam M ],
               [ lam M',               |lam| s (I - w^ w^') ]]

-- half the true off-diagonal coupling plus a LOCAL shift on only the six
variables of that row. 3x the nnz of ``'gershgorin'``.

Both keep a FIXED sparsity pattern across SQP iterations (explicit zeros at
setup, values-only ``update(Px=...)``) -- OSQP rejects pattern changes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from dvfopt._logging import logger
from dvfopt.core.primitives.isqp import HAS_CLARABEL, HAS_OSQP, _backtrack, _HybridQP

try:
    import osqp
except ImportError:  # pragma: no cover
    osqp = None
try:
    import clarabel
except ImportError:  # pragma: no cover
    clarabel = None

SIG = 0.5 * np.sqrt(3.0)  # the (doubly degenerate) singular value of every Hc_i

# (A, B, C) corner ids into [TL, TR, BL, BR] for row blocks T1, T2, U1, U2.
_ORDER = np.array([(1, 2, 3), (0, 2, 1), (0, 2, 3), (1, 0, 3)], dtype=np.int64)

# (vertex of the y-index, vertex of the x-index, value) -> H[y_Q, x_P].
_PAIRS = ((0, 1, +0.5), (0, 2, -0.5), (1, 0, -0.5), (1, 2, +0.5), (2, 0, +0.5), (2, 1, -0.5))


# ---------------------------------------------------------------------------
# QP solve statistics (module-global; the harness resets per run)
# ---------------------------------------------------------------------------

STATS = {
    "qp_solves": 0,
    "admm_iters": 0,
    "ip_solves": 0,
    "ip_iters": 0,
    "tau_sum": 0.0,  # convexification shift, accumulated over assembles
    "tau_max": 0.0,
    "tau_n": 0,
}


def reset_stats():
    for k in STATS:
        STATS[k] = 0.0 if k.startswith("tau_") and k != "tau_n" else 0


def _record(res):
    st = str(getattr(res.info, "status", ""))
    STATS["qp_solves"] += 1
    if st.startswith("clarabel"):
        STATS["ip_solves"] += 1
        STATS["ip_iters"] += int(res.info.iter)
    else:
        STATS["admm_iters"] += int(res.info.iter)


class _OSQPY:
    """``osqp.OSQP()`` plus solve accounting."""

    def __init__(self):
        self._r = osqp.OSQP()

    def setup(self, *a, **k):
        self._r.setup(*a, **k)

    def update(self, **k):
        self._r.update(**k)

    def solve(self):
        res = self._r.solve()
        _record(res)
        return res


class _HybridQPY(_HybridQP):
    """:class:`~dvfopt.core.primitives.isqp._HybridQP` that also hands back the
    interior-point solve's DUALS in OSQP's sign convention (the stock class
    returns only ``x``, which would starve the Newton term exactly on the cold /
    stale-warm-start solves), plus solve accounting."""

    def solve(self):
        res = super().solve()
        _record(res)
        return res

    def _solve_ip(self):
        try:
            fu, fl = np.isfinite(self._up), np.isfinite(self._lo)
            a_csr = self._a.tocsr()
            a_ip = sparse.vstack([a_csr[fu], -a_csr[fl]], format="csc")
            b_ip = np.concatenate([self._up[fu], -self._lo[fl]])
            st = clarabel.DefaultSettings()
            st.verbose = False
            st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-3
            sol = clarabel.DefaultSolver(
                self._p, self._q, a_ip, b_ip, [clarabel.NonnegativeConeT(b_ip.size)], st
            ).solve()
            x = np.asarray(sol.x, dtype=np.float64)
            if str(sol.status) != "Solved" or not np.all(np.isfinite(x)):
                return None
            # Clarabel: Px + q + A_ip' z = 0, z >= 0. Finite-l rows enter A_ip as
            # -A, so their OSQP-convention dual is y = -z; finite-u rows give y = z.
            z = np.asarray(sol.z, dtype=np.float64)
            y = np.zeros(self._lo.size)
            nfu = int(fu.sum())
            y[fu] = z[:nfu]
            y[fl] = -z[nfu : nfu + int(fl.sum())]
            self._real.warm_start(x=x)
            return SimpleNamespace(
                x=x,
                y=y,
                info=SimpleNamespace(iter=int(sol.iterations), status=f"clarabel-{sol.status}"),
            )
        except Exception as exc:  # any IP trouble is a fall-through, never a failure
            logger.debug(f"hybrid QP: interior-point solve failed ({exc!r}); using OSQP")
            return None


def _make_qp(qp_backend, ip_cold, ip_after_admm_iters):
    if qp_backend == "osqp":
        return _OSQPY()
    if qp_backend != "hybrid":
        raise ValueError(f"unknown qp_backend {qp_backend!r}; valid: 'osqp', 'hybrid'")
    if not HAS_CLARABEL:
        return _OSQPY()
    return _HybridQPY(ip_cold, ip_after_admm_iters)


# ---------------------------------------------------------------------------
# Constant per-row constraint Hessians
# ---------------------------------------------------------------------------


def triangle_abc(enforced_idx, ph, pw):
    """``(n_rows, 3)`` pixel indices ``(A, B, C)`` of each enforced triangle row.

    Row layout is block-major: ``row = b*(ph-1)*(pw-1) + i*(pw-1) + j``.
    """
    m = (ph - 1) * (pw - 1)
    r = np.asarray(enforced_idx, dtype=np.int64)
    b, cell = r // m, r % m
    i, j = cell // (pw - 1), cell % (pw - 1)
    tl = i * pw + j
    corners = np.stack([tl, tl + 1, tl + pw, tl + pw + 1], axis=1)  # TL, TR, BL, BR
    return np.take_along_axis(corners, _ORDER[b], axis=1)


def _entry_table(convexify, coupling=1.0):
    """``(chan_r, vert_r, chan_c, vert_c, coef, use_abs)`` per triangle row.
    Channel 0 = dy (var = pixel), 1 = dx (var = pixel + N)."""
    t = [(0, q, 1, p, v * coupling, False) for (q, p, v) in _PAIRS]
    if convexify == "gershgorin":
        return t
    if convexify != "psd_row":
        raise ValueError(f"unknown convexify {convexify!r}; valid: 'gershgorin', 'psd_row'")
    t = [(a, b, c, d, v * 0.5, e) for (a, b, c, d, v, e) in t]  # PSD projection halves it
    for ch in (0, 1):  # |lam| * s * (I - w^ w^') / 2 on each channel block
        t += [(ch, k, ch, k, SIG / 3.0, True) for k in range(3)]
        t += [(ch, k, ch, ll, -SIG / 6.0, True) for k, ll in ((0, 1), (0, 2), (1, 2))]
    return t


class NewtonHess:
    """Assembles ``H_obj + sum_i lam_i Hc_i`` (convexified, restricted to the free
    variables) into an upper-triangular CSC ``P`` with a FIXED sparsity pattern."""

    def __init__(
        self,
        abc,
        n_var,
        free_idx,
        n_slack,
        convexify="gershgorin",
        lam_scale=1.0,
        lam_cap=None,
        coupling=1.0,
    ):
        self.convexify = convexify
        self.lam_scale = float(lam_scale)
        # Cap on |lam|. The elastic formulation pins a row whose slack is positive
        # at exactly lam = -rho (1e3), so the raw multipliers are a big-M penalty
        # artifact, not NLP multipliers -- a cap tests the Newton term in the regime
        # where the convexification shift does not swamp the curvature it models.
        self.lam_cap = None if lam_cap is None else float(lam_cap)
        n_pix = n_var // 2
        free_idx = np.asarray(free_idx, dtype=np.int64)
        nf = free_idx.size
        self.nf, self.n_slack = nf, int(n_slack)
        inv = np.full(n_var, -1, np.int64)
        inv[free_idx] = np.arange(nf)
        rr, cc, co, ss, ab = [], [], [], [], []
        for ch_r, v_r, ch_c, v_c, coef, use_abs in _entry_table(convexify, coupling):
            r = inv[abc[:, v_r] + ch_r * n_pix]
            c = inv[abc[:, v_c] + ch_c * n_pix]
            keep = (r >= 0) & (c >= 0)
            r, c = r[keep], c[keep]
            rr.append(np.minimum(r, c))
            cc.append(np.maximum(r, c))
            co.append(np.full(r.size, coef))
            ss.append(np.nonzero(keep)[0])
            ab.append(np.full(r.size, use_abs))
        self.r = np.concatenate(rr) if rr else np.zeros(0, np.int64)
        self.c = np.concatenate(cc) if cc else np.zeros(0, np.int64)
        self.coef = np.concatenate(co) if co else np.zeros(0)
        self.src = np.concatenate(ss) if ss else np.zeros(0, np.int64)
        self.use_abs = np.concatenate(ab) if ab else np.zeros(0, bool)

        # Pattern = Newton entries UNION the objective diagonal; the slack block
        # stays empty (its Hessian is zero). Explicit zeros are kept, so the
        # pattern never changes as lam changes.
        n = nf + self.n_slack
        rows = np.concatenate([self.r, np.arange(nf)])
        cols = np.concatenate([self.c, np.arange(nf)])
        order = np.lexsort((rows, cols))  # CSC order: by column, then row
        key = cols.astype(np.int64) * n + rows
        ks = key[order]
        new = np.ones(ks.size, bool)
        new[1:] = ks[1:] != ks[:-1]
        self.pos = np.empty(rows.size, np.int64)
        self.pos[order] = np.cumsum(new) - 1
        uniq_r, uniq_c = rows[order][new], cols[order][new]
        indptr = np.zeros(n + 1, np.int64)
        np.add.at(indptr, uniq_c + 1, 1)
        indptr = np.cumsum(indptr)
        self.nnz = int(uniq_r.size)
        self.P = sparse.csc_matrix(
            (np.zeros(self.nnz), uniq_r.astype(np.int32), indptr.astype(np.int32)), shape=(n, n)
        )
        self.diag_pos = self.pos[self.r.size :]  # data slots of the (k, k) entries

    def assemble(self, lam, hdv):
        """Refresh ``P``'s VALUES for multipliers ``lam`` (OSQP sign) and objective
        curvature ``hdv``. Returns ``(P, tau)``."""
        lam = np.minimum(np.asarray(lam, dtype=np.float64), 0.0)
        if self.lam_cap is not None:
            lam = np.maximum(lam, -self.lam_cap)
        lam = self.lam_scale * lam
        li = lam[self.src]
        v = self.coef * np.where(self.use_abs, np.abs(li), li)
        if self.convexify == "gershgorin":
            av = np.abs(v)
            rs = np.bincount(self.r, av, self.nf)
            cs = np.bincount(self.c, av, self.nf)
            tau = float(np.sqrt(rs.max(initial=0.0) * cs.max(initial=0.0)))
        else:
            tau = 0.0
        STATS["tau_sum"] += tau
        STATS["tau_max"] = max(STATS["tau_max"], tau)
        STATS["tau_n"] += 1
        self.P.data = np.bincount(self.pos, np.concatenate([v, hdv + tau]), self.nnz)
        return self.P, tau

    def quad(self, z, diag_vals):
        """``z' P_sym z`` for the upper-triangular stored ``P``."""
        return 2.0 * float(z @ self.P.dot(z)) - float(
            (diag_vals * z[: self.nf] * z[: self.nf]).sum()
        )


def newton_for_sub(sub, convexify="gershgorin", lam_scale=1.0, lam_cap=None, coupling=1.0):
    """Build the :class:`NewtonHess` for a windowed-engine ``WindowSub``.

    Recomputes the ENFORCED row set with the engine's own locality adapter, so the
    Hessian rows line up with the rows ``sub.cons`` / ``sub.cons_jac`` return.
    """
    from dvfopt.core.windowed._locality import _locality_of

    ph, pw = sub.constraint.shape
    enforced_idx, _ = _locality_of(sub.constraint).influenced(
        sub.constraint, sub.free_mask, ph, pw, (False, False, False, False)
    )
    assert enforced_idx.size == sub.n_enforced, "enforced-row set drifted from the engine's"
    return NewtonHess(
        triangle_abc(enforced_idx, ph, pw),
        sub.flat0.size,
        sub.free_idx,
        enforced_idx.size,
        convexify,
        lam_scale,
        lam_cap,
        coupling,
    )


# ---------------------------------------------------------------------------
# The driver (a copy of isqp_solve; only the QP Hessian changes)
# ---------------------------------------------------------------------------


def isqp_newton_solve(
    flat0,
    cons,
    cons_jac,
    obj_grad,
    maxiter,
    rho=1e3,
    tol=1e-7,
    obj=None,
    hess_diag=None,
    free_idx=None,
    trace=None,
    trust_region=True,
    protect=1.0,
    osqp_eps=None,
    osqp_max_iter=None,
    monotone=False,
    log_every=0,
    qp_backend="osqp",
    ip_cold=True,
    ip_after_admm_iters=800,
    tr_delta=2.0,
    tr_max=16.0,
    newton=None,
    ls_salvage=False,
):
    """:func:`dvfopt.core.primitives.isqp.isqp_solve` with ``newton`` (a
    :class:`NewtonHess`) folding the Lagrangian Hessian into the QP's ``P``.
    ``newton=None`` reproduces the stock driver exactly (verified by the
    ``--parity`` leg of ``benchmarks/newton_sqp_proto.py``)."""
    if not HAS_OSQP:
        raise ImportError("isqp_newton_solve requires osqp")

    x = np.asarray(flat0, dtype=np.float64).copy()
    n = x.size
    free = np.arange(n) if free_idx is None else np.asarray(free_idx)
    if hess_diag is None:

        def hess_diag(_f):
            return np.full(n, 2.0)

    if obj is None:

        def obj(y):
            return float((y - flat0) @ (y - flat0))

    def build_j(f):
        jj = cons_jac(f)
        j = jj if sparse.issparse(jj) else sparse.csc_matrix(np.asarray(jj))
        return j[:, free] if free_idx is not None else j

    def merit_w(y, w):
        return obj(y) + float(w @ np.clip(-np.asarray(cons(y)), 0, None))

    def _emit(rec):
        if trace is not None:
            trace["iters"].append(rec)
        if log_every and (rec["it"] % log_every == 0 or not rec.get("stepped", True)):
            print(f"    [newton] {rec}", flush=True)

    prob = None
    a_pat = None
    p_pat = None
    lam = None
    it = 0
    exit_reason = "maxiter"
    tr_delta, tr_max = float(tr_delta), float(tr_max)
    tr_min = 1e-6
    if trace is not None:
        trace["iters"] = []
    while it < maxiter:
        it += 1
        c = np.asarray(cons(x))
        viol = np.clip(-c, 0.0, None)
        j = build_j(x)
        m = c.size
        nf = j.shape[1]
        if lam is None:
            lam = np.zeros(m)  # first iteration: lam = 0 == today's behaviour
        s_up = (viol + 1e-6) if monotone else np.full(m, np.inf)
        eye_m = sparse.eye(m, format="csc")
        hdv = hess_diag(x)[free]
        if newton is None:
            p = sparse.block_diag([sparse.diags(hdv), sparse.csc_matrix((m, m))], format="csc")
            pdiag = hdv
        else:
            p, _tau = newton.assemble(lam, hdv)
            pdiag = p.data[newton.diag_pos]
        gx = obj_grad(x)[free]
        rho_vec = np.full(m, float(rho))
        if protect != 1.0:
            rho_vec[c >= 0.0] = rho * protect
        q = np.concatenate([gx, rho_vec])
        if trust_region:
            a = sparse.bmat(
                [[j, eye_m], [None, eye_m], [sparse.eye(nf, format="csc"), None]], format="csc"
            )
            lo = np.concatenate([-c, np.zeros(m), np.full(nf, -tr_delta)])
            up = np.concatenate([np.full(m, np.inf), s_up, np.full(nf, tr_delta)])
        else:
            a = sparse.bmat([[j, eye_m], [None, eye_m]], format="csc")
            lo = np.concatenate([-c, np.zeros(m)])
            up = np.concatenate([np.full(m, np.inf), s_up])
        same_pattern = (
            a_pat is not None
            and a.indices.shape == a_pat[1].shape
            and (a.indptr == a_pat[0]).all()
            and (a.indices == a_pat[1]).all()
        )
        if prob is not None and same_pattern:
            if p_pat is not None and not (
                p.indices.shape == p_pat[1].shape
                and (p.indptr == p_pat[0]).all()
                and (p.indices == p_pat[1]).all()
            ):
                raise RuntimeError("P sparsity pattern changed between iterations")
            prob.update(q=q, l=lo, u=up, Px=p.data, Ax=a.data)
        else:
            prob = _make_qp(qp_backend, ip_cold, ip_after_admm_iters)
            eps_kw = {} if osqp_eps is None else {"eps_abs": osqp_eps, "eps_rel": osqp_eps}
            prob.setup(
                p,
                q,
                a,
                lo,
                up,
                verbose=False,
                warm_starting=True,
                polishing=True,
                max_iter=8000 if osqp_max_iter is None else int(osqp_max_iter),
                **eps_kw,
            )
            a_pat = (a.indptr.copy(), a.indices.copy())
            p_pat = (p.indptr.copy(), p.indices.copy())
        res = prob.solve()
        z = np.asarray(res.x)
        if not np.all(np.isfinite(z)):
            exit_reason = "osqp-nonfinite"
            break
        yd = getattr(res, "y", None)
        if yd is not None:
            yd = np.asarray(yd, dtype=np.float64)
            if yd.size >= m and np.all(np.isfinite(yd[:m])):
                lam = yd[:m].copy()  # multipliers for the NEXT QP's Lagrangian Hessian
        d = np.zeros(n)
        d[free] = z[:nf]
        dn = float(np.linalg.norm(d))
        ph0 = obj(x) + float(rho_vec @ viol)

        def mfun(y, _w=rho_vec):
            return merit_w(y, _w)

        rec = {
            "it": it,
            "max_viol": float(viol.max(initial=0.0)),
            "n_viol": int((c < -1e-9).sum()),
            "merit": ph0,
            "step_norm": dn,
            "osqp_status": str(getattr(res.info, "status", "?")).strip(),
            "delta": (tr_delta if trust_region else None),
        }
        if dn < tol:
            rec["stepped"] = False
            _emit(rec)
            exit_reason = "step-tol"
            break
        if trust_region:
            s_slack = z[nf : nf + m]
            quad = (
                0.5 * float(z[:nf] @ (hdv * z[:nf]))
                if newton is None
                else 0.5 * newton.quad(z, pdiag)
            )
            pred = float(rho_vec @ viol) - (float(gx @ z[:nf]) + quad + float(rho_vec @ s_slack))
            act = ph0 - mfun(x + d)
            ratio = act / pred if pred > 1e-12 else float("nan")
            rec["ratio"] = ratio
            if pred <= 1e-8:
                x, stepped = _backtrack(mfun, x, d, ph0)
                if not stepped:
                    exit_reason = "model-flat"
            else:
                stepped = act > 0.0 and ratio > 1e-3
                if stepped:
                    x = x + d
                    if ratio > 0.75 and dn >= 0.9 * tr_delta:
                        tr_delta = min(tr_delta * 2.0, tr_max)
                    elif ratio < 0.25:
                        tr_delta = max(tr_delta * 0.5, tr_min)
                else:
                    if ls_salvage:
                        # CONTROL (not Newton): a rejected direction is thrown away
                        # today and re-derived by a whole extra QP solve. Salvage
                        # whatever progress it still carries with the legacy
                        # backtracking line search before shrinking.
                        x = _backtrack(mfun, x, d, ph0)[0]
                    tr_delta *= 0.25
                    if tr_delta < tr_min:
                        exit_reason = "tr-collapse"
            rec["stepped"] = bool(stepped)
            _emit(rec)
            if exit_reason in ("model-flat", "tr-collapse"):
                break
        else:
            x, stepped = _backtrack(mfun, x, d, ph0)
            rec["stepped"] = bool(stepped)
            _emit(rec)
            if not stepped:
                exit_reason = "linesearch-stall"
                break
    feasible = bool((np.asarray(cons(x)) >= -1e-6).all())
    if trace is not None:
        trace["exit"] = exit_reason
        trace["feasible"] = feasible
        trace["nit"] = it
    return x, it, feasible


# ---------------------------------------------------------------------------
# Binding into the windowed engine
# ---------------------------------------------------------------------------


def bind(
    convexify="gershgorin",
    lam_scale=1.0,
    newton=True,
    lam_cap=None,
    ls_salvage=False,
    coupling=1.0,
):
    """Route the windowed engine's isqp inner through this driver. Returns an
    ``unbind()`` callable. ``newton=False`` binds the driver with NO Lagrangian
    term (the parity / instrumentation baseline)."""
    import dvfopt.core.windowed._common as _c
    import dvfopt.core.windowed._inners as _i

    orig = _c.solve_window_inner

    def patched(sub, inner, maxiter, **kw):
        if inner not in _i._ISQP_LABELS:
            return orig(sub, inner, maxiter, **kw)
        nh = newton_for_sub(sub, convexify, lam_scale, lam_cap, coupling) if newton else None
        return isqp_newton_solve(
            sub.flat0,
            sub.cons,
            sub.cons_jac,
            sub.obj_grad,
            maxiter,
            obj=sub.obj,
            hess_diag=sub.hess_diag,
            free_idx=sub.free_idx,
            newton=nh,
            ls_salvage=ls_salvage,
            **kw,
        )

    _c.solve_window_inner = patched

    def unbind():
        _c.solve_window_inner = orig

    return unbind


__all__ = [
    "STATS",
    "NewtonHess",
    "bind",
    "isqp_newton_solve",
    "newton_for_sub",
    "reset_stats",
    "triangle_abc",
]
