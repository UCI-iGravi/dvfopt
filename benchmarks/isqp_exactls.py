"""Exact 1-D line search in the elastic-QP SQP — research prototype.

RESEARCH PROTOTYPE (benchmarks only, no library change). A patched copy of
:func:`dvfopt.core.primitives.isqp.isqp_solve` whose step ACCEPTANCE changes;
the QP itself (diagonal ``H_obj``, elastic slack, trust-region box) is untouched
-- explicitly NO Newton term (see ``benchmarks/isqp_newton.py`` for why that one
does not work).

The exact 1-D model
-------------------
Every 2-tri / bilinear row is a signed triangle area, i.e. an inhomogeneous
BILINEAR form in the decision variables, so along any line it is EXACTLY
quadratic::

    c_i(x + a d) = c_i(x) + a (J d)_i + a^2 q_i(d),
    q_i(d) = 1/2 d' Hc_i d = sum over the 6 constant pairs of  v * d_yQ * d_xP

with the same ``(Q, P, v)`` table and the same constant row Hessians verified to
4.4e-12 by ``benchmarks/newton_sqp_proto.py --check``. No evaluation of ``cons``
is needed to know the whole line -- three numbers per row.

Two closed-form tools follow (``step_rule``):

``'cap'`` (A) -- MAXIMAL FEASIBILITY-PRESERVING STEP. For a row that currently
    holds (``c_i >= 0``), the first ``a > 0`` at which it breaks is a root of a
    quadratic; ``a_max`` is the min over such rows. Scaling the QP step to
    ``a_max`` means no satisfied row is broken by the step (the NONLINEAR version
    of what ``monotone=True`` does linearly). Acceptance stays the stock ratio
    test, applied to the scaled step.

``'exact'`` (B) -- EXACT MERIT LINE MINIMISER. The merit
    ``m(a) = f(a) + sum_i w_i max(0, -c_i(a))`` is piecewise quadratic in ``a``
    with breakpoints exactly at the roots of ``c_i(a) = 0``: ``f`` is quadratic
    (``NoneObjective``: 0; L2: quadratic) and each hinge switches on/off only at a
    root. So the global minimiser on ``[0, a_hi]`` is available in closed form --
    sort the (<= 2m) roots, sweep the active set with a cumulative sum, and take
    the best of every interval's endpoints and parabola vertex. O(m log m), no
    ``cons`` evaluation at all. This REPLACES the ratio test / backtracking; the
    trust region still bounds the QP and is still adapted, now from the achieved
    ``a*``.

``'exact_tr'`` -- (B), but ONLY on the trust-region path: the engine's no-TR
    fallback rung (``trust_region=False``, a deliberately different escape from a
    stuck window) keeps the legacy backtracking it was tuned for.

``'exact_bail'`` -- (B) plus the ratio test's OWN futility threshold as a
    termination signal only (see the ``exact_bail`` branch): an exact minimiser
    always finds SOME decrease, so it never fires the fast ``tr-collapse`` bail
    that hands a hopeless window straight to the engine's escalation ladder.

``'both'`` -- (A) then (B) on ``[0, a_max]``.

``'tr'`` -- the stock driver (the instrumented baseline; ``--parity`` proves it is
    byte-identical to the library path).

The quadratic term, without any per-family table
------------------------------------------------
:class:`LineModel` builds ``q_i(d)`` from the constant row Hessians, which needs
the row -> triangle map and the pack convention. It does not have to: since the
row IS quadratic, ``q = c(x + d) - c(x) - J d`` EXACTLY, and the stock ratio test
already evaluates ``cons(x + d)`` every iteration -- so the identity form is free
and family-agnostic (it holds for every 2D family here: 2tri, bilinear, jdet and
finite rows are all bilinear forms in ``(dy, dx)``). Passing ``line=None`` with a
non-``'tr'`` rule selects it; ``--check`` asserts the two agree to 1e-10.

Objective along the line
------------------------
``f(a)`` is fitted as a parabola through ``obj`` at ``a = 0, 1/2, 1`` -- EXACT for
``NoneObjective`` (f == 0) and ``L2Objective`` (quadratic), which is every
configuration measured here. For the eps-smoothed L1 objective it is a
quadratic approximation of a convex function on the segment, so the minimiser
is approximate there.
"""

from __future__ import annotations

import numpy as np

# Reuse the verified pieces of the Newton prototype: the (Q, P, v) pair table,
# the row -> triangle map, the QP wrappers and their solve accounting.
from isqp_newton import _PAIRS, STATS, _make_qp, triangle_abc
from isqp_newton import reset_stats as _reset_qp_stats
from scipy import sparse

from dvfopt.core.primitives.isqp import HAS_OSQP, _backtrack

_QI = np.array([q for q, _p, _v in _PAIRS])
_PI = np.array([p for _q, p, _v in _PAIRS])
_VV = np.array([v for _q, _p, v in _PAIRS])

# Line-search accounting (module-global; the harness resets per run).
LS_STATS = {
    "ls_calls": 0,
    "alpha_sum": 0.0,
    "alpha_min": 1.0,
    "n_alpha_full": 0,  # a* == a_hi (the QP step taken whole)
    "n_no_progress": 0,  # a* == 0 / no merit decrease -> trust-region shrink
    "n_rejected": 0,  # stock ratio-test rejections (baseline / 'cap')
    "cap_calls": 0,
    "cap_sum": 0.0,
    "n_cap_active": 0,  # a_max < 1 (the cap actually bound)
    "n_cap_skip": 0,  # a_max below the floor -> cap ignored, full step used
    "n_events": 0,  # breakpoints scanned
}


def reset_stats():
    _reset_qp_stats()
    for k in LS_STATS:
        LS_STATS[k] = 1.0 if k == "alpha_min" else (0.0 if "sum" in k else 0)


class LineModel:
    """Per-row quadratic coefficient ``q_i(d)`` of the enforced rows.

    ``abc`` is ``(m, 3)`` patch pixel indices of each enforced triangle row's
    ordered vertices; the DY_FIRST pack puts pixel ``p``'s dy at ``p`` and its dx
    at ``p + n_pix``.
    """

    def __init__(self, abc, n_pix):
        self.yv = np.ascontiguousarray(abc)
        self.xv = self.yv + int(n_pix)

    def quad(self, d):
        dy = d[self.yv]  # (m, 3)
        dx = d[self.xv]
        return (dy[:, _QI] * dx[:, _PI] * _VV).sum(1)


def line_model_for_sub(sub):
    """:class:`LineModel` for a windowed-engine ``WindowSub`` (rows recomputed
    with the engine's OWN locality adapter, so they line up with ``sub.cons``)."""
    from dvfopt.core.windowed._locality import _locality_of

    ph, pw = sub.constraint.shape
    enforced_idx, _ = _locality_of(sub.constraint).influenced(
        sub.constraint, sub.free_mask, ph, pw, (False, False, False, False)
    )
    assert enforced_idx.size == sub.n_enforced, "enforced-row set drifted from the engine's"
    return LineModel(triangle_abc(enforced_idx, ph, pw), ph * pw)


# ---------------------------------------------------------------------------
# closed-form line tools
# ---------------------------------------------------------------------------


def line_events(c0, g, q, a_hi):
    """Roots of ``c_i(a) = c0 + g a + q a^2`` inside ``(0, a_hi)``.

    Returns ``(roots, flags, rows)`` with ``flag = +1`` where the row ENTERS the
    violated region (``c`` goes negative) and ``-1`` where it LEAVES. A row with
    ``q > 0`` is negative between its roots; with ``q < 0``, outside them; a
    linear row (``q == 0``) switches once.
    """
    lin = q == 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        disc = g * g - 4.0 * q * c0
        sq = np.sqrt(np.where(disc > 0.0, disc, 0.0))
        # numerically stable quadratic: t = -(g + sign(g) sqrt(disc)) / 2,
        # roots are t/q and c0/t (never the cancelling (-g + sq) / 2q form)
        t = -0.5 * (g + np.where(g >= 0.0, 1.0, -1.0) * sq)
        ra = np.where(lin, np.inf, t / np.where(lin, 1.0, q))
        rb = np.where(t == 0.0, ra, c0 / np.where(t == 0.0, 1.0, t))
        rb = np.where(lin, np.inf, rb)
        rlin = np.where(lin & (g != 0.0), -c0 / np.where(g == 0.0, 1.0, g), np.inf)
    r1, r2 = np.minimum(ra, rb), np.maximum(ra, rb)
    quad_ok = (~lin) & (disc >= 0.0)
    up = q > 0.0
    roots = np.concatenate([r1, r2, rlin])
    flags = np.concatenate(
        [
            np.where(up, 1.0, -1.0),  # first root of a quadratic row
            np.where(up, -1.0, 1.0),  # second root
            np.where(g < 0.0, 1.0, -1.0),  # linear row
        ]
    )
    idx = np.arange(c0.size)
    rows = np.concatenate([idx, idx, idx])
    keep = (
        np.concatenate([quad_ok, quad_ok, lin & (g != 0.0)])
        & np.isfinite(roots)
        & (roots > 0.0)
        & (roots < a_hi)
    )
    return roots[keep], flags[keep], rows[keep]


def max_feasible_step(c0, g, q, a_hi=1.0):
    """(A) Largest ``a`` in ``(0, a_hi]`` at which NO currently-satisfied row
    (``c0 >= 0``) has gone below its target."""
    r, fl, rows = line_events(c0, g, q, a_hi)
    if r.size == 0:
        return a_hi
    block = (fl > 0.0) & (c0[rows] >= 0.0)
    return float(r[block].min()) if block.any() else a_hi


def exact_line_min(c0, g, q, w, fco, a_hi=1.0):
    """(B) Exact minimiser of the piecewise-quadratic merit on ``[0, a_hi]``.

    ``fco = (f0, f1, f2)`` are the objective's coefficients along the line.
    Returns ``(a_star, m_star, m_zero, n_events)``.
    """
    r, fl, rows = line_events(c0, g, q, a_hi)
    order = np.argsort(r, kind="stable")
    r, fl, rows = r[order], fl[order], rows[order]
    # active-set coefficient sums: a violated row contributes w*(-c) to the merit
    wc, wg, wq = w * (-c0), w * (-g), w * (-q)
    act = c0 < 0.0
    a0 = np.concatenate([[wc[act].sum()], wc[rows] * fl]).cumsum()
    a1 = np.concatenate([[wg[act].sum()], wg[rows] * fl]).cumsum()
    a2 = np.concatenate([[wq[act].sum()], wq[rows] * fl]).cumsum()
    k0, k1, k2 = fco[0] + a0, fco[1] + a1, fco[2] + a2
    edges = np.concatenate([[0.0], r, [a_hi]])
    lo, hi = edges[:-1], edges[1:]
    pos = k2 > 0.0
    vert = np.clip(np.where(pos, -k1 / np.where(pos, 2.0 * k2, 1.0), lo), lo, hi)
    aa = np.concatenate([lo, hi, vert])
    kk0, kk1, kk2 = np.tile(k0, 3), np.tile(k1, 3), np.tile(k2, 3)
    vv = kk0 + kk1 * aa + kk2 * aa * aa
    b = int(np.argmin(vv))
    return float(aa[b]), float(vv[b]), float(k0[0]), int(r.size)


def obj_line_coeffs(obj, x, d):
    """``(f0, f1, f2)`` of ``obj(x + a d)`` from a 3-point parabola fit --
    EXACT for a quadratic objective (``NoneObjective`` / ``L2Objective``)."""
    f0 = float(obj(x))
    fh = float(obj(x + 0.5 * d))
    f1 = float(obj(x + d))
    return f0, 4.0 * fh - f1 - 3.0 * f0, 2.0 * f1 + 2.0 * f0 - 4.0 * fh


# ---------------------------------------------------------------------------
# the driver (a copy of isqp_solve; only step acceptance changes)
# ---------------------------------------------------------------------------

CAP_FLOOR = 1e-4  # a_max below this is a degenerate block -> ignore the cap


def isqp_exactls_solve(
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
    step_rule="tr",
    line=None,
):
    """:func:`dvfopt.core.primitives.isqp.isqp_solve` with a ``step_rule``.

    ``step_rule='tr'`` (and ``line=None``) reproduces the stock driver exactly.
    ``'cap'`` / ``'exact'`` / ``'both'`` need ``line`` (a :class:`LineModel`).
    """
    if not HAS_OSQP:
        raise ImportError("isqp_exactls_solve requires osqp")
    if step_rule not in ("tr", "cap", "exact", "exact_tr", "exact_bail", "both"):
        raise ValueError(f"unknown step_rule {step_rule!r}")
    # ``line=None`` on a non-'tr' rule selects the TABLE-FREE quadratic term (see
    # the module docstring): q = cons(x+d) - c - J d, exact for any quadratic row
    # family, paid for by an evaluation the stock ratio test already makes.
    # 'exact_tr' scopes the exact minimiser to the trust-region path and leaves the
    # engine's no-TR fallback rung on the legacy backtracking it was tuned for.
    use_exact = step_rule in ("exact", "exact_bail", "both") or (
        step_rule == "exact_tr" and trust_region
    )
    use_line = step_rule != "tr" and (use_exact or step_rule in ("cap", "both"))

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
            print(f"    [exactls] {rec}", flush=True)

    prob = None
    a_pat = None
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
        s_up = (viol + 1e-6) if monotone else np.full(m, np.inf)
        eye_m = sparse.eye(m, format="csc")
        hdv = hess_diag(x)[free]
        p = sparse.block_diag([sparse.diags(hdv), sparse.csc_matrix((m, m))], format="csc")
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
        res = prob.solve()
        z = np.asarray(res.x)
        if not np.all(np.isfinite(z)):
            exit_reason = "osqp-nonfinite"
            break
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

        # --- closed-form line tools -------------------------------------
        alpha = 1.0
        if use_line:
            gl = np.asarray(j @ z[:nf])  # (J d)_i, exact linear term
            # exact quadratic term: the per-row constant Hessian (table) or, with no
            # LineModel, the identity q = c(x+d) - c(x) - J d (one cons evaluation)
            ql = line.quad(d) if line is not None else np.asarray(cons(x + d)) - c - gl
        if step_rule in ("cap", "both"):
            a_max = max_feasible_step(c, gl, ql, 1.0)
            LS_STATS["cap_calls"] += 1
            LS_STATS["cap_sum"] += a_max
            if a_max < 1.0:
                LS_STATS["n_cap_active"] += 1
            if a_max < CAP_FLOOR:
                LS_STATS["n_cap_skip"] += 1
                a_max = 1.0  # degenerate block (a row sitting on its target)
            alpha = a_max
            rec["a_max"] = round(a_max, 6)

        if use_exact:
            fco = obj_line_coeffs(obj, x, d)
            a_star, m_star, m_zero, n_ev = exact_line_min(c, gl, ql, rho_vec, fco, alpha)
            LS_STATS["ls_calls"] += 1
            LS_STATS["alpha_sum"] += a_star
            LS_STATS["alpha_min"] = min(LS_STATS["alpha_min"], a_star)
            LS_STATS["n_events"] += n_ev
            if a_star >= alpha - 1e-12:
                LS_STATS["n_alpha_full"] += 1
            dec = m_zero - m_star
            rec["alpha"] = round(a_star, 6)
            rec["dec"] = dec
            stepped = a_star > 0.0 and m_star < m_zero
            futile = False
            if step_rule == "exact_bail" and trust_region:
                # An exact minimiser ALWAYS finds some decrease, so it never fires the
                # ratio test's fast bail-out -- and a window that cannot be solved at
                # this size then grinds instead of escalating. Reuse the ratio test's
                # OWN futility threshold (actual < 1e-3 x the QP's predicted decrease)
                # purely as a TERMINATION signal; the step itself is still taken.
                s_slack = z[nf : nf + m]
                pred = float(rho_vec @ viol) - (
                    float(gx @ z[:nf])
                    + 0.5 * float(z[:nf] @ (hdv * z[:nf]))
                    + float(rho_vec @ s_slack)
                )
                futile = pred > 1e-8 and dec <= 1e-3 * pred
                rec["futile"] = futile
            if stepped:
                x = x + a_star * d
                if a_star * dn < tol:
                    rec["stepped"] = True
                    _emit(rec)
                    exit_reason = "step-tol"
                    break
            else:
                LS_STATS["n_no_progress"] += 1
            if not trust_region:
                if not stepped:
                    exit_reason = "linesearch-stall"
            elif futile or not stepped:
                tr_delta *= 0.25
                if tr_delta < tr_min:
                    exit_reason = "tr-collapse"
            elif a_star >= 0.9 and dn >= 0.9 * tr_delta:
                tr_delta = min(tr_delta * 2.0, tr_max)
            elif a_star < 0.25:
                tr_delta = max(tr_delta * 0.5, tr_min)
            rec["stepped"] = bool(stepped)
            _emit(rec)
            if exit_reason in ("tr-collapse", "linesearch-stall"):
                break
            continue
        # ---------------------------------------------------------------

        if alpha != 1.0:  # 'cap' alone: stock acceptance on the SCALED step
            d = alpha * d
            z = alpha * z
            dn *= alpha
        if trust_region:
            s_slack = z[nf : nf + m]
            pred = float(rho_vec @ viol) - (
                float(gx @ z[:nf]) + 0.5 * float(z[:nf] @ (hdv * z[:nf])) + float(rho_vec @ s_slack)
            )
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
                    LS_STATS["n_rejected"] += 1
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


def bind(step_rule="exact", line_model=True):
    """Route the windowed engine's isqp inner through this driver. Returns an
    ``unbind()`` callable. ``step_rule='tr'`` is the instrumented baseline.
    ``line_model=False`` takes the table-free quadratic term instead."""
    import dvfopt.core.windowed._common as _c
    import dvfopt.core.windowed._inners as _i

    orig = _c.solve_window_inner

    def patched(sub, inner, maxiter, **kw):
        if inner not in _i._ISQP_LABELS:
            return orig(sub, inner, maxiter, **kw)
        return isqp_exactls_solve(
            sub.flat0,
            sub.cons,
            sub.cons_jac,
            sub.obj_grad,
            maxiter,
            obj=sub.obj,
            hess_diag=sub.hess_diag,
            free_idx=sub.free_idx,
            step_rule=step_rule,
            line=line_model_for_sub(sub) if (line_model and step_rule != "tr") else None,
            **kw,
        )

    _c.solve_window_inner = patched

    def unbind():
        _c.solve_window_inner = orig

    return unbind


__all__ = [
    "CAP_FLOOR",
    "LS_STATS",
    "STATS",
    "LineModel",
    "bind",
    "exact_line_min",
    "isqp_exactls_solve",
    "line_events",
    "line_model_for_sub",
    "max_feasible_step",
    "obj_line_coeffs",
    "reset_stats",
]
