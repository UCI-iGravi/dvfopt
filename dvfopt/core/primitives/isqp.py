"""Elastic-QP I-SLSQP solver (sparse OSQP subproblem, trust region,
monotone/protect elastic modes, pyslsqp-style trace) — promoted verbatim from
``benchmarks/slsqp_variants.py`` (PRs #61-64).

Gauss-Newton SQP whose linearized subproblem is ALWAYS feasible: a
non-negative elastic slack absorbs constraint inconsistency (penalized by
``rho``), so the method never bounces on an infeasible QP — the failure mode
that makes plain SLSQP oscillate. The QP is solved by OSQP over a SPARSE
system with a warm-started iterate. :func:`isqp_solve` documents the knobs,
the ``trace`` dict fields, and the exit-reason taxonomy.

One deliberate signature change from the benchmark original
(``_isqp_solve_osqp``): the ``constraint=`` kwarg (full-grid CPR-coloring
fast-path) is dropped — primitives may not import ``dvfopt.constraints``.
Callers that want CPR coloring pass a ``cons_jac`` built from
:mod:`dvfopt.core.primitives.coloring`; the windowed path always passed
``constraint=None`` and is unchanged.

Requires the optional ``osqp`` dependency: ``HAS_OSQP`` is False when it is
not importable and :func:`isqp_solve` raises ImportError.
"""

import numpy as np

try:
    import osqp

    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False


def _backtrack(merit, x, d, phi0, alpha_min=1e-4):
    """Backtracking line search on *merit* along *d*. Returns ``(x_new, stepped)``.

    Only accepts a step that strictly decreases the merit; if backtracking bottoms
    out without descent it returns ``(x, False)`` so the caller can stop — never
    take a non-improving step (an L1 QP with near-zero curvature can hand back a
    divergent direction, and stepping regardless sends the iterate to infinity).
    """
    alpha = 1.0
    while alpha > alpha_min:
        if merit(x + alpha * d) < phi0:
            return x + alpha * d, True
        alpha *= 0.5
    return x, False


def isqp_solve(
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
    monotone=False,
    log_every=0,
):
    """Optimized I-SLSQP: elastic-QP SQP where the QP is solved by OSQP over a
    SPARSE system with a warm-started iterate. Each iteration solves a QP for a
    step ``d`` plus non-negative slack ``s`` that is ALWAYS feasible (the slack
    absorbs constraint inconsistency, penalized by ``rho``), so the method never
    bounces on an infeasible linearized subproblem — the failure mode that makes
    plain SLSQP oscillate.

    ``cons_jac(f)`` supplies the constraint Jacobian (dense or ``scipy.sparse``).
    The benchmark original's ``constraint=`` CPR-coloring fast-path is dropped
    here (primitives may not import ``dvfopt.constraints``); callers that want
    coloring build ``cons_jac`` from :mod:`dvfopt.core.primitives.coloring`.

    ``protect`` (>1 to enable) multiplies the slack cost of rows that are currently
    SATISFIED: with a uniform elastic cost the QP happily digs one deep new fold
    (cost rho*depth) to fill dozens of shallow shortfalls (gain rho*sum) — the
    whack-a-mole seen on the z=0 dense cluster is built into the uniform
    formulation. Asymmetric costs (SNOPT-style elastic: relax only what is already
    broken) forbid that trade to first order. ``osqp_eps`` tightens the OSQP
    subproblem tolerances (default ~1e-3 leaves a ~1e-5 violation noise floor).

    ``monotone=True`` caps each slack at that row's CURRENT violation
    (``s_i <= viol_i + eps``): linearly, no row may get worse — a per-row hard
    filter (Fletcher-Leyffer style). This subsumes ``protect`` (satisfied rows get
    a hard floor at 0) and closes its leak (sign-based protection leaves
    already-slightly-violated rows as a cheap dumping ground the QP drives deep).

    ``log_every=N`` prints a live progress line every N iterations (and on every
    rejected step) so long solves are observable mid-run, not only post-mortem.

    ``trust_region`` (default on) bounds the step inside the QP (``|d_i| <= delta``)
    and adapts ``delta`` by the actual-vs-predicted merit reduction. The z=0 stall
    trace showed why this is needed: the unbounded QP proposes |d|~2-4 steps whose
    linearization is only valid at a fraction of that length, so the backtracking
    line search crawls (~0.03% merit/iter) and then dies on the first direction with
    no decreasing alpha — a backtrack can only SHORTEN a stale direction, while a
    trust region RE-COMPUTES it under the bound and shrinks instead of quitting.
    ``trust_region=False`` restores the legacy backtracking behaviour.

    ``trace`` (optional dict) turns on pyslsqp-style convergence tracking: it is
    filled with ``iters`` (per-iteration ``max_viol`` / ``n_viol`` / ``merit`` /
    ``step_norm`` / ``osqp_status`` / ``delta`` / ``ratio`` / ``stepped``), the
    ``exit`` reason (``osqp-nonfinite`` / ``step-tol`` / ``linesearch-stall`` /
    ``tr-collapse`` / ``maxiter``), ``feasible`` and ``nit`` — so a stall is
    attributable, not silent.

    quadprog is dense O((n+m)^3); OSQP exploits that the constraint Jacobian is
    ~99% zeros (each Jdet cell touches a small stencil) and reuses the previous
    step as an ADMM warm start, so consecutive near-identical subproblems solve
    far faster. Same merit line search, same convergence behaviour — only the QP
    backend and Jacobian assembly change.

    ``free_idx`` (optional) restricts the optimisation to those variable indices —
    every other variable is frozen at ``flat0``. The windowed driver uses this to
    hold a patch's context ring fixed while only its interior moves; ``cons`` /
    ``cons_jac`` should already be restricted to the enforced constraint rows.
    """
    if not HAS_OSQP:
        raise ImportError(
            "isqp_solve requires osqp (optional dependency) — pip install dvfopt[solvers]"
        )
    from scipy import sparse

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
        jj = cons_jac(f)  # windowed path may already return a sparse enforced-row jac
        j = jj if sparse.issparse(jj) else sparse.csc_matrix(np.asarray(jj))
        return j[:, free] if free_idx is not None else j  # restrict to free columns

    def merit_w(y, w):
        return obj(y) + float(w @ np.clip(-np.asarray(cons(y)), 0, None))

    def _emit(rec):
        """Record + optionally live-log one iteration (every Nth, and every reject)."""
        if trace is not None:
            trace["iters"].append(rec)
        if log_every and (rec["it"] % log_every == 0 or not rec.get("stepped", True)):
            ratio = rec.get("ratio")
            fd = f"{rec['delta']:.3g}" if rec.get("delta") is not None else "-"
            fr = f"{ratio:.2f}" if isinstance(ratio, float) and np.isfinite(ratio) else "-"
            print(
                f"    [isqp] it={rec['it']:4d} viol={rec['max_viol']:.5f} "
                f"n_viol={rec['n_viol']} |d|={rec['step_norm']:.2e} "
                f"delta={fd} ratio={fr} stepped={rec.get('stepped')}",
                flush=True,
            )

    prob = None  # reused across SQP iterations (setup once, then update in place)
    a_pat = None  # (indptr, indices) of A at setup — guards the in-place update
    it = 0
    exit_reason = "maxiter"
    tr_delta, tr_min, tr_max = 2.0, 1e-6, 16.0  # trust-region radius (grid units)
    if trace is not None:
        trace["iters"] = []
    while it < maxiter:
        it += 1
        c = np.asarray(cons(x))  # want >= 0
        viol = np.clip(-c, 0.0, None)
        j = build_j(x)  # (m, n_free), sparse
        m = c.size
        nf = j.shape[1]
        # slack upper bound: inf (classic elastic) or current violation (monotone —
        # linearly no row may get worse; 1e-6 headroom avoids degenerate blocking)
        s_up = (viol + 1e-6) if monotone else np.full(m, np.inf)
        eye_m = sparse.eye(m, format="csc")
        # z = [d (n_free); s (m)];  min 1/2 z^T P z + q^T z  s.t.  l <= A z <= u
        hdv = hess_diag(x)[free]  # objective curvature over free vars
        hd = sparse.diags(hdv)
        p = sparse.block_diag([hd, sparse.csc_matrix((m, m))], format="csc")
        gx = obj_grad(x)[free]
        rho_vec = np.full(m, float(rho))
        if protect != 1.0:
            # asymmetric elastic: currently-satisfied rows are expensive to break, so
            # the QP cannot fund shallow fills by digging new deep folds (whack-a-mole)
            rho_vec[c >= 0.0] = rho * protect
        q = np.concatenate([gx, rho_vec])
        if trust_region:
            # extra identity rows box the step: -delta <= d_i <= delta (see docstring)
            a = sparse.bmat(
                [[j, eye_m], [None, eye_m], [sparse.eye(nf, format="csc"), None]], format="csc"
            )
            lo = np.concatenate([-c, np.zeros(m), np.full(nf, -tr_delta)])
            up = np.concatenate([np.full(m, np.inf), s_up, np.full(nf, tr_delta)])
        else:
            a = sparse.bmat([[j, eye_m], [None, eye_m]], format="csc")  # [Jd+s ; s]
            lo = np.concatenate([-c, np.zeros(m)])
            up = np.concatenate([np.full(m, np.inf), s_up])
        # The Jacobian sparsity pattern is fixed across SQP iterations (the stencil
        # and free set don't change), so factor the KKT once and update values in
        # place — a big saving in the windowed regime (hundreds of small solves).
        same_pattern = (
            a_pat is not None
            and a.indices.shape == a_pat[1].shape
            and (a.indptr == a_pat[0]).all()
            and (a.indices == a_pat[1]).all()
        )
        if prob is not None and same_pattern:
            prob.update(q=q, l=lo, u=up, Px=p.data, Ax=a.data)
        else:
            prob = osqp.OSQP()
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
                max_iter=8000,
                **eps_kw,
            )
            a_pat = (a.indptr.copy(), a.indices.copy())
        res = prob.solve()  # OSQP retains the last solution -> auto warm start on update
        z = np.asarray(res.x)
        if not np.all(np.isfinite(z)):
            exit_reason = "osqp-nonfinite"
            break
        d = np.zeros(n)
        d[free] = z[:nf]  # scatter the free-var step back into the full vector
        dn = float(np.linalg.norm(d))
        ph0 = obj(x) + float(rho_vec @ viol)  # merit(x), reusing the computed c

        def mfun(y, _w=rho_vec):  # this iteration's weighted merit
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
            # ratio test: predicted merit reduction from the QP model (at d=0 the
            # feasible slack is the current violation, so q(0) = rho*sum(viol))
            # vs the ACTUAL nonlinear merit reduction at the full bounded step.
            s_slack = z[nf : nf + m]
            pred = float(rho_vec @ viol) - (
                float(gx @ z[:nf]) + 0.5 * float(z[:nf] @ (hdv * z[:nf])) + float(rho_vec @ s_slack)
            )
            act = ph0 - mfun(x + d)
            ratio = act / pred if pred > 1e-12 else float("nan")
            rec["ratio"] = ratio
            if pred <= 1e-8:
                # model-flat regime (at/near the subproblem optimum): the ratio test
                # is numerical noise there, so polish along d with the legacy
                # backtracking and stop once even that cannot decrease the merit.
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
                    tr_delta *= 0.25  # reject: shrink and RE-SOLVE a new direction
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
    # 1e-6 matches OSQP's practical subproblem accuracy; callers enforce folds via a
    # margin_delta (>=1e-3) shifted target, so this is 3+ orders inside that slack.
    feasible = bool((np.asarray(cons(x)) >= -1e-6).all())
    if trace is not None:
        trace["exit"] = exit_reason
        trace["feasible"] = feasible
        trace["nit"] = it
    return x, it, feasible


__all__ = ['HAS_OSQP', 'isqp_solve']
