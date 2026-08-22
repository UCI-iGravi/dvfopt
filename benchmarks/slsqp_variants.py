"""Full-grid constrained-Jdet correction with swappable SQP solvers.

Compares SLSQP variants / SQP solvers on the SAME small full-grid feasibility
problem (minimize L2 change subject to per-cell Jdet >= threshold), reusing
dvfopt's ``JdetConstraint2D`` (its ``values`` + point-dependent ``adjoint`` give
the exact constraint Jacobian, so every solver gets analytic gradients — a fair
comparison). Full-grid is only tractable on SMALL grids (~<=40x40); that is why
the real solver windows. Use ``crop_fold_region`` to carve a tractable patch out
of a moderate real slice.

Solvers: ``scipy-slsqp`` (scipy 1.16+ C rewrite of Kraft — the canonical base),
``scipy-trust-constr`` (trust-region SQP), and ``pyslsqp`` (the original Kraft
Fortran + QoL). PySLSQP ships wheels only for Python <=3.12, so it is available
only where importable — ``available_solvers()`` reflects that.
"""

import importlib.util
import time

import numpy as np

from dvfopt.constraints import JdetConstraint2D

SOLVERS = (
    "scipy-slsqp",
    "slsqp-traced",  # vendored scipy C driver + pyslsqp-style tracing (slsqp_traced.py)
    "scipy-trust-constr",
    "scipy-slsqp+trust-constr",  # escalation: SLSQP, fall back to trust-constr on a leftover fold
    "pyslsqp",
    "isqp-proto",
    "isqp-osqp",
)

# Extra module each solver needs beyond scipy (None => scipy builtin, always available).
_SOLVER_DEP = {"pyslsqp": "pyslsqp", "isqp-proto": "quadprog", "isqp-osqp": "osqp"}


def available_solvers():
    """Solvers importable in this environment (pyslsqp needs a py<=3.12 wheel;
    isqp-proto needs quadprog)."""
    return tuple(
        s
        for s in SOLVERS
        if s not in _SOLVER_DEP or importlib.util.find_spec(_SOLVER_DEP[s]) is not None
    )


def dense_jacobian(constraint, flat):
    """Dense (m, n) constraint Jacobian at *flat*: row i = adjoint(flat, e_i).

    dvfopt exposes the constraint's adjoint (Jᵀv) but not J itself; applying it to
    each unit constraint vector recovers the exact rows. Only tractable for small
    constraint counts (crops / windows).
    """
    m = constraint.n_constraints
    eye = np.eye(m)
    return np.stack([constraint.adjoint(flat, eye[i]) for i in range(m)])


def colored_jacobian(constraint, flat, pattern, colors, stride=None):
    """Sparse (m, n) constraint Jacobian via CPR coloring — one adjoint call per
    COLOUR instead of one per row.

    Constraints sharing a colour have DISJOINT variable supports, so probing the
    adjoint with their indicator returns each of their rows superposed on
    non-overlapping columns, and ``colvals[pattern[r]]`` recovers row ``r`` exactly.
    ``pattern``/``colors`` are precomputed once per grid shape (Jdet: pixel grid,
    ``(i%3)*3+j%3``; 2-tri: cell grid, ``triangle*4 + (i%2)*2 + j%2``). ``stride`` is
    accepted for back-compat but ignored — the colour count is ``colors.max()+1``.
    Returns a ``scipy.sparse`` CSC matrix.
    """
    from scipy import sparse

    m, n = constraint.n_constraints, constraint.n_variables
    rows, cols, vals = [], [], []
    for cid in range(int(colors.max()) + 1):
        grp = np.nonzero(colors == cid)[0]
        if grp.size == 0:
            continue
        v = np.zeros(m)
        v[grp] = 1.0
        colvals = constraint.adjoint(flat, v)  # length n; disjoint supports per grp
        for r in grp:
            pr = pattern[r]
            rows.append(np.full(pr.size, r))
            cols.append(pr)
            vals.append(colvals[pr])
    return sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, n)
    )


def jacobian_coloring(constraint, flat0, stride=3, probes=4, seed=0):
    """Precompute (pattern, colors, stride) for :func:`colored_jacobian`.

    ``pattern[r]`` = the nonzero column indices of Jacobian row ``r``, taken as the
    UNION of nonzeros over ``probes`` random perturbations of ``flat0`` (a single
    point can accidentally zero a structurally-nonzero entry, which then corrupts
    the coloring). ``colors[r]`` = ``(i%stride)*stride + j%stride`` over the
    ``H*W`` constraint grid. stride 3 is exact for the radius-1 Jdet stencil.
    """
    h, w = constraint.shape
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(probes):
        b = np.abs(dense_jacobian(constraint, flat0 + rng.normal(0, 0.4, flat0.size))) > 0
        acc = b if acc is None else (acc | b)
    pattern = [np.nonzero(acc[r])[0] for r in range(acc.shape[0])]
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    colors = ((ii % stride) * stride + (jj % stride)).ravel()
    return pattern, colors, stride


def _problem(phi_dydx, threshold, objective="l2", eps=1e-4):
    """Build the correction problem for a ``(2,H,W)`` field.

    Returns ``(constraint, flat0, cons, cons_jac, obj, obj_grad, hess_diag, (h,w))``.
    ``objective`` picks the minimal-displacement metric minimised subject to the
    per-cell Jdet constraint:

    - ``"l2"``: ``||f-f0||^2`` — Hessian is the constant ``2I`` (smooth corrections).
    - ``"l1"``: the eps-smoothed L1 ``sum(sqrt(d^2+eps^2)-eps)`` (matching dvfopt's
      :class:`L1Objective`) — differentiable, so every gradient solver handles it;
      its Gauss-Newton diagonal Hessian is ``eps^2/(d^2+eps^2)^{3/2}``. L1 favours
      SPARSE corrections (few pixels moved a lot) vs L2's spread-out smoothing.

    ``hess_diag(f)`` returns the length-n diagonal of the (Gauss-Newton) objective
    Hessian, consumed by the elastic-QP solvers.
    """
    h, w = phi_dydx.shape[1:]
    c = JdetConstraint2D(shape=(h, w))
    flat0 = np.asarray(c.flatten(phi_dydx), dtype=np.float64)
    n = flat0.size

    def cons(f):  # >= 0
        return np.asarray(c.values(f)) - threshold

    def cons_jac(f):  # dense (m, n) — row i = grad of constraint i
        return dense_jacobian(c, f)

    if objective == "l2":

        def obj(f):
            d = f - flat0
            return float(d @ d)

        def obj_grad(f):
            return 2.0 * (f - flat0)

        def hess_diag(f):
            return np.full(n, 2.0)

    elif objective == "l1":

        def obj(f):
            d = f - flat0
            return float((np.sqrt(d * d + eps * eps) - eps).sum())

        def obj_grad(f):
            d = f - flat0
            return d / np.sqrt(d * d + eps * eps)

        def hess_diag(f):
            # Gauss-Newton diagonal, floored to a proximal/trust-region term: the
            # true GN curvature collapses to ~0 for |d|>>eps, leaving the elastic
            # QP under-determined so the step doesn't reduce the merit and the
            # solver stalls before feasibility. The floor only regularizes the
            # STEP; the L1-shaped gradient still drives sparse corrections.
            d = f - flat0
            return np.maximum(eps * eps / np.power(d * d + eps * eps, 1.5), 0.1)

    else:
        raise ValueError(f"unknown objective {objective!r} (use 'l1' or 'l2')")

    return c, flat0, cons, cons_jac, obj, obj_grad, hess_diag, (h, w)


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


def _isqp_solve(
    flat0, cons, cons_jac, obj_grad, maxiter, rho=1e3, tol=1e-7, obj=None, hess_diag=None
):
    """Gauss-Newton SQP with an elastic-QP subproblem — the I-SLSQP-style prototype.

    The QP curvature comes from ``hess_diag`` (the objective's Gauss-Newton
    diagonal Hessian — constant ``2`` for L2, ``eps^2/(d^2+eps^2)^{3/2}`` for the
    smoothed L1) and the merit uses ``obj``; both default to the L2 case.
    Each iteration solves a QP for a step ``d`` plus non-negative slack ``s`` that
    is ALWAYS feasible (the slack absorbs constraint inconsistency, penalized by
    ``rho``), so the method never bounces on an infeasible linearized subproblem —
    the failure mode that makes plain SLSQP oscillate. The QP is solved by quadprog
    (Goldfarb-Idnani dual active-set), replacing Kraft's fragile dual-LSQ.
    """
    import quadprog

    x = np.asarray(flat0, dtype=np.float64).copy()
    n = x.size
    if hess_diag is None:

        def hess_diag(_f):
            return np.full(n, 2.0)

    if obj is None:

        def obj(y):
            return float((y - flat0) @ (y - flat0))

    def merit(y):
        return obj(y) + rho * np.clip(-np.asarray(cons(y)), 0, None).sum()

    it = 0
    while it < maxiter:
        it += 1
        c = np.asarray(cons(x))  # want >= 0
        j = np.asarray(cons_jac(x))  # (m, n)
        m = c.size
        # QP over z = [d (n); s (m)]:  min 1/2 z^T G z - a^T z  s.t.  C^T z >= b
        big = np.zeros((n + m, n + m))
        big[:n, :n] = np.diag(np.maximum(hess_diag(x), 1e-6))  # quadprog needs SPD
        big[n:, n:] = 1e-6 * np.eye(m)  # tiny reg so G is positive-definite for quadprog
        a = np.concatenate([-obj_grad(x), -rho * np.ones(m)])
        cmat = np.zeros((n + m, 2 * m))
        cmat[:n, :m] = j.T
        cmat[n:, :m] = np.eye(m)  # J d + s >= -c
        cmat[n:, m:] = np.eye(m)  # s >= 0
        b = np.concatenate([-c, np.zeros(m)])
        try:
            d = quadprog.solve_qp(big, a, cmat, b, 0)[0][:n]
        except ValueError:
            break  # QP failed (should be rare with the elastic slack)
        if np.linalg.norm(d) < tol:
            break
        x, stepped = _backtrack(merit, x, d, merit(x))
        if not stepped:
            break
    feasible = bool((np.asarray(cons(x)) >= -1e-9).all())
    return x, it, feasible


def _isqp_solve_osqp(
    flat0,
    cons,
    cons_jac,
    obj_grad,
    maxiter,
    rho=1e3,
    tol=1e-7,
    constraint=None,
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
    """Optimized I-SLSQP: the same elastic-QP SQP as ``_isqp_solve`` but the QP is
    solved by OSQP over a SPARSE system with a warm-started iterate, and the
    constraint Jacobian is rebuilt by CPR coloring (``stride**2`` adjoint calls,
    not ``m``) when a ``constraint`` is supplied.

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
    import osqp
    from scipy import sparse

    x = np.asarray(flat0, dtype=np.float64).copy()
    n = x.size
    free = np.arange(n) if free_idx is None else np.asarray(free_idx)
    coloring = jacobian_coloring(constraint, flat0) if constraint is not None else None
    if hess_diag is None:

        def hess_diag(_f):
            return np.full(n, 2.0)

    if obj is None:

        def obj(y):
            return float((y - flat0) @ (y - flat0))

    def build_j(f):
        if coloring is not None:
            j = colored_jacobian(constraint, f, *coloring)
        else:
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
                float(gx @ z[:nf])
                + 0.5 * float(z[:nf] @ (hdv * z[:nf]))
                + float(rho_vec @ s_slack)
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


def full_grid_correct(phi_dydx, solver, threshold=0.01, maxiter=200, objective="l2", eps=1e-4):
    """Correct a small ``(2, H, W)`` field with *solver*. Returns (phi_out, info).

    ``objective`` is ``"l2"`` (smooth) or ``"l1"`` (eps-smoothed, sparse) — see
    :func:`_problem`. ``info`` records both ``l1_move`` and ``l2_move`` so the two
    objectives' correction footprints can be compared directly.
    """
    c, flat0, cons, cons_jac, obj, obj_grad, hess_diag, (h, w) = _problem(
        phi_dydx, threshold, objective=objective, eps=eps
    )

    def _run_slsqp(x0):
        from scipy.optimize import minimize

        r = minimize(
            obj,
            x0,
            jac=obj_grad,
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
            options={"maxiter": maxiter, "ftol": 1e-8},
        )
        return r.x, int(r.nit), bool(r.success)

    def _run_trust(x0):
        from scipy.optimize import NonlinearConstraint, minimize

        nlc = NonlinearConstraint(cons, 0.0, np.inf, jac=cons_jac)
        r = minimize(
            obj,
            x0,
            jac=obj_grad,
            method="trust-constr",
            constraints=[nlc],
            options={"maxiter": maxiter, "gtol": 1e-8, "xtol": 1e-10},
        )
        return r.x, int(r.niter), bool(r.status in (1, 2))

    def _infeasible(x):  # any strictly-negative determinant left?
        return bool((np.asarray(c.values(x)) < 0.0).any())

    t = time.perf_counter()
    if solver == "scipy-slsqp":
        out, nit, ok = _run_slsqp(flat0)
    elif solver == "scipy-trust-constr":
        out, nit, ok = _run_trust(flat0)
    elif solver == "scipy-slsqp+trust-constr":
        # escalation: fast SLSQP first; only if it leaves a fold fall back to the
        # robust (but slow) trust-constr from the original field, then keep whichever
        # is LESS folded (trust-constr can also fail to converge at a low maxiter, so
        # the hybrid must never do worse than SLSQP alone). SLSQP speed on the easy
        # majority, trust-constr robustness on the hard cases.
        out, nit, ok = _run_slsqp(flat0)
        if _infeasible(out):
            out2, nit2, ok2 = _run_trust(flat0)
            nit += nit2
            if np.asarray(c.values(out2)).min() > np.asarray(c.values(out)).min():
                out, ok = out2, ok2
    elif solver == "slsqp-traced":
        # vendored scipy C driver + pyslsqp-style tracing, run with tracing ON so
        # the benchmark charges it the full QoL cost (scipy-slsqp = untracked base)
        from slsqp_traced import minimize_slsqp_traced

        tr: dict = {}
        r = minimize_slsqp_traced(
            obj,
            flat0,
            jac=obj_grad,
            constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
            maxiter=maxiter,
            ftol=1e-8,
            trace=tr,
        )
        out, nit, ok = r.x, int(r.nit), bool(r.success)
    elif solver == "pyslsqp":  # original Kraft Fortran + QoL (py<=3.12 wheels only)
        import os

        from pyslsqp import optimize

        r = optimize(
            flat0.copy(),
            obj=obj,
            grad=obj_grad,
            con=cons,
            jac=cons_jac,
            meq=0,
            maxiter=maxiter,
            acc=1e-8,
            iprint=0,
            save_itr=None,  # no per-iteration HDF5 recorder
            summary_filename=os.devnull,  # discard the summary file
        )
        out = np.asarray(r["x"])
        nit, ok = int(r.get("num_majiter", -1)), bool(r.get("success", True))
    elif solver == "isqp-proto":
        # L1's weak curvature needs a heavier feasibility penalty; L2 solves the
        # hardest dense crops better at the lighter default.
        rho = 1e4 if objective == "l1" else 1e3
        out, nit, ok = _isqp_solve(
            flat0, cons, cons_jac, obj_grad, maxiter, rho=rho, obj=obj, hess_diag=hess_diag
        )
    elif solver == "isqp-osqp":
        rho = 1e4 if objective == "l1" else 1e3
        out, nit, ok = _isqp_solve_osqp(
            flat0,
            cons,
            cons_jac,
            obj_grad,
            maxiter,
            rho=rho,
            constraint=c,
            obj=obj,
            hess_diag=hess_diag,
        )
    else:
        raise ValueError(f"unknown solver {solver!r}")

    dt = time.perf_counter() - t
    jac_after = np.asarray(c.values(out))
    jac_before = np.asarray(c.values(flat0))
    move = out - flat0
    info = {
        "solver": solver,
        "objective": objective,
        "folds_before": int((jac_before < threshold).sum()),
        "folds_after": int((jac_after < threshold).sum()),
        "min_before": float(jac_before.min()),
        "min_after": float(jac_after.min()),
        "l2_move": float(np.linalg.norm(move)),  # spread of the correction
        "l1_move": float(np.abs(move).sum()),  # total displacement mass (sparsity proxy)
        "n_moved": int((np.abs(move) > 1e-6).sum()),  # how many variables actually changed
        "n_iter": nit,
        # feasibility (no strictly-negative determinants) is the real goal; a
        # feasible result at maxiter shouldn't read as a failure.
        "success": int((jac_after < 0.0).sum()) == 0,
        "converged": bool(ok),  # the solver's own termination flag
        "time_s": dt,
    }
    return c.unflatten(out), info


def crop_fold_region(phi_slice, size=28, threshold=0.01):
    """Crop a ``size x size`` patch of a ``(3,1,H,W)`` slice around its worst fold.

    Returns a ``(2, size, size)`` ``[dy, dx]`` field for the full-grid solvers.
    """
    from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d

    dy, dx = phi_slice[1, 0], phi_slice[2, 0]
    jac = _numpy_jdet_2d(dy, dx)
    yy, xx = np.unravel_index(int(np.argmin(jac)), jac.shape)
    h, w = jac.shape
    y0 = int(np.clip(yy - size // 2, 0, h - size))
    x0 = int(np.clip(xx - size // 2, 0, w - size))
    return np.stack([dy[y0 : y0 + size, x0 : x0 + size], dx[y0 : y0 + size, x0 : x0 + size]])
