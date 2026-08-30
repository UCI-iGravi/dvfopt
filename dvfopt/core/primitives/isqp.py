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

QP backend
----------
``qp_backend='hybrid'`` (see :func:`_make_qp`) adds an interior-point escape
hatch on top of the warm-started ADMM path. ``'osqp'`` (this module's default)
is the pre-hybrid path, byte for byte.

Step rule
---------
``step_rule='exact_ls'`` replaces the trust-region ratio test's accept/reject
with the EXACT minimiser of the merit function along the QP step (2D only —
:func:`_exact_line_min`). ``'tr'`` (this module's default) is the stock path,
byte for byte.
"""

from types import SimpleNamespace

import numpy as np

from dvfopt._logging import logger

try:
    import osqp

    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

try:
    import clarabel

    HAS_CLARABEL = True
except ImportError:
    HAS_CLARABEL = False

_WARNED_NO_CLARABEL = False


class _HybridQP:
    """OSQP with a Clarabel (interior-point) escape hatch, behind OSQP's own
    ``setup`` / ``update`` / ``solve`` surface — a drop-in for ``osqp.OSQP()``.

    Why: on real giant-tile QPs (16k vars, 27k rows) Clarabel solves in ~0.25 s /
    15-25 IP iterations at ~1e-9 feasibility where OSQP needs 0.4-2.2 s /
    700-4000 ADMM iterations at ~1e-3 — but an in-engine *warm-started* OSQP
    averages 0.175 s/solve, so always-Clarabel is a net loss (raw B0039 z16:
    381 s vs 300 s, and 34% more SQP iterations). The win is using IP exactly
    where the warm start is worth nothing or has gone stale:

    - the window's **cold** first solve (``ip_cold``), and
    - right after an ADMM solve that ran ``>= ip_after_admm_iters`` iterations
      (the tail signal that the warm start stopped helping).

    Measured on raw B0039 z16: 262 s vs 300 s for OSQP-only (-13%), 0 simplex
    folds, damage 0, and a smaller move (L2 325 vs 346). Policy sweep:
    cold-only 296 s, threshold 400 -> 289 s, 800 -> 262 s (best), 1500 ->
    269 s, no-cold/800 -> 281 s.

    An IP solve seeds OSQP's warm start with its own solution, so the next ADMM
    solve starts from the IP point. Any IP failure (bad status, non-finite,
    exception) silently falls through to ADMM — the backend can only be faster,
    never less feasible.
    """

    def __init__(self, ip_cold=True, ip_after_admm_iters=800):
        self._real = osqp.OSQP()
        self._ip_cold = bool(ip_cold)
        self._ip_after = int(ip_after_admm_iters)
        self._last_admm_iters = None  # None == cold (no ADMM solve yet)
        self._p = self._q = self._a = self._lo = self._up = None

    def setup(self, p, q, a, lo, up, **kw):
        self._p, self._q, self._a, self._lo, self._up = p, q, a, lo, up
        self._real.setup(p, q, a, lo, up, **kw)

    def update(self, q=None, l=None, u=None, Px=None, Ax=None):  # `l`: OSQP's own name
        # Px/Ax carry new VALUES for the setup-time sparsity pattern, so the
        # stored csc matrices are refreshed in place (we own the only reference).
        if q is not None:
            self._q = q
        if l is not None:
            self._lo = l
        if u is not None:
            self._up = u
        if Px is not None:
            self._p.data = Px
        if Ax is not None:
            self._a.data = Ax
        self._real.update(q=q, l=l, u=u, Px=Px, Ax=Ax)

    def solve(self):
        cold = self._last_admm_iters is None
        if (self._ip_cold and cold) or (not cold and self._last_admm_iters >= self._ip_after):
            res = self._solve_ip()
            if res is not None:
                self._last_admm_iters = 0  # the next ADMM solve starts from the IP point
                return res
        res = self._real.solve()
        self._last_admm_iters = int(res.info.iter)
        return res

    def _solve_ip(self):
        """One Clarabel solve of the stored QP; ``None`` on any failure (-> ADMM).

        ``l <= A z <= u`` becomes the conic form Clarabel wants: finite-``u``
        rows as ``A z + s = u``, finite-``l`` rows as ``-A z + s = -l``, stacked
        under one nonnegative cone. ``P`` is passed as stored — OSQP's
        upper-triangular convention is Clarabel's too (and this driver's ``P`` is
        diagonal anyway).
        """
        from scipy import sparse

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
            self._real.warm_start(x=x)
            return SimpleNamespace(
                x=x,
                info=SimpleNamespace(iter=int(sol.iterations), status=f"clarabel-{sol.status}"),
            )
        except Exception as exc:  # any IP trouble is a fall-through, never a failure
            logger.debug(f"hybrid QP: interior-point solve failed ({exc!r}); using OSQP")
            return None


def _make_qp(qp_backend, ip_cold, ip_after_admm_iters):
    """QP object for *qp_backend*: a plain ``osqp.OSQP()`` for ``'osqp'`` (the
    pre-hybrid path, byte for byte) or :class:`_HybridQP` for ``'hybrid'``.
    Without ``clarabel`` installed, ``'hybrid'`` IS ``'osqp'`` (logged once)."""
    if qp_backend == "osqp":
        return osqp.OSQP()
    if qp_backend != "hybrid":
        raise ValueError(f"unknown qp_backend {qp_backend!r}; valid: 'osqp', 'hybrid'")
    if not HAS_CLARABEL:
        global _WARNED_NO_CLARABEL
        if not _WARNED_NO_CLARABEL:
            _WARNED_NO_CLARABEL = True
            logger.debug("qp_backend='hybrid' but clarabel is not importable; using OSQP only")
        return osqp.OSQP()
    return _HybridQP(ip_cold, ip_after_admm_iters)


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


def _line_events(c0, g, q, a_hi):
    """Roots of ``c_i(a) = c0 + g a + q a**2`` inside ``(0, a_hi)``, vectorised.

    Returns ``(roots, flags, rows)`` with ``flag = +1`` where the row ENTERS the
    violated region (``c`` goes negative) and ``-1`` where it LEAVES — the
    breakpoints of the merit's hinge terms. A row with ``q > 0`` is negative
    BETWEEN its roots, with ``q < 0`` outside them; a linear row (``q == 0``)
    switches once.
    """
    lin = q == 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        disc = g * g - 4.0 * q * c0
        sq = np.sqrt(np.where(disc > 0.0, disc, 0.0))
        # numerically stable quadratic: t = -(g + sign(g) sqrt(disc)) / 2, roots are
        # t/q and c0/t (never the cancelling (-g + sq) / 2q form)
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


def _exact_line_min(c0, g, q, w, fco, a_hi=1.0):
    """Exact minimiser of the merit ``m(a) = f(a) + sum_i w_i max(0, -c_i(a))``
    on ``[0, a_hi]``, where ``c_i(a) = c0 + g a + q a**2`` and ``f`` has the
    quadratic coefficients ``fco = (f0, f1, f2)``.

    ``m`` is piecewise quadratic with breakpoints exactly at the roots of the
    rows (:func:`_line_events`): sort them, sweep the active set with a
    cumulative sum over ``(w*-c0, w*-g, w*-q)``, and take the best of every
    interval's endpoints and its parabola vertex. O(m log m), fully vectorised.
    Returns ``(a_star, m_star, m_zero)``.
    """
    r, fl, rows = _line_events(c0, g, q, a_hi)
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
    return float(aa[b]), float(vv[b]), float(k0[0])


def _ftol_stop(ftol, feas_tol, viol, merit_before, merit_after):
    """The ``ftol`` exit: feasible within ``feas_tol`` and an accepted step that
    moved the merit by no more than ``ftol`` relative (see :func:`isqp_solve`)."""
    return bool(
        ftol
        and viol.max(initial=0.0) <= feas_tol
        and (merit_before - merit_after) <= ftol * max(abs(merit_before), 1.0)
    )


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
    osqp_max_iter=None,
    monotone=False,
    log_every=0,
    qp_backend='osqp',
    ip_cold=True,
    ip_after_admm_iters=800,
    tr_delta=2.0,
    tr_max=16.0,
    step_rule='tr',
    exact_ls_fallback_steps=0,
    ftol=0.0,
    feas_tol=1e-6,
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
    subproblem tolerances (default ~1e-3 leaves a ~1e-5 violation noise floor);
    ``osqp_max_iter`` caps the ADMM iterations per subproblem (``None`` = OSQP
    setup default of 8000). A low cap trades subproblem accuracy for speed — the
    windowed engine runs 2000 for normal window solves and 500 for its no-trust-
    region fallback, both measured to keep feasibility while ~2x faster.

    ``monotone=True`` caps each slack at that row's CURRENT violation
    (``s_i <= viol_i + eps``): linearly, no row may get worse — a per-row hard
    filter (Fletcher-Leyffer style). This subsumes ``protect`` (satisfied rows get
    a hard floor at 0) and closes its leak (sign-based protection leaves
    already-slightly-violated rows as a cheap dumping ground the QP drives deep).

    ``qp_backend`` picks the QP solver behind the subproblem: ``'osqp'``
    (default here — warm-started ADMM, byte-identical to the pre-hybrid driver)
    or ``'hybrid'`` (:class:`_HybridQP`: interior-point Clarabel on a window's
    cold first solve and after any ADMM solve that ran ``>= ip_after_admm_iters``
    iterations, warm-started OSQP otherwise; ``ip_cold=False`` drops the cold
    leg). The windowed engine defaults to ``'hybrid'`` — 13% faster on raw
    B0039 z16 at unchanged feasibility; see :class:`_HybridQP` for the numbers.

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

    ``tr_delta`` (initial radius, grid units) / ``tr_max`` (cap) size that trust
    region; the shrink floor stays fixed at 1e-6. The default 2.0 is what every
    measured windowed number was taken at. A tighter start trades fidelity for
    speed — raw B0039 z16, windowed engine defaults: ``tr_delta=1.0`` runs 267 s
    / 1022 SQP iterations at L2 move 344 vs 300 s / 1320 / L2 325 at 2.0 (-11%
    wall, -23% iterations, a bigger departure from the input). ``tr_max`` never
    binds on the measured B0039 windows.

    ``step_rule`` picks how the QP step ``d`` is turned into an iterate.
    ``'tr'`` (this driver's default) is the stock trust-region ratio test:
    accept the whole step or reject it and re-solve under a shrunken radius.
    ``'exact_ls'`` instead takes the EXACT minimiser of the merit along ``d``.
    It is exact because every 2D row family here (2tri, bilinear, jdet, finite)
    is a BILINEAR form in ``(dy, dx)``, so along the line ``x + a d`` a row is
    exactly quadratic::

        c(a) = c + a (J d) + a**2 q,     q = cons(x + d) - c - J d

    and that ``cons(x + d)`` is the evaluation the ratio test already makes — so
    the model is free (no extra constraint evaluation) and needs no per-family
    Hessian table. The merit ``m(a) = f(a) + rho . max(0, -c(a))`` is then
    piecewise quadratic and its global minimiser on ``[0, 1]`` (the trust region
    already bounds the QP) is closed form — :func:`_exact_line_min`. The
    objective along the line is fitted from ``obj`` at ``a = 0, 1/2, 1``: EXACT
    for a quadratic objective (``NoneObjective`` / ``L2Objective``), an
    approximation for the eps-smoothed L1 — so the TRUE merit at ``a*`` is
    checked before the step is taken and this iteration falls back to the ``'tr'``
    acceptance if it did not decrease. The trust region is still built, still
    bounds the QP and is still adapted (now from the achieved ``a*``), and the
    ratio test's own futility threshold (achieved <= 1e-3 x predicted) is kept as
    the ``tr-collapse`` termination signal — an exact minimiser always finds SOME
    decrease, so without it a hopeless window grinds instead of handing off to
    the caller's escalation ladder.

    ``feas_tol`` is the violation the returned ``feasible`` flag tolerates (the
    windowed engine passes half its margin: its rows are shifted by the margin so
    that a solve landing a hair short of the active bound is still fold-free, and a
    distance objective parks the solution ON the active rows at ADMM precision).
    ``ftol`` (0 = off) is a relative objective-decrease stop for such
    feasible-within-``feas_tol`` iterates: once an accepted step moves the merit by
    less than ``ftol * max(|merit|, 1)`` the window exits ``'ftol'``. A zero
    objective never needs it (its feasible step is 0, caught by ``tol``); a distance
    objective otherwise polishes along the active rows at a median relative merit
    decrease of 3e-4 per iteration (traced z=440) for fidelity nobody can measure.

    ``exact_ls_fallback_steps`` (0 = off) closes the rest of that hole. The
    futility test only fires when the achieved decrease is small RELATIVE to the
    QP's prediction; a window whose ``a*`` has collapsed keeps clearing that bar
    by a hair while going nowhere, and on the caller's no-trust-region rung there
    is no futility test at all. So count CONSECUTIVE accepted exact steps with
    ``a* < 0.25`` (the same threshold the trust region shrinks at) and, after that
    many, STOP with ``exit`` reason ``'a-collapse'`` — the step is still taken, and
    the caller's escalation ladder (no-TR retry, backend retry, grow) is what
    actually clears such a window. ``'tr-collapse'`` is only cheap because it
    stops; this is the same stop on the signal ``'exact_ls'`` can see.
    **Handing the remaining iterations to the ``'tr'`` acceptance instead of
    stopping was measured and is WORSE** — mid-run the ratio test lands in the
    regime where it ACCEPTS tiny steps rather than rejecting them, so it grinds
    too (z0_sliver: 2350 SQP iterations vs 1684 with no bail at all).
    The consecutive-run length is the discriminator measurement picked: over the
    crop set's first-round windows the longest run of ``a* < 0.25`` is 2 on
    ``z16_twist`` (the window ``'exact_ls'`` turns from a 108-iteration failure
    into a 46-iteration solve) against 4 on ``z0_sliver`` and 6 on ``z0_cluster``,
    so a default of 3 spares the winner and fires on the pathologies. See
    :func:`dvfopt.core.windowed.windowed_correct` for the engine-level table.

    ``'exact_ls'`` is **2D only** (a 6-tet volume row is trilinear, hence cubic
    along a line); :func:`dvfopt.core.windowed.windowed_correct` — the only
    caller — guards that at its entry. Measured there on raw B0039 z16: 200 s /
    563 SQP iterations vs 244 s / 780 at ``'tr'`` (-18% / -28%), 0 folds, damage
    0 and a SMALLER move (L2 268 vs 280); over a 9-real-slice B0039 sample, 9/9
    wall AND iteration wins, -19% wall / -27% iterations in total, with a smaller
    L2 move on every slice. A maximal fold-free step cap was measured and REFUTED
    (it strangles the elastic mechanism); do not add one.

    ``trace`` (optional dict) turns on pyslsqp-style convergence tracking: it is
    filled with ``iters`` (per-iteration ``max_viol`` / ``n_viol`` / ``merit`` /
    ``step_norm`` / ``osqp_status`` / ``delta`` / ``ratio`` / ``stepped``, plus
    ``alpha`` / ``rule`` under ``step_rule='exact_ls'``), the
    ``exit`` reason (``osqp-nonfinite`` / ``step-tol`` / ``linesearch-stall`` /
    ``tr-collapse`` / ``a-collapse`` / ``maxiter``), ``feasible`` and ``nit`` — so a stall is
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
    if step_rule not in ('tr', 'exact_ls'):
        raise ValueError(f"unknown step_rule {step_rule!r}; valid: 'tr', 'exact_ls'")
    from scipy import sparse

    n_small = 0  # consecutive accepted exact steps with a* < the shrink threshold
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
    tr_delta, tr_max = float(tr_delta), float(tr_max)  # trust-region radius (grid units)
    tr_min = 1e-6  # shrink floor -> 'tr-collapse' exit
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
        res = prob.solve()  # OSQP retains the last solution -> auto warm start on update
        z = np.asarray(res.x)
        if not np.all(np.isfinite(z)):
            exit_reason = "osqp-nonfinite"
            break
        d = np.zeros(n)
        d[free] = z[:nf]  # scatter the free-var step back into the full vector
        dn = float(np.linalg.norm(d))
        fx = obj(x)
        ph0 = fx + float(rho_vec @ viol)  # merit(x), reusing the computed c

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
        if step_rule == 'exact_ls':
            # Exact merit line minimisation in place of the ratio test's
            # accept/reject. The rows are exactly quadratic along the line (see the
            # docstring), and q reuses the cons(x + d) the ratio test already makes.
            gl = np.asarray(j @ z[:nf])  # (J d)_i, exact linear term
            ql = np.asarray(cons(x + d)) - c - gl  # exact quadratic term, no table
            fh, f1 = float(obj(x + 0.5 * d)), float(obj(x + d))
            fco = (fx, 4.0 * fh - f1 - 3.0 * fx, 2.0 * f1 + 2.0 * fx - 4.0 * fh)
            a_star, _m_star, _m0 = _exact_line_min(c, gl, ql, rho_vec, fco, 1.0)
            # Guard: only the objective part of the line model is fitted (exact for
            # a quadratic objective, approximate for L1), so verify the TRUE merit
            # before stepping — 'exact_ls' can then never regress a window.
            m_true = mfun(x + a_star * d) if a_star > 0.0 else ph0
            rec['alpha'] = a_star
            rec['rule'] = 'exact_ls'
            if m_true < ph0:
                x = x + a_star * d
                # a*-collapse bail. A collapsed a* is the exact minimiser nibbling
                # along a direction it cannot use — but it still DECREASES the merit,
                # so the futility test below (which compares against the QP's
                # predicted decrease) need not fire, and the window grinds instead of
                # escalating. Count consecutive collapses on the SAME threshold the
                # trust region shrinks at and, after that many, STOP — the caller's
                # escalation ladder is what actually clears such a window, and
                # 'tr-collapse' is only cheap because it stops. (Handing the rest of
                # the call to the ratio test instead was measured WORSE: mid-run it
                # lands in the regime where the ratio test ACCEPTS tiny steps rather
                # than rejecting them, so it grinds too — z0_sliver 2350 SQP
                # iterations vs 1684 with no bail at all.) Also covers the caller's
                # no-trust-region rung, which has no futility test whatsoever.
                small = a_star < 0.25
                n_small = n_small + 1 if small else 0
                if exact_ls_fallback_steps and n_small >= exact_ls_fallback_steps:
                    rec['stepped'] = True
                    _emit(rec)
                    exit_reason = "a-collapse"
                    break
                if a_star * dn < tol:
                    rec['stepped'] = True
                    _emit(rec)
                    exit_reason = "step-tol"
                    break
                if _ftol_stop(ftol, feas_tol, viol, ph0, m_true):
                    rec['stepped'] = True
                    _emit(rec)
                    exit_reason = "ftol"
                    break
                if trust_region:
                    # An exact minimiser always finds SOME decrease, so it never fires
                    # the ratio test's fast bail-out — and a window that cannot be
                    # solved at this size then grinds instead of escalating. Reuse the
                    # ratio test's OWN futility threshold as a termination signal (the
                    # step is still taken); measured load-bearing on sliver-scale
                    # windows (z0_sliver 229 s -> 139 s).
                    pred = float(rho_vec @ viol) - (
                        float(gx @ z[:nf])
                        + 0.5 * float(z[:nf] @ (hdv * z[:nf]))
                        + float(rho_vec @ z[nf : nf + m])
                    )
                    if pred > 1e-8 and (ph0 - m_true) <= 1e-3 * pred:
                        tr_delta *= 0.25
                        if tr_delta < tr_min:
                            exit_reason = "tr-collapse"
                    elif a_star >= 0.9 and dn >= 0.9 * tr_delta:
                        tr_delta = min(tr_delta * 2.0, tr_max)
                    elif small:  # a_star < 0.25 — the collapse threshold above
                        tr_delta = max(tr_delta * 0.5, tr_min)
                rec['stepped'] = True
                _emit(rec)
                if exit_reason == "tr-collapse":
                    break
                continue
            rec['rule'] = 'tr'  # the fitted objective misled -> stock acceptance
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
                    if _ftol_stop(ftol, feas_tol, viol, ph0, ph0 - act):
                        exit_reason = "ftol"
                else:
                    tr_delta *= 0.25  # reject: shrink and RE-SOLVE a new direction
                    if tr_delta < tr_min:
                        exit_reason = "tr-collapse"
            rec["stepped"] = bool(stepped)
            _emit(rec)
            if exit_reason in ("model-flat", "tr-collapse", "ftol"):
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
    feasible = bool((np.asarray(cons(x)) >= -float(feas_tol)).all())
    if trace is not None:
        trace["exit"] = exit_reason
        trace["feasible"] = feasible
        trace["nit"] = it
    return x, it, feasible


__all__ = ['HAS_CLARABEL', 'HAS_OSQP', 'isqp_solve']
