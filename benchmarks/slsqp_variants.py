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

The elastic-QP I-SLSQP core and the CPR-coloring Jacobian helpers were
PROMOTED into the library (``dvfopt.core.primitives.isqp`` and
``dvfopt.core.primitives.coloring``, PRs #61-64): this module re-exports the
coloring helpers and keeps :func:`_isqp_solve_osqp` only as a thin back-compat
shim over :func:`dvfopt.core.primitives.isqp.isqp_solve`.
"""

import importlib.util
import time

import numpy as np

from dvfopt.constraints import JdetConstraint2D

# Re-exported for harness/tests — the implementations were promoted into the
# library (see the module docstring). _backtrack backs the quadprog proto below.
from dvfopt.core.primitives.coloring import colored_jacobian, dense_jacobian, jacobian_coloring
from dvfopt.core.primitives.isqp import _backtrack, isqp_solve

SOLVERS = (
    "scipy-slsqp",
    "slsqp-traced",  # vendored scipy C driver + pyslsqp-style tracing (dvfopt.core.primitives.slsqp)
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


def _isqp_solve_osqp(flat0, cons, cons_jac, obj_grad, maxiter, *, constraint=None, **kw):
    """Back-compat shim: the optimized elastic-QP I-SLSQP solver was promoted to
    :func:`dvfopt.core.primitives.isqp.isqp_solve` (which dropped the
    ``constraint=`` CPR-coloring fast path — primitives may not import
    ``dvfopt.constraints``). This harness shim reproduces the old signature
    exactly: when ``constraint`` is supplied, the coloring is built once here
    and wrapped into ``cons_jac`` before delegating (proven identical to the
    old fast path by the promotion's identity gates); all other kwargs
    (``rho``/``tol``/``obj``/``hess_diag``/``free_idx``/``trace``/... ) pass
    straight through.
    """
    if constraint is not None:
        pattern_colors = jacobian_coloring(constraint, flat0)

        def _colored_jac(f):
            return colored_jacobian(constraint, f, *pattern_colors)

        cons_jac = _colored_jac
    return isqp_solve(flat0, cons, cons_jac, obj_grad, maxiter, **kw)


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
        from dvfopt.core.primitives.slsqp import minimize_slsqp_traced

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
