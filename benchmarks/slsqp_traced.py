"""Vendored scipy SLSQP driver with pyslsqp-style convergence tracing.

scipy's C rewrite (>=1.15) moved only the numerical core to C
(``scipy.optimize._slsqplib.slsqp``, reverse communication); the driver loop is
still pure Python (``scipy/optimize/_slsqp_py.py``). This module vendors that
driver and adds an optional ``trace`` dict filled per MAJOR iteration with the
solver internals pyslsqp tracks (objective, max constraint violation,
Lagrangian-gradient optimality, step length ``alpha``, line-search count, BFGS
resets, inconsistent-QP flag, cumulative nfev/ngev, multipliers) — pyslsqp's
QoL at the exact speed of ``minimize(method='SLSQP')``: same C core, same loop;
with ``trace=None`` the arithmetic path is identical to scipy's.

Differences from the scipy original (deliberate trims, not behaviour changes):
finite-difference fallbacks are dropped (this repo always has analytic
jacobians — a constraint without ``'jac'`` raises), and iprint/disp/callback
plumbing is replaced by the trace dict.

Pinned against scipy's PRIVATE API (state-dict/workspace layout of
``_slsqplib``): written from scipy 1.18.0. A layout change in a future scipy
breaks this loudly (import/shape error), not silently — pin scipy.
"""

import numpy as np
import scipy

try:
    from scipy.linalg.lapack import HAS_ILP64
    from scipy.optimize._optimize import (
        OptimizeResult,
        _clip_x_for_func,
        _prepare_scalar_function,
    )
    from scipy.optimize._slsqplib import slsqp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        f"slsqp_traced vendors scipy>=1.15 private internals (found scipy {scipy.__version__}); "
        "pin scipy to a 1.15-1.18 release"
    ) from e

EXIT_MODES = {
    -1: "Gradient evaluation required (g & a)",
    0: "Optimization terminated successfully",
    1: "Function evaluation required (f & c)",
    2: "More equality constraints than independent variables",
    3: "More than 3*n iterations in LSQ subproblem",
    4: "Inequality constraints incompatible",
    5: "Singular matrix E in LSQ subproblem",
    6: "Singular matrix C in LSQ subproblem",
    7: "Rank-deficient equality constraint subproblem HFTI",
    8: "Positive directional derivative for linesearch",
    9: "Iteration limit reached",
}


def minimize_slsqp_traced(
    func,
    x0,
    jac,
    constraints=(),
    bounds=None,
    maxiter=100,
    ftol=1.0e-6,
    trace=None,
    save_x=False,
):
    """Drop-in for ``minimize(..., method='SLSQP')`` with convergence tracing.

    ``constraints`` are old-style dicts (``type``/``fun``/``jac`` — ``jac``
    required). If ``trace`` is a dict it is filled with ``iters`` (one record
    per major iteration), ``exit``/``mode``/``nit``/``nfev``/``ngev`` and the
    final ``multipliers``; ``save_x=True`` additionally snapshots the iterate
    into each record (pyslsqp's iterate saving). Returns a scipy
    ``OptimizeResult`` (including ``multipliers``).
    """
    x = np.asarray(x0, dtype=np.float64).ravel().copy()

    if bounds is None or len(bounds) == 0:
        new_bounds = (-np.inf, np.inf)
    else:
        from scipy.optimize._constraints import old_bound_to_new

        new_bounds = old_bound_to_new(bounds)
    x = np.clip(x, new_bounds[0], new_bounds[1])

    if isinstance(constraints, dict):
        constraints = (constraints,)
    cons = {"eq": (), "ineq": ()}
    for ic, con in enumerate(constraints):
        ctype = con["type"].lower()
        if ctype not in ("eq", "ineq"):
            raise ValueError(f"Unknown constraint type '{con['type']}'.")
        if "fun" not in con or con.get("jac") is None:
            raise ValueError(f"Constraint {ic} needs 'fun' AND analytic 'jac' (vendored driver).")
        cons[ctype] += ({"fun": con["fun"], "jac": con["jac"], "args": con.get("args", ())},)

    meq = sum(len(np.atleast_1d(c["fun"](x, *c["args"]))) for c in cons["eq"])
    mieq = sum(len(np.atleast_1d(c["fun"](x, *c["args"]))) for c in cons["ineq"])
    m = meq + mieq
    n = len(x)

    if bounds is None or len(bounds) == 0:
        xl = np.full(n, np.nan)
        xu = np.full(n, np.nan)
    else:
        bnds = np.asarray(bounds, dtype=float)
        if bnds.shape[0] != n:
            raise IndexError("length of bounds is not compatible with x0")
        xl, xu = bnds[:, 0].copy(), bnds[:, 1].copy()
        infbnd = ~np.isfinite(bnds)  # the C code marks infinite bounds with nans
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    sf = _prepare_scalar_function(func, x, jac=jac, bounds=new_bounds)
    wrapped_fun = _clip_x_for_func(sf.fun, new_bounds)
    wrapped_grad = _clip_x_for_func(sf.grad, new_bounds)

    # Internal solver state matching the C struct SLSQP_static_vars (see the
    # comment block in scipy/optimize/_slsqp_py.py) — modified in place by slsqp().
    state_dict = {
        "acc": ftol,
        "alpha": 0.0,
        "f0": 0.0,
        "gs": 0.0,
        "h1": 0.0,
        "h2": 0.0,
        "h3": 0.0,
        "h4": 0.0,
        "t": 0.0,
        "t0": 0.0,
        "tol": 10.0 * ftol,
        "exact": 0,
        "inconsistent": 0,
        "reset": 0,
        "iter": 0,
        "itermax": int(maxiter),
        "line": 0,
        "m": m,
        "meq": meq,
        "mode": 0,
        "n": n,
    }

    indices = np.zeros([max(m + 2 * n + 2, 1)], dtype=np.int64 if HAS_ILP64 else np.int32)
    # Worst-case workspace (SLSQP+LSQ+LSEI+LDP+NNLS summed; see scipy source).
    buffer_size = (
        n * (n + 1) // 2
        + 3 * m * n
        - (m + 5 * n + 7) * meq
        + 9 * m
        + 8 * n * n
        + 35 * n
        + meq * meq
        + 28
    )
    if mieq == 0:
        buffer_size += 2 * n * (n + 1)
    buffer = np.zeros(max(buffer_size, 1), dtype=np.float64)

    fx = wrapped_fun(x)
    g = wrapped_grad(x)
    mult = np.zeros([max(1, m + 2 * n + 2)], dtype=np.float64)
    C = np.zeros([max(1, m), n], dtype=np.float64, order="F")
    d = np.zeros([max(1, m)], dtype=np.float64)
    _eval_con_normals(C, x, cons, m, meq)
    _eval_constraint(d, x, cons, m, meq)

    if trace is not None:
        trace["iters"] = []
    iter_prev = 0

    while True:
        slsqp(state_dict, fx, g, C, d, x, mult, xl, xu, buffer, indices)

        if state_dict["mode"] == 1:  # objective and constraint evaluation required
            fx = sf.fun(x)
            _eval_constraint(d, x, cons, m, meq)
        if state_dict["mode"] == -1:  # gradient evaluation required
            g = sf.grad(x)
            _eval_con_normals(C, x, cons, m, meq)

        if state_dict["iter"] > iter_prev and trace is not None:
            # major iteration completed — fx/g/C/d are all current at the new x here
            viol = 0.0
            if meq:
                viol = float(np.abs(d[:meq]).max())
            if m > meq:
                viol = max(viol, float(np.clip(-d[meq:], 0.0, None).max()))
            lag_g = g - C.T @ mult[:m] if m else g  # ineq: L = f - mult.c, mult >= 0
            rec = {
                "it": int(state_dict["iter"]),
                "obj": float(fx),
                "max_viol": viol,
                "opt": float(np.linalg.norm(lag_g)),
                "alpha": float(state_dict["alpha"]),
                "line": int(state_dict["line"]),
                "reset": int(state_dict["reset"]),
                "inconsistent": int(state_dict["inconsistent"]),
                "nfev": int(sf.nfev),
                "ngev": int(sf.ngev),
            }
            if save_x:
                rec["x"] = np.copy(x)
            trace["iters"].append(rec)

        if abs(state_dict["mode"]) != 1:  # not an evaluation request => done
            break
        iter_prev = state_dict["iter"]

    if trace is not None:
        trace["exit"] = EXIT_MODES[state_dict["mode"]]
        trace["mode"] = int(state_dict["mode"])
        trace["nit"] = int(state_dict["iter"])
        trace["nfev"] = int(sf.nfev)
        trace["ngev"] = int(sf.ngev)
        trace["multipliers"] = np.copy(mult[:m])

    return OptimizeResult(
        x=x,
        fun=fx,
        jac=g,
        nit=int(state_dict["iter"]),
        nfev=sf.nfev,
        njev=sf.ngev,
        status=int(state_dict["mode"]),
        message=EXIT_MODES[state_dict["mode"]],
        success=(state_dict["mode"] == 0),
        multipliers=mult[:m],
    )


def _eval_constraint(d, x, cons, m, meq):
    """In-place constraint values at x: eq rows first, then ineq (scipy order)."""
    if m == 0:
        return
    row = 0
    for con in cons["eq"]:
        temp = np.atleast_1d(con["fun"](x, *con["args"])).ravel()
        d[row : row + len(temp)] = temp
        row += len(temp)
    row = meq
    for con in cons["ineq"]:
        temp = np.atleast_1d(con["fun"](x, *con["args"])).ravel()
        d[row : row + len(temp)] = temp
        row += len(temp)


def _eval_con_normals(C, x, cons, m, meq):
    """In-place constraint jacobian rows at x: eq first, then ineq."""
    if m == 0:
        return
    row = 0
    for con in cons["eq"]:
        temp = np.atleast_2d(con["jac"](x, *con["args"]))
        C[row : row + temp.shape[0], :] = temp
        row += temp.shape[0]
    row = meq
    for con in cons["ineq"]:
        temp = np.atleast_2d(con["jac"](x, *con["args"]))
        C[row : row + temp.shape[0], :] = temp
        row += temp.shape[0]


if __name__ == "__main__":
    # self-check: byte-identical to scipy's minimize(method='SLSQP') on a random
    # inequality-constrained least-distance problem, and the trace converges.
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    n = 40
    x0 = rng.normal(0, 1, n)
    tgt = rng.normal(0, 1, n)
    A = rng.normal(0, 1, (10, n))
    b = A @ tgt + np.abs(rng.normal(0, 1, 10))  # infeasible at tgt -> active constraints

    def f(x):
        return float((x - tgt) @ (x - tgt))

    def gf(x):
        return 2.0 * (x - tgt)

    cons = [{"type": "ineq", "fun": lambda x: A @ x - b, "jac": lambda x: A}]
    opts = {"maxiter": 100, "ftol": 1e-8}
    ref = minimize(f, x0, jac=gf, method="SLSQP", constraints=cons, options=opts)
    tr: dict = {}
    r = minimize_slsqp_traced(f, x0, jac=gf, constraints=cons, maxiter=100, ftol=1e-8, trace=tr)
    assert (r.status, r.nit) == (ref.status, ref.nit), (r.status, ref.status, r.nit, ref.nit)
    assert np.array_equal(r.x, ref.x), float(np.abs(r.x - ref.x).max())
    assert tr["iters"], "no trace records"
    last = tr["iters"][-1]
    assert last["max_viol"] < 1e-8, last
    print(
        f"self-check OK: identical to scipy (nit={r.nit}, fun={r.fun:.6f}), "
        f"{len(tr['iters'])} trace records, final opt={last['opt']:.2e} "
        f"viol={last['max_viol']:.2e} alpha={last['alpha']:.3f} exit='{tr['exit']}'"
    )
