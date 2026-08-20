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

SOLVERS = ("scipy-slsqp", "scipy-trust-constr", "pyslsqp")


def available_solvers():
    """Solvers importable in this environment (pyslsqp needs a py<=3.12 wheel)."""
    return tuple(
        s for s in SOLVERS if s.startswith("scipy") or importlib.util.find_spec(s) is not None
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


def _problem(phi_dydx, threshold):
    """Build (constraint, flat0, cons_fn, cons_jac_fn, obj, obj_grad) for a (2,H,W) field."""
    h, w = phi_dydx.shape[1:]
    c = JdetConstraint2D(shape=(h, w))
    flat0 = np.asarray(c.flatten(phi_dydx), dtype=np.float64)

    def cons(f):  # >= 0
        return np.asarray(c.values(f)) - threshold

    def cons_jac(f):  # dense (m, n) — row i = grad of constraint i
        return dense_jacobian(c, f)

    def obj(f):
        d = f - flat0
        return float(d @ d)

    def obj_grad(f):
        return 2.0 * (f - flat0)

    return c, flat0, cons, cons_jac, obj, obj_grad, (h, w)


def full_grid_correct(phi_dydx, solver, threshold=0.01, maxiter=200):
    """Correct a small ``(2, H, W)`` field with *solver*. Returns (phi_out, info)."""
    c, flat0, cons, cons_jac, obj, obj_grad, (h, w) = _problem(phi_dydx, threshold)
    t = time.perf_counter()
    if solver == "scipy-slsqp":
        from scipy.optimize import minimize

        r = minimize(
            obj,
            flat0,
            jac=obj_grad,
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
            options={"maxiter": maxiter, "ftol": 1e-8},
        )
        out, nit, ok = r.x, int(r.nit), bool(r.success)
    elif solver == "scipy-trust-constr":
        from scipy.optimize import NonlinearConstraint, minimize

        nlc = NonlinearConstraint(cons, 0.0, np.inf, jac=cons_jac)
        r = minimize(
            obj,
            flat0,
            jac=obj_grad,
            method="trust-constr",
            constraints=[nlc],
            options={"maxiter": maxiter, "gtol": 1e-8, "xtol": 1e-10},
        )
        out, nit, ok = r.x, int(r.niter), bool(r.status in (1, 2))
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
    else:
        raise ValueError(f"unknown solver {solver!r}")

    dt = time.perf_counter() - t
    jac_after = np.asarray(c.values(out))
    jac_before = np.asarray(c.values(flat0))
    info = {
        "solver": solver,
        "folds_before": int((jac_before < threshold).sum()),
        "folds_after": int((jac_after < threshold).sum()),
        "min_before": float(jac_before.min()),
        "min_after": float(jac_after.min()),
        "l2_move": float(np.linalg.norm(out - flat0)),
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
