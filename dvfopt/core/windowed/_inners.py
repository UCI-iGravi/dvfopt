"""Window sub-problem container + inner-solver dispatch for the windowed engine.

:class:`WindowSub` is the frozen-ring reduced problem the engine builds (one
per window); :func:`solve_window_inner` hands it to an inner solver selected
by label. All inners only move the window's free variables, so the engine's
no-damage invariant holds regardless of the choice.
"""

from dataclasses import dataclass

import numpy as np

from dvfopt.core.primitives.isqp import isqp_solve

_ISQP_LABELS = ("isqp", "isqp-osqp")
_SLSQP_LABELS = ("slsqp", "scipy-slsqp")
_SLSQP_TC_LABELS = ("slsqp+trust-constr", "scipy-slsqp+trust-constr")
# Public: every accepted inner label (canonical names + aliases), so callers
# (e.g. WindowedWrapperStrategy) can validate eagerly at construction.
INNER_LABELS = _ISQP_LABELS + _SLSQP_LABELS + _SLSQP_TC_LABELS


@dataclass
class WindowSub:
    """A window sub-problem ready to hand to the inner solver."""

    constraint: object
    flat0: np.ndarray
    cons: object
    cons_jac: object
    obj: object
    obj_grad: object
    hess_diag: object
    free_idx: np.ndarray
    free_mask: np.ndarray  # (ph, pw) which patch pixels are free (for paste-back)
    patch_box: tuple  # (py0, py1, px0, px1) global coords
    n_enforced: int


def solve_window_inner(
    sub,
    inner,
    maxiter,
    trace=None,
    trust_region=True,
    osqp_max_iter=None,
    qp_backend="osqp",
    ip_cold=True,
    ip_after_admm_iters=800,
):
    """Solve a built window sub-problem with the chosen inner solver, returning
    ``(x_full, n_iter, feasible)`` — ``x_full`` is the full patch flat vector.

    - ``"isqp"`` (default; alias ``"isqp-osqp"``): the tuned elastic-QP SQP over
      the free vars (:func:`dvfopt.core.primitives.isqp.isqp_solve`), UNCHANGED —
      the path every existing test and the no-damage invariant assume.
    - ``"slsqp"`` / ``"slsqp+trust-constr"`` (aliases ``"scipy-slsqp"`` /
      ``"scipy-slsqp+trust-constr"``): the SLSQP leg runs through
      ``dvfopt.core.primitives.slsqp.minimize_slsqp_traced`` — scipy's own C-core
      driver (verified byte-identical to ``minimize(method='SLSQP')``; see
      ``benchmarks/trace_parity_check.py``) with optional pyslsqp-style tracing —
      on the REDUCED free-variable problem (frozen vars pinned at ``sub.flat0``, so
      no-damage still holds by construction). ``+trust-constr`` escalates to scipy
      trust-constr only when SLSQP leaves an enforced row folded, and keeps
      whichever iterate reaches the higher constraint minimum (never worse than
      SLSQP alone).

    ``trust_region`` / ``osqp_max_iter`` / ``qp_backend`` / ``ip_cold`` /
    ``ip_after_admm_iters`` are ``isqp``-only knobs (ignored by the SLSQP legs):
    the engine's per-window fallback re-solves a failed window with
    ``trust_region=False`` (legacy line search), caps the OSQP ADMM iterations
    per subproblem, and ``qp_backend='hybrid'`` routes the cold / long-tail QPs
    to interior-point Clarabel (see
    :class:`dvfopt.core.primitives.isqp._HybridQP`). Their defaults are
    :func:`isqp_solve`'s own.

    ``trace`` (optional dict) is threaded to the inner solver — ``isqp`` and
    the traced SLSQP leg both fill it with per-iteration records + an explicit
    exit reason (house style: ``trace['iters']`` / ``trace['exit']``). Default
    ``None`` keeps behavior byte-identical to the untraced path.
    """
    if inner in _ISQP_LABELS:
        return isqp_solve(
            sub.flat0,
            sub.cons,
            sub.cons_jac,
            sub.obj_grad,
            maxiter,
            obj=sub.obj,
            hess_diag=sub.hess_diag,
            free_idx=sub.free_idx,
            trace=trace,
            trust_region=trust_region,
            osqp_max_iter=osqp_max_iter,
            qp_backend=qp_backend,
            ip_cold=ip_cold,
            ip_after_admm_iters=ip_after_admm_iters,
        )
    if inner not in _SLSQP_LABELS + _SLSQP_TC_LABELS:
        raise ValueError(f"unknown inner {inner!r}; valid labels: {list(INNER_LABELS)}")

    from dvfopt.core.primitives.slsqp import minimize_slsqp_traced

    free = np.asarray(sub.free_idx)
    x0 = sub.flat0.copy()
    if free.size == 0 or sub.n_enforced == 0:
        return x0, 0, True  # nothing to move / no enforced row -> already done

    def embed(zf):
        x = x0.copy()
        x[free] = zf
        return x

    def cons_z(zf):
        return sub.cons(embed(zf))  # enforced rows only (built restricted)

    def jac_z_dense(zf):
        return sub.cons_jac(embed(zf))[:, free].toarray()  # enforced rows, free cols

    def obj_z(zf):
        return sub.obj(embed(zf))

    def grad_z(zf):
        return sub.obj_grad(embed(zf))[free]

    z0 = x0[free]
    r = minimize_slsqp_traced(
        obj_z,
        z0,
        jac=grad_z,
        constraints=[{"type": "ineq", "fun": cons_z, "jac": jac_z_dense}],
        maxiter=maxiter,
        ftol=1e-8,
        trace=trace,
    )
    zf = r.x
    if inner in _SLSQP_TC_LABELS and cons_z(zf).min() < 0:
        from scipy.optimize import NonlinearConstraint, minimize

        r2 = minimize(
            obj_z,
            zf,  # warm-start the escalation from SLSQP's (closest-to-feasible) iterate
            jac=grad_z,
            method="trust-constr",
            constraints=[NonlinearConstraint(cons_z, 0.0, np.inf, jac=jac_z_dense)],
            options={"maxiter": maxiter, "xtol": 1e-10},
        )
        if cons_z(r2.x).min() > cons_z(zf).min():  # keep the better; never worse
            zf = r2.x
    x = embed(zf)
    return x, 0, bool(sub.cons(x).min() >= -1e-9)


__all__ = ['INNER_LABELS', 'WindowSub', 'solve_window_inner']
