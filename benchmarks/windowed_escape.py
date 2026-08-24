"""Residual-escape modes for the windowed fold-corrector.

The distance-minimising window solve in :mod:`dvfopt.core.windowed` can leave a
few folds behind. That residual is **not** a geometric floor — it is an
OBJECTIVE-BASIN trap: un-folding a deeply inverted cell needs a large node move
that the L1/L2 anchor penalises, so the distance-minimising SQP halts one fold
short. (Verified on B0039 z=16: the *same* window, same freedom, clears to zero
folds the moment the objective is switched off.)

These modes clear that residual within a *single* objective:

- ``baseline``  — the current inner (objective on from the folded start); leaves the residual.
- ``twophase``  — pure feasibility (objective off) to cross the barrier, then re-anchor
  distance from the feasible point. The reference two-phase fix.
- ``weighted``  — Option A: one solve, hard constraints, a per-variable fidelity weight
  that tapers to ~0 on the fold neighbourhood, so the constraint drives the un-fold with
  no fidelity resistance THERE while staying faithful elsewhere.
- ``penalty``   — Option B: one solve, soft fold-penalty + rho continuation (no hard
  barrier), so there is a downhill path *through* the degenerate (zero-area) state.

Every mode moves ONLY a window's free pixels, so the no-damage invariant holds exactly
as in the engine's ``_solve_window``. :func:`repair_residuals` is the public entry
point — additive over :func:`dvfopt.core.windowed.windowed_correct`'s output. It keeps
the historical family-string API (``'jdet'`` / ``'2tri'`` / ``'finite'``), converted to
:mod:`dvfopt.constraints` instances at the library boundary.
"""

import time

import numpy as np
from scipy import ndimage

from dvfopt.constraints import FiniteJdetConstraint2D, JdetConstraint2D, SimplexConstraint2D
from dvfopt.core.primitives.isqp import isqp_solve
from dvfopt.core.windowed import LOCALITY, build_subproblem, find_windows
from dvfopt.core.windowed import min_field as _engine_min_field
from dvfopt.objectives import L1Objective, L2Objective, NoneObjective

_ESCAPE_MODES = ("baseline", "twophase", "weighted", "penalty")

# Historical family-string API -> registered constraint types (the boundary
# where this harness meets the promoted engine).
_FAMILY = {"jdet": JdetConstraint2D, "2tri": SimplexConstraint2D, "finite": FiniteJdetConstraint2D}


def _constraint_of(family, shape):
    """A shape-bound constraint instance for a family string."""
    try:
        ctype = _FAMILY[family]
    except KeyError:
        raise ValueError(f"unknown family {family!r} (choose from {tuple(_FAMILY)})") from None
    return ctype(shape=tuple(shape))


def _objective_of(objective, eps):
    """'l2' / 'l1' / 'none' string -> the dvfopt Objective the engine consumes."""
    if objective == "l2":
        return L2Objective()
    if objective == "l1":
        return L1Objective(eps=eps)
    if objective == "none":
        return NoneObjective()
    raise ValueError(f"unknown objective {objective!r}")


def min_field(family, phi_dydx):
    """Family-string adapter over the engine's per-location constraint map
    (see :func:`dvfopt.core.windowed.min_field`)."""
    return _engine_min_field(_constraint_of(family, phi_dydx.shape[1:]), phi_dydx)


def pixel_fold_mask(family, phi_dydx, threshold):
    """Boolean ``(H, W)`` pixel mask of folds (constraint value < threshold)."""
    return min_field(family, phi_dydx) < threshold


def _fold_weight_field(taint, taper):
    """Per-pixel fidelity weight (Option A): 0 on a folded pixel, ramping to 1 by
    ``taper`` px away. ``taint`` is a (ph, pw) bool pixel mask of the fold region."""
    if not taint.any():
        return np.ones(taint.shape)
    dist = ndimage.distance_transform_edt(~taint)
    return np.clip(dist / max(taper, 1e-9), 0.0, 1.0)


def _weighted_obj_fns(flat0, wvar, objective, eps):
    """Per-variable-weighted L1/L2 obj / grad / GN-Hessian (the Option A objective)."""
    if objective == "l2":
        return (
            lambda f: float(((f - flat0) * wvar) @ (f - flat0)),
            lambda f: 2.0 * wvar * (f - flat0),
            lambda f: 2.0 * wvar,
        )
    if objective == "l1":

        def obj(f):
            d = f - flat0
            return float((wvar * (np.sqrt(d * d + eps * eps) - eps)).sum())

        def grad(f):
            d = f - flat0
            return wvar * d / np.sqrt(d * d + eps * eps)

        def hess(f):
            d = f - flat0
            return wvar * np.maximum(eps * eps / np.power(d * d + eps * eps, 1.5), 0.1)

        return obj, grad, hess
    raise ValueError(f"unknown objective {objective!r}")


def _escape_solve(
    sub,
    mode,
    family,
    threshold,
    maxiter,
    objective="l2",
    eps=1e-2,
    taper=6.0,
    rho_schedule=(1e2, 1e3, 1e4, 1e5),
):
    """Solve a built window sub-problem with a residual-escape strategy, returning
    ``(x_full, n_iter, feasible)``. Frozen vars stay at ``sub.flat0`` (no-damage)."""
    if mode == "baseline":
        return isqp_solve(
            sub.flat0.copy(),
            sub.cons,
            sub.cons_jac,
            sub.obj_grad,
            maxiter,
            obj=sub.obj,
            hess_diag=sub.hess_diag,
            free_idx=sub.free_idx,
        )

    if mode == "twophase":
        # phase 1: pure feasibility (objective off) to cross the fold barrier;
        # phase 2: re-anchor distance from the now-feasible point to recover fidelity.
        xf, n1, _ = isqp_solve(
            sub.flat0.copy(),
            sub.cons,
            sub.cons_jac,
            (lambda f: np.zeros_like(f)),
            maxiter,
            rho=1e5,
            obj=(lambda f: 0.0),
            hess_diag=sub.hess_diag,
            free_idx=sub.free_idx,
        )
        xp, n2, ok = isqp_solve(
            xf,
            sub.cons,
            sub.cons_jac,
            sub.obj_grad,
            maxiter,
            obj=sub.obj,
            hess_diag=sub.hess_diag,
            free_idx=sub.free_idx,
        )
        return xp, n1 + n2, ok

    if mode == "weighted":
        # Option A: one solve, hard constraints, per-variable fidelity weight ~0 on the
        # fold neighbourhood -> 1 far away. The flat objective at the fold lets the
        # constraint drive the un-fold with no fidelity resistance there; faithful else.
        c = sub.constraint
        patch = np.asarray(c.unflatten(sub.flat0))
        taint = pixel_fold_mask(family, patch, threshold)  # (ph, pw)
        wpix = np.maximum(_fold_weight_field(taint, taper), 1e-3)  # floor keeps QP nonsingular
        wvar = np.asarray(c.flatten(np.stack([wpix, wpix])))
        obj, grad, hess = _weighted_obj_fns(sub.flat0, wvar, objective, eps)
        return isqp_solve(
            sub.flat0.copy(),
            sub.cons,
            sub.cons_jac,
            grad,
            maxiter,
            obj=obj,
            hess_diag=hess,
            free_idx=sub.free_idx,
        )

    if mode == "penalty":
        # Option B: soft fold-penalty + continuation, distance objective on THROUGHOUT
        # (uniform fidelity). Realised through the elastic-QP's own slack penalty `rho`:
        # a low rho lets the coordinated QP step VIOLATE constraints (cross the coupled
        # multi-node fold ridge a scalar penalty method stalls on); ramping rho up then
        # enforces feasibility. Warm-started across the ramp; a final high-rho pass snaps
        # to exact 0 folds (a soft penalty is only asymptotically feasible).
        x = sub.flat0.copy()
        nit = 0
        ok = False
        for rho in rho_schedule:
            x, n, ok = isqp_solve(
                x,
                sub.cons,
                sub.cons_jac,
                sub.obj_grad,
                maxiter,
                rho=rho,
                obj=sub.obj,
                hess_diag=sub.hess_diag,
                free_idx=sub.free_idx,
            )
            nit += n
        return x, nit, ok

    raise ValueError(f"unknown escape mode {mode!r}")


def repair_residuals(
    phi_dydx,
    family="2tri",
    threshold=0.01,
    mode="weighted",
    objective="l2",
    margin=20,
    maxiter=800,
    eps=1e-2,
    margin_delta=1e-3,
    taper=6.0,
    rho_schedule=(1e2, 1e3, 1e4, 1e5),
    max_passes=3,
):
    """Clear the folds a distance-minimising solve leaves behind, using a residual-
    escape ``mode`` (see ``_ESCAPE_MODES``). Additive over
    :func:`dvfopt.core.windowed.windowed_correct`'s output — re-windows each residual
    cluster (large ``margin``) and applies the escape solve, pasting back only free
    pixels so the no-damage invariant holds. Returns ``(phi_out, report dict)``."""
    if mode not in _ESCAPE_MODES:
        raise ValueError(f"unknown escape mode {mode!r} (choose from {_ESCAPE_MODES})")
    phi = np.array(phi_dydx, dtype=np.float64, copy=True)
    phi_in = np.array(phi_dydx, dtype=np.float64, copy=True)
    H, W = phi.shape[1:]
    con = _constraint_of(family, (H, W))
    obj = _objective_of(objective, eps)
    ring = LOCALITY[type(con)].ring
    margin = max(margin, ring)
    orig_fold = min_field(family, phi) < threshold
    fb = int(orig_fold.sum())
    touched = np.zeros((H, W), bool)
    t0 = time.perf_counter()
    n_windows = 0
    total_iter = 0
    min_margin = np.inf
    for _p in range(max_passes):
        mask = pixel_fold_mask(family, phi, threshold)
        if not mask.any():
            break
        for box in find_windows(mask, margin, ring):
            fy0, fy1, fx0, fx1 = box
            touched[max(0, fy0 - ring) : fy1 + ring, max(0, fx0 - ring) : fx1 + ring] = True
            sub = build_subproblem(con, phi, box, threshold, obj, margin_delta)
            x, nit, _ = _escape_solve(
                sub,
                mode,
                family,
                threshold,
                maxiter,
                objective=objective,
                eps=eps,
                taper=taper,
                rho_schedule=rho_schedule,
            )
            patch_out = np.asarray(sub.constraint.unflatten(x))
            py0, py1, px0, px1 = sub.patch_box
            dst = phi[:, py0:py1, px0:px1]
            dst[:, sub.free_mask] = patch_out[:, sub.free_mask]
            n_windows += 1
            total_iter += nit
            if sub.n_enforced:
                min_margin = min(min_margin, float(sub.cons(x).min()))
    after_fold = min_field(family, phi) < threshold
    move = phi - phi_in
    return phi, {
        "mode": mode,
        "family": family,
        "objective": objective,
        "folds_before": fb,
        "folds_after": int(after_fold.sum()),
        "damage": int((after_fold & ~orig_fold & ~touched).sum()),
        "n_windows": n_windows,
        "iters": total_iter,
        "l1_move": float(np.abs(move).sum()),
        "l2_move": float(np.linalg.norm(move)),
        "min_margin": (None if not np.isfinite(min_margin) else float(min_margin)),
        "time_s": time.perf_counter() - t0,
    }
