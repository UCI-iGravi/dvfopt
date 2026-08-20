"""Windowed fold-correction driver — solves only where the folds are.

Designed from the measured B0039 fold geometry (folds cover ~3-6.5% of a slice,
in many small clusters) rather than from any existing dvfopt loop. A full-grid
solve wastes ~95% of its work on fold-free area; this restricts each solve to a
small window around a fold cluster and freezes a context ring, so the rest of the
slice is untouched *by construction*.

Core invariant (no-damage): moving a pixel changes only constraints whose support
touches it. Both constraint families have finite support, so a window can enforce
*every* constraint a free pixel influences and freeze everything else — no window
solve can create a fold elsewhere. Two subtleties this module handles explicitly:

- **Finite-difference Jdet.** ``det(I+grad phi)`` is evaluated with ``np.gradient``
  (central differences), so a constraint on a patch's *interior-cut* edge uses a
  one-sided difference that DISAGREES with the global field. Such rows are not
  enforced and their pixels are frozen. At a true image border the global field is
  ALSO one-sided there, so those rows are valid and may be enforced/freed. The
  ``damage`` report field (folds created outside every window) verifies the
  invariant held (must be 0).
- **Window coupling.** Clusters are merged (dilate-then-label) so no window's free
  set lands in another's context ring; windows are then independent and only free
  pixels are pasted back.

The inner solve is ``slsqp_variants._isqp_solve_osqp`` (elastic-QP SQP, L1/L2,
warm-started), restricted to the window's free variables. Everything here is the
window *builder*; the solver is unchanged.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slsqp_variants as sv

from dvfopt.constraints import JdetConstraint2D
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d

# Frozen-ring width per family: how far in from an interior patch edge a pixel must
# be before every constraint it influences is enforceable with the correct
# (global-matching) evaluation. Jdet needs 2 (central-diff row must itself be 1 in
# from the edge, and a free pixel's influenced rows are its 4 neighbours).
_RING = {"jdet": 2}


def _constraint(family, shape):
    if family == "jdet":
        return JdetConstraint2D(shape=shape)
    raise ValueError(f"unknown family {family!r}")


_COLORING_CACHE = {}  # (family, ph, pw) -> (pattern, colors, stride)


def _cached_coloring(family, c, shape):
    """CPR coloring for a patch shape. The Jacobian sparsity pattern depends only on
    the shape (not the field values), so it is computed once per shape and reused —
    shapes recur constantly across a volume, so this turns the ~4-dense-build probe
    setup from per-window into per-shape."""
    key = (family, *shape)
    hit = _COLORING_CACHE.get(key)
    if hit is None:
        hit = sv.jacobian_coloring(c, np.random.default_rng(0).normal(0, 0.5, c.n_variables))
        _COLORING_CACHE[key] = hit
    return hit


def pixel_fold_mask(family, phi_dydx, threshold):
    """Boolean ``(H, W)`` mask of folded pixels (Jdet < threshold)."""
    if family == "jdet":
        return _numpy_jdet_2d(phi_dydx[0], phi_dydx[1]) < threshold
    raise ValueError(f"unknown family {family!r}")


def _eval_valid_jdet(ph, pw, at_top, at_bot, at_left, at_right):
    """``(ph, pw)`` bool: constraint rows whose patch central-difference matches the
    global field. Invalid only on an *interior* (non-image-border) patch edge."""
    ax0 = np.ones(ph, bool)
    if not at_top:
        ax0[0] = False
    if not at_bot:
        ax0[-1] = False
    ax1 = np.ones(pw, bool)
    if not at_left:
        ax1[0] = False
    if not at_right:
        ax1[-1] = False
    return ax0[:, None] & ax1[None, :]


@dataclass
class _Sub:
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


def _objective_fns(flat0, objective, eps):
    """L1 (eps-smoothed) or L2 obj / grad / GN-diagonal-Hessian over the full patch."""
    if objective == "l2":
        return (
            lambda f: float((f - flat0) @ (f - flat0)),
            lambda f: 2.0 * (f - flat0),
            lambda f: np.full(f.size, 2.0),
        )
    if objective == "l1":

        def obj(f):
            d = f - flat0
            return float((np.sqrt(d * d + eps * eps) - eps).sum())

        def grad(f):
            d = f - flat0
            return d / np.sqrt(d * d + eps * eps)

        def hess(f):
            d = f - flat0
            return np.maximum(eps * eps / np.power(d * d + eps * eps, 1.5), 0.1)

        return obj, grad, hess
    raise ValueError(f"unknown objective {objective!r}")


def build_subproblem(
    family, phi_dydx, free_box, threshold, objective="l2", eps=1e-2, margin_delta=1e-3
):
    """Build the window sub-problem for a free box ``(fy0, fy1, fx0, fx1)`` (global).

    Expands the free box by the family ring to a patch, instantiates the constraint
    on the patch, selects the free pixels and the enforced constraint rows (those a
    free pixel influences AND that evaluate correctly), and returns a :class:`_Sub`.

    ``margin_delta`` is enforced ON TOP of ``threshold`` (constraints are driven to
    ``threshold + margin_delta``) so that OSQP's ~1e-5 tolerance landing a hair
    short of the active bound still clears the strict ``< threshold`` fold check.
    """
    H, W = phi_dydx.shape[1:]
    ring = _RING[family]
    fy0, fy1, fx0, fx1 = free_box
    py0, py1 = max(0, fy0 - ring), min(H, fy1 + ring)
    px0, px1 = max(0, fx0 - ring), min(W, fx1 + ring)
    patch = np.ascontiguousarray(phi_dydx[:, py0:py1, px0:px1])
    ph, pw = patch.shape[1:]
    c = _constraint(family, (ph, pw))
    flat0 = np.asarray(c.flatten(patch), dtype=np.float64)

    # free pixels (patch-local): the free box, clipped into the patch
    free_mask = np.zeros((ph, pw), bool)
    free_mask[fy0 - py0 : fy1 - py0, fx0 - px0 : fx1 - px0] = True

    # constraint rows that evaluate correctly on this patch
    valid = _eval_valid_jdet(ph, pw, py0 == 0, py1 == H, px0 == 0, px1 == W)
    # rows influenced by a free pixel: a Jdet row depends on its 4 neighbours, so a
    # row is influenced by the free set iff it is 4-adjacent to a free pixel.
    influenced = ndimage.binary_dilation(free_mask, ndimage.generate_binary_structure(2, 1))
    enforced = influenced & valid
    enforced_idx = np.nonzero(enforced.ravel())[0]  # Jdet row = pixel raveled (ph,pw)

    # free variable indices in the constraint's own pack (never hand-packed)
    free_phi = np.stack([free_mask, free_mask]).astype(float)
    free_idx = np.nonzero(np.asarray(c.flatten(free_phi)) > 0.5)[0]

    coloring = _cached_coloring(family, c, (ph, pw))
    target = threshold + margin_delta

    def cons(f):
        return (np.asarray(c.values(f)) - target)[enforced_idx]

    def cons_jac(f):  # sparse (n_enforced, n_vars) — enforced rows only
        return sv.colored_jacobian(c, f, *coloring).tocsr()[enforced_idx]

    obj, grad, hess = _objective_fns(flat0, objective, eps)
    return _Sub(
        c,
        flat0,
        cons,
        cons_jac,
        obj,
        grad,
        hess,
        free_idx,
        free_mask,
        (py0, py1, px0, px1),
        enforced_idx.size,
    )


def find_windows(mask, margin, ring):
    """Free boxes ``(fy0, fy1, fx0, fx1)`` around fold clusters.

    Dilating the fold mask by ``margin+ring`` before labelling merges any clusters
    whose (free box + ring) regions could touch, so the resulting windows are
    independent — no window's free set falls in another's context ring.
    """
    grow = margin + ring
    dil = ndimage.binary_dilation(mask, iterations=grow)
    lbl, n = ndimage.label(dil)
    boxes = []
    H, W = mask.shape
    for sy, sx in ndimage.find_objects(lbl):
        # free box = fold cluster's bbox + margin (the dilated bbox minus the ring)
        fy0, fy1 = max(0, sy.start + ring), min(H, sy.stop - ring)
        fx0, fx1 = max(0, sx.start + ring), min(W, sx.stop - ring)
        boxes.append((fy0, fy1, fx0, fx1))
    return boxes


@dataclass
class WindowRec:
    z: int = -1
    fy0: int = 0
    fx0: int = 0
    ph: int = 0
    pw: int = 0
    n_free: int = 0
    n_enforced: int = 0
    inner_iters: int = 0
    grows: int = 0
    min_before: float = 0.0
    min_after: float = 0.0
    feasible: bool = False
    time_s: float = 0.0


@dataclass
class SliceReport:
    n_windows: int = 0
    folds_before: int = 0
    folds_after: int = 0
    min_before: float = 0.0
    min_after: float = 0.0
    # no-damage invariant: folds created OUTSIDE every window's free region. Must be
    # 0 — it is the proof that windowing (incl. the finite-difference Jdet context
    # ring) never damages untouched area. Distinct from in-window residual (a fold
    # left inside a window because that solve did not fully converge).
    damage: int = 0
    residual_in_window: int = 0
    rounds: int = 0
    time_s: float = 0.0
    windows: list = field(default_factory=list)


def windowed_correct(
    phi_dydx,
    family="jdet",
    objective="l2",
    threshold=0.01,
    margin=3,
    maxiter=400,
    eps=1e-2,
    max_rounds=4,
    z=-1,
    margin_delta=1e-3,
):
    """Correct a full ``(2, H, W)`` slice by solving one small window per fold
    cluster. Returns ``(phi_out, SliceReport)``."""
    phi = np.array(phi_dydx, dtype=np.float64, copy=True)
    ring = _RING[family]
    H, W = phi.shape[1:]
    j0 = _numpy_jdet_2d(phi[0], phi[1])
    orig_fold = j0 < threshold
    rep = SliceReport(folds_before=int(orig_fold.sum()), min_before=float(j0.min()))
    touched = np.zeros((H, W), bool)  # union of all free regions solved
    t0 = time.perf_counter()

    prev_nfold = None
    for _rnd in range(max_rounds):
        mask = pixel_fold_mask(family, phi, threshold)
        nfold = int(mask.sum())
        if nfold == 0:
            break
        if prev_nfold is not None and nfold >= prev_nfold:
            break  # no progress — stop rather than spin
        prev_nfold = nfold
        rep.rounds += 1
        for box in find_windows(mask, margin, ring):
            fy0, fy1, fx0, fx1 = box
            touched[fy0:fy1, fx0:fx1] = True
            _solve_window(
                phi,
                family,
                box,
                threshold,
                objective,
                eps,
                maxiter,
                ring,
                z,
                rep,
                margin_delta=margin_delta,
            )

    jf = _numpy_jdet_2d(phi[0], phi[1])
    after_fold = jf < threshold
    new = after_fold & ~orig_fold
    rep.folds_after = int(after_fold.sum())
    rep.min_after = float(jf.min())
    rep.damage = int((new & ~touched).sum())  # invariant: MUST be 0
    rep.residual_in_window = int((after_fold & touched).sum())
    rep.n_windows = len(rep.windows)
    rep.time_s = time.perf_counter() - t0
    return phi, rep


def _solve_window(
    phi, family, box, threshold, objective, eps, maxiter, ring, z, rep, _grow=0, margin_delta=1e-3
):
    """Solve one window; on infeasibility grow the blocked sides once and retry."""
    H, W = phi.shape[1:]
    sub = build_subproblem(family, phi, box, threshold, objective, eps, margin_delta)
    t = time.perf_counter()
    x, nit, ok = sv._isqp_solve_osqp(
        sub.flat0,
        sub.cons,
        sub.cons_jac,
        sub.obj_grad,
        maxiter,
        constraint=None,
        obj=sub.obj,
        hess_diag=sub.hess_diag,
        free_idx=sub.free_idx,
    )
    dt = time.perf_counter() - t
    patch_out = np.asarray(sub.constraint.unflatten(x))
    py0, py1, px0, px1 = sub.patch_box
    # paste back ONLY free pixels (frozen ring is unchanged and may be shared)
    fm = sub.free_mask
    dst = phi[:, py0:py1, px0:px1]
    dst[:, fm] = patch_out[:, fm]

    jpatch = _numpy_jdet_2d(phi[0, py0:py1, px0:px1], phi[1, py0:py1, px0:px1])
    rec = WindowRec(
        z=z,
        fy0=box[0],
        fx0=box[2],
        ph=py1 - py0,
        pw=px1 - px0,
        n_free=sub.free_idx.size,
        n_enforced=sub.n_enforced,
        inner_iters=nit,
        grows=_grow,
        min_after=float(jpatch.min()),
        feasible=bool(ok),
        time_s=dt,
    )
    rep.windows.append(rec)

    # grow-on-failure: if still infeasible and the window can expand, widen and retry
    if not ok and _grow < 2:
        fy0, fy1, fx0, fx1 = box
        gy0, gy1 = max(0, fy0 - 4), min(H, fy1 + 4)
        gx0, gx1 = max(0, fx0 - 4), min(W, fx1 + 4)
        if (gy0, gy1, gx0, gx1) != box:
            _solve_window(
                phi,
                family,
                (gy0, gy1, gx0, gx1),
                threshold,
                objective,
                eps,
                maxiter,
                ring,
                z,
                rep,
                _grow + 1,
                margin_delta,
            )
