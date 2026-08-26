"""Windowed fold-correction engine — solves only where the folds are.

dvfopt's **third shared engine** (after ``barrier/_core.py`` and
``schwarz/_common.py``): domain decomposition orthogonal to the inner
solve, carrying no method logic of its own. Promoted from
``benchmarks/windowed_isqp.py`` (PRs #61-64).

Designed from the measured B0039 fold geometry (folds cover ~3-6.5% of a slice,
in many small clusters) rather than from any existing dvfopt loop. A full-grid
solve wastes ~95% of its work on fold-free area; this restricts each solve to a
small window around a fold cluster and freezes a context ring, so the rest of the
slice is untouched *by construction*.

Core invariant (no-damage): moving a pixel changes only constraints whose support
touches it. Every registered constraint family has finite support, so a window can
enforce *every* constraint a free pixel influences and freeze everything else — no
window solve can create a fold elsewhere. Two subtleties this module handles
explicitly:

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

The "inner" contract
--------------------

Each window becomes a :class:`~dvfopt.core.windowed._inners.WindowSub` — a
frozen-ring REDUCED problem: a patch-shaped constraint clone, the patch flat
vector, ``cons``/``cons_jac`` restricted to the enforced rows (driven to
``threshold + margin_delta``), the objective triplet, and the free-variable
indices. :func:`~dvfopt.core.windowed._inners.solve_window_inner` dispatches it
by label (``'isqp'`` default / ``'slsqp'`` / ``'slsqp+trust-constr'``) and
returns ``(x_full, n_iter, feasible)``; frozen variables MUST stay at
``sub.flat0`` and only free pixels are pasted back, so the no-damage invariant
holds for any inner. This is deliberately NOT ``Strategy.fit`` on a crop — a
crop-level Strategy cannot express frozen variables or row restriction (the
seam gap ``core/slsqp_windowed`` documents in its own FOLLOW-UP comment).

Per-constraint locality (ring width, fold map, influenced rows) comes from the
:mod:`~dvfopt.core.windowed._locality` registry. The giant-region tiler
(:func:`_solve_giant_schwarz`) is windowing-specific — ring-inset overlapping
tiles with damage accounting — and deliberately does NOT reuse
``core/schwarz/_common.py``, whose crop-Strategy contract cannot freeze rings.
"""

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage

from dvfopt._logging import log_warning
from dvfopt.objectives import L2Objective, _kind_eps

from ._inners import _ISQP_LABELS, WindowSub, solve_window_inner
from ._locality import _locality_of, min_field, pixel_fold_mask


@dataclass(frozen=True)
class _InnerOpts:
    """Inner-solve and giant-tiler knobs threaded from :func:`windowed_correct`.

    Bundled rather than passed as seven more positionals through the round loop /
    giant tiler / mop. See :func:`windowed_correct` for what each one does.
    """

    no_tr_fallback: bool = True
    fallback_maxiter: int = 200
    qp_max_iter: int | None = 2000  # None -> OSQP's own default (8000)
    qp_max_iter_fallback: int | None = 500
    giant_tile: int = 64
    giant_max_sweeps: int = 8
    giant_tile_fit: bool = True
    qp_backend: str = 'hybrid'
    ip_cold: bool = True
    ip_after_admm_iters: int = 800


def _objective_fns(flat0, objective):
    """L1 (eps-smoothed) or L2 obj / grad / GN-diagonal-Hessian over the full patch.

    Adapted from the :class:`~dvfopt.objectives.Objective` via ``_kind_eps`` —
    the L1 smoothing eps comes from the objective itself
    (``L1Objective(eps=...)``), not from any engine knob.
    """
    kind, eps = _kind_eps(objective)
    if kind == "l2":
        return (
            lambda f: float((f - flat0) @ (f - flat0)),
            lambda f: 2.0 * (f - flat0),
            lambda f: np.full(f.size, 2.0),
        )
    if kind == "l1":

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
    if kind == "none":
        # Pure feasibility: no distance anchor, only the elastic-QP constraint drive.
        # The flat unit Hessian keeps the QP positive-definite. Use this to clear an
        # objective-basin trap the distance objective pins (see the z=16 analysis).
        return (
            lambda f: 0.0,
            lambda f: np.zeros_like(f),
            lambda f: np.full(f.size, 2.0),
        )
    raise ValueError(f"unknown objective {kind!r}")


def build_subproblem(constraint, phi_dydx, free_box, threshold, objective=None, margin_delta=1e-3):
    """Build the window sub-problem for a free box ``(fy0, fy1, fx0, fx1)`` (global).

    Expands the free box by the family ring to a patch, instantiates a
    patch-shaped clone of ``constraint``'s type, selects the free pixels and the
    enforced constraint rows (those a free pixel influences AND that evaluate
    correctly), and returns a :class:`~dvfopt.core.windowed._inners.WindowSub`.

    ``margin_delta`` is enforced ON TOP of ``threshold`` (constraints are driven to
    ``threshold + margin_delta``) so that OSQP's ~1e-5 tolerance landing a hair
    short of the active bound still clears the strict ``< threshold`` fold check.
    """
    H, W = phi_dydx.shape[1:]
    loc = _locality_of(constraint)
    ring = loc.ring
    fy0, fy1, fx0, fx1 = free_box
    py0, py1 = max(0, fy0 - ring), min(H, fy1 + ring)
    px0, px1 = max(0, fx0 - ring), min(W, fx1 + ring)
    patch = np.ascontiguousarray(phi_dydx[:, py0:py1, px0:px1])
    ph, pw = patch.shape[1:]
    c = type(constraint)(shape=(ph, pw))
    flat0 = np.asarray(c.flatten(patch), dtype=np.float64)

    # free pixels (patch-local): the free box, clipped into the patch
    free_mask = np.zeros((ph, pw), bool)
    free_mask[fy0 - py0 : fy1 - py0, fx0 - px0 : fx1 - px0] = True

    enforced_idx, jac_of = loc.influenced(
        c, free_mask, ph, pw, (py0 == 0, py1 == H, px0 == 0, px1 == W)
    )

    # free variable indices in the constraint's own pack (never hand-packed)
    free_phi = np.stack([free_mask, free_mask]).astype(float)
    free_idx = np.nonzero(np.asarray(c.flatten(free_phi)) > 0.5)[0]

    target = threshold + margin_delta

    def cons(f):
        return (np.asarray(c.values(f)) - target)[enforced_idx]

    def cons_jac(f):  # sparse (n_enforced, n_vars) — enforced rows only
        return jac_of(f)[enforced_idx]

    obj, grad, hess = _objective_fns(flat0, L2Objective() if objective is None else objective)
    return WindowSub(
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

    Dilating the fold mask by ``margin+ring`` before labelling merges clusters whose
    (free box + ring) regions could touch, so a window's free set does not fall in
    another's context ring. (Two diagonally-offset clusters can still yield
    overlapping bounding boxes and hence overlapping free boxes — that only causes
    redundant/overwritten work, never damage, since each window enforces its full
    influence set; see the module invariant.)

    The dilated bbox is inset by ``ring`` to recover the ``cluster + margin`` free
    box — but NOT on a side that reached the image border, where the fold sits on
    the border line and must stay free (mirrors the Schwarz tiler's border guard).
    """
    grow = margin + ring
    dil = ndimage.binary_dilation(mask, iterations=grow)
    lbl, n = ndimage.label(dil)
    boxes = []
    H, W = mask.shape
    for sy, sx in ndimage.find_objects(lbl):
        fy0 = sy.start + ring if sy.start > 0 else 0  # keep image-border folds free
        fy1 = sy.stop - ring if sy.stop < H else H
        fx0 = sx.start + ring if sx.start > 0 else 0
        fx1 = sx.stop - ring if sx.stop < W else W
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
    fallback: bool = False  # the no-trust-region retry ran on this window
    backend_fallback: bool = False  # the QP-backend retry (hybrid -> osqp) ran
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
    damage_coords: list = field(default_factory=list)  # (y,x) of damage folds — debug
    giant_boxes: list = field(default_factory=list)  # free boxes routed to the tiler
    residual_in_window: int = 0
    # connected fold regions too big for a single QP (> max_window_area) that were
    # cleared by overlapping-tile Schwarz decomposition instead. Reported so the
    # cost/quality of the tiled path can be tracked separately from single windows.
    giant_regions: int = 0
    # terminal "mop" pass: after the round loop plateaus, the residual (which the
    # B0039 z=0 diagnostic showed is boundary-stuck INSIDE giants — a big frozen-
    # exterior window clears what a small one can't) is re-windowed with a large
    # margin. mop_windows counts those solves; mop_cleared = folds cleared by them.
    mop_windows: int = 0
    mop_cleared: int = 0
    # windows whose interior-point trajectory failed and were retried on plain
    # warm-started ADMM (qp_backend='hybrid' only; see _solve_window).
    backend_fallbacks: int = 0
    rounds: int = 0
    time_s: float = 0.0
    windows: list = field(default_factory=list)
    # per-stage convergence entries, filled only when ``record_history=True``
    # (each: name / n_iter / n_neg / min_T / wall_s; the 'final' entry adds
    # ``extras`` with damage / n_windows / giant_regions / mop_cleared /
    # l1_move / l2_move). Empty list otherwise.
    history: list = field(default_factory=list)


def windowed_correct(
    phi_in,
    inner="isqp",
    *,
    constraint,
    objective=None,
    threshold,
    margin=3,
    maxiter=400,
    max_rounds=8,
    margin_delta=1e-3,
    max_window_area=3000,
    mop_margin=25,
    no_tr_fallback=True,
    fallback_maxiter=200,
    qp_max_iter=2000,
    qp_max_iter_fallback=500,
    giant_tile=64,
    giant_max_sweeps=8,
    giant_tile_fit=True,
    qp_backend='hybrid',
    ip_cold=True,
    ip_after_admm_iters=800,
    time_budget_s=None,
    verbose=1,
    record_history=False,
    step_callback=None,
):
    """Correct a full ``(2, H, W)`` slice by solving one small window per fold
    cluster. Returns ``(phi_out, SliceReport)``.

    ``constraint`` is a registered 2D constraint instance
    (:class:`~dvfopt.constraints.JdetConstraint2D`,
    :class:`~dvfopt.constraints.SimplexConstraint2D`,
    :class:`~dvfopt.constraints.SimplexConstraint2DBilinear`, or
    :class:`~dvfopt.constraints.FiniteJdetConstraint2D` — see
    :data:`~dvfopt.core.windowed._locality.LOCALITY`); ``objective`` is a
    :class:`~dvfopt.objectives.Objective` (``None`` -> ``L2Objective()``; the
    L1 smoothing eps rides on ``L1Objective(eps=...)``).

    ``max_window_area`` caps a window's free-box area; larger merged clusters are
    cleared by overlapping-tile Schwarz decomposition (``report.giant_regions``)
    instead of an intractable near-full-grid QP. ``margin`` is clamped to at least
    the family ring (the frozen inset band must stay fold-free).

    ``inner`` selects the per-window solver — ``"isqp"`` (default, the tuned
    elastic-QP SQP), ``"slsqp"``, or ``"slsqp+trust-constr"`` (see
    :func:`~dvfopt.core.windowed._inners.solve_window_inner` for the aliases).
    All inners only move the window's free pixels, so the no-damage invariant
    holds regardless of the choice.

    After the round loop plateaus, a terminal **mop** pass re-windows any residual
    with ``mop_margin`` (>> ``margin``): the diagnostic on the densest slices shows
    the plateau is boundary-stuck folds inside the giants that a small window's
    tight frozen boundary can't clear but a large frozen-exterior window can (the
    analogue of the 2.5D pipeline's ``mop_interior_3d``). ``mop_margin=0`` disables.

    Four knobs tune the inner solves (all ``isqp``-only, defaults measured on
    the hard B0039 crops):

    - ``no_tr_fallback`` (default on) — a window that fails to reach its target
      is retried ONCE, same box, with the trust region off (legacy backtracking
      line search) before grow-on-failure. The TR ratio test freezes on
      sliver-scale violations (~1e-4, inside OSQP's own noise) that the line
      search still clears; the retry warm-starts from the failed iterate and
      keeps whichever result has the higher constraint minimum (never worse).
      ``fallback_maxiter`` is that retry's SQP iteration budget (the line search
      otherwise runs far past convergence).
    - ``qp_max_iter`` / ``qp_max_iter_fallback`` — OSQP ADMM iteration cap per
      subproblem for normal / fallback solves (``None`` = OSQP's 8000 default).
      2000/500 keeps the hard crops at zero simplex folds at ~2x the speed.
    - ``qp_backend`` (default ``'hybrid'``) / ``ip_cold`` /
      ``ip_after_admm_iters`` — which QP solver backs each subproblem.
      ``'hybrid'`` runs interior-point Clarabel on a window's cold first solve
      and after any ADMM solve that hit ``>= ip_after_admm_iters`` iterations
      (the stale-warm-start signal), warm-started OSQP otherwise: raw B0039 z16
      262 s vs 300 s (-13%), zero simplex folds, damage 0, smaller move (L2 325
      vs 346). ``'osqp'`` restores the pre-hybrid path byte for byte, and is
      what ``'hybrid'`` degrades to when ``clarabel`` is not installed. See
      :class:`dvfopt.core.primitives.isqp._HybridQP` for the policy sweep.

    ``giant_tile`` / ``giant_max_sweeps`` size the overlapping-tile Schwarz
    decomposition of an over-``max_window_area`` region: square tiles of
    ``giant_tile`` px stepped by ``giant_tile - (2 * ring + 2)``, swept at most
    ``giant_max_sweeps`` times (the sweep loop stops early once the region is
    clear or stops improving). Bigger tiles mean fewer Schwarz seams and fewer
    sweeps: on a full raw B0039 z16 slice (bilinear rows, objective ``none``)
    ``giant_tile=64`` ran 362 s / 22 windows / 1 round / no mop vs 685 s /
    264 windows / 3 rounds at 32 — 1.9x faster, zero simplex folds and zero
    damage either way, and a *smaller* move (L2 316 vs 404). 64 is the default.

    ``giant_tile_fit=True`` (default) makes ``giant_tile`` a *target* rather
    than a literal size: the effective tile is fitted per region so an integer
    number of near-equal tiles covers its longest side (:func:`_fit_tile`).
    Tile size matters through grid *alignment* — the sweep-round count — not
    through size itself. ``False`` restores the literal ``giant_tile``.

    ``time_budget_s`` (``None`` = unlimited) is checked at round boundaries and
    before each window solve; on expiry the engine stops, logs a warning, and
    finishes accounting on the best-so-far field. ``record_history=True`` fills
    ``report.history`` with per-stage convergence entries; ``step_callback``
    receives ``{'phi': ..., 'stage': ...}`` snapshots after each round / giant /
    mop (``KeyboardInterrupt`` propagates as the documented Stop). All three are
    no-ops at their defaults — the default path is byte-identical to the
    promoted benchmark driver. ``verbose`` reserves the standard solver
    verbosity contract (the engine itself emits no progress lines; warnings
    surface through the ``dvfopt`` logger regardless).
    """
    loc = _locality_of(constraint)
    opts = _InnerOpts(
        no_tr_fallback,
        fallback_maxiter,
        qp_max_iter,
        qp_max_iter_fallback,
        giant_tile,
        giant_max_sweeps,
        giant_tile_fit,
        qp_backend,
        ip_cold,
        ip_after_admm_iters,
    )
    objective = L2Objective() if objective is None else objective
    phi = np.array(phi_in, dtype=np.float64, copy=True)
    ring = loc.ring
    margin = max(margin, ring)  # inset band must be fold-free margin, never < ring
    H, W = phi.shape[1:]
    j0 = min_field(constraint, phi)
    orig_fold = j0 < threshold
    rep = SliceReport(folds_before=int(orig_fold.sum()), min_before=float(j0.min()))
    touched = np.zeros((H, W), bool)  # union of every window's ENFORCED footprint
    t0 = time.perf_counter()
    deadline = None if time_budget_s is None else t0 + float(time_budget_s)

    def _expired():
        return deadline is not None and time.perf_counter() > deadline

    def _fire(stage, phi_snap):
        """Forward an intermediate phi snapshot to ``step_callback`` so
        the live-viz GUI can scrub through each cluster splice. Buggy
        callbacks are silenced; KeyboardInterrupt propagates as the
        documented stop signal."""
        if step_callback is None:
            return
        try:
            step_callback({'phi': np.asarray(phi_snap).copy(), 'stage': stage})
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log_warning(f'step_callback raised {type(exc).__name__}: {exc}; continuing')

    def _stage_entry(name, w_start):
        """One history entry: stage name + current fold stats + wall clock."""
        mf = min_field(constraint, phi)
        return {
            "name": name,
            "n_iter": int(sum(r.inner_iters for r in rep.windows[w_start:])),
            "n_neg": int((mf < threshold).sum()),
            "min_T": float(mf.min()),
            "wall_s": time.perf_counter() - t0,
        }

    budget_hit = False
    prev_nfold = None
    for _rnd in range(max_rounds):
        if _expired():
            budget_hit = True
            break
        mask = pixel_fold_mask(constraint, phi, threshold)
        nfold = int(mask.sum())
        if nfold == 0:
            break
        if prev_nfold is not None and nfold >= prev_nfold:
            break  # no progress — stop rather than spin
        prev_nfold = nfold
        rep.rounds += 1
        round_w0 = len(rep.windows)
        for box in find_windows(mask, margin, ring):
            if _expired():
                budget_hit = True
                break
            fy0, fy1, fx0, fx1 = box
            # touched = the ENFORCED footprint (free box dilated by ring), not the
            # bare free box: a free pixel influences constraints up to `ring` beyond
            # the free box, so an infeasible solve could leave a violated row there.
            # Marking it touched makes any such residual count as residual, never
            # damage — so damage=0 is by construction, not merely for feasible solves.
            touched[max(0, fy0 - ring) : fy1 + ring, max(0, fx0 - ring) : fx1 + ring] = True
            if (fy1 - fy0) * (fx1 - fx0) > max_window_area:
                # too big for one QP -> overlapping-tile Schwarz decomposition
                rep.giant_regions += 1
                rep.giant_boxes.append(box)
                giant_w0 = len(rep.windows)
                _solve_giant_schwarz(
                    phi,
                    constraint,
                    box,
                    threshold,
                    objective,
                    maxiter,
                    ring,
                    rep,
                    margin_delta,
                    inner=inner,
                    opts=opts,
                )
                if record_history:
                    rep.history.append(_stage_entry("giant", giant_w0))
                _fire("giant", phi)
                continue
            _solve_window(
                phi,
                constraint,
                box,
                threshold,
                objective,
                maxiter,
                ring,
                rep,
                margin_delta=margin_delta,
                inner=inner,
                opts=opts,
            )
        if record_history:
            rep.history.append(_stage_entry(f"round{rep.rounds}", round_w0))
        _fire(f"round{rep.rounds}", phi)
        if budget_hit:
            break

    if budget_hit:
        log_warning("windowed_correct: time budget exhausted; stopping with best-so-far field")

    # terminal mop: clear the boundary-stuck residual the round loop plateaued on
    if mop_margin > 0 and not budget_hit:
        before_mop = int(pixel_fold_mask(constraint, phi, threshold).sum())
        if before_mop > 0:
            mop_w0 = len(rep.windows)
            _mop_pass(
                phi,
                constraint,
                threshold,
                objective,
                maxiter,
                ring,
                rep,
                margin_delta,
                touched,
                mop_margin,
                max_window_area,
                inner=inner,
                opts=opts,
            )
            rep.mop_cleared = before_mop - int(pixel_fold_mask(constraint, phi, threshold).sum())
            if record_history:
                rep.history.append(_stage_entry("mop", mop_w0))
            _fire("mop", phi)

    jf = min_field(constraint, phi)
    after_fold = jf < threshold
    new = after_fold & ~orig_fold
    rep.folds_after = int(after_fold.sum())
    rep.min_after = float(jf.min())
    damage_mask = new & ~touched
    rep.damage = int(damage_mask.sum())  # invariant: MUST be 0
    rep.damage_coords = [tuple(int(v) for v in c) for c in np.argwhere(damage_mask)[:20]]
    rep.residual_in_window = int((after_fold & touched).sum())
    rep.n_windows = len(rep.windows)
    rep.time_s = time.perf_counter() - t0
    if record_history:
        move = phi - np.asarray(phi_in, dtype=np.float64)
        rep.history.append(
            {
                "name": "final",
                "n_iter": 0,
                "n_neg": rep.folds_after,
                "min_T": rep.min_after,
                "wall_s": rep.time_s,
                "extras": dict(
                    damage=rep.damage,
                    n_windows=rep.n_windows,
                    giant_regions=rep.giant_regions,
                    mop_cleared=rep.mop_cleared,
                    l1_move=float(np.abs(move).sum()),
                    l2_move=float(np.linalg.norm(move.ravel())),
                ),
            }
        )
    return phi, rep


def _mop_pass(
    phi,
    constraint,
    threshold,
    objective,
    maxiter,
    ring,
    rep,
    margin_delta,
    touched,
    mop_margin,
    max_window_area,
    max_sweeps=3,
    inner="isqp",
    opts=None,
):
    """Clear the boundary-stuck residual the round loop plateaued on. Each residual
    cluster is re-solved with a LARGE margin so its neighbourhood is free — no tight
    interior frozen boundary near the fold, the condition the z=0 diagnostic showed
    clears folds a small window can't. Solved WHOLE (tiling would just re-introduce
    the frozen boundaries) up to a generous cap; the rare over-cap cluster falls back
    to Schwarz. Big windows may overlap — harmless, each enforces its own footprint.
    Sweeps until no further progress, i.e. the genuine local floor."""
    H, W = phi.shape[1:]
    whole_cap = 4 * max_window_area  # the mop is allowed much larger single QPs
    for _sweep in range(max_sweeps):
        mask = pixel_fold_mask(constraint, phi, threshold)
        n = int(mask.sum())
        if n == 0:
            break
        lbl, _ = ndimage.label(mask)  # raw residual clusters (per connected component)
        for sy, sx in ndimage.find_objects(lbl):
            fy0, fy1 = max(0, sy.start - mop_margin), min(H, sy.stop + mop_margin)
            fx0, fx1 = max(0, sx.start - mop_margin), min(W, sx.stop + mop_margin)
            box = (fy0, fy1, fx0, fx1)
            touched[max(0, fy0 - ring) : fy1 + ring, max(0, fx0 - ring) : fx1 + ring] = True
            rep.mop_windows += 1
            if (fy1 - fy0) * (fx1 - fx0) > whole_cap:
                _solve_giant_schwarz(
                    phi,
                    constraint,
                    box,
                    threshold,
                    objective,
                    maxiter,
                    ring,
                    rep,
                    margin_delta,
                    inner=inner,
                    opts=opts,
                )
            else:
                _solve_window(
                    phi,
                    constraint,
                    box,
                    threshold,
                    objective,
                    maxiter,
                    ring,
                    rep,
                    margin_delta=margin_delta,
                    inner=inner,
                    opts=opts,
                )
        if int(pixel_fold_mask(constraint, phi, threshold).sum()) >= n:
            break  # no progress -> genuine local floor


def _fit_tile(h, w, target, lo_frac=0.75, hi_frac=1.5):
    """Fit a giant region's tile size to its geometry: the largest tile no bigger
    than ``target`` that covers the region's longest side with an integer number
    of near-equal tiles, clamped to ``[lo_frac, hi_frac] * target``.

    Tile size matters through *alignment* — how many Schwarz sweep rounds the
    tiling needs — not through the size itself. A tile that leaves a thin
    remainder strip along the long side costs an extra round to propagate
    through. Measured on the raw B0039 z16 giant (a 125x152 box): tile 64
    happens to align (1 round, 374 s) while 56 and 80 do not (2 rounds, ~600 s);
    the fitted ``_fit_tile(125, 152, 64) == 51`` aligns by construction
    (1 round, 345 s). The clamp keeps a region smaller than ``target`` from
    collapsing the tile — and with it the point of tiling. A heuristic, not a
    guarantee: tiles step by ``tile - overlap``, so exact integer coverage of
    the side is approximate, and only the region's longest side is fitted.
    """
    n = max(1, -(-max(h, w) // target))  # tiles along the longest side
    tile = -(-max(h, w) // n)
    return int(min(max(tile, math.ceil(lo_frac * target)), math.ceil(hi_frac * target)))


def _solve_giant_schwarz(
    phi,
    constraint,
    giant_box,
    threshold,
    objective,
    maxiter,
    ring,
    rep,
    margin_delta,
    inner="isqp",
    opts=None,
):
    """Clear a large connected fold region by overlapping-tile (additive Schwarz)
    decomposition. Each tile is an ordinary window (frozen ring = current iterate);
    tiles overlap so a fold on one tile's seam is interior to a neighbour, and
    repeated sweeps propagate the correction across the whole region. Returns folds
    remaining in the region. Tile size / sweep cap come from ``opts``
    (``giant_tile`` / ``giant_max_sweeps`` / ``giant_tile_fit``, see
    :func:`windowed_correct` and :func:`_fit_tile`).

    Windowing-specific and NOT :mod:`dvfopt.core.schwarz` — that engine's
    crop-Strategy contract cannot freeze rings or restrict enforced rows, which
    is exactly what the damage accounting here relies on.

    No-damage by construction: the tiled region is INSET by ``ring`` from each
    interior giant edge, so a tile-free pixel can only influence constraints inside
    the giant box (= ``touched``); nothing outside can be changed, let alone folded.
    The inset band is fold-free margin (needs ``margin >= ring``), so insetting
    leaves no fold unfixed. Image-border edges are not inset — no "outside" there.
    Without the inset, an infeasible edge-tile solve can leave a boundary guard row
    just outside the giant violated -> a damage fold (observed on B0039 z=0)."""
    opts = _InnerOpts() if opts is None else opts
    tile, max_sweeps = opts.giant_tile, opts.giant_max_sweeps
    H, W = phi.shape[1:]
    fy0, fy1, fx0, fx1 = giant_box
    if opts.giant_tile_fit:
        tile = _fit_tile(fy1 - fy0, fx1 - fx0, tile)
    it0 = fy0 + (ring if fy0 > 0 else 0)  # inset interior edges; keep image borders
    it1 = fy1 - (ring if fy1 < H else 0)
    ix0 = fx0 + (ring if fx0 > 0 else 0)
    ix1 = fx1 - (ring if fx1 < W else 0)
    overlap = 2 * ring + 2  # free regions must overlap so seams are some tile's interior
    step = max(1, tile - overlap)
    tiles = [
        (ty, min(ty + tile, it1), tx, min(tx + tile, ix1))
        for ty in range(it0, it1, step)
        for tx in range(ix0, ix1, step)
    ]
    prev = None
    for _sweep in range(max_sweeps):
        for tb in tiles:
            if tb[1] > tb[0] and tb[3] > tb[2]:
                _solve_window(
                    phi,
                    constraint,
                    tb,
                    threshold,
                    objective,
                    maxiter,
                    ring,
                    rep,
                    margin_delta=margin_delta,
                    allow_grow=False,
                    inner=inner,
                    opts=opts,
                )
        nf = int((min_field(constraint, phi)[fy0:fy1, fx0:fx1] < threshold).sum())
        if nf == 0 or (prev is not None and nf >= prev):
            return nf  # cleared, or no further progress (geometric floor)
        prev = nf
    return prev if prev is not None else 0


def _solve_window(
    phi,
    constraint,
    box,
    threshold,
    objective,
    maxiter,
    ring,
    rep,
    _grow=0,
    margin_delta=1e-3,
    allow_grow=True,
    inner="isqp",
    opts=None,
):
    """Solve one window; on infeasibility grow the blocked sides once and retry.
    Returns the inner solver's feasibility flag. ``allow_grow=False`` (used by the
    Schwarz tiler) keeps a tile at fixed size — growing a tile defeats tiling.
    ``inner`` picks the per-window solver (see
    :func:`~dvfopt.core.windowed._inners.solve_window_inner`); ``opts`` carries
    the inner knobs (:class:`_InnerOpts`)."""
    H, W = phi.shape[1:]
    opts = _InnerOpts() if opts is None else opts
    sub = build_subproblem(constraint, phi, box, threshold, objective, margin_delta)
    t = time.perf_counter()

    def _attempt(backend):
        """One full inner attempt at *backend*, always from the window's ORIGINAL
        start state: the solve plus (if it fails) the no-trust-region retry.
        Returns ``(x, n_iter, feasible, used_no_tr_retry)``."""
        x, nit, ok = solve_window_inner(
            sub,
            inner,
            maxiter,
            osqp_max_iter=opts.qp_max_iter,
            qp_backend=backend,
            ip_cold=opts.ip_cold,
            ip_after_admm_iters=opts.ip_after_admm_iters,
        )
        no_tr = False
        if not ok and opts.no_tr_fallback and inner in _ISQP_LABELS:
            # The trust-region ratio test freezes on sliver-scale violations
            # (~1e-4, inside OSQP's own noise) that the legacy backtracking line
            # search still clears -- so retry the SAME window once with the TR off
            # before paying for a grow. Warm-started from the failed iterate: the
            # inner never moves frozen variables, so the ring stays pinned, and the
            # objective closures stay anchored at the ORIGINAL flat0 (only the
            # start point moves).
            no_tr = True
            x2, nit2, ok2 = solve_window_inner(
                replace(sub, flat0=x),
                inner,
                opts.fallback_maxiter,
                trust_region=False,
                osqp_max_iter=opts.qp_max_iter_fallback,
                qp_backend=backend,
                ip_cold=opts.ip_cold,
                ip_after_admm_iters=opts.ip_after_admm_iters,
            )
            nit += nit2
            if ok2 or sub.cons(x2).min() > sub.cons(x).min():  # keep the better
                x, ok = x2, ok2
        return x, nit, ok, no_tr

    x, nit, ok, fell_back = _attempt(opts.qp_backend)
    # Backend rung of the escalation ladder, ahead of grow-on-failure. The
    # interior-point legs change the SQP trajectory, and on some windows they steer
    # it into a basin with no escape: measured, the z0_cluster crop ends one triangle
    # genuinely INVERTED at -1.2e-4 under 'hybrid' where plain ADMM clears it, and
    # growing does not recover it. So retry the whole attempt on plain warm-started
    # OSQP from the ORIGINAL start state -- the IP trajectory is exactly what led
    # astray, so warm-starting from its failed iterate would keep the window in that
    # basin. Two guards keep the retry off the paths that do not need it:
    #
    # - A GENUINE fold, not merely `not ok`: `ok` tests the margin-shifted target
    #   (threshold + margin_delta), so a window landing at 0.0109 against a 0.011
    #   target reports not-ok while being perfectly fold-free. Since
    #   `cons = values - (threshold + margin_delta)`, an enforced row is truly below
    #   `threshold` exactly when `cons < -margin_delta`.
    # - Real windows only, never a giant tile (`allow_grow=False`). A second full
    #   attempt per tile defeats tiling for the same reason growing one does: the
    #   Schwarz loop already re-sweeps a tile that ends folded, and the terminal mop
    #   re-windows whatever survives with a large margin (mop windows DO take this
    #   rung). Measured on raw B0039 z16, retrying tiles cost 505 s vs 264 s at an
    #   identical (zero) fold count and a worse move (L2 362 vs 325).
    backend_fell_back = False
    if not ok and allow_grow and opts.qp_backend != "osqp" and inner in _ISQP_LABELS:
        backend_fell_back = bool(sub.cons(x).min() < -margin_delta)
    if backend_fell_back:
        x2, nit2, ok2, no_tr2 = _attempt("osqp")
        nit += nit2
        if ok2 or sub.cons(x2).min() > sub.cons(x).min():  # keep the better; never worse
            x, ok, fell_back = x2, ok2, no_tr2
    dt = time.perf_counter() - t
    patch_out = np.asarray(sub.constraint.unflatten(x))
    py0, py1, px0, px1 = sub.patch_box
    # paste back ONLY free pixels (frozen ring is unchanged and may be shared)
    fm = sub.free_mask
    dst = phi[:, py0:py1, px0:px1]
    dst[:, fm] = patch_out[:, fm]

    jpatch = min_field(constraint, phi[:, py0:py1, px0:px1])
    rec = WindowRec(
        fy0=box[0],
        fx0=box[2],
        ph=py1 - py0,
        pw=px1 - px0,
        n_free=sub.free_idx.size,
        n_enforced=sub.n_enforced,
        inner_iters=nit,
        grows=_grow,
        fallback=fell_back,
        backend_fallback=backend_fell_back,
        min_after=float(jpatch.min()),
        feasible=bool(ok),
        time_s=dt,
    )
    rep.windows.append(rec)
    rep.backend_fallbacks += int(backend_fell_back)

    # grow-on-failure: if still infeasible and the window can expand, widen and retry
    if allow_grow and not ok and _grow < 2:
        fy0, fy1, fx0, fx1 = box
        gy0, gy1 = max(0, fy0 - 4), min(H, fy1 + 4)
        gx0, gx1 = max(0, fx0 - 4), min(W, fx1 + 4)
        if (gy0, gy1, gx0, gx1) != box:
            return _solve_window(
                phi,
                constraint,
                (gy0, gy1, gx0, gx1),
                threshold,
                objective,
                maxiter,
                ring,
                rep,
                _grow + 1,
                margin_delta,
                inner=inner,
                opts=opts,
            )
    return bool(ok)


__all__ = [
    'SliceReport',
    'WindowRec',
    'build_subproblem',
    'find_windows',
    'windowed_correct',
]
