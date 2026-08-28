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
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from dvfopt._logging import log_warning
from dvfopt.objectives import L2Objective, _kind_eps, make_objective

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
    tr_delta: float = 2.0
    tr_max: float = 16.0
    step_rule: str = 'exact_ls'
    exact_ls_fallback_steps: int = 3
    patience_retry: bool = True


@dataclass(frozen=True)
class _ReanchorOpts:
    """Knobs of the optional post-feasibility re-anchor stage (:func:`_reanchor_pass`).

    ``kind`` is ``'none'`` (the stage never runs — the default) / ``'l2'`` / ``'l1'``.
    See :func:`windowed_correct` for what the rest do.
    """

    kind: str = 'none'
    maxiter: int = 60
    sweeps: int = 3
    tile: int = 48


_REANCHOR_KINDS = ('none', 'l2', 'l1')
_REANCHOR_MIN_GAIN = 0.01  # stop sweeping once a sweep buys < 1% of the L2 move
_REANCHOR_TOL = 1e-9  # accepted rows must clear `threshold` by this much
_REANCHOR_OVERLAP = 8  # tile overlap, in px (48 stepped by 40 in the prototype)


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


def build_subproblem(
    constraint, phi_dydx, free_box, threshold, objective=None, margin_delta=1e-3, free_extra=None
):
    """Build the window sub-problem for a free box ``(fy0, fy1, fx0, fx1)`` (global).

    Expands the free box by the family ring to a patch, instantiates a
    patch-shaped clone of ``constraint``'s type, selects the free pixels and the
    enforced constraint rows (those a free pixel influences AND that evaluate
    correctly), and returns a :class:`~dvfopt.core.windowed._inners.WindowSub`.

    ``margin_delta`` is enforced ON TOP of ``threshold`` (constraints are driven to
    ``threshold + margin_delta``) so that OSQP's ~1e-5 tolerance landing a hair
    short of the active bound still clears the strict ``< threshold`` fold check.

    ``free_extra`` (optional global ``(H, W)`` bool mask) is INTERSECTED with the
    free box, so a caller can free a subset of it — the re-anchor stage frees only
    pixels the main solve moved. ``None`` (default) frees the whole box. The patch
    is still the box expanded by the family ring, so the enforced rows a free pixel
    influences are all in-patch either way.
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
    if free_extra is not None:
        free_mask &= free_extra[py0:py1, px0:px1]

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
    patience_fallback: bool = False  # the no-bail exact_ls retry (patience rung) ran
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
    patience_fallbacks: int = 0  # windows that took the patience rung
    # coarse-to-fine warm start (see _coarse_warm_start). -1 == the stage did not
    # run (disabled, no folds, or the field too small for a meaningful coarse
    # problem); coarse_solve_s stays 0.0 then.
    coarse_solve_s: float = 0.0
    coarse_folds_before: int = -1
    coarse_folds_after: int = -1
    coarse_iters: int = 0  # SQP iterations spent on the coarse grid
    warm_folds: int = -1  # folds left after applying the prolongated correction
    # optional post-feasibility re-anchor stage (``reanchor != 'none'``): tiles over
    # the MOVED region re-solved against the distance-to-INPUT objective, each kept
    # only if it stays fold-free. ``reanchor_sweeps_run == 0`` means the stage did
    # not run (off, the field still folded, or the budget was spent).
    reanchor_sweeps_run: int = 0
    reanchor_tiles: int = 0
    reanchor_accepted: int = 0
    reanchor_l2_before: float = 0.0  # ||phi - input|| before / after the stage
    reanchor_l2_after: float = 0.0
    # terminal harmonic re-seed stage (``reseed_rounds > 0``): residual fold clusters the
    # round loop + mop plateaued on are re-seeded (interior replaced by the harmonic
    # interpolation of their ring) and polished again. ``reseed_rounds_run == 0`` means
    # the stage did not run (field already fold-free, or off).
    reseed_rounds_run: int = 0
    reseed_px: int = 0  # pixels re-seeded (all rounds)
    reseed_folds_before: int = -1  # folds when the stage started / when it ended
    reseed_folds_after: int = -1
    rounds: int = 0
    time_s: float = 0.0
    windows: list = field(default_factory=list)
    # per-stage convergence entries, filled only when ``record_history=True``
    # (each: name / n_iter / n_neg / min_T / wall_s; the 'final' entry adds
    # ``extras`` with damage / n_windows / giant_regions / mop_cleared /
    # l1_move / l2_move). Empty list otherwise.
    history: list = field(default_factory=list)


def _restrict(phi, factor):
    """Box-average ``factor`` x ``factor`` blocks of a ``(2, H, W)`` field.

    Displacements are divided by ``factor`` so they stay in COARSE pixel units —
    the coarse field is then an ordinary deformation field on its own grid and the
    same constraint/threshold means the same thing there. A trailing partial block
    (odd ``H``/``W``) is dropped.
    """
    hc, wc = (d // factor for d in phi.shape[1:])
    trimmed = phi[:, : factor * hc, : factor * wc]
    return trimmed.reshape(2, hc, factor, wc, factor).mean(axis=(2, 4)) / factor


def _prolongate(delta_c, shape, factor):
    """Bilinear ``factor`` x upsample of a coarse-grid CORRECTION back to ``shape``.

    Displacements are multiplied by ``factor`` (the inverse of :func:`_restrict`'s
    rescale). Rows/cols the integer factor cannot cover (odd ``H``/``W``) stay
    zero — the fine solve handles that strip itself.
    """
    h, w = shape
    out = np.zeros((2, h, w))
    for c in range(2):
        up = ndimage.zoom(delta_c[c] * factor, factor, order=1)
        hh, ww = min(h, up.shape[0]), min(w, up.shape[1])
        out[c, :hh, :ww] = up[:hh, :ww]
    return out


def _coarse_warm_start(phi, constraint, objective, threshold, factor, margin, ring, inner, sub_kw):
    """Warm-start delta from a solve of the SAME problem on a coarser grid.

    Restrict (:func:`_restrict`), run the engine on the coarse field
    (``coarse_to_fine=False`` — never recursive), prolongate the correction back
    (:func:`_prolongate`), and MASK it to the fine-level window free boxes
    :func:`find_windows` would open on the original fold mask. The mask is what
    keeps the no-damage invariant: the warm start can only move pixels the engine
    was going to free anyway, so healthy area outside every fold neighbourhood
    stays byte-identical. Returns ``(delta, coarse_report, boxes)`` — the caller
    marks ``boxes`` as touched, because the warm start IS a move over them: on a
    run cut by ``time_budget_s`` before the fine loop reaches a box, a fold the
    prolongated correction created there is a residual inside a fold
    neighbourhood, not damage to untouched area (measured: raw B0039 z16 under a
    40 s budget booked 3 such folds as damage before this).

    Why it pays: the coarse solve is ~1/factor**2 the work and lands the fine
    windows near a solution, so their SQP loops converge in far fewer iterations.
    Raw B0039 z16 (3890 simplex folds, bilinear rows, objective ``none``,
    maxiter 600): 205 s / 909 SQP iterations — 841 fine plus a 16 s, 68-iteration
    coarse solve — vs 283 s / 1320 cold. -28% wall, -31% iterations, 0 folds and
    damage 0 either way, and a slightly SMALLER move (L2 320.6 vs 325.1).
    """
    coarse = _restrict(phi, factor)
    out_c, rep_c = windowed_correct(
        coarse,
        inner,
        constraint=type(constraint)(shape=coarse.shape[1:]),
        objective=objective,
        threshold=threshold,
        coarse_to_fine=False,
        **sub_kw,
    )
    delta = _prolongate(out_c - coarse, phi.shape[1:], factor)
    allow = np.zeros(phi.shape[1:], bool)
    fine_mask = pixel_fold_mask(constraint, phi, threshold)
    boxes = find_windows(fine_mask, margin, ring)
    for fy0, fy1, fx0, fx1 in boxes:
        allow[fy0:fy1, fx0:fx1] = True
    delta[:, ~allow] = 0.0
    return delta, rep_c, boxes


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
    tr_delta=2.0,
    tr_max=16.0,
    step_rule='exact_ls',
    exact_ls_fallback_steps=3,
    patience_retry=True,
    coarse_to_fine=True,
    coarse_factor=4,
    reanchor='none',
    reanchor_maxiter=60,
    reanchor_sweeps=3,
    reanchor_tile=48,
    reseed_rounds=3,
    reseed_radius=2,
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

    ``tr_delta`` (2.0) / ``tr_max`` (16.0) size the ``isqp`` inner's trust
    region — initial radius and cap, in grid units. The default is what every
    measured windowed number was taken at; ``tr_delta=1.0`` buys speed with
    fidelity (raw B0039 z16: 267 s / 1022 SQP iterations / L2 move 344 vs 300 s
    / 1320 / L2 325). ``tr_max`` never binds on the measured B0039 windows.

    ``step_rule`` (default ``'exact_ls'``) picks how the ``isqp`` inner turns a
    QP step into an iterate. ``'exact_ls'`` replaces the trust-region ratio
    test's accept/reject with the EXACT minimiser of the merit along the step —
    free, because the 2D rows are exactly quadratic along a line and the model's
    quadratic term reuses the ``cons(x + d)`` the ratio test already evaluates
    (see :func:`~dvfopt.core.primitives.isqp.isqp_solve`). Measured on raw B0039
    z16: 200 s / 563 SQP iterations vs 244 s / 780 at ``'tr'`` (-18% / -28%),
    0 folds, damage 0, smaller move (L2 268 vs 280); across a 9-real-slice
    sample, 9/9 wall AND iteration wins (-19% wall / -27% iterations in total)
    with a smaller L2 move on every slice. It applies on the no-trust-region
    fallback rung too — scoping it out of that rung was measured WORSE (re-measured
    on the shipped implementation: ``z0_sliver`` 1918 SQP iterations vs 1684).
    ``'tr'`` restores the ratio-test path byte for byte. ``'exact_ls'`` is
    2D-only and rejected at this entry otherwise.

    ``exact_ls_fallback_steps`` (default 3, 0 = off) is what keeps ``'exact_ls'``
    from grinding on a window it cannot solve. The exact minimiser always finds
    SOME decrease, so it need not trip the ratio test's futility test and NEVER
    trips it on the no-trust-region rung (which has none) — so after this many
    consecutive steps with ``a* < 0.25`` the window stops (``exit='a-collapse'``)
    and hands itself to the escalation ladder, exactly as ``'tr-collapse'`` does.
    3 is the measured setting: it never fires on the window ``'exact_ls'`` turns
    from a 108-iteration failure into a 46-iteration solve (longest collapse run
    there: 2) and fires immediately on ``z0_sliver`` (run of 4) and ``z0_cluster``
    (6). Bilinear rows, objective ``none``, threshold 0.01, maxiter 600, engine
    defaults; **every row 0 simplex folds, damage 0**:

    ==================  =========  =========  ====================
    case                ``'tr'``   exact_ls   exact_ls + bail (3)
    ==================  =========  =========  ====================
    z16_twist (crop)    128        47         47
    z0_cluster (crop)   387        328        287
    z0_sliver (crop)    540        1684       **212**
    raw B0039 z16       780        563        **396**
    ==================  =========  =========  ====================

    (SQP iterations, coarse warm start included.) The L2 move falls too —
    z0_sliver 25.3 / 39.7 / **19.4**, raw z16 280.3 / 268.4 / **268.0** — so the
    bail is not buying speed with fidelity. Handing the remaining iterations to
    the ``'tr'`` acceptance instead of stopping was measured and is WORSE
    (z0_sliver 2350 iterations): mid-run the ratio test accepts tiny steps rather
    than rejecting them, so it grinds too. Stopping is the whole mechanism.

    ``patience_retry`` (default True) is the bail's counterpart, the LAST rung of
    the window escalation ladder before grow: a window still GENUINELY folded
    after the solve, the no-TR retry and the backend retry continues its exact-LS
    iteration from the best iterate with the bail OFF. The bail is cheap because
    the rungs above clear most windows it stops; on a window pinned by a
    prescribed correspondence whose displacement disagrees with its neighbours
    by tens of pixels (every residual cluster of the 7-brain cohort sweep sits on
    such a pin) the tiny ``a*`` steps are slow but productive and no other rung
    can continue them. Measured on a crop of the B0304 z181 residual: the full
    ladder ends folded at -0.044 after 101 s, the bail-free continuation clears
    it to +0.011 in 1 s. Real windows and mop windows only (never a giant tile),
    and never on a window that is merely short of the margin-shifted target.
    ``report.patience_fallbacks`` / ``WindowRec.patience_fallback`` count it.

    ``coarse_to_fine=True`` (default) prepends a **coarse-grid warm start**: the
    same problem is solved on a ``coarse_factor`` x coarsened field and the
    prolongated correction seeds the fine solve, so the fine windows start near a
    solution instead of cold (raw B0039 z16: 205 s / 909 SQP iterations — 841 fine
    plus a 16 s, 68-iteration coarse solve — vs 283 s / 1320 cold, at a slightly
    smaller L2 move, 320.6 vs 325.1).
    The warm-start delta is MASKED to the window free boxes the fine engine would
    open anyway, so the no-damage invariant holds unchanged and the final damage
    accounting still runs against the ORIGINAL input. It is skipped — leaving the
    path byte-identical to ``coarse_to_fine=False`` — when the field has no folds
    or when ``min(H, W) < 4 * max(giant_tile, coarse_factor)`` (below that the
    coarse problem is too small to be a meaningful preview, and its own solve is
    not amortised; the ``coarse_factor`` leg only bites for absurd factors).
    ``report.coarse_solve_s`` / ``coarse_folds_before`` / ``coarse_folds_after``
    / ``coarse_iters`` / ``warm_folds`` record the stage (``-1`` = skipped).

    ``giant_tile_fit=True`` (default) makes ``giant_tile`` a *target* rather
    than a literal size: the effective tile is fitted per region so an integer
    number of near-equal tiles covers its longest side (:func:`_fit_tile`).
    Tile size matters through grid *alignment* — the sweep-round count — not
    through size itself. ``False`` restores the literal ``giant_tile``.

    ``reanchor`` (``'none'`` — the DEFAULT, ``'l2'``, ``'l1'``) adds an optional
    **post-feasibility re-anchor stage**, opt-in because fidelity is a concern
    separate from the zero-fold certificate. With ``objective='none'`` (the robust
    recipe — pure feasibility keeps the inner out of the objective-basin traps a
    distance anchor pins it in) the correction is only close to the input by
    construction; when the field comes out feasible this stage recovers the
    fidelity, with no fold left to trap it: the MOVED region is tiled
    (``reanchor_tile`` px, overlapping), each tile is re-solved minimising the
    chosen distance to the INPUT under the same constraint rows
    (``reanchor_maxiter`` inner iterations), and a tile is kept only if every
    enforced row stays at or above ``threshold`` — otherwise it is reverted. Up to
    ``reanchor_sweeps`` sweeps, stopping once a sweep buys < 1% of the L2 move. The
    stage frees only pixels the main solve already moved, so **no-damage accounting
    is unaffected** (see :func:`_reanchor_pass`); the whole stage is reverted and a
    warning logged if a global re-check finds a fold anyway. Measured on
    already-feasible fields: B0039 z16 L2 move 76.7 -> 59.9, z0 194 -> 170, 0 folds
    throughout. ``report.reanchor_sweeps_run`` / ``reanchor_tiles`` /
    ``reanchor_accepted`` / ``reanchor_l2_before`` / ``reanchor_l2_after`` record it.

    ``reseed_rounds`` (default 3, 0 = off) / ``reseed_radius`` (default 2) add a
    **terminal harmonic re-seed stage** for the residual the round loop AND the mop
    plateau on. Such residual clusters are, measured on every non-clean slice of
    the 7-brain cohort, cells whose corner images have been driven onto the
    ROTATED orientation branch (both edge factors of the area negative, product
    positive): locally fold-free, but not joinable to the surrounding field, and
    the seam between the branches is a merit MAXIMUM no local step crosses -- every
    rung, step rule and trust radius fails there identically. The stage resets the
    branch: each residual cluster's neighbourhood (its cells' corner pixels
    dilated by ``reseed_radius``) is replaced by the discrete-harmonic
    interpolation of its ring, and the engine polishes the re-seeded field
    (recursively, with this stage off). Up to ``reseed_rounds`` rounds. Measured
    on the five plateaued cohort slices (29-79 residual cells each, every rung
    exhausted): **0 simplex and 0 bilinear folds on all five, damage 0, 10-40 s**.
    The re-seeded pixels are fold neighbourhoods, and the polish's windows join
    ``touched``, so the no-damage invariant is unchanged; the stage never fires on
    a field the mop cleared, so those runs are byte-identical.
    ``report.reseed_rounds_run`` / ``reseed_px`` / ``reseed_folds_before`` /
    ``reseed_folds_after`` record it.

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
    if step_rule not in ('tr', 'exact_ls'):
        raise ValueError(f"unknown step_rule {step_rule!r}; valid: 'tr', 'exact_ls'")
    if reanchor not in _REANCHOR_KINDS:
        raise ValueError(f"unknown reanchor {reanchor!r}; valid: {list(_REANCHOR_KINDS)}")
    if step_rule == 'exact_ls' and np.asarray(phi_in).ndim != 3:
        # The exact line model needs rows that are BILINEAR in (dy, dx) — true of
        # every 2D family here, false in 3D (a 6-tet volume is trilinear, hence
        # cubic along a line). Guarded here, at the only caller, not in the driver.
        raise ValueError("step_rule='exact_ls' requires a 2D (2, H, W) field")
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
        tr_delta,
        tr_max,
        step_rule,
        exact_ls_fallback_steps,
        patience_retry,
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

    # Coarse-grid warm start: solve small, prolongate the correction, then run the
    # normal fine loop from the warmed field. Skipped when there is nothing to do
    # or the field is too small for the coarse problem to be a useful preview.
    if coarse_to_fine and rep.folds_before > 0 and min(H, W) >= 4 * max(giant_tile, coarse_factor):
        t_coarse = time.perf_counter()
        delta, rep_c, warm_boxes = _coarse_warm_start(
            phi,
            constraint,
            objective,
            threshold,
            coarse_factor,
            margin,
            ring,
            inner,
            dict(
                margin=margin,
                maxiter=maxiter,
                max_rounds=max_rounds,
                margin_delta=margin_delta,
                max_window_area=max_window_area,
                mop_margin=mop_margin,
                time_budget_s=time_budget_s,
                verbose=verbose,
                **asdict(opts),
            ),
        )
        phi += delta
        for fy0, fy1, fx0, fx1 in warm_boxes:  # the warm start is a move over these
            touched[max(0, fy0 - ring) : fy1 + ring, max(0, fx0 - ring) : fx1 + ring] = True
        rep.coarse_solve_s = time.perf_counter() - t_coarse
        rep.coarse_folds_before = rep_c.folds_before
        rep.coarse_folds_after = rep_c.folds_after
        rep.coarse_iters = int(sum(w.inner_iters for w in rep_c.windows))
        rep.warm_folds = int(pixel_fold_mask(constraint, phi, threshold).sum())
        if record_history:
            entry = _stage_entry("coarse", len(rep.windows))
            entry["n_iter"] = rep.coarse_iters  # coarse windows are not in rep.windows
            rep.history.append(entry)
        _fire("coarse", phi)

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
                    expired=_expired,
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

    # Terminal harmonic re-seed: the residual both the round loop and the mop
    # plateau on sits on the rotated orientation branch (see the docstring); reset
    # the branch and polish. Bounded rounds, deadline-aware, off on a clean field.
    if reseed_rounds > 0 and not budget_hit:
        _reseed_stage(
            phi,
            constraint,
            threshold,
            objective,
            inner,
            opts,
            rep,
            touched,
            ring,
            reseed_rounds,
            reseed_radius,
            dict(
                margin=margin,
                maxiter=maxiter,
                max_rounds=max_rounds,
                margin_delta=margin_delta,
                max_window_area=max_window_area,
                mop_margin=mop_margin,
                verbose=verbose,
                **asdict(opts),
            ),
            expired=_expired,
        )
        if record_history and rep.reseed_rounds_run:
            rep.history.append(_stage_entry("reseed", len(rep.windows)))
        if rep.reseed_rounds_run:
            _fire("reseed", phi)

    # Post-feasibility re-anchor: recover fidelity now that no fold is left to trap
    # the inner in an objective basin. Only on a fold-free field, and reverted whole
    # if it somehow breaks that (per-tile verification should make this unreachable).
    if reanchor != 'none' and not budget_hit and not _expired():
        if int(pixel_fold_mask(constraint, phi, threshold).sum()) == 0:
            saved = phi.copy()
            _reanchor_pass(
                phi,
                np.asarray(phi_in, dtype=np.float64),
                constraint,
                threshold,
                _ReanchorOpts(reanchor, reanchor_maxiter, reanchor_sweeps, reanchor_tile),
                margin_delta,
                rep,
                inner,
                opts,
                expired=_expired,
            )
            if int(pixel_fold_mask(constraint, phi, threshold).sum()) > 0:
                log_warning(
                    "windowed_correct: re-anchor stage created a fold despite per-tile "
                    "verification; reverting the whole stage"
                )
                phi[:] = saved
                rep.reanchor_accepted = 0
                rep.reanchor_l2_after = rep.reanchor_l2_before
            if record_history:
                rep.history.append(_stage_entry("reanchor", len(rep.windows)))
            _fire("reanchor", phi)

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


def _reanchor_tile(
    phi, phi_ref, constraint, box, threshold, obj_ref, moved, ropts, margin_delta, inner, opts
):
    """Re-solve one tile against the distance-to-INPUT objective; keep it or revert.

    Returns True if the tile was accepted. The subproblem is the engine's OWN
    (:func:`build_subproblem` on the current field), with only the objective
    triplet swapped for one anchored at the input patch — so the enforced-row set,
    the frozen ring and the paste-back are exactly the main solve's.
    """
    sub = build_subproblem(constraint, phi, box, threshold, None, margin_delta, free_extra=moved)
    if sub.free_idx.size == 0 or sub.n_enforced == 0:
        return False
    py0, py1, px0, px1 = sub.patch_box
    ref = np.asarray(sub.constraint.flatten(np.ascontiguousarray(phi_ref[:, py0:py1, px0:px1])))
    # Same helper the engine builds its own objective with, re-anchored at the INPUT
    # patch instead of the current one (L1's eps rides on the Objective, as there).
    obj, grad, hess = _objective_fns(ref, obj_ref)
    sub = replace(sub, obj=obj, obj_grad=grad, hess_diag=hess)
    x, _nit, _ok = solve_window_inner(
        sub,
        inner,
        ropts.maxiter,
        osqp_max_iter=opts.qp_max_iter,
        qp_backend=opts.qp_backend,
        ip_cold=opts.ip_cold,
        ip_after_admm_iters=opts.ip_after_admm_iters,
        tr_delta=opts.tr_delta,
        tr_max=opts.tr_max,
        step_rule=opts.step_rule,
    )
    # Verify-and-revert. `cons = values - (threshold + margin_delta)`, so an
    # enforced row is still fold-free exactly when `cons >= -margin_delta`; the
    # second test refuses a tile that did not actually buy fidelity, so the stage
    # is monotone in the move it is minimising and can never make the field worse.
    if sub.cons(x).min() < -margin_delta + _REANCHOR_TOL or obj(x) >= obj(sub.flat0):
        return False
    patch_out = np.asarray(sub.constraint.unflatten(x))
    dst = phi[:, py0:py1, px0:px1]
    dst[:, sub.free_mask] = patch_out[:, sub.free_mask]
    return True


def _reanchor_pass(
    phi, phi_ref, constraint, threshold, ropts, margin_delta, rep, inner, opts, expired=None
):
    """Pull a FEASIBLE corrected field back toward ``phi_ref`` (the input) in place.

    The robust recipe solves with ``objective='none'`` — pure feasibility, which
    keeps the windowed isqp out of the objective-basin traps a distance anchor pins
    it in, but leaves the correction close to the input only by construction. This
    stage recovers the fidelity afterwards, when there is no fold left to trap it:
    tile the MOVED region, re-solve each tile minimising the distance to the input
    under the same constraint rows, and accept the tile only if every enforced row
    stays at or above ``threshold`` (per-tile verify-and-revert).

    No-damage is untouched. The free set of every tile is intersected with the
    moved mask, so the stage only ever moves pixels the main solve already moved —
    the moved set can shrink, never grow — and those pixels are inside ``touched``
    by construction, as are the rows they influence (``touched`` is the free boxes
    dilated by the ring). Damage accounting therefore reads exactly the same.

    Measured (benchmarks prototype, on already-feasible fields): B0039 z16 L2 move
    76.7 -> 59.9 and z0 194 -> 170, at 0 folds throughout.
    """
    moved = np.any(np.abs(phi - phi_ref) > 1e-9, axis=0)
    if not moved.any():
        return
    obj_ref = make_objective(ropts.kind)
    tile = max(1, ropts.tile)
    step = max(1, tile - _REANCHOR_OVERLAP)  # overlap so a seam is a neighbour's interior
    ys, xs = np.nonzero(moved)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    def l2_move():
        return float(np.linalg.norm((phi - phi_ref).ravel()))

    rep.reanchor_l2_before = rep.reanchor_l2_after = prev = l2_move()
    for _sweep in range(max(0, ropts.sweeps)):
        rep.reanchor_sweeps_run += 1
        for ty in range(y0, y1, step):
            for tx in range(x0, x1, step):
                box = (ty, min(ty + tile, y1), tx, min(tx + tile, x1))
                if not moved[box[0] : box[1], box[2] : box[3]].any():
                    continue
                if expired is not None and expired():
                    rep.reanchor_l2_after = l2_move()
                    return
                rep.reanchor_tiles += 1
                rep.reanchor_accepted += int(
                    _reanchor_tile(
                        phi,
                        phi_ref,
                        constraint,
                        box,
                        threshold,
                        obj_ref,
                        moved,
                        ropts,
                        margin_delta,
                        inner,
                        opts,
                    )
                )
        cur = l2_move()
        rep.reanchor_l2_after = cur
        if prev - cur < _REANCHOR_MIN_GAIN * prev:
            break  # a sweep that buys < 1% of the move is not worth the next one
        prev = cur


def _harmonic_fill(phi, mask):
    """Replace ``phi[:, mask]`` by the discrete-harmonic (4-neighbour Laplacian)
    interpolation of ``phi`` on the mask's boundary, in place. One sparse solve per
    channel over the masked pixels (a few hundred on real residuals)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n == 0:
        return
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [], [], []
    rhs = np.zeros((phi.shape[0], n))
    for k, (y, x) in enumerate(zip(ys, xs)):
        deg = 0
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            y2, x2 = y + dy, x + dx
            if not (0 <= y2 < H and 0 <= x2 < W):
                continue
            deg += 1
            if mask[y2, x2]:
                rows.append(k)
                cols.append(idx[y2, x2])
                vals.append(-1.0)
            else:
                rhs[:, k] += phi[:, y2, x2]
        rows.append(k)
        cols.append(k)
        vals.append(float(deg))
    lap = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))
    for ch in range(phi.shape[0]):
        phi[ch, ys, xs] = spsolve(lap, rhs[ch])


def _reseed_stage(
    phi,
    constraint,
    threshold,
    objective,
    inner,
    opts,
    rep,
    touched,
    ring,
    rounds,
    radius,
    sub_kw,
    expired,
):
    """Harmonic re-seed of every residual fold cluster, then a recursive polish; in place.

    A residual cell's corner pixels (the cell and its +1 row/column) dilated by
    ``radius`` form the re-seed mask; its interior is replaced by the harmonic
    interpolation of the ring, which puts the cluster back on the ring's orientation
    branch. The polish is :func:`windowed_correct` on the re-seeded field with this
    stage off (never recursive) and the coarse warm start off; its windows' patch
    boxes join ``touched`` so the outer damage accounting stays exact. Stops when
    the field is fold-free, the deadline passes, or a round makes no progress.
    """
    fold0 = pixel_fold_mask(constraint, phi, threshold)
    if not fold0.any():
        return
    rep.reseed_folds_before = int(fold0.sum())
    prev = rep.reseed_folds_before
    for _ in range(rounds):
        fold = pixel_fold_mask(constraint, phi, threshold)
        nf = int(fold.sum())
        if nf == 0 or expired():
            break
        rep.reseed_rounds_run += 1
        corners = fold.copy()
        corners[1:, :] |= fold[:-1, :]
        corners[:, 1:] |= fold[:, :-1]
        corners[1:, 1:] |= fold[:-1, :-1]
        mask = ndimage.binary_dilation(corners, iterations=radius)
        _harmonic_fill(phi, mask)
        rep.reseed_px += int(mask.sum())
        touched |= ndimage.binary_dilation(mask, iterations=ring)
        out, rep_in = windowed_correct(
            phi,
            inner,
            constraint=constraint,
            objective=objective,
            threshold=threshold,
            coarse_to_fine=False,
            reseed_rounds=0,
            reanchor="none",
            time_budget_s=None,
            **sub_kw,
        )
        phi[...] = out
        for w in rep_in.windows:  # the polish's enforced footprints
            py0, px0 = max(0, w.fy0 - ring), max(0, w.fx0 - ring)
            touched[py0 : py0 + w.ph, px0 : px0 + w.pw] = True
        rep.windows.extend(rep_in.windows)
        rep.backend_fallbacks += rep_in.backend_fallbacks
        rep.patience_fallbacks += rep_in.patience_fallbacks
        nf_after = int(pixel_fold_mask(constraint, phi, threshold).sum())
        if nf_after >= prev:
            break  # no progress -> stop rather than churn
        prev = nf_after
    rep.reseed_folds_after = int(pixel_fold_mask(constraint, phi, threshold).sum())


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
    expired=None,
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
            if expired is not None and expired():
                # ``time_budget_s`` is checked between tiles as between windows —
                # a giant region is many window solves, not one (measured: a 40 s
                # budget ran 189 s on raw B0039 z16 before this check existed).
                return prev if prev is not None else -1
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
            tr_delta=opts.tr_delta,
            tr_max=opts.tr_max,
            step_rule=opts.step_rule,
            exact_ls_fallback_steps=opts.exact_ls_fallback_steps,
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
                tr_delta=opts.tr_delta,
                tr_max=opts.tr_max,
                step_rule=opts.step_rule,
                exact_ls_fallback_steps=opts.exact_ls_fallback_steps,
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
    # Patience rung, last before grow: the a*-collapse bail (`exact_ls_fallback_steps`)
    # stops a window after a few tiny exact-LS steps and hands it to the rungs above --
    # which is what makes the bail cheap on windows those rungs then clear. But on a
    # window pinned by a prescribed correspondence whose displacement disagrees with
    # its neighbours by tens of pixels (the cohort's residual clusters), the tiny
    # steps are slow but PRODUCTIVE, and none of the rungs above can continue them:
    # the no-TR line search stalls, the backend retry bails again, growing repeats
    # the pattern. Measured on a crop of the B0304 z181 residual: the full ladder
    # ends folded at -0.044 after 101 s; the same window with the bail OFF clears
    # to +0.011 in 1 s (L2 move 46 -- it walked the pin out). So, only when the
    # window is still GENUINELY folded after every rung above, continue the
    # exact-LS iteration from the best iterate with the bail off. Same guards as
    # the backend rung (genuine fold, real windows and mop windows only).
    patience_fell_back = False
    if (
        not ok
        and allow_grow
        and opts.patience_retry
        and opts.step_rule == 'exact_ls'
        and opts.exact_ls_fallback_steps
        and inner in _ISQP_LABELS
        and sub.cons(x).min() < -margin_delta
    ):
        patience_fell_back = True
        x2, nit2, ok2 = solve_window_inner(
            replace(sub, flat0=x),
            inner,
            maxiter,
            osqp_max_iter=opts.qp_max_iter,
            qp_backend=opts.qp_backend,
            ip_cold=opts.ip_cold,
            ip_after_admm_iters=opts.ip_after_admm_iters,
            tr_delta=opts.tr_delta,
            tr_max=opts.tr_max,
            step_rule=opts.step_rule,
            exact_ls_fallback_steps=0,
        )
        nit += nit2
        if ok2 or sub.cons(x2).min() > sub.cons(x).min():  # keep the better; never worse
            x, ok = x2, ok2
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
        patience_fallback=patience_fell_back,
        min_after=float(jpatch.min()),
        feasible=bool(ok),
        time_s=dt,
    )
    rep.windows.append(rec)
    rep.backend_fallbacks += int(backend_fell_back)
    rep.patience_fallbacks += int(patience_fell_back)

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
