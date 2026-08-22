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
import finite_jdet as fj
import slsqp_variants as sv

from dvfopt.constraints import JdetConstraint2D, TriConstraint2D
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d

# Frozen-ring width per family: how far in from an interior patch edge a pixel must
# be before every constraint it influences is enforceable with the correct
# (global-matching) evaluation.
#   jdet: 2 — central-diff rows must be 1 in from the edge, and a free pixel's
#            influenced rows are its 4 neighbours (finite-difference support).
#   2tri: 1 — cell areas are EXACT (no finite-diff), and a free pixel's influenced
#            cells are its <=4 corner cells, all in-patch once it is 1 in.
#   finite: 1 — forward-diff cell det is EXACT and depends on 3 corners; a free
#            pixel's <=3 influenced cells are all in-patch once it is 1 in (like 2tri).
_RING = {"jdet": 2, "2tri": 1, "finite": 1}


def _constraint(family, shape):
    if family == "jdet":
        return JdetConstraint2D(shape=shape)
    if family == "2tri":
        return TriConstraint2D(shape=shape)
    if family == "finite":
        return fj.FiniteJdetConstraint2D(shape)
    raise ValueError(f"unknown family {family!r}")


def min_field(family, phi_dydx):
    """Per-location constraint value on the ``(H, W)`` pixel grid (folds are where
    it is ``< threshold``).

    - jdet: the pixel Jacobian determinant.
    - 2tri: each cell ``(i, j)``'s min triangle area, placed at pixel ``(i, j)``;
      the last pixel row/col have no cell and are set to ``+inf``.
    - finite: each cell ``(i, j)``'s forward-diff determinant, placed at pixel
      ``(i, j)``; the last pixel row/col have no cell and are set to ``+inf``.
    """
    H, W = phi_dydx.shape[1:]
    if family == "jdet":
        return _numpy_jdet_2d(phi_dydx[0], phi_dydx[1])
    if family == "2tri":
        c = TriConstraint2D(shape=(H, W))
        vals = np.asarray(c.values(c.flatten(phi_dydx)))
        m = (H - 1) * (W - 1)
        cellmin = np.minimum(vals[:m], vals[m:]).reshape(H - 1, W - 1)
        out = np.full((H, W), np.inf)
        out[: H - 1, : W - 1] = cellmin
        return out
    if family == "finite":
        c = fj.FiniteJdetConstraint2D((H, W))
        vals = np.asarray(c.values(c.flatten(phi_dydx))).reshape(H - 1, W - 1)
        out = np.full((H, W), np.inf)
        out[: H - 1, : W - 1] = vals
        return out
    raise ValueError(f"unknown family {family!r}")


_COLORING_CACHE = {}  # (family, ph, pw) -> (pattern, colors, None)


def _pattern_union(c, probes=4, seed=0):
    """Sparsity pattern of ``c``'s Jacobian, as the union of nonzeros over ``probes``
    random points (a single point can zero a structurally-nonzero entry)."""
    rng = np.random.default_rng(seed)
    flat0 = rng.normal(0, 0.5, c.n_variables)
    acc = None
    for _ in range(probes):
        b = np.abs(sv.dense_jacobian(c, flat0 + rng.normal(0, 0.4, flat0.size))) > 0
        acc = b if acc is None else (acc | b)
    return [np.nonzero(acc[r])[0] for r in range(acc.shape[0])]


def _cached_coloring(family, c, shape):
    """CPR coloring ``(pattern, colors, None)`` for a patch shape, cached (the
    Jacobian sparsity pattern depends only on the shape, and shapes recur across a
    volume). Jdet uses the pixel-grid stride-3 colouring; 2-tri uses a cell-grid
    ``triangle*4 + (i%2)*2 + j%2`` colouring (8 colours) — both give one adjoint
    call per colour instead of one per constraint row, exact for their stencils."""
    key = (family, *shape)
    hit = _COLORING_CACHE.get(key)
    if hit is None:
        if family == "jdet":
            hit = sv.jacobian_coloring(c, np.random.default_rng(0).normal(0, 0.5, c.n_variables))
        else:  # 2tri: cell grid, 2 triangles per cell share a 2x2-corner support
            ph, pw = shape
            ii, jj = np.meshgrid(np.arange(ph - 1), np.arange(pw - 1), indexing="ij")
            cellcol = ((ii % 2) * 2 + (jj % 2)).ravel()
            colors = np.concatenate([cellcol, 4 + cellcol])  # T1: 0-3, T2: 4-7
            hit = (_pattern_union(c), colors, None)
        _COLORING_CACHE[key] = hit
    return hit


def pixel_fold_mask(family, phi_dydx, threshold):
    """Boolean ``(H, W)`` pixel mask of folds (constraint value < threshold)."""
    return min_field(family, phi_dydx) < threshold


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
    if objective == "none":
        # Pure feasibility: no distance anchor, only the elastic-QP constraint drive.
        # The flat unit Hessian keeps the QP positive-definite. Use this to clear an
        # objective-basin trap the distance objective pins (see the z=16 analysis).
        return (
            lambda f: 0.0,
            lambda f: np.zeros_like(f),
            lambda f: np.full(f.size, 2.0),
        )
    raise ValueError(f"unknown objective {objective!r}")


def _enforced_rows_and_jac(family, c, free_mask, ph, pw, borders):
    """Return ``(enforced_idx, jac_of)`` for a patch.

    ``enforced_idx`` are the constraint rows a free pixel influences AND that
    evaluate correctly (grid + finite-difference rules differ per family).
    ``jac_of(f)`` returns the full sparse CSR Jacobian (rows sliced by the caller).
    """
    cross = ndimage.generate_binary_structure(2, 1)
    if family == "jdet":
        valid = _eval_valid_jdet(ph, pw, *borders)
        influenced = ndimage.binary_dilation(free_mask, cross)  # Jdet row depends on 4 neighbours
        enforced_idx = np.nonzero((influenced & valid).ravel())[0]  # row = pixel raveled
        coloring = _cached_coloring(family, c, (ph, pw))

        def jac_of(f):
            return sv.colored_jacobian(c, f, *coloring).tocsr()

        return enforced_idx, jac_of

    if family == "2tri":
        # cell (i,j) is influenced iff any of its 4 corner pixels is free; cell areas
        # are exact so every cell evaluates correctly (no image-border special case).
        fm = free_mask
        cell = fm[:-1, :-1] | fm[1:, :-1] | fm[:-1, 1:] | fm[1:, 1:]  # (ph-1, pw-1)
        cell_flat = np.nonzero(cell.ravel())[0]
        m = (ph - 1) * (pw - 1)
        enforced_idx = np.concatenate([cell_flat, m + cell_flat])  # T1 and T2 rows
        coloring = _cached_coloring(family, c, (ph, pw))

        def jac_of(f):
            # coloring: 8 adjoint calls, not a full dense (2M x 2N) rebuild per iter
            # (the native jacobian() densifies — ~43% of a 2-tri window's time).
            return sv.colored_jacobian(c, f, *coloring).tocsr()

        return enforced_idx, jac_of

    if family == "finite":
        # forward-diff cell (i,j) depends on its 3 forward corners (i,j),(i,j+1),
        # (i+1,j); influenced iff any is free. One row per cell (row = cell raveled
        # C-order, matching FiniteJdetConstraint2D.values). The analytic sparse
        # jacobian is cheap — no coloring needed.
        fm = free_mask
        cell = fm[:-1, :-1] | fm[:-1, 1:] | fm[1:, :-1]  # (ph-1, pw-1)
        enforced_idx = np.nonzero(cell.ravel())[0]

        def jac_of(f):
            return c.jacobian(f).tocsr()

        return enforced_idx, jac_of

    raise ValueError(f"unknown family {family!r}")


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

    enforced_idx, jac_of = _enforced_rows_and_jac(
        family, c, free_mask, ph, pw, (py0 == 0, py1 == H, px0 == 0, px1 == W)
    )

    # free variable indices in the constraint's own pack (never hand-packed)
    free_phi = np.stack([free_mask, free_mask]).astype(float)
    free_idx = np.nonzero(np.asarray(c.flatten(free_phi)) > 0.5)[0]

    target = threshold + margin_delta

    def cons(f):
        return (np.asarray(c.values(f)) - target)[enforced_idx]

    def cons_jac(f):  # sparse (n_enforced, n_vars) — enforced rows only
        return jac_of(f)[enforced_idx]

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
    max_rounds=8,
    z=-1,
    margin_delta=1e-3,
    max_window_area=3000,
    mop_margin=25,
    inner="isqp-osqp",
):
    """Correct a full ``(2, H, W)`` slice by solving one small window per fold
    cluster. Returns ``(phi_out, SliceReport)``.

    ``max_window_area`` caps a window's free-box area; larger merged clusters are
    cleared by overlapping-tile Schwarz decomposition (``report.giant_regions``)
    instead of an intractable near-full-grid QP. ``margin`` is clamped to at least
    the family ring (the frozen inset band must stay fold-free).

    ``inner`` selects the per-window solver — ``"isqp-osqp"`` (default, the tuned
    elastic-QP SQP), ``"scipy-slsqp"``, or ``"scipy-slsqp+trust-constr"``. All
    inners only move the window's free pixels, so the no-damage invariant holds
    regardless of the choice.

    After the round loop plateaus, a terminal **mop** pass re-windows any residual
    with ``mop_margin`` (>> ``margin``): the diagnostic on the densest slices shows
    the plateau is boundary-stuck folds inside the giants that a small window's
    tight frozen boundary can't clear but a large frozen-exterior window can (the
    analogue of the 2.5D pipeline's ``mop_interior_3d``). ``mop_margin=0`` disables.
    """
    phi = np.array(phi_dydx, dtype=np.float64, copy=True)
    ring = _RING[family]
    margin = max(margin, ring)  # inset band must be fold-free margin, never < ring
    H, W = phi.shape[1:]
    j0 = min_field(family, phi)
    orig_fold = j0 < threshold
    rep = SliceReport(folds_before=int(orig_fold.sum()), min_before=float(j0.min()))
    touched = np.zeros((H, W), bool)  # union of every window's ENFORCED footprint
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
                _solve_giant_schwarz(
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
                    margin_delta,
                    inner=inner,
                )
                continue
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
                inner=inner,
            )

    # terminal mop: clear the boundary-stuck residual the round loop plateaued on
    if mop_margin > 0:
        before_mop = int(pixel_fold_mask(family, phi, threshold).sum())
        if before_mop > 0:
            _mop_pass(
                phi,
                family,
                threshold,
                objective,
                eps,
                maxiter,
                ring,
                z,
                rep,
                margin_delta,
                touched,
                mop_margin,
                max_window_area,
                inner=inner,
            )
            rep.mop_cleared = before_mop - int(pixel_fold_mask(family, phi, threshold).sum())

    jf = min_field(family, phi)
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
    return phi, rep


def _mop_pass(
    phi,
    family,
    threshold,
    objective,
    eps,
    maxiter,
    ring,
    z,
    rep,
    margin_delta,
    touched,
    mop_margin,
    max_window_area,
    max_sweeps=3,
    inner="isqp-osqp",
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
        mask = pixel_fold_mask(family, phi, threshold)
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
                    family,
                    box,
                    threshold,
                    objective,
                    eps,
                    maxiter,
                    ring,
                    z,
                    rep,
                    margin_delta,
                    inner=inner,
                )
            else:
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
                    inner=inner,
                )
        if int(pixel_fold_mask(family, phi, threshold).sum()) >= n:
            break  # no progress -> genuine local floor


def _solve_giant_schwarz(
    phi,
    family,
    giant_box,
    threshold,
    objective,
    eps,
    maxiter,
    ring,
    z,
    rep,
    margin_delta,
    tile=32,
    max_sweeps=8,
    inner="isqp-osqp",
):
    """Clear a large connected fold region by overlapping-tile (additive Schwarz)
    decomposition. Each tile is an ordinary window (frozen ring = current iterate);
    tiles overlap so a fold on one tile's seam is interior to a neighbour, and
    repeated sweeps propagate the correction across the whole region. Returns folds
    remaining in the region.

    No-damage by construction: the tiled region is INSET by ``ring`` from each
    interior giant edge, so a tile-free pixel can only influence constraints inside
    the giant box (= ``touched``); nothing outside can be changed, let alone folded.
    The inset band is fold-free margin (needs ``margin >= ring``), so insetting
    leaves no fold unfixed. Image-border edges are not inset — no "outside" there.
    Without the inset, an infeasible edge-tile solve can leave a boundary guard row
    just outside the giant violated -> a damage fold (observed on B0039 z=0)."""
    H, W = phi.shape[1:]
    fy0, fy1, fx0, fx1 = giant_box
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
                    family,
                    tb,
                    threshold,
                    objective,
                    eps,
                    maxiter,
                    ring,
                    z,
                    rep,
                    margin_delta=margin_delta,
                    allow_grow=False,
                    inner=inner,
                )
        nf = int((min_field(family, phi)[fy0:fy1, fx0:fx1] < threshold).sum())
        if nf == 0 or (prev is not None and nf >= prev):
            return nf  # cleared, or no further progress (geometric floor)
        prev = nf
    return prev if prev is not None else 0


def _inner_solve(sub, inner, maxiter, trace=None):
    """Solve a built window sub-problem with the chosen inner solver, returning
    ``(x_full, n_iter, feasible)`` — ``x_full`` is the full patch flat vector.

    - ``"isqp-osqp"`` (default): the tuned elastic-QP SQP over the free vars,
      UNCHANGED — the path every existing test and the no-damage invariant assume.
    - ``"scipy-slsqp"`` / ``"scipy-slsqp+trust-constr"``: the SLSQP leg runs through
      ``slsqp_traced.minimize_slsqp_traced`` — scipy's own C-core driver (verified
      byte-identical to ``minimize(method='SLSQP')``; see
      ``benchmarks/trace_parity_check.py``) with optional pyslsqp-style tracing —
      on the REDUCED free-variable problem (frozen vars pinned at ``sub.flat0``, so
      no-damage still holds by construction). ``+trust-constr`` escalates to scipy
      trust-constr only when SLSQP leaves an enforced row folded, and keeps
      whichever iterate reaches the higher constraint minimum (never worse than
      SLSQP alone).

    ``trace`` (optional dict) is threaded to the inner solver — ``isqp-osqp`` and
    the traced SLSQP leg both fill it with per-iteration records + an explicit
    exit reason (house style: ``trace['iters']`` / ``trace['exit']``). Default
    ``None`` keeps behavior byte-identical to the untraced path.
    """
    if inner == "isqp-osqp":
        return sv._isqp_solve_osqp(
            sub.flat0,
            sub.cons,
            sub.cons_jac,
            sub.obj_grad,
            maxiter,
            constraint=None,
            obj=sub.obj,
            hess_diag=sub.hess_diag,
            free_idx=sub.free_idx,
            trace=trace,
        )
    if inner not in ("scipy-slsqp", "scipy-slsqp+trust-constr"):
        raise ValueError(f"unknown inner {inner!r}")

    from slsqp_traced import minimize_slsqp_traced

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
    if inner == "scipy-slsqp+trust-constr" and cons_z(zf).min() < 0:
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


def _solve_window(
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
    _grow=0,
    margin_delta=1e-3,
    allow_grow=True,
    inner="isqp-osqp",
):
    """Solve one window; on infeasibility grow the blocked sides once and retry.
    Returns the inner solver's feasibility flag. ``allow_grow=False`` (used by the
    Schwarz tiler) keeps a tile at fixed size — growing a tile defeats tiling.
    ``inner`` picks the per-window solver (see :func:`_inner_solve`)."""
    H, W = phi.shape[1:]
    sub = build_subproblem(family, phi, box, threshold, objective, eps, margin_delta)
    t = time.perf_counter()
    x, nit, ok = _inner_solve(sub, inner, maxiter)
    dt = time.perf_counter() - t
    patch_out = np.asarray(sub.constraint.unflatten(x))
    py0, py1, px0, px1 = sub.patch_box
    # paste back ONLY free pixels (frozen ring is unchanged and may be shared)
    fm = sub.free_mask
    dst = phi[:, py0:py1, px0:px1]
    dst[:, fm] = patch_out[:, fm]

    jpatch = min_field(family, phi[:, py0:py1, px0:px1])
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
    if allow_grow and not ok and _grow < 2:
        fy0, fy1, fx0, fx1 = box
        gy0, gy1 = max(0, fy0 - 4), min(H, fy1 + 4)
        gx0, gx1 = max(0, fx0 - 4), min(W, fx1 + 4)
        if (gy0, gy1, gx0, gx1) != box:
            return _solve_window(
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
                inner=inner,
            )
    return bool(ok)
