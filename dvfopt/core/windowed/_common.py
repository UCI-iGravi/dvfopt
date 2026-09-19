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

import itertools
import math
import time
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from dvfopt._logging import log_info, log_warning, logger
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
    qp_max_iter: int | None = 1000  # None -> OSQP's own default (8000)
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
    ftol: float = (
        1e-2  # isqp inner: relative objective-decrease stop for feasible windows (0 = off)
    )
    patience_retry: bool = True
    orientation_delta: object = None  # float -> linear orientation rows in every window (2D)
    giant_workers: int = (
        0  # >1: RAS (Jacobi) giant-tile sweeps on the shared spawn pool (0 = serial)
    )
    polish: object = (
        None  # 'l2' | 'l1': per-window anchored polish after a successful solve (None = off)
    )
    polish_maxiter: int = 30
    ladder: bool = (
        True  # False -> one attempt per window: no retries, no grow (the mop's big windows)
    )
    orientation_rows: str = 'full'  # 'full' (edge + anti-diagonal rows) | 'edges' (edge rows only)


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
    overlap: int = 8  # tile overlap; resolved per dimension at entry (`_REANCHOR_OVERLAP`)


_REANCHOR_KINDS = ('none', 'l2', 'l1')
_REANCHOR_MIN_GAIN = 0.01  # stop sweeping once a sweep buys < 1% of the L2 move
_REANCHOR_TOL = 1e-9  # accepted rows must clear `threshold` by this much
_REANCHOR_OVERLAP = 8  # 2D re-anchor tile overlap, in px (48 stepped by 40 in the prototype)

DEFAULTS_BY_DIM: dict[str, dict[int, int | bool]] = {
    # knob -> {dim: default}. The 2D column IS the engine's signature default (never
    # change it here without changing the signature); the 3D column is the phase-3
    # measured table (CHANGELOG "3D windowed engine, phase 3"). A knob a caller leaves
    # at its 2D default takes the column for the field's dimension; any other explicit
    # value is honoured in every dimension — so `giant_tile=12` works on a 3D field, and
    # `mop_margin=0` still disables the mop. (`giant_tile=64` on a 3D field therefore
    # reads as "the 3D default"; pass 65 if you really want a 64-voxel tile, or
    # `dim_defaults=False` to take every knob literally — the escape for a 3D run that
    # must use a 2D-default value.)
    'giant_tile': {2: 64, 3: 16},  # 16^3 voxels ~= the 2D 64^2 tile by count
    'mop_margin': {2: 25, 3: 6},  # residual + 6/side -> a 13-17^3 mop window
    # sub20 whole 19 it / L2 19.1 vs tiled 437 it / 23.3; 14000 is 3x the wall (QP-bound)
    'max_window_area': {2: 3000, 3: 8000},
    'reanchor_tile': {2: 48, 3: 16},
    # module-internal, not a kwarg: the 2D column is a LITERAL 8 so that patching
    # `_REANCHOR_OVERLAP` (benchmarks/windowed_3d_sweep.py) reads as "not at the 2D
    # default" and is honoured on 3D. ov4: -20.5% L2 at half the wall vs ov8 -24%;
    # outside the 2% rule, so 8 stands. The entry block passes the constant in.
    'reanchor_overlap': {2: 8, 3: 8},
    # measured and REJECTED as False: the ADMM-only start wins the 17^3 whole window
    # (107 -> 26 it) but breaks the tiled crops (sliver 699 -> 1154 it, 2 rounds, mop +
    # re-seed; moderate 672 -> 1449) — phase 3
    'ip_cold': {2: True, 3: True},
    # measured and REJECTED at 2000 / 1000: wins the 17^3 whole window (107 -> 22 it)
    # but breaks the moderate crop (672 -> 1592 it, 2 rounds, mop + re-seed) and taxes
    # twist / sliver (+5 / +22 % it, +36 / +18 % wall); pass qp_max_iter=2000 explicitly
    # for whole-window sizes — phase 3
    'qp_max_iter': {2: 1000, 3: 1000},
    'qp_max_iter_fallback': {2: 500, 3: 500},  # tracks qp_max_iter (half)
    'ip_after_admm_iters': {2: 800, 3: 800},  # ip400 / ip200: no gain on either case
    # OFF on 3D (spike 2, benchmarks/output/spike_3d_2/REPORT.md). The stage needs 64
    # voxels per axis (`min(shape) >= 4 * giant_tile`), so it had never run on the 24^3
    # crops the 3D column was tuned on. Measured on a 64^3 cut: factor 4 is neutral
    # (-1.3 % SQP iterations) but INTRODUCES 12 negative axial edge projections (4
    # rotated cells) where the stage off introduces 0; factor 2 introduces 568 (95
    # rotated cells) and ends with more folds at equal wall. On the 2.5D residual's
    # densest box the default warm start took 493 -> 23,277 folds and the minimum
    # -0.035 -> -13.1 (L2 move 625 vs 39.9 with it off), with `damage` 0 throughout —
    # the damage counter cannot see it, because every new fold sits inside the set the
    # solve was entitled to move. Sharp edge (the table's rule is value equality):
    # `coarse_to_fine=True` on a 3D field reads as "the 2D default" and resolves to
    # False; `dim_defaults=False` is the escape that takes every knob literally.
    'coarse_to_fine': {2: True, 3: False},
}


def resolve_dim_defaults(dim, **knobs):
    """``knobs`` with every entry that sits at its 2D default replaced by the
    :data:`DEFAULTS_BY_DIM` column for ``dim`` (the identity for ``dim == 2``)."""
    out = dict(knobs)
    for k, col in DEFAULTS_BY_DIM.items():
        if k in out and out[k] == col[2]:
            out[k] = col[dim]
    return out


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
    constraint,
    phi_dydx,
    free_box,
    threshold,
    objective=None,
    margin_delta=1e-3,
    free_extra=None,
    orientation_delta=None,
    orientation_rows='full',
):
    """Build the window sub-problem for a free box of per-axis ``(lo, hi)`` pairs
    (global; ``(fy0, fy1, fx0, fx1)`` in 2D, ``(fz0, fz1, fy0, fy1, fx0, fx1)`` in 3D).

    Expands the free box by the family ring to a patch, instantiates a
    patch-shaped clone of ``constraint``'s type, selects the free pixels and the
    enforced constraint rows (those a free pixel influences AND that evaluate
    correctly), and returns a :class:`~dvfopt.core.windowed._inners.WindowSub`.

    ``margin_delta`` is enforced ON TOP of ``threshold`` (constraints are driven to
    ``threshold + margin_delta``) so that OSQP's ~1e-5 tolerance landing a hair
    short of the active bound still clears the strict ``< threshold`` fold check.

    ``orientation_delta`` (optional float) appends the LINEAR orientation rows of
    :func:`_orientation_rows` (deformed grid edges keep a positive projection of at
    least ``orientation_delta`` on their own direction, plus the anti-diagonal
    convexity rows) for every edge / cell with a free pixel. A cell on the rotated
    orientation branch violates them, so with the rows the QP never heads there --
    and they are linear, hence exact in the QP (no thin-cell linearisation error).
    2D, ``PhiPack.DY_FIRST`` families only. On a 3D patch the rows are
    :func:`_orientation_rows_3d` (axial edges on all three axes, ``'edges'`` only).

    ``free_extra`` (optional global bool mask of the field's spatial shape) is
    INTERSECTED with the free box, so a caller can free a subset of it — the
    re-anchor stage frees only pixels the main solve moved. ``None`` (default)
    frees the whole box. The patch is still the box expanded by the family ring,
    so the enforced rows a free pixel influences are all in-patch either way.
    """
    shape = phi_dydx.shape[1:]
    loc = _locality_of(constraint)
    ring = loc.ring
    patch_box = _pad_box(free_box, shape, ring)
    patch = np.ascontiguousarray(phi_dydx[(slice(None), *_box_slices(patch_box))])
    pshape = patch.shape[1:]
    c = type(constraint)(shape=pshape)
    flat0 = np.asarray(c.flatten(patch), dtype=np.float64)

    # free pixels (patch-local): the free box, clipped into the patch
    local = tuple(int(free_box[i]) - int(patch_box[2 * (i // 2)]) for i in range(len(free_box)))
    free_mask = np.zeros(pshape, bool)
    free_mask[_box_slices(local)] = True
    if free_extra is not None:
        free_mask &= free_extra[_box_slices(patch_box)]

    borders = tuple(
        b
        for ax, n in enumerate(shape)
        for b in (patch_box[2 * ax] == 0, patch_box[2 * ax + 1] == n)
    )
    enforced_idx, jac_of = loc.influenced(c, free_mask, *pshape, borders)

    # free variable indices in the constraint's own pack (never hand-packed)
    free_phi = np.stack([free_mask] * phi_dydx.shape[0]).astype(float)
    free_idx = np.nonzero(np.asarray(c.flatten(free_phi)) > 0.5)[0]

    target = threshold + margin_delta

    def cons(f):
        return (np.asarray(c.values(f)) - target)[enforced_idx]

    def cons_jac(f):  # sparse (n_enforced, n_vars) — enforced rows only
        return jac_of(f)[enforced_idx]

    n_rows = enforced_idx.size
    if orientation_delta is not None:
        if len(pshape) == 3:
            if orientation_rows != 'edges':
                raise ValueError(
                    "3D orientation rows: only kind='edges' exists (no convexity rows in 3D)"
                )
            a_or, b_or = _orientation_rows_3d(c, free_mask, float(orientation_delta))
        else:
            a_or, b_or = _orientation_rows(
                c, free_mask, float(orientation_delta), kind=orientation_rows
            )
        base_cons, base_jac = cons, cons_jac

        def cons(f, _a=a_or, _b=b_or):
            return np.concatenate([base_cons(f), _a @ np.asarray(f) + _b])

        def cons_jac(f, _a=a_or):
            return sparse.vstack([sparse.csr_matrix(base_jac(f)), _a], format='csr')

        n_rows += a_or.shape[0]

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
        patch_box,
        n_rows,
    )


def _orientation_rows(c, free_mask, delta, kind='full'):
    """Sparse ``(A, b)`` with ``A @ x + b >= 0`` the linear orientation rows of a patch.

    For every horizontal edge ``1 + dx[i,j+1] - dx[i,j] >= delta``, every vertical edge
    ``1 + dy[i+1,j] - dy[i,j] >= delta``, and every cell the two anti-diagonal convexity
    rows ``1 + dx[i,j+1] - dx[i+1,j] >= delta`` / ``1 + dy[i+1,j] - dy[i,j+1] >= delta``
    (:mod:`dvfopt.jacobian.monotonicity`'s injectivity conditions), restricted to
    edges / cells with at least one free pixel. ``x`` is the patch flat vector in the
    ``DY_FIRST`` pack (``[dy, dx]``), asserted on the constraint's ``pack``.
    """
    from dvfopt.constraints import PhiPack

    if getattr(c, 'pack', None) != PhiPack.DY_FIRST or free_mask.ndim != 2:
        raise ValueError('orientation rows need a 2D DY_FIRST (simplex-family) constraint')
    ph, pw = free_mask.shape
    n = ph * pw
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    cm = np.ones((ph - 1, pw - 1), bool)
    h_ok = np.ones((ph, pw - 1), bool)
    v_ok = np.ones((ph - 1, pw), bool)

    def add(ia, ib):  # row: x[ia] - x[ib] + (1 - delta) >= 0
        nonlocal r
        rows.extend((r, r))
        cols.extend((ia, ib))
        vals.extend((1.0, -1.0))
        rhs.append(1.0 - delta)
        r += 1

    fm = free_mask
    for i in range(ph):
        for j in range(pw - 1):
            if h_ok[i, j] and (fm[i, j] or fm[i, j + 1]):
                add(n + i * pw + j + 1, n + i * pw + j)
    for i in range(ph - 1):
        for j in range(pw):
            if v_ok[i, j] and (fm[i, j] or fm[i + 1, j]):
                add((i + 1) * pw + j, i * pw + j)
    if kind == 'full':  # anti-diagonal convexity rows; 'edges' keeps only the h/v edge rows
        for i in range(ph - 1):
            for j in range(pw - 1):
                if cm[i, j] and (fm[i, j] or fm[i, j + 1] or fm[i + 1, j] or fm[i + 1, j + 1]):
                    add(n + i * pw + j + 1, n + (i + 1) * pw + j)
                    add((i + 1) * pw + j, i * pw + j + 1)
    a = sparse.csr_matrix((vals, (rows, cols)), shape=(r, 2 * n))
    return a, np.asarray(rhs)


def _orientation_rows_3d(c, free_mask, delta):
    """Sparse ``(A, b)`` with ``A @ x + b >= 0`` the axial edge rows of a 3D patch.

    Every grid edge with at least one FREE endpoint keeps a positive projection of at
    least ``delta`` on its own axis: ``1 + dx[k,i,j+1] - dx[k,i,j] >= delta`` for the
    x-edges (dx block), likewise y-edges on dy and z-edges on dz — the ``'edges'`` kind
    of :func:`_orientation_rows` in the ``DX_FIRST`` pack ``[dx | dy | dz]``. The row
    matrix is :func:`dvfopt.jacobian.monotonicity.axial_gap_matrix` with its
    ``endpoints='any'`` filter: the free-to-frozen-ring edges are exactly the rows that
    stop a free voxel rotating against its pinned neighbour (the
    ``core.marching._mono_rows.mono_block`` predicate), so a both-endpoints-free filter
    would drop the load-bearing rows. Frozen-frozen rows ARE dropped: a violated constant
    row would sit in the elastic slack for the whole solve and distort the merit.
    """
    from dvfopt.constraints import PhiPack
    from dvfopt.jacobian.monotonicity import axial_gap_matrix

    if getattr(c, 'pack', None) != PhiPack.DX_FIRST or free_mask.ndim != 3:
        raise ValueError('3D orientation rows need a 3D DX_FIRST (simplex-family) constraint')
    n = free_mask.size
    a = axial_gap_matrix(free_mask.shape, free=free_mask, endpoints='any')
    if a is None:
        return sparse.csr_matrix((0, 3 * n)), np.zeros(0)
    return a, np.full(a.shape[0], 1.0 - float(delta))


def _orientation_violation_mask(phi, delta):
    """Grid points adjacent to an axial edge whose deformed projection on its own axis
    is below ``delta`` — the pixel-level violations of the edge rows of
    :func:`_orientation_rows` / :func:`_orientation_rows_3d`, in any rank.

    For axis ``a`` the edge projection is ``1 + diff(phi[a], axis=a)`` (channel ``a``
    carries the displacement along axis ``a`` in every pack, so this reads the
    ``(C, *shape)`` field and never the flat vector). Both endpoints of a violated edge
    are marked, which is the role a fold cell's corners play in :func:`_reseed_stage`.
    """
    bad = np.zeros(phi.shape[1:], bool)
    for ax in range(phi.ndim - 1):
        e = (1.0 + np.diff(phi[ax], axis=ax)) < delta
        lo = [slice(None)] * bad.ndim
        hi = [slice(None)] * bad.ndim
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        bad[tuple(lo)] |= e
        bad[tuple(hi)] |= e
    return bad


def _box_slices(box):
    """``(a0, a1, b0, b1, ...)`` -> ``(slice(a0, a1), slice(b0, b1), ...)`` — one
    slice per axis, so ``arr[(slice(None), *_box_slices(box))]`` crops a
    ``(C, *shape)`` field to the box in any dimension."""
    return tuple(slice(int(box[i]), int(box[i + 1])) for i in range(0, len(box), 2))


def _box_size(box):
    """Grid points inside ``box`` (area in 2D, volume in 3D)."""
    n = 1
    for i in range(0, len(box), 2):
        n *= max(0, int(box[i + 1]) - int(box[i]))
    return n


def _pad_box(box, shape, pad):
    """Grow every side of ``box`` by ``pad`` grid points, clipped to ``shape``."""
    out = []
    for ax, n in enumerate(shape):
        out.append(max(0, int(box[2 * ax]) - pad))
        out.append(min(int(n), int(box[2 * ax + 1]) + pad))
    return tuple(out)


def _folds_in_box(constraint, phi, box, threshold):
    """``int((min_field(constraint, phi)[_box_slices(box)] < threshold).sum())`` computed on
    the box padded by ONE grid point per side (clipped to the field), never on the whole
    field. Every cell / cube / central-difference stencil whose value lands inside ``box``
    reads at most one grid point beyond it, so the padded sub-field reproduces those values
    exactly (elementwise arithmetic — bit-identical, not merely close); the sub-field's own
    last plane / row / column is ``+inf``, or one-sided for ``jdet``, exactly where the whole
    field's is (the true border) and lies outside ``box`` otherwise."""
    shape = phi.shape[1:]
    pb = _pad_box(box, shape, 1)
    sub = min_field(constraint, phi[(slice(None), *_box_slices(pb))])
    inner = tuple(
        slice(box[2 * a] - pb[2 * a], box[2 * a + 1] - pb[2 * a]) for a in range(len(shape))
    )
    return int((sub[inner] < threshold).sum())


def _resolve_opts_for_test(dim):
    """Test hook: the :class:`_InnerOpts` :func:`windowed_correct` builds at its OWN
    defaults for a ``dim``-dimensional field (signature defaults -> per-dimension table
    -> the 3D ``step_rule`` degrade), so a test can call :func:`_solve_window` /
    :func:`_ras_tile_task` directly with the shipped knobs."""
    import inspect

    sig = inspect.signature(windowed_correct).parameters
    kw = resolve_dim_defaults(
        dim, **{f.name: sig[f.name].default for f in fields(_InnerOpts) if f.name in sig}
    )
    if dim == 3:
        kw['step_rule'] = 'tr'  # 'exact_ls' is 2D-only (see windowed_correct)
    return _InnerOpts(**kw)


def find_windows(mask, margin, ring):
    """Free boxes around fold clusters, per-axis ``(lo, hi)`` pairs in axis order.

    Dilating the fold mask by ``margin+ring`` before labelling merges clusters whose
    (free box + ring) regions could touch, so a window's free set does not fall in
    another's context ring. (Two diagonally-offset clusters can still yield
    overlapping bounding boxes and hence overlapping free boxes — that only causes
    redundant/overwritten work, never damage, since each window enforces its full
    influence set; see the module invariant.)

    The dilated bbox is inset by ``ring`` to recover the ``cluster + margin`` free
    box — but NOT on a side that reached the image border, where the fold sits on
    the border line and must stay free (mirrors the Schwarz tiler's border guard).

    Works in any dimension: boxes are per-axis ``(lo, hi)`` pairs in axis order.
    """
    grow = margin + ring
    dil = ndimage.binary_dilation(mask, iterations=grow)
    lbl, n = ndimage.label(dil)
    boxes = []
    for sl in ndimage.find_objects(lbl):
        box = []
        for s, n_ax in zip(sl, mask.shape):
            box.append(s.start + ring if s.start > 0 else 0)  # keep image-border folds free
            box.append(s.stop - ring if s.stop < n_ax else n_ax)
        boxes.append(tuple(box))
    return boxes


@dataclass
class WindowRec:
    z: int = -1
    fy0: int = 0
    fx0: int = 0
    ph: int = 0
    pw: int = 0
    patch_box: tuple = ()  # per-axis (lo, hi) pairs of the patch, any dimension
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
    polish_windows: int = 0
    polish_accepted: int = 0
    patience_fallbacks: int = 0  # windows that took the patience rung
    # resumable runs (``checkpoint_dir=``): the last checkpoint unit done at resume
    # ('' on an uninterrupted run, 'finished' when the mirror was already complete).
    resumed_from: str = ''
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
    # 3D certificate (SimplexConstraint3D only; -1 on 2D fields): the fixed-diagonal
    # 6-tet fold count at 0 (folds_after is at `threshold`) and the best-of-4-diagonals
    # floor — cubes no main-diagonal split certifies — at `threshold` and at 0
    # (`tetrahedron_sign.n_neg_best_diagonal`, `<=` semantics; the rows are driven
    # to threshold + margin_delta so no cell parks exactly on the threshold).
    folds_after_zero: int = -1
    best_diag_floor_after: int = -1
    best_diag_floor_after_zero: int = -1
    rounds: int = 0
    time_s: float = 0.0
    windows: list = field(default_factory=list)
    # per-stage convergence entries, filled only when ``record_history=True``
    # (each: name / n_iter / n_neg / min_T / wall_s; the 'final' entry adds
    # ``extras`` with damage / n_windows / giant_regions / mop_cleared /
    # l1_move / l2_move). Empty list otherwise.
    history: list = field(default_factory=list)


def _restrict(phi, factor):
    """Box-average ``factor``-blocks of a ``(C, *shape)`` field, any rank.

    Displacements are divided by ``factor`` so they stay in COARSE grid units —
    the coarse field is then an ordinary deformation field on its own grid and the
    same constraint/threshold means the same thing there. A trailing partial block
    along any axis is dropped.
    """
    coarse = [d // factor for d in phi.shape[1:]]
    trimmed = phi[(slice(None), *(slice(0, factor * n) for n in coarse))]
    blocked = trimmed.reshape(phi.shape[0], *(v for n in coarse for v in (n, factor)))
    return blocked.mean(axis=tuple(range(2, 2 * len(coarse) + 1, 2))) / factor


def _prolongate(delta_c, shape, factor):
    """Multilinear ``factor`` x upsample of a coarse-grid CORRECTION back to ``shape``.

    Displacements are multiplied by ``factor`` (the inverse of :func:`_restrict`'s
    rescale). Planes/rows/cols the integer factor cannot cover (odd sizes) stay
    zero — the fine solve handles that strip itself.
    """
    out = np.zeros((delta_c.shape[0], *shape))
    for c in range(delta_c.shape[0]):
        up = ndimage.zoom(delta_c[c] * factor, factor, order=1)
        keep = tuple(slice(0, min(n, u)) for n, u in zip(shape, up.shape))
        out[(c, *keep)] = up[keep]
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

    Why it pays: the coarse solve is ~1/factor**ndim the work and lands the fine
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
    for box in boxes:
        allow[_box_slices(box)] = True
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
    qp_max_iter=1000,
    qp_max_iter_fallback=500,
    giant_tile=64,
    giant_max_sweeps=8,
    giant_tile_fit=True,
    giant_workers=0,
    qp_backend='hybrid',
    ip_cold=True,
    ip_after_admm_iters=800,
    tr_delta=2.0,
    tr_max=16.0,
    step_rule='exact_ls',
    exact_ls_fallback_steps=3,
    ftol=1e-2,
    patience_retry=True,
    orientation_delta=0.01,
    orientation_rows='edges',
    polish=None,
    polish_maxiter=30,
    coarse_to_fine=True,
    coarse_factor=4,
    reanchor='none',
    reanchor_maxiter=60,
    reanchor_sweeps=3,
    reanchor_tile=48,
    reseed_rounds=3,
    reseed_radius=2,
    time_budget_s=None,
    checkpoint_dir=None,
    touched_out=None,
    dim_defaults=True,
    verbose=1,
    record_history=False,
    step_callback=None,
):
    """Correct a full ``(2, H, W)`` slice — or, with
    :class:`~dvfopt.constraints.SimplexConstraint3D`, a ``(3, D, H, W)`` volume
    (3D: every stage runs (phase 2 of the 3D port)) — by solving one small
    window per fold cluster. Returns ``(phi_out, SliceReport)``.

    ``constraint`` is a registered constraint instance
    (:class:`~dvfopt.constraints.JdetConstraint2D`,
    :class:`~dvfopt.constraints.SimplexConstraint2D`,
    :class:`~dvfopt.constraints.SimplexConstraint2DBilinear`,
    :class:`~dvfopt.constraints.FiniteJdetConstraint2D`, or
    :class:`~dvfopt.constraints.SimplexConstraint3D` — see
    :data:`~dvfopt.core.windowed._locality.LOCALITY`); ``objective`` is a
    :class:`~dvfopt.objectives.Objective` (``None`` -> ``L2Objective()``; the
    L1 smoothing eps rides on ``L1Objective(eps=...)``).

    ``max_window_area`` caps a window's free-box area; larger merged clusters are
    cleared by overlapping-tile Schwarz decomposition (``report.giant_regions``)
    instead of an intractable near-full-grid QP. ``margin`` is clamped to at least
    the family ring (the frozen inset band must stay fold-free).

    **Per-dimension defaults.** A knob listed in :data:`DEFAULTS_BY_DIM`
    (``giant_tile``, ``mop_margin``, ``max_window_area``, ``reanchor_tile``,
    the QP caps) that the caller LEAVES at its 2D default is replaced at entry by that
    knob's column for the field's dimension — so a 3D solve gets the phase-3
    measured 3D table without a second set of ``*_3d`` parameters. Any other
    explicit value is honoured in every dimension (``giant_tile=12`` on a 3D field
    is 12; ``mop_margin=0`` still disables the mop). Phase-3 ruled the 3D column at
    ``max_window_area=8000`` (an 8000-voxel 20^3 region solves whole in 19 iterations vs 437
    tiled); ``qp_max_iter=2000``/``qp_max_iter_fallback=1000`` was measured and
    REJECTED — it wins the 17^3 whole window (107 -> 22 iterations) but breaks the
    moderate crop (672 -> 1592 iterations, 2 rounds, mop + re-seed) and taxes
    twist / sliver (+5 / +22% iterations, +36 / +18% wall), so the QP caps keep
    their 2D values on 3D and ``qp_max_iter=2000`` is the explicit opt-in for
    whole-window sizes; ``ip_cold=False``
    was measured and rejected (it wins the 17^3 whole window, 107 -> 26 iterations,
    but breaks the tiled crops — sliver 699 -> 1154 iterations, 2 rounds, mop +
    re-seed; moderate 672 -> 1449). The rule has a sharp edge: a value that IS the 2D
    default reads as "leave it at the 2D default" and still resolves to the 3D
    column, so ``dim_defaults=False`` is the escape — it takes every knob literally,
    with no per-dimension resolution.

    ``inner`` selects the per-window solver — ``"isqp"`` (default, the tuned
    elastic-QP SQP), ``"slsqp"``, or ``"slsqp+trust-constr"`` (see
    :func:`~dvfopt.core.windowed._inners.solve_window_inner` for the aliases).
    All inners only move the window's free pixels, so the no-damage invariant
    holds regardless of the choice.

    After the round loop plateaus, a terminal **mop** pass re-windows any residual
    with ``mop_margin`` (>> ``margin``): the diagnostic on the densest slices shows
    the plateau is boundary-stuck folds inside the giants that a small window's
    tight frozen boundary can't clear but a large frozen-exterior window can (the
    analogue of the 2.5D pipeline's ``mop_interior_3d``). ``mop_margin=0``
    disables. ``mop_margin`` 6 on 3D gives a 13-17^3 mop window (2.2k-4.9k voxels); the
    mop's one-attempt threshold and its ``whole_cap`` (4x) follow ``max_window_area``,
    which the phase-3 table sets to 8000 on 3D — so a 3D mop window of that size runs
    the full ladder and ``whole_cap`` is 32000 voxels (phase 2 measured the mop at the
    3000 / 12000 thresholds; the cluster crop is the case that fires it — CHANGELOG
    phase 3).

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
    On a 3D field the default is 16 per axis (16³ voxels is the 2D 64² tile by
    count; phase 1 measured 11 s per SQP iteration at 17³ and 69 s at 25³, so the
    tile must stay near 16).

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
    2D-only: on a 3D field it degrades to ``'tr'`` (DEBUG log).

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

    ``coarse_to_fine=True`` (the 2D default; **False on 3D** — it is a
    :data:`DEFAULTS_BY_DIM` row, so an explicit ``True`` on a 3D field reads as the
    2D default and resolves to ``False``, and ``dim_defaults=False`` is the escape)
    prepends a **coarse-grid warm start**: the
    same problem is solved on a ``coarse_factor`` x coarsened field and the
    prolongated correction seeds the fine solve, so the fine windows start near a
    solution instead of cold (raw B0039 z16: 205 s / 909 SQP iterations — 841 fine
    plus a 16 s, 68-iteration coarse solve — vs 283 s / 1320 cold, at a slightly
    smaller L2 move, 320.6 vs 325.1).
    The warm-start delta is MASKED to the window free boxes the fine engine would
    open anyway, so the no-damage invariant holds unchanged and the final damage
    accounting still runs against the ORIGINAL input. It is skipped — leaving the
    path byte-identical to ``coarse_to_fine=False`` — when the field has no folds
    or when ``min(shape) < 4 * max(giant_tile, coarse_factor)`` (the
    per-dimension-resolved tile; below that the coarse problem is too
    small to be a meaningful preview, and its own solve is not amortised;
    the ``coarse_factor`` leg only bites for absurd factors).
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

    ``orientation_delta`` (``None`` = off; a float, e.g. ``0.01``) appends the LINEAR
    orientation rows (:func:`_orientation_rows`: deformed grid edges keep a
    positive projection of at least ``orientation_delta`` on their own direction,
    plus the anti-diagonal convexity rows) to every window sub-problem. The residual
    the ladder plateaus on is a cluster driven onto the ROTATED orientation branch
    (see ``reseed_rounds``); a rotated cell violates these rows, so with them the QP
    never heads there -- prevention, where the re-seed is repair -- and they are
    linear, hence exact in the QP. Measured from raw with the re-seed off: B0304 ext
    z128 8956 -> 0 folds (3643 s, L2 1405 vs 1395 for the re-seed path), B0032 ext
    z1 4556 -> 0 in 1066 s / 1 round / 31 windows where the plain engine left 70
    after 8902 s / 374 windows (L2 1988 vs 1973). 2D simplex-family only; raises
    on other packs.

    ``giant_workers`` (default 0 = serial) solves each giant-tile sweep as
    restricted additive Schwarz on the shared spawn pool: all tiles of a sweep
    from the same snapshot concurrently, each pasting back only its disjoint
    step-grid core. Trades within-sweep propagation (possibly more sweeps) for
    parallel tiles; the serial multiplicative sweep is the default and
    byte-identical at 0.

    ``polish='l2'|'l1'`` (default None = off) adds a per-window anchored polish:
    after a window solves, the SAME box is re-solved against the distance to its
    pre-solve patch from the warm feasible point (``polish_maxiter`` inner
    iterations, the ``ftol`` stop), verify-and-revert as in the reanchor stage.
    Meant for ``objective='none'``: the cheap feasibility solve plus a short
    anchored polish, instead of anchored-QP prices for the whole fight.

    ``reseed_rounds`` (default 3, 0 = off) / ``reseed_radius`` (default 2) add a
    **terminal harmonic re-seed stage** for the residual the round loop AND the mop
    plateau on. Such residual clusters are, measured on every non-clean slice of
    the 7-brain cohort, cells whose corner images have been driven onto the
    ROTATED orientation branch (both edge factors of the area negative, product
    positive): locally fold-free, but not joinable to the surrounding field, and
    the seam between the branches is a merit MAXIMUM no local step crosses -- every
    rung, step rule and trust radius fails there identically. The stage resets the
    branch: each residual cluster's neighbourhood (its cells' corner pixels, plus the
    endpoints of every edge violating the orientation rows when ``orientation_delta``
    is on — a rotated strip has POSITIVE areas, so it is invisible to the fold mask
    and re-seeding around the fold at its end would fill from a rotated ring —
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

    ``dim_defaults=False`` takes every knob literally — no per-dimension resolution;
    the measurement escape for a 3D run that must use a 2D-default value, e.g.
    ``giant_tile=64``.

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

    **Checkpoint / resume.** ``checkpoint_dir`` (``None`` = off, the default —
    byte-identical to before) mirrors progress through
    :class:`dvfopt.checkpoint.RunCheckpoint`: a unit is marked after the coarse
    stage, after each round, after each giant-tiler sweep, after the mop, after
    the re-seed stage and after the re-anchor stage, each mark mirroring the
    current field to ``<checkpoint_dir>/field.npy`` / ``state.json`` and the
    current ``touched`` mask to ``<checkpoint_dir>/touched.npy``. A resumed run
    reloads the mirrored field and ``touched.npy``, restores the counters of the
    last row into the report, skips every stage whose unit is done, and
    re-enters the round loop with a fresh fold mask — so its ``rep.windows`` /
    ``rep.history`` cover only the post-resume work and the round loop's
    no-progress check restarts; the certificate and the damage accounting
    (against the ORIGINAL input, through the restored ``touched``) are the same
    invariants as an uninterrupted run. ``time_budget_s`` counts from the
    resume. ``report.resumed_from`` names the last done unit at resume (``''``
    on an uninterrupted run, ``'finished'`` when the mirror was already
    complete — the whole run then just reloads). A checkpoint whose stored
    knobs/shape/input hash disagree with this call raises ``ValueError``.
    ``touched_out`` (optional bool array of the field's spatial shape) is
    OR-ed with the engine's final ``touched`` mask in place, letting a caller
    compose no-damage accounting across multiple ``windowed_correct`` calls
    (e.g. a banded driver over sub-volumes).
    """
    if step_rule not in ('tr', 'exact_ls'):
        raise ValueError(f"unknown step_rule {step_rule!r}; valid: 'tr', 'exact_ls'")
    if reanchor not in _REANCHOR_KINDS:
        raise ValueError(f"unknown reanchor {reanchor!r}; valid: {list(_REANCHOR_KINDS)}")
    loc = _locality_of(constraint)  # raises IncompatibleConstraintError if unregistered
    dim = int(constraint.dim)
    if np.asarray(phi_in).ndim != dim + 1:
        raise ValueError(
            f"windowed_correct: a {dim}D constraint needs a rank-{dim + 1} ({dim}, ...) field; "
            f"got rank {np.asarray(phi_in).ndim}"
        )
    is3d = dim == 3
    knobs = dict(
        giant_tile=giant_tile,
        mop_margin=mop_margin,
        max_window_area=max_window_area,
        reanchor_tile=reanchor_tile,
        reanchor_overlap=_REANCHOR_OVERLAP,
        qp_max_iter=qp_max_iter,
        qp_max_iter_fallback=qp_max_iter_fallback,
        ip_cold=ip_cold,
        ip_after_admm_iters=ip_after_admm_iters,
        coarse_to_fine=coarse_to_fine,
    )
    r = resolve_dim_defaults(dim, **knobs) if dim_defaults else knobs
    giant_tile, mop_margin, max_window_area = r['giant_tile'], r['mop_margin'], r['max_window_area']
    reanchor_tile, reanchor_overlap = r['reanchor_tile'], r['reanchor_overlap']
    qp_max_iter, ip_after_admm_iters = r['qp_max_iter'], r['ip_after_admm_iters']
    qp_max_iter_fallback, ip_cold = r['qp_max_iter_fallback'], r['ip_cold']
    coarse_to_fine = r['coarse_to_fine']
    if step_rule == 'exact_ls' and is3d:
        # The exact line model needs rows that are BILINEAR in the displacements — true
        # of every 2D family here, false of a 6-tet volume (trilinear, hence cubic along
        # a line). The cubic model was built and measured in phase 3 and REFUTED for the
        # default (wins the sparse crops, loses the dense ones — CHANGELOG); degrade to
        # the ratio test, the way orientation_delta is dropped on non-DY_FIRST packs below.
        logger.debug(
            "windowed_correct: step_rule='exact_ls' is 2D-only; using 'tr' on this 3D field"
        )
        step_rule = 'tr'
    if is3d and orientation_delta is not None and orientation_rows != 'edges':
        raise ValueError("3D orientation rows: only kind='edges' exists (no convexity rows in 3D)")
    if orientation_delta is not None:
        from dvfopt.constraints import PhiPack

        # The simplex families provide edge rows; Jdet / finite keep the plain rows
        # (an explicit request on such a sub-problem still raises in build_subproblem).
        family_rows = constraint.pack == PhiPack.DY_FIRST or is3d
        if not family_rows:
            orientation_delta = None
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
        ftol,
        patience_retry,
        orientation_delta,
        orientation_rows=orientation_rows,
        giant_workers=giant_workers,
        polish=polish,
        polish_maxiter=polish_maxiter,
    )
    objective = L2Objective() if objective is None else objective
    phi = np.array(phi_in, dtype=np.float64, copy=True)
    ring = loc.ring
    margin = max(margin, ring)  # inset band must be fold-free margin, never < ring
    shape = phi.shape[1:]
    j0 = min_field(constraint, phi)
    orig_fold = j0 < threshold
    rep = SliceReport(folds_before=int(orig_fold.sum()), min_before=float(j0.min()))
    touched = np.zeros(shape, bool)  # union of every window's ENFORCED footprint

    ck = None
    done_units: set = set()
    if checkpoint_dir is not None:
        from dvfopt.checkpoint import RunCheckpoint, atomic_replace

        meta = dict(
            engine='windowed',
            inner=str(inner),
            constraint=type(constraint).__name__,
            objective=type(objective).__name__,
            threshold=float(threshold),
            margin=int(margin),
            maxiter=int(maxiter),
            max_rounds=int(max_rounds),
            max_window_area=int(max_window_area),
            mop_margin=int(mop_margin),
            orientation_delta=None if orientation_delta is None else float(orientation_delta),
            orientation_rows=str(orientation_rows),
            reseed_rounds=int(reseed_rounds),
            reanchor=str(reanchor),
        )
        ck = RunCheckpoint(checkpoint_dir, phi_in, meta, slab=lambda _u: Ellipsis).open()
        tp = ck.dir / 'touched.npy'
        if ck.finished:
            phi[...] = ck.field
            touched[...] = np.load(tp) if tp.exists() else False
            rep.resumed_from = 'finished'
            log_info(f'[windowed resume] finished run reloaded from {ck.dir}')
        elif ck.done:
            phi[...] = ck.field
            touched[...] = np.load(tp) if tp.exists() else False
            last = ck.done[-1]
            row = ck.rows.get(str(last), {})
            rep.rounds = int(row.get('rounds', 0))
            rep.giant_regions = int(row.get('giant_regions', 0))
            rep.mop_windows = int(row.get('mop_windows', 0))
            rep.mop_cleared = int(row.get('mop_cleared', 0))
            rep.reseed_rounds_run = int(row.get('reseed_rounds_run', 0))
            rep.resumed_from = str(last)
            done_units = set(map(str, ck.done))
            log_info(f'[windowed resume] {len(ck.done)} units from {ck.dir}, last {last!r}')

    def _mark(unit):
        if ck is None:
            return
        tmp = ck.dir / 'touched.npy.tmp'
        with open(tmp, 'wb') as f:  # np.save(path) would append another .npy suffix
            np.save(f, touched)
        atomic_replace(tmp, ck.dir / 'touched.npy')
        ck.mark(
            unit,
            phi,
            row=dict(
                rounds=rep.rounds,
                n_windows=len(rep.windows),
                giant_regions=rep.giant_regions,
                mop_windows=rep.mop_windows,
                mop_cleared=rep.mop_cleared,
                reseed_rounds_run=rep.reseed_rounds_run,
                folds=int(pixel_fold_mask(constraint, phi, threshold).sum()),
            ),
        )

    def _skip(unit):
        return unit in done_units or rep.resumed_from == 'finished'

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
    if (
        coarse_to_fine
        and rep.folds_before > 0
        and min(shape) >= 4 * max(opts.giant_tile, coarse_factor)
        and not _skip('coarse')
    ):
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
                # the RESOLVED values: on 3D they are no longer at their 2D default,
                # so `resolve_dim_defaults` passes them through unchanged
                mop_margin=mop_margin,
                time_budget_s=time_budget_s,
                dim_defaults=dim_defaults,
                verbose=verbose,
                **_engine_kwargs(opts),
            ),
        )
        phi += delta
        for wb in warm_boxes:  # the warm start is a move over these
            touched[_box_slices(_pad_box(wb, shape, ring))] = True
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
        _mark('coarse')

    budget_hit = False
    prev_nfold = None
    if rep.resumed_from != 'finished':
        for _rnd in range(rep.rounds, max_rounds):
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
                # touched = the ENFORCED footprint (free box dilated by ring), not the
                # bare free box: a free pixel influences constraints up to `ring` beyond
                # the free box, so an infeasible solve could leave a violated row there.
                # Marking it touched makes any such residual count as residual, never
                # damage — so damage=0 is by construction, not merely for feasible solves.
                touched[_box_slices(_pad_box(box, shape, ring))] = True
                if _box_size(box) > max_window_area:
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
                        on_sweep=lambda s, g=rep.giant_regions: _mark(f'giant:{g}:{s}'),
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
            _mark(f'round:{rep.rounds}')
            if budget_hit:
                break

    if budget_hit:
        log_warning("windowed_correct: time budget exhausted; stopping with best-so-far field")

    def _run_reseed():
        """The harmonic re-seed stage + polish (see ``reseed_rounds``); no-op on a clean field."""
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
                dim_defaults=dim_defaults,
                verbose=verbose,
                **_engine_kwargs(opts),
            ),
            expired=_expired,
        )
        if record_history and rep.reseed_rounds_run:
            rep.history.append(_stage_entry("reseed", len(rep.windows)))
        if rep.reseed_rounds_run:
            _fire("reseed", phi)

    # terminal mop: clear the boundary-stuck residual the round loop plateaued on
    if mop_margin > 0 and not budget_hit and not _skip('mop'):
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
            _mark('mop')

    if reseed_rounds > 0 and not budget_hit and not _skip('reseed'):
        _run_reseed()
        _mark('reseed')

    # Post-feasibility re-anchor: recover fidelity now that no fold is left to trap
    # the inner in an objective basin. Only on a fold-free field, and reverted whole
    # if it somehow breaks that (per-tile verification should make this unreachable).
    if reanchor != 'none' and not budget_hit and not _expired() and not _skip('reanchor'):
        if int(pixel_fold_mask(constraint, phi, threshold).sum()) == 0:
            saved = phi.copy()
            _reanchor_pass(
                phi,
                np.asarray(phi_in, dtype=np.float64),
                constraint,
                threshold,
                _ReanchorOpts(
                    reanchor,
                    reanchor_maxiter,
                    reanchor_sweeps,
                    reanchor_tile,
                    reanchor_overlap,
                ),
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
            _mark('reanchor')

    # Only a run that actually completed gets stamped 'done' — a budget-expired exit
    # leaves stage='run' with its last mark as the resume point (R7): stamping 'done'
    # here unconditionally would make a time-budget-cut run look finished, so the next
    # call would reload the UNFINISHED field and skip every stage, permanently.
    if ck is not None and not ck.finished and not budget_hit and not _expired():
        ck.finish(phi)

    jf = min_field(constraint, phi)
    after_fold = jf < threshold
    new = after_fold & ~orig_fold
    rep.folds_after = int(after_fold.sum())
    rep.min_after = float(jf.min())
    damage_mask = new & ~touched
    rep.damage = int(damage_mask.sum())  # invariant: MUST be 0
    rep.damage_coords = [tuple(int(v) for v in c) for c in np.argwhere(damage_mask)[:20]]
    rep.residual_in_window = int((after_fold & touched).sum())
    if is3d:
        from dvfopt.jacobian.tetrahedron_sign import best_diagonal_min_volume

        best_min, _ = best_diagonal_min_volume(phi)
        rep.folds_after_zero = int((jf <= 0).sum())
        rep.best_diag_floor_after = int((best_min <= threshold).sum())
        rep.best_diag_floor_after_zero = int((best_min <= 0.0).sum())
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
    if touched_out is not None:
        touched_out |= touched
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
    Sweeps until no further progress, i.e. the genuine local floor. Any rank."""
    shape = phi.shape[1:]
    whole_cap = 4 * max_window_area  # the mop is allowed much larger single QPs
    for _sweep in range(max_sweeps):
        mask = pixel_fold_mask(constraint, phi, threshold)
        n = int(mask.sum())
        if n == 0:
            break
        lbl, _ = ndimage.label(mask)  # raw residual clusters (per connected component)
        for sl in ndimage.find_objects(lbl):
            box = _pad_box(tuple(v for s in sl for v in (s.start, s.stop)), shape, mop_margin)
            touched[_box_slices(_pad_box(box, shape, ring))] = True
            rep.mop_windows += 1
            if _box_size(box) > whole_cap:
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
                # A mop window above max_window_area gets ONE attempt: no retries, no
                # grow. Measured on the full-resolution B0039 exterior z=1 trace, these
                # whole-cluster windows (up to 9 660 free pixels, 3.7 s per SQP
                # iteration) ran the entire ladder -- 10 calls per box -- on the
                # rotated-branch residual for 12 367 of 15 657 s (79%), and the re-seed
                # stage then cleared it in 7 s. Small mop windows keep the full ladder
                # (the sliver-type residual needs it and is cheap).
                big = _box_size(box) > max_window_area
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
                    opts=replace(opts, ladder=False) if big else opts,
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
    psl = (slice(None), *_box_slices(sub.patch_box))
    ref = np.asarray(sub.constraint.flatten(np.ascontiguousarray(phi_ref[psl])))
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
    dst = phi[psl]
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
    tile the MOVED region (``reanchor_tile``, overlapping by ``ropts.overlap``),
    re-solve each tile minimising the distance to the input
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
    # overlap so a seam is a neighbour's interior; same floor as the tiler (a 3D
    # re-anchor tile of 16 -> step 8; 8 -> step 4, not 1)
    step = max(tile - ropts.overlap, tile // 2, 1)
    pts = np.nonzero(moved)
    lo = [int(p.min()) for p in pts]
    hi = [int(p.max()) + 1 for p in pts]

    def l2_move():
        return float(np.linalg.norm((phi - phi_ref).ravel()))

    rep.reanchor_l2_before = rep.reanchor_l2_after = prev = l2_move()
    for _sweep in range(max(0, ropts.sweeps)):
        rep.reanchor_sweeps_run += 1
        for starts in itertools.product(*(range(a, b, step) for a, b in zip(lo, hi))):
            box = tuple(v for t, b in zip(starts, hi) for v in (t, min(t + tile, b)))
            if not moved[_box_slices(box)].any():
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


def _engine_kwargs(opts):
    """``_InnerOpts`` as ``windowed_correct`` kwargs for a recursive solve (the coarse
    warm start, the re-seed polish): every knob except ``ladder``, which is per-window
    (the mop's big windows) and not a ``windowed_correct`` parameter. ``dim_defaults``
    is not here either — it is not an ``_InnerOpts`` field — but the two recursive call
    sites pass it explicitly alongside these: under ``False`` the knobs ARE the 2D
    defaults the flag exists to keep, so a recursive solve without it would resolve them
    to the 3D column."""
    return {k: v for k, v in asdict(opts).items() if k != "ladder"}


def _harmonic_fill(phi, mask):
    """Replace ``phi[:, mask]`` by the discrete-harmonic (2·ndim-neighbour Laplacian)
    interpolation of ``phi`` on the mask's boundary, in place — any rank. One sparse
    solve per channel over the masked grid points (a few hundred on real residuals).
    The neighbour order (last axis first, ``+1`` before ``-1``) is the 2D engine's
    ``(0, 1), (0, -1), (1, 0), (-1, 0)`` and fixes the boundary sum's float order."""
    shape = mask.shape
    ndim = len(shape)
    pts = np.nonzero(mask)
    n = len(pts[0])
    if n == 0:
        return
    idx = np.full(shape, -1, dtype=np.int64)
    idx[pts] = np.arange(n)
    offsets = [
        tuple((d if a == ax else 0) for a in range(ndim))
        for ax in reversed(range(ndim))
        for d in (1, -1)
    ]
    rows, cols, vals = [], [], []
    rhs = np.zeros((phi.shape[0], n))
    for k, p in enumerate(zip(*pts)):
        deg = 0
        for off in offsets:
            q = tuple(int(pi) + oi for pi, oi in zip(p, off))
            if not all(0 <= qi < ni for qi, ni in zip(q, shape)):
                continue
            deg += 1
            if mask[q]:
                rows.append(k)
                cols.append(int(idx[q]))
                vals.append(-1.0)
            else:
                rhs[:, k] += phi[(slice(None), *q)]
        rows.append(k)
        cols.append(k)
        vals.append(float(deg))
    lap = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))
    for ch in range(phi.shape[0]):
        phi[(ch, *pts)] = spsolve(lap, rhs[ch])


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

    A residual cell's corner pixels (the cell and its +1 shift along every axis),
    plus — when the engine's orientation rows are on (``opts.orientation_delta``) —
    every endpoint of an edge violating them (:func:`_orientation_violation_mask`),
    dilated by ``radius`` form the re-seed mask; its interior is replaced by the
    harmonic interpolation of the ring, which puts the cluster back on the ring's
    orientation branch. The orientation violations are load-bearing: a 180-degree
    rotated cell has a POSITIVE determinant, so a rotated strip is area-feasible and
    invisible to the fold mask, and re-seeding only around the fold at its end fills
    from a ring that is itself rotated (measured on ``m2_ffd_brainpair_coarse``: the
    stage stalled there at 6 folds; with the strip in the mask the field certifies).
    The polish is :func:`windowed_correct` on the re-seeded
    field with this stage off (never recursive) and the coarse warm start off; its
    windows' patch boxes join ``touched`` so the outer damage accounting stays
    exact. Stops when the field is fold-free, the deadline passes, or a round makes
    no progress. Any rank.
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
        # a cell's 2**ndim corner grid points ({0, +1} offsets per axis)
        corners = ndimage.binary_dilation(
            fold, structure=np.ones((2,) * fold.ndim, bool), origin=-1
        )
        if opts.orientation_delta is not None:  # rotated cells are area-feasible: see above
            corners |= _orientation_violation_mask(phi, float(opts.orientation_delta))
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
            **{k: v for k, v in sub_kw.items() if k != "ladder"},
        )
        phi[...] = out
        for w in rep_in.windows:  # the polish's enforced footprints (= the padded patches)
            touched[_box_slices(w.patch_box)] = True
        rep.windows.extend(rep_in.windows)
        rep.backend_fallbacks += rep_in.backend_fallbacks
        rep.patience_fallbacks += rep_in.patience_fallbacks
        nf_after = int(pixel_fold_mask(constraint, phi, threshold).sum())
        if nf_after >= prev:
            break  # no progress -> stop rather than churn
        prev = nf_after
    rep.reseed_folds_after = int(pixel_fold_mask(constraint, phi, threshold).sum())


def _fit_tile_nd(extents, target, lo_frac=0.75, hi_frac=1.5):
    """:func:`_fit_tile` over a region's per-axis extents (any rank): the largest tile
    no bigger than ``target`` that covers the LONGEST extent with an integer number of
    near-equal tiles, clamped to ``[lo_frac, hi_frac] * target``."""
    longest = max(int(v) for v in extents)
    n = max(1, -(-longest // target))  # tiles along the longest side
    tile = -(-longest // n)
    return int(min(max(tile, math.ceil(lo_frac * target)), math.ceil(hi_frac * target)))


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
    return _fit_tile_nd((h, w), target, lo_frac, hi_frac)


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
    on_sweep=None,
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
    just outside the giant violated -> a damage fold (observed on B0039 z=0).

    Any rank: the giant box, the inset region, the tiles and the RAS cores are
    per-axis ``(lo, hi)`` pairs."""
    opts = _InnerOpts() if opts is None else opts
    tile, max_sweeps = opts.giant_tile, opts.giant_max_sweeps
    shape = phi.shape[1:]
    ndim = len(shape)
    extents = [giant_box[2 * a + 1] - giant_box[2 * a] for a in range(ndim)]
    if opts.giant_tile_fit:
        tile = _fit_tile_nd(extents, tile)
    inset = []
    for a, n in enumerate(shape):  # inset interior faces by the ring; keep image borders
        lo, hi = giant_box[2 * a], giant_box[2 * a + 1]
        inset += [lo + (ring if lo > 0 else 0), hi - (ring if hi < n else 0)]
    inset = tuple(inset)
    overlap = 2 * ring + 2  # free regions must overlap so seams are some tile's interior
    # floor: a tile near the overlap must not degenerate to a 1-voxel step (3D tiles are small)
    step = max(tile - overlap, tile // 2, 1)
    tiles = _giant_tiles(inset, tile, step, shape, ring)
    if (
        not tiles
    ):  # an over-cap region thinner than 2 * ring on some interior axis has no inset to tile
        log_warning(f"windowed_correct: giant region {giant_box} has no tileable inset; skipped")
        return -1

    def _nonempty(b):
        return all(b[2 * a + 1] > b[2 * a] for a in range(ndim))

    prev = None
    ras = int(getattr(opts, 'giant_workers', 0) or 0)
    cores = _ras_cores(tiles, step, inset) if ras > 1 else None
    for _sweep in range(max_sweeps):
        if ras > 1:
            # Restricted additive Schwarz: every tile solves from the SAME
            # iterate concurrently; each pastes back only its disjoint
            # step-grid core, so writes cannot conflict. Trades the
            # multiplicative sweep's within-sweep propagation for
            # parallelism; the sweep loop's progress check is unchanged.
            from dvfopt.core._pool import pool_map

            if expired is not None and expired():
                return prev if prev is not None else -1
            # Each task carries ONLY its tile's ring+1-padded patch (~tile^ndim floats,
            # never the field): `build_subproblem` pads the local box by `ring` and clips
            # to the patch, which clips exactly where the true image border clipped the +1
            # — so the enforced rows and the border flags are those of the whole-field
            # solve (asserted byte-for-byte in tests/test_windowed_phase4.py). The whole
            # arg list is cut from `phi` BEFORE the first paste-back below, so every tile
            # of a sweep sees the same Jacobi iterate with no snapshot copy at all.
            args = []
            for tb, core in zip(tiles, cores):
                if not (_nonempty(tb) and _nonempty(core)):
                    continue
                pb = _pad_box(tb, shape, ring + 1)
                off = tuple(pb[2 * a] for a in range(ndim))
                args.append(
                    (
                        phi[(slice(None), *_box_slices(pb))].copy(),
                        off,
                        constraint,
                        tuple(tb[i] - off[i // 2] for i in range(len(tb))),
                        tuple(core[i] - off[i // 2] for i in range(len(core))),
                        threshold,
                        objective,
                        maxiter,
                        ring,
                        margin_delta,
                        inner,
                        opts,
                    )
                )
            for core, vals, sub_rep in pool_map(_ras_tile_task, args, ras):
                phi[(slice(None), *_box_slices(core))] = vals
                rep.windows.extend(sub_rep.windows)
                rep.backend_fallbacks += sub_rep.backend_fallbacks
                rep.patience_fallbacks += sub_rep.patience_fallbacks
            nf = _folds_in_box(constraint, phi, giant_box, threshold)
            if on_sweep is not None:
                on_sweep(_sweep)
            if nf == 0 or (prev is not None and nf >= prev):
                return nf
            prev = nf
            continue
        for tb in tiles:
            if expired is not None and expired():
                # ``time_budget_s`` is checked between tiles as between windows —
                # a giant region is many window solves, not one (measured: a 40 s
                # budget ran 189 s on raw B0039 z16 before this check existed).
                return prev if prev is not None else -1
            if _nonempty(tb):
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
        nf = _folds_in_box(constraint, phi, giant_box, threshold)
        if on_sweep is not None:
            on_sweep(_sweep)
        if nf == 0 or (prev is not None and nf >= prev):
            return nf  # cleared, or no further progress (geometric floor)
        prev = nf
    return prev if prev is not None else 0


def _giant_tiles(inset, tile, step, shape, ring):
    """The tile boxes covering an inset region (any rank): starts every ``step`` per
    axis, each tile ``tile`` long and clipped to the inset; ``itertools.product`` iterates
    axis 0 outermost — the 2D ``for ty ... for tx`` order. A trailing start whose
    ring-padded patch would be thinner than the constraint's 3-voxel minimum on that
    axis (a 1-voxel remainder strip on an image border, where the pad is clipped) is
    dropped: the strip is already inside the previous tile (``tile - step >= 2``), and
    a window that thin is refused by ``validate_dvf`` (hit on the 17³ B0039
    sub-volume: inset 17, fitted tile 12, step 8 -> starts 0, 8, 16)."""
    ndim = len(shape)
    axes = []
    for a in range(ndim):
        lo, hi = inset[2 * a], inset[2 * a + 1]
        starts = list(range(lo, hi, step))
        while len(starts) > 1:
            s = starts[-1]
            if min(shape[a], min(s + tile, hi) + ring) - max(0, s - ring) >= 3:
                break
            starts.pop()
        axes.append(starts)
    return [
        tuple(v for a, t in enumerate(starts) for v in (t, min(t + tile, inset[2 * a + 1])))
        for starts in itertools.product(*axes)
    ]


def _ras_cores(tiles, step, inset):
    """Disjoint step-grid cores of the tiles (any rank): a tile starting at ``t`` on an
    axis owns ``[t, next start)`` there, the last start on each axis owning up to the
    inset edge — a partition of the inset region (tiles overlap, cores do not). With a
    full ``step`` grid of starts this is ``[t, min(t + step, inset_hi))``; when
    :func:`_giant_tiles` dropped a trailing strip, the previous tile's core absorbs it."""
    ndim = len(inset) // 2
    nxt = []
    for a in range(ndim):
        starts = sorted({tb[2 * a] for tb in tiles})
        nxt.append(
            {
                t: (starts[i + 1] if i + 1 < len(starts) else inset[2 * a + 1])
                for i, t in enumerate(starts)
            }
        )
    return [tuple(v for a in range(ndim) for v in (tb[2 * a], nxt[a][tb[2 * a]])) for tb in tiles]


def _ras_tile_task(args):
    """Pool worker: solve ONE giant tile on its ring+1-padded patch (never the whole
    field) and return the tile's core values, boxes translated back to global.

    ``args`` carries the patch, its global ``offset``, and the tile / core boxes in
    PATCH-LOCAL coordinates (picklable; see ``giant_workers``). The patch is padded
    one grid point beyond the window ring, so the sub-problem's own ring padding
    clips exactly where the true image border clipped it — the enforced rows, the
    border flags and hence the result are those of a whole-field solve."""
    from dvfopt.core._pool import pin_worker_threads

    pin_worker_threads()
    patch, off, constraint, tb, core = args[:5]
    threshold, objective, maxiter, ring, margin_delta, inner, opts = args[5:]
    phi = np.array(patch, dtype=np.float64, copy=True)
    rep = SliceReport()
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
    # patch-local boxes -> global. `fy0` / `fx0` are the FREE box's last two axes
    # (`_solve_window`: `box[-4]` / `box[-2]`), populated in every dimension, so both
    # always shift; `ph` / `pw` are patch extents and are translation-invariant.
    nd = len(off)
    for w in rep.windows:
        if w.patch_box:
            w.patch_box = tuple(w.patch_box[i] + off[i // 2] for i in range(len(w.patch_box)))
        w.fy0 += off[nd - 2]
        w.fx0 += off[nd - 1]
    core_global = tuple(core[i] + off[i // 2] for i in range(len(core)))
    return core_global, phi[(slice(None), *_box_slices(core))].copy(), rep


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
    shape = phi.shape[1:]
    opts = _InnerOpts() if opts is None else opts
    sub = build_subproblem(
        constraint,
        phi,
        box,
        threshold,
        objective,
        margin_delta,
        orientation_delta=opts.orientation_delta,
        orientation_rows=opts.orientation_rows,
    )
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
            feas_tol=0.5 * margin_delta,
            ftol=opts.ftol,
        )
        no_tr = False
        if not ok and opts.no_tr_fallback and opts.ladder and inner in _ISQP_LABELS:
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
                feas_tol=0.5 * margin_delta,
                ftol=opts.ftol,
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
    if (
        not ok
        and allow_grow
        and opts.ladder
        and opts.qp_backend != "osqp"
        and inner in _ISQP_LABELS
    ):
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
        and opts.ladder
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
            feas_tol=0.5 * margin_delta,
            ftol=opts.ftol,
        )
        nit += nit2
        if ok2 or sub.cons(x2).min() > sub.cons(x).min():  # keep the better; never worse
            x, ok = x2, ok2
    dt = time.perf_counter() - t
    patch_out = np.asarray(sub.constraint.unflatten(x))
    psl = (slice(None), *_box_slices(sub.patch_box))
    # paste back ONLY free pixels (frozen ring is unchanged and may be shared)
    fm = sub.free_mask
    dst = phi[psl]
    dst[:, fm] = patch_out[:, fm]

    if ok and opts.polish and allow_grow:
        # Per-window anchored polish (the reanchor stage's verify-and-revert, run
        # immediately while the window is warm): re-solve the SAME box on the
        # current field against the distance to the window's PRE-SOLVE patch
        # (phase 1's ``sub.flat0``), so a cheap pure-feasibility solve recovers
        # the in-solve anchor's fidelity at a few anchored iterations instead of
        # paying anchored-QP prices for the whole feasibility fight. Rows are the
        # engine's own (orientation rows included); a polish that ends any row
        # below ``-margin_delta + _REANCHOR_TOL`` or does not reduce the distance
        # is reverted, so it can never cost feasibility or fidelity.
        from dvfopt.objectives import L1Objective, L2Objective

        rep.polish_windows += 1
        psub = build_subproblem(
            constraint,
            phi,
            box,
            threshold,
            None,
            margin_delta,
            orientation_delta=opts.orientation_delta,
            orientation_rows=opts.orientation_rows,
        )
        if psub.free_idx.size and psub.n_enforced:
            obj_ref = L1Objective() if opts.polish == 'l1' else L2Objective()
            pobj, pgrad, phess = _objective_fns(sub.flat0, obj_ref)
            psub = replace(psub, obj=pobj, obj_grad=pgrad, hess_diag=phess)
            px_, pnit, _pok = solve_window_inner(
                psub,
                inner,
                opts.polish_maxiter,
                osqp_max_iter=opts.qp_max_iter,
                qp_backend=opts.qp_backend,
                ip_cold=opts.ip_cold,
                ip_after_admm_iters=opts.ip_after_admm_iters,
                tr_delta=opts.tr_delta,
                tr_max=opts.tr_max,
                step_rule=opts.step_rule,
                ftol=opts.ftol,
            )
            nit += pnit
            if psub.cons(px_).min() >= -margin_delta + _REANCHOR_TOL and pobj(px_) < pobj(
                psub.flat0
            ):
                ppatch = np.asarray(psub.constraint.unflatten(px_))
                pdst = phi[psl]
                pdst[:, psub.free_mask] = ppatch[:, psub.free_mask]
                rep.polish_accepted += 1

    jpatch = min_field(constraint, phi[psl])
    rec = WindowRec(
        fy0=box[-4],
        fx0=box[-2],
        ph=sub.patch_box[-3] - sub.patch_box[-4],
        pw=sub.patch_box[-1] - sub.patch_box[-2],
        patch_box=tuple(int(v) for v in sub.patch_box),
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
    if allow_grow and opts.ladder and not ok and _grow < 2:
        grown = _pad_box(box, shape, 4)
        if grown != tuple(box):
            return _solve_window(
                phi,
                constraint,
                grown,
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
