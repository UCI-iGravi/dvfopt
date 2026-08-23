"""Per-constraint locality knowledge for the windowed engine.

The engine needs, per constraint family: the frozen-ring width, the fold
map on the ``(H, W)`` pixel grid, and the enforced-rows + patch-Jacobian
builder for a free mask. No such concept exists on
:class:`~dvfopt.constraints.Constraint` (yet — folding it into the base
contract is the stage-2 refactor once 3D forces the question), so it
lives here as a small :class:`WindowLocality` adapter registered per
constraint *type* in :data:`LOCALITY`.

Frozen-ring width per family: how far in from an interior patch edge a pixel must
be before every constraint it influences is enforceable with the correct
(global-matching) evaluation.

  jdet: 2 — central-diff rows must be 1 in from the edge, and a free pixel's
           influenced rows are its 4 neighbours (finite-difference support).
  2tri: 1 — cell areas are EXACT (no finite-diff), and a free pixel's influenced
           cells are its <=4 corner cells, all in-patch once it is 1 in.
  finite: 1 — forward-diff cell det is EXACT and depends on 3 corners; a free
           pixel's <=3 influenced cells are all in-patch once it is 1 in (like 2tri).

``TriConstraint2DFullCoverage`` is deliberately NOT registered: its 2
extra opposite-diagonal corner rows need their own influenced-row
handling, which is out of scope here (stage 2).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

from dvfopt.constraints import FiniteJdetConstraint2D, JdetConstraint2D, TriConstraint2D
from dvfopt.core.primitives.coloring import colored_jacobian, dense_jacobian, jacobian_coloring
from dvfopt.exceptions import IncompatibleConstraintError
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d


@dataclass(frozen=True)
class WindowLocality:
    """Windowing adapter for one constraint family.

    ``ring`` — frozen-ring width (see the module docstring).

    ``min_field`` — ``(2, H, W)`` field -> ``(H, W)`` per-location
    constraint value (+inf pad on cell grids).

    ``influenced`` — ``(constraint, free_mask, ph, pw, borders) ->
    (enforced_idx, jac_of)``: the constraint rows a free pixel influences
    AND that evaluate correctly, plus the full-patch sparse-Jacobian
    builder ``jac_of(f)`` (rows sliced by the caller).
    """

    ring: int
    min_field: Callable
    influenced: Callable


# ---------------------------------------------------------------------------
# Per-family fold maps
# ---------------------------------------------------------------------------


def _min_field_jdet(phi_dydx):
    return _numpy_jdet_2d(phi_dydx[0], phi_dydx[1])


def _min_field_2tri(phi_dydx):
    H, W = phi_dydx.shape[1:]
    c = TriConstraint2D(shape=(H, W))
    vals = np.asarray(c.values(c.flatten(phi_dydx)))
    m = (H - 1) * (W - 1)
    cellmin = np.minimum(vals[:m], vals[m:]).reshape(H - 1, W - 1)
    out = np.full((H, W), np.inf)
    out[: H - 1, : W - 1] = cellmin
    return out


def _min_field_finite(phi_dydx):
    H, W = phi_dydx.shape[1:]
    c = FiniteJdetConstraint2D(shape=(H, W))
    vals = np.asarray(c.values(c.flatten(phi_dydx))).reshape(H - 1, W - 1)
    out = np.full((H, W), np.inf)
    out[: H - 1, : W - 1] = vals
    return out


# ---------------------------------------------------------------------------
# CPR coloring cache (per constraint type + patch shape)
# ---------------------------------------------------------------------------

_COLORING_CACHE = {}  # (constraint type, ph, pw) -> (pattern, colors, None)


def _pattern_union(c, probes=4, seed=0):
    """Sparsity pattern of ``c``'s Jacobian, as the union of nonzeros over ``probes``
    random points (a single point can zero a structurally-nonzero entry)."""
    rng = np.random.default_rng(seed)
    flat0 = rng.normal(0, 0.5, c.n_variables)
    acc = None
    for _ in range(probes):
        b = np.abs(dense_jacobian(c, flat0 + rng.normal(0, 0.4, flat0.size))) > 0
        acc = b if acc is None else (acc | b)
    return [np.nonzero(acc[r])[0] for r in range(acc.shape[0])]


def _cached_coloring(c, shape):
    """CPR coloring ``(pattern, colors, None)`` for a patch shape, cached (the
    Jacobian sparsity pattern depends only on the shape, and shapes recur across a
    volume). Jdet uses the pixel-grid stride-3 colouring; 2-tri uses a cell-grid
    ``triangle*4 + (i%2)*2 + j%2`` colouring (8 colours) — both give one adjoint
    call per colour instead of one per constraint row, exact for their stencils."""
    key = (type(c), *shape)
    hit = _COLORING_CACHE.get(key)
    if hit is None:
        if isinstance(c, JdetConstraint2D):
            hit = jacobian_coloring(c, np.random.default_rng(0).normal(0, 0.5, c.n_variables))
        else:  # 2tri: cell grid, 2 triangles per cell share a 2x2-corner support
            ph, pw = shape
            ii, jj = np.meshgrid(np.arange(ph - 1), np.arange(pw - 1), indexing="ij")
            cellcol = ((ii % 2) * 2 + (jj % 2)).ravel()
            colors = np.concatenate([cellcol, 4 + cellcol])  # T1: 0-3, T2: 4-7
            hit = (_pattern_union(c), colors, None)
        _COLORING_CACHE[key] = hit
    return hit


# ---------------------------------------------------------------------------
# Per-family enforced rows + patch-Jacobian builders
# ---------------------------------------------------------------------------


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


def _influenced_jdet(c, free_mask, ph, pw, borders):
    cross = ndimage.generate_binary_structure(2, 1)
    valid = _eval_valid_jdet(ph, pw, *borders)
    influenced = ndimage.binary_dilation(free_mask, cross)  # Jdet row depends on 4 neighbours
    enforced_idx = np.nonzero((influenced & valid).ravel())[0]  # row = pixel raveled
    coloring = _cached_coloring(c, (ph, pw))

    def jac_of(f):
        return colored_jacobian(c, f, *coloring).tocsr()

    return enforced_idx, jac_of


def _influenced_2tri(c, free_mask, ph, pw, borders):
    # cell (i,j) is influenced iff any of its 4 corner pixels is free; cell areas
    # are exact so every cell evaluates correctly (no image-border special case).
    fm = free_mask
    cell = fm[:-1, :-1] | fm[1:, :-1] | fm[:-1, 1:] | fm[1:, 1:]  # (ph-1, pw-1)
    cell_flat = np.nonzero(cell.ravel())[0]
    m = (ph - 1) * (pw - 1)
    enforced_idx = np.concatenate([cell_flat, m + cell_flat])  # T1 and T2 rows
    coloring = _cached_coloring(c, (ph, pw))

    def jac_of(f):
        # coloring: 8 adjoint calls, not a full dense (2M x 2N) rebuild per iter
        # (the native jacobian() densifies — ~43% of a 2-tri window's time).
        return colored_jacobian(c, f, *coloring).tocsr()

    return enforced_idx, jac_of


def _influenced_finite(c, free_mask, ph, pw, borders):
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


# ---------------------------------------------------------------------------
# Registry + public dispatchers
# ---------------------------------------------------------------------------

LOCALITY: dict[type, WindowLocality] = {
    JdetConstraint2D: WindowLocality(
        ring=2, min_field=_min_field_jdet, influenced=_influenced_jdet
    ),
    TriConstraint2D: WindowLocality(ring=1, min_field=_min_field_2tri, influenced=_influenced_2tri),
    FiniteJdetConstraint2D: WindowLocality(
        ring=1, min_field=_min_field_finite, influenced=_influenced_finite
    ),
}


def _locality_of(constraint) -> WindowLocality:
    try:
        return LOCALITY[type(constraint)]
    except KeyError:
        raise IncompatibleConstraintError(
            f'{type(constraint).__name__} is not supported by the windowed '
            f'engine; supported constraint types: '
            f'{", ".join(t.__name__ for t in LOCALITY)}'
        ) from None


def min_field(constraint, phi_dydx):
    """Per-location constraint value on the ``(H, W)`` pixel grid (folds are where
    it is ``< threshold``).

    - jdet: the pixel Jacobian determinant.
    - 2tri: each cell ``(i, j)``'s min triangle area, placed at pixel ``(i, j)``;
      the last pixel row/col have no cell and are set to ``+inf``.
    - finite: each cell ``(i, j)``'s forward-diff determinant, placed at pixel
      ``(i, j)``; the last pixel row/col have no cell and are set to ``+inf``.
    """
    return _locality_of(constraint).min_field(phi_dydx)


def pixel_fold_mask(constraint, phi_dydx, threshold):
    """Boolean ``(H, W)`` pixel mask of folds (constraint value < threshold)."""
    return min_field(constraint, phi_dydx) < threshold


__all__ = [
    'LOCALITY',
    'WindowLocality',
    'min_field',
    'pixel_fold_mask',
]
