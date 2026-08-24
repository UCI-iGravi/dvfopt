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
  bilinear: 1 — the 2tri cells with both diagonals (4 rows/cell); same locality.
  finite: 1 — forward-diff cell det is EXACT and depends on 3 corners; a free
           pixel's <=3 influenced cells are all in-patch once it is 1 in (like 2tri).

``TriConstraint2DFullCoverage`` is deliberately NOT registered: its 2
extra opposite-diagonal corner rows need their own influenced-row
handling, which is out of scope here (stage 2).
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from scipy import ndimage

from dvfopt.constraints import (
    FiniteJdetConstraint2D,
    JdetConstraint2D,
    TriConstraint2D,
    TriConstraint2DBilinear,
)
from dvfopt.core.primitives.coloring import colored_jacobian, jacobian_coloring
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


def _min_field_cells(cls, phi_dydx):
    """Fold map of a per-cell family ``cls`` (``k`` rows per cell, laid out
    ``[row0.ravel, row1.ravel, ...]``): each cell's min row at its TL pixel; the
    last pixel row/col have no cell (+inf)."""
    H, W = phi_dydx.shape[1:]
    c = cls(shape=(H, W))
    vals = np.asarray(c.values(c.flatten(phi_dydx))).reshape(-1, H - 1, W - 1)
    out = np.full((H, W), np.inf)
    out[: H - 1, : W - 1] = vals.min(0)
    return out


# ---------------------------------------------------------------------------
# CPR coloring cache (per constraint type + patch shape)
# ---------------------------------------------------------------------------

_COLORING_CACHE = {}  # (constraint type, ph, pw) -> (pattern, colors, None)

# Corner (TL, TR, BL, BR) indices touched by each triangle-row block, in the
# row order of ``tri_areas_flat`` / ``tri_areas_flat_bilinear``:
# T1=(TR,BL,BR)  T2=(TL,BL,TR)  U1=(TL,BL,BR)  U2=(TR,TL,BR).
_BLOCK_CORNERS = ((1, 2, 3), (0, 1, 2), (0, 2, 3), (0, 1, 3))


def _cell_pattern(ph, pw, k):
    """Exact Jacobian sparsity of the ``k``-rows-per-cell triangle families by
    index arithmetic (row = 3 corners x 2 channels) — no dense probing, whose
    ``np.eye(m)`` is O(m^2) memory (a cap-sized bilinear mop window would need
    ~19 GB). Same column sets, in the same order, as the probed pattern."""
    HW = ph * pw
    ii, jj = np.meshgrid(np.arange(ph - 1), np.arange(pw - 1), indexing='ij')
    tl = (ii * pw + jj).ravel()
    corners = np.stack([tl, tl + 1, tl + pw, tl + pw + 1], axis=1)  # (m, 4)
    pattern = []
    for b in range(k):
        cols = corners[:, list(_BLOCK_CORNERS[b])]
        pattern.extend(np.sort(np.concatenate([cols, cols + HW], axis=1), axis=1))
    return pattern


def _cached_coloring(c, shape):
    """CPR coloring ``(pattern, colors, None)`` for a patch shape, cached (the
    Jacobian sparsity pattern depends only on the shape, and shapes recur across a
    volume). Jdet uses the pixel-grid stride-3 colouring; the cell families use a
    cell-grid ``row_block*4 + (i%2)*2 + j%2`` colouring (4 colours per row block:
    8 for 2tri, 16 for bilinear) — both give one adjoint call per colour instead of
    one per constraint row, exact for their stencils."""
    key = (type(c), *shape)
    hit = _COLORING_CACHE.get(key)
    if hit is None:
        if isinstance(c, JdetConstraint2D):
            hit = jacobian_coloring(c, np.random.default_rng(0).normal(0, 0.5, c.n_variables))
        else:  # 2tri/bilinear: cell grid, k triangles per cell share a 2x2-corner support
            ph, pw = shape
            ii, jj = np.meshgrid(np.arange(ph - 1), np.arange(pw - 1), indexing='ij')
            cellcol = ((ii % 2) * 2 + (jj % 2)).ravel()
            k = c.n_constraints // cellcol.size
            colors = np.concatenate([4 * b + cellcol for b in range(k)])
            hit = (_cell_pattern(ph, pw, k), colors, None)
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
    k = c.n_constraints // m  # rows per cell: 2 (2tri) or 4 (bilinear)
    assert k * m == c.n_constraints, 'per-cell family with a non-cell row tail'
    enforced_idx = np.concatenate([b * m + cell_flat for b in range(k)])
    coloring = _cached_coloring(c, (ph, pw))

    def jac_of(f):
        # coloring: 4k adjoint calls, not a full dense (kM x 2N) rebuild per iter
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
    TriConstraint2D: WindowLocality(
        ring=1, min_field=partial(_min_field_cells, TriConstraint2D), influenced=_influenced_2tri
    ),
    TriConstraint2DBilinear: WindowLocality(
        ring=1,
        min_field=partial(_min_field_cells, TriConstraint2DBilinear),
        influenced=_influenced_2tri,
    ),
    FiniteJdetConstraint2D: WindowLocality(
        ring=1,
        min_field=partial(_min_field_cells, FiniteJdetConstraint2D),
        influenced=_influenced_finite,
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
    - 2tri / bilinear / finite: each cell ``(i, j)``'s min row (2, 4 or 1 rows per
      cell), placed at pixel ``(i, j)``; the last pixel row/col have no cell and
      are set to ``+inf``.
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
