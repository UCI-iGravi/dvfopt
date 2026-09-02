"""Axial edge-monotonicity rows for the marching engines (the 2D campaign's
'edges' prevention ported to the 2.5D paths).

Each deformed grid edge keeps a positive projection on its own axis:

    x-gaps:  1 + dx[s, i, j+1] - dx[s, i, j]  >=  delta
    y-gaps:  1 + dy[s, i+1, j] - dy[s, i, j]  >=  delta

(the z-family is identically 1 under the 2.5D precondition ``dz == 0`` and is
omitted). The rows are LINEAR in the ``[dx | dy]`` flat layout, so the elastic
SLP's linearization is exact for them — the same argument as the windowed
engine's orientation rows (they exclude the rotated orientation branch instead
of repairing it; measured to be what removed the 2D residual plateaus).

The block is appended to the LP only; the exact-violation acceptance oracle is
deliberately left on the fold gauges alone (the rows are pressure, the folds
are the certificate) — the mirror of the mop's acceptance-only 2-tri term.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy import sparse


@functools.lru_cache(maxsize=32)  # bounded: a sweep sees many distinct crop shapes
def axial_mono_rows(D: int, H: int, W: int):
    """Sparse ``A`` with ``1 + A @ [dx_flat | dy_flat] >= delta`` the axial
    monotonicity rows of a ``(D, H, W)`` grid (``D = 1`` for a single plane).
    Cached per shape (bounded LRU); the caller slices columns to its free set
    and filters rows to those touching a free column.
    """
    n = D * H * W

    def _idx(s, i, j):
        return (s * H + i) * W + j

    rows, cols, vals = [], [], []
    r = 0
    for s in range(D):
        for i in range(H):
            for j in range(W - 1):  # x-gap on dx
                rows.extend((r, r))
                cols.extend((_idx(s, i, j + 1), _idx(s, i, j)))
                vals.extend((1.0, -1.0))
                r += 1
        for i in range(H - 1):
            for j in range(W):  # y-gap on dy
                rows.extend((r, r))
                cols.extend((n + _idx(s, i + 1, j), n + _idx(s, i, j)))
                vals.extend((1.0, -1.0))
                r += 1
    a = sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(r, 2 * n)
    )
    return a


def mono_block(a_full, dxdy_flat, free_cols, delta, active_window):
    """The ``(J, T, thr)`` elastic-LP block for :func:`axial_mono_rows`.

    ``dxdy_flat`` is the CURRENT ``[dx | dy]`` flat vector (frozen values
    included — they enter ``T`` as constants, which is exact since the rows are
    linear); ``free_cols`` indexes that layout. Rows that touch no free column
    are dropped, then the caller-side active-window prefilter is applied.
    Returns ``None`` when nothing is active.
    """
    j_free = a_full.tocsc()[:, free_cols].tocsr()
    touch = np.diff(j_free.indptr) > 0
    if not touch.any():
        return None
    t_all = 1.0 + a_full @ np.asarray(dxdy_flat, dtype=np.float64)
    keep = np.where(touch & (t_all < delta + active_window))[0]
    if keep.size == 0:
        return None
    return j_free[keep], t_all[keep], float(delta)
