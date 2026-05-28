"""Sparse gradient operators matching ``np.gradient``.

These build the sparse matrix ``G`` such that
``G @ f.ravel() == np.gradient(f.reshape(shape), axis=axis).ravel()``,
allowing the per-pixel / per-voxel Python loops in
``gradients.py`` / ``gradients3d.py`` to be replaced with vectorized
sparse linear algebra (orders of magnitude faster for moderate windows).

``np.gradient`` uses central differences interior and forward/backward
differences at the boundaries, both with unit spacing:

* interior ``j``:    ``(f[j+1] - f[j-1]) / 2``     (coefficients ``-0.5, 0.5``)
* ``j == 0``:        ``f[1] - f[0]``               (coefficients ``-1, 1``)
* ``j == n-1``:      ``f[-1] - f[-2]``             (coefficients ``-1, 1``)
* ``n == 1``:        zero operator (no gradient).
"""
from __future__ import annotations

import functools

import numpy as np
import scipy.sparse


@functools.lru_cache(maxsize=32)
def gradient_operator(shape, axis):
    """Return sparse ``(N, N)`` gradient operator along ``axis`` for arrays of ``shape``.

    Cached on ``(shape, axis)`` — the operator depends only on geometry, so
    repeated SLSQP calls with the same window size reuse one matrix.
    """
    shape = tuple(int(s) for s in shape)
    axis = int(axis)
    ndim = len(shape)
    N = int(np.prod(shape))
    n = shape[axis]
    if n == 1:
        return scipy.sparse.csr_matrix((N, N))

    # Stride along ``axis`` in the flat C-order layout.
    # For shape (s0, s1, s2): axis=0 stride=s1*s2, axis=1 stride=s2, axis=2 stride=1.
    stride = 1
    for k in range(axis + 1, ndim):
        stride *= shape[k]

    flat = np.arange(N)
    pa = (flat // stride) % n  # position along ``axis`` for each flat index

    p_int = flat[(pa >= 1) & (pa <= n - 2)]
    p_low = flat[pa == 0]
    p_high = flat[pa == n - 1]

    rows = np.concatenate([
        p_int, p_int,
        p_low, p_low,
        p_high, p_high,
    ])
    cols = np.concatenate([
        p_int - stride, p_int + stride,
        p_low,          p_low + stride,
        p_high - stride, p_high,
    ])
    vals = np.concatenate([
        np.full(p_int.size, -0.5), np.full(p_int.size, 0.5),
        np.full(p_low.size, -1.0), np.full(p_low.size, 1.0),
        np.full(p_high.size, -1.0), np.full(p_high.size, 1.0),
    ])
    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))


def scale_rows(d, sparse_matrix):
    """Return ``diag(d.ravel()) @ sparse_matrix`` — scales each row by its corresponding ``d`` value.

    Wraps the ``scipy.sparse.diags @ M`` idiom and forces a CSR result for
    cheap horizontal stacking downstream.
    """
    diag = scipy.sparse.diags(np.asarray(d).ravel(), format="csr")
    return (diag @ sparse_matrix).tocsr()
