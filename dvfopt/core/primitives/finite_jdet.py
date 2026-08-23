"""Flat forward-difference Jacobian-determinant form + analytic sparse
jacobian and adjoint (phi pack: [dx, dy]).

Per cell (i,j), with forward differences
    a = dx[i,j+1]-dx[i,j]   b = dx[i+1,j]-dx[i,j]
    c = dy[i,j+1]-dy[i,j]   d = dy[i+1,j]-dy[i,j]
the forward-diff determinant is  J = (1+a)(1+d) - b*c  (the deformed-
parallelogram area from the two forward edges — a local 2-pixel stencil, so
unlike central diff it is NOT blind to high-frequency modes). Cells:
(H-1)x(W-1), padded nowhere — callers pad.
"""

import numpy as np
from scipy import sparse


def _grids(phi_flat, h, w):
    n = h * w
    dx = phi_flat[:n].reshape(h, w)
    dy = phi_flat[n:].reshape(h, w)
    a = dx[:-1, 1:] - dx[:-1, :-1]
    b = dx[1:, :-1] - dx[:-1, :-1]
    c = dy[:-1, 1:] - dy[:-1, :-1]
    d = dy[1:, :-1] - dy[:-1, :-1]
    return dx, dy, a, b, c, d


def finite_jdet_flat(phi_flat, h, w):
    """Forward-diff determinants, length (h-1)*(w-1)."""
    _, _, a, b, c, d = _grids(np.asarray(phi_flat, dtype=np.float64), h, w)
    return ((1 + a) * (1 + d) - b * c).ravel()


def finite_jdet_jacobian(phi_flat, h, w):
    """Analytic sparse CSR (m, 2hw) Jacobian. 6 nonzeros per cell (3 dx, 3 dy)."""
    n = h * w
    _, _, a, b, c, d = _grids(np.asarray(phi_flat, dtype=np.float64), h, w)
    hc, wc = h - 1, w - 1
    ii, jj = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    rows = np.arange(hc * wc)
    a, b, c, d = a.ravel(), b.ravel(), c.ravel(), d.ravel()
    p = ii * w + jj  # flat pixel index of corner (i,j) in an HxW grid
    # dJ/d(dx[i,j]) = -(1+d)+c ; dx[i,j+1] = (1+d) ; dx[i+1,j] = -c
    # dJ/d(dy[i,j]) =  b-(1+a) ; dy[i,j+1] = -b     ; dy[i+1,j] = (1+a)
    r = np.concatenate([rows] * 6)
    cidx = np.concatenate([p, p + 1, p + w, n + p, n + p + 1, n + p + w])
    val = np.concatenate([-(1 + d) + c, (1 + d), -c, b - (1 + a), -b, (1 + a)])
    return sparse.csr_matrix((val, (r, cidx)), shape=(hc * wc, 2 * n))


def finite_jdet_grad_T_v(phi_flat, h, w, v):
    """Adjoint: jacobian(phi_flat).T @ v, length 2hw."""
    return finite_jdet_jacobian(phi_flat, h, w).T @ np.asarray(v)


__all__ = ['finite_jdet_flat', 'finite_jdet_grad_T_v', 'finite_jdet_jacobian']
