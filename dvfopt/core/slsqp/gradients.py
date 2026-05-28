"""Analytical gradient (Jacobian matrix) of the 2D Jdet constraint.

The 2D Jacobian determinant at pixel (i, j) using central differences:

    J(i, j) = (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx

where ``ddx_dx = np.gradient(dx, axis=1)``, etc.

np.gradient stencils (unit spacing):

* interior:  central  ``(f[k+1] - f[k-1]) / 2``
* j == 0:    forward  ``f[1] - f[0]``
* j == n-1:  backward ``f[-1] - f[-2]``

Let ``a = 1 + ddx_dx``, ``b = 1 + ddy_dy``, ``c = ddx_dy``, ``d = ddy_dx``::

    dJ/d(dx) = b * d(ddx_dx)/d(dx) - d * d(ddx_dy)/d(dx)
             = b * G_x.T   - d * G_y.T      (as rows of the constraint Jacobian)
    dJ/d(dy) = a * d(ddy_dy)/d(dy) - c * d(ddy_dx)/d(dy)
             = a * G_y.T   - c * G_x.T

Each ``G_axis`` is the sparse gradient operator for that axis (cached on
geometry), so the full constraint Jacobian is assembled as a couple of
sparse hstacks instead of a per-pixel Python loop.
"""

import numpy as np
import scipy.sparse

from dvfopt.core.slsqp._grad_op import gradient_operator, scale_rows


def _gradient_stencil_axis1(j, sx):
    """Return ``(indices, coefficients)`` for ``np.gradient`` along axis=1 at column ``j``."""
    if sx == 1:
        return [j], [0.0]
    if j == 0:
        return [0, 1], [-1.0, 1.0]
    if j == sx - 1:
        return [sx - 2, sx - 1], [-1.0, 1.0]
    return [j - 1, j + 1], [-0.5, 0.5]


def _gradient_stencil_axis0(i, sy):
    """Return ``(row_indices, coefficients)`` for ``np.gradient`` along axis=0 at row ``i``."""
    if sy == 1:
        return [i], [0.0]
    if i == 0:
        return [0, 1], [-1.0, 1.0]
    if i == sy - 1:
        return [sy - 2, sy - 1], [-1.0, 1.0]
    return [i - 1, i + 1], [-0.5, 0.5]


def _interior_keep_mask(sy, sx, exclude_boundaries):
    """Return a flat boolean mask (length sy*sx) selecting rows for the constraint."""
    if not exclude_boundaries:
        return np.ones(sy * sx, dtype=bool)
    keep = np.zeros((sy, sx), dtype=bool)
    if sy > 2 and sx > 2:
        keep[1:-1, 1:-1] = True
    return keep.ravel()


def jdet_constraint_jacobian_2d(phi_flat, submatrix_size, exclude_boundaries=True):
    """Sparse Jacobian matrix of the Jdet constraint w.r.t. ``phi_flat``.

    Parameters
    ----------
    phi_flat : 1-D array
        Packed as ``[dx_flat, dy_flat]``.
    submatrix_size : int or tuple
        ``(sy, sx)`` sub-window size.
    exclude_boundaries : bool
        When True, the constraint covers only interior pixels ``(1:-1, 1:-1)``.

    Returns
    -------
    scipy.sparse.csr_matrix, shape ``(n_constraints, len(phi_flat))``
    """
    sy, sx = submatrix_size if isinstance(submatrix_size, tuple) else (submatrix_size, submatrix_size)
    pixels = sy * sx
    shape = (sy, sx)

    dx = phi_flat[:pixels].reshape(shape)
    dy = phi_flat[pixels:].reshape(shape)

    # Match _numpy_jdet_2d's central-difference convention.
    a = 1 + np.gradient(dx, axis=1)   # 1 + ddx_dx
    b = 1 + np.gradient(dy, axis=0)   # 1 + ddy_dy
    c = np.gradient(dx, axis=0)       # ddx_dy
    d = np.gradient(dy, axis=1)       # ddy_dx

    G_x = gradient_operator(shape, axis=1)  # along x (axis=1 of (sy, sx))
    G_y = gradient_operator(shape, axis=0)  # along y (axis=0)

    # dx column block: dJ/d(dx) = b * G_x - d * G_y
    M_dx = scale_rows(b, G_x) - scale_rows(d, G_y)
    # dy column block: dJ/d(dy) = a * G_y - c * G_x
    M_dy = scale_rows(a, G_y) - scale_rows(c, G_x)

    J = scipy.sparse.hstack([M_dx, M_dy], format="csr")

    keep = _interior_keep_mask(sy, sx, exclude_boundaries)
    if not keep.all():
        J = J[keep, :]
    return J


def shoelace_constraint_jacobian_2d(phi_flat, submatrix_size, exclude_boundaries=True):
    """Sparse Jacobian of the shoelace quad-area constraint w.r.t. ``phi_flat``.

    Each quad cell ``(r, c)`` has area depending on its 4 corner vertices.
    The gradient has 8 nonzeros per row (4 from dx, 4 from dy).
    """
    sy, sx = submatrix_size if isinstance(submatrix_size, tuple) else (submatrix_size, submatrix_size)
    pixels = sy * sx

    dx = phi_flat[:pixels].reshape(sy, sx)
    dy = phi_flat[pixels:].reshape(sy, sx)

    ref_y, ref_x = np.mgrid[:sy, :sx]
    def_x = ref_x + dx
    def_y = ref_y + dy

    # Cell (r, c) corners: TL=(r,c), TR=(r,c+1), BR=(r+1,c+1), BL=(r+1,c).
    if exclude_boundaries:
        r_lo, r_hi = 1, sy - 2  # cells r in [1, sy-3]
        c_lo, c_hi = 1, sx - 2
    else:
        r_lo, r_hi = 0, sy - 1
        c_lo, c_hi = 0, sx - 1
    n_cells_y = max(0, r_hi - r_lo)
    n_cells_x = max(0, c_hi - c_lo)
    n_rows = n_cells_y * n_cells_x

    if n_rows == 0:
        return scipy.sparse.csr_matrix((0, 2 * pixels))

    rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
    rr = rr.ravel(); cc = cc.ravel()
    row_idx = np.arange(n_rows)

    # Corner positions (deformed) for every cell in the selection.
    x0 = def_x[rr,     cc    ]; y0 = def_y[rr,     cc    ]  # TL
    x1 = def_x[rr,     cc + 1]; y1 = def_y[rr,     cc + 1]  # TR
    x2 = def_x[rr + 1, cc + 1]; y2 = def_y[rr + 1, cc + 1]  # BR
    x3 = def_x[rr + 1, cc    ]; y3 = def_y[rr + 1, cc    ]  # BL

    # Flat dx-column index for each corner.
    dx_tl =     rr      * sx + cc
    dx_tr =     rr      * sx + cc + 1
    dx_br = (rr + 1)    * sx + cc + 1
    dx_bl = (rr + 1)    * sx + cc

    # ∂Area/∂dx values: 0.5 * (y1-y3), 0.5 * (y2-y0), 0.5 * (y3-y1), 0.5 * (y0-y2)
    rows = np.concatenate([row_idx, row_idx, row_idx, row_idx,
                           row_idx, row_idx, row_idx, row_idx])
    cols = np.concatenate([
        dx_tl, dx_tr, dx_br, dx_bl,
        pixels + dx_tl, pixels + dx_tr, pixels + dx_br, pixels + dx_bl,
    ])
    vals = np.concatenate([
        0.5 * (y1 - y3), 0.5 * (y2 - y0), 0.5 * (y3 - y1), 0.5 * (y0 - y2),
        0.5 * (x3 - x1), 0.5 * (x0 - x2), 0.5 * (x1 - x3), 0.5 * (x2 - x0),
    ])
    return scipy.sparse.csr_matrix((vals, (rows, cols)),
                                   shape=(n_rows, 2 * pixels))


def triangle_constraint_jacobian_2d(phi_flat, submatrix_size, exclude_boundaries=True):
    """Sparse Jacobian of the 4-triangle-per-cell constraint.

    For a triangle with vertices A, B, C the signed area is
    ``0.5 * ((x_B - x_A)(y_C - y_A) - (x_C - x_A)(y_B - y_A))`` with 6
    closed-form partials:

        ∂a/∂x_A = 0.5 * (y_B - y_C)     ∂a/∂y_A = 0.5 * (x_C - x_B)
        ∂a/∂x_B = 0.5 * (y_C - y_A)     ∂a/∂y_B = 0.5 * (x_A - x_C)
        ∂a/∂x_C = 0.5 * (y_A - y_B)     ∂a/∂y_C = 0.5 * (x_B - x_A)

    Row layout matches :func:`triangle_constraint`: T1 block, then T2, T3, T4.
    """
    sy, sx = submatrix_size if isinstance(submatrix_size, tuple) else (submatrix_size, submatrix_size)
    pixels = sy * sx

    dx = phi_flat[:pixels].reshape(sy, sx)
    dy = phi_flat[pixels:].reshape(sy, sx)

    ref_y, ref_x = np.mgrid[:sy, :sx]
    X = ref_x + dx
    Y = ref_y + dy

    if exclude_boundaries:
        r_lo, r_hi = 1, sy - 2
        c_lo, c_hi = 1, sx - 2
    else:
        r_lo, r_hi = 0, sy - 1
        c_lo, c_hi = 0, sx - 1
    n_cells_y = max(0, r_hi - r_lo)
    n_cells_x = max(0, c_hi - c_lo)
    n_cells = n_cells_y * n_cells_x

    if n_cells == 0:
        return scipy.sparse.csr_matrix((0, 2 * pixels))

    rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
    rr = rr.ravel(); cc = cc.ravel()
    cell_idx = np.arange(n_cells)

    # Per-cell corner (row, col) tuples.
    A_TL = (rr,     cc    )
    A_TR = (rr,     cc + 1)
    A_BR = (rr + 1, cc + 1)
    A_BL = (rr + 1, cc    )

    # Triangle definitions: (A, B, C) corners and the row offset.
    # T1=(TL,TR,BR), T2=(TL,BR,BL), T3=(TL,TR,BL), T4=(TR,BR,BL).
    triangles = [
        (A_TL, A_TR, A_BR),
        (A_TL, A_BR, A_BL),
        (A_TL, A_TR, A_BL),
        (A_TR, A_BR, A_BL),
    ]

    rows_all = []
    cols_all = []
    vals_all = []
    for tri_no, (A, B, C) in enumerate(triangles):
        row_offset = tri_no * n_cells
        row_idx = cell_idx + row_offset
        xa, ya = X[A], Y[A]
        xb, yb = X[B], Y[B]
        xc, yc = X[C], Y[C]
        lin_A = A[0] * sx + A[1]
        lin_B = B[0] * sx + B[1]
        lin_C = C[0] * sx + C[1]

        # dx partials at (A, B, C)
        rows_all.append(np.concatenate([row_idx, row_idx, row_idx]))
        cols_all.append(np.concatenate([lin_A, lin_B, lin_C]))
        vals_all.append(np.concatenate([
            0.5 * (yb - yc), 0.5 * (yc - ya), 0.5 * (ya - yb),
        ]))
        # dy partials at (A, B, C)
        rows_all.append(np.concatenate([row_idx, row_idx, row_idx]))
        cols_all.append(np.concatenate([
            pixels + lin_A, pixels + lin_B, pixels + lin_C,
        ]))
        vals_all.append(np.concatenate([
            0.5 * (xc - xb), 0.5 * (xa - xc), 0.5 * (xb - xa),
        ]))

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all)
    return scipy.sparse.csr_matrix((vals, (rows, cols)),
                                   shape=(4 * n_cells, 2 * pixels))


def injectivity_constraint_jacobian_2d(phi_flat, submatrix_size, exclude_boundaries=True):
    """Sparse Jacobian of the injectivity (monotonicity) constraint.

    h_mono[i,j] = 1 + dx[i,j+1] - dx[i,j]    →  ∂/∂dx[i,j]=-1,  ∂/∂dx[i,j+1]=+1
    v_mono[i,j] = 1 + dy[i+1,j] - dy[i,j]    →  ∂/∂dy[i,j]=-1,  ∂/∂dy[i+1,j]=+1
    d1[r,c]    = 1 + dx[r,c+1] - dx[r+1,c]   →  ∂/∂dx[r,c+1]=+1, ∂/∂dx[r+1,c]=-1
    d2[r,c]    = 1 + dy[r+1,c] - dy[r,c+1]   →  ∂/∂dy[r+1,c]=+1, ∂/∂dy[r,c+1]=-1

    All rows are constant (2 nonzeros each, independent of phi_flat values).
    """
    sy, sx = submatrix_size if isinstance(submatrix_size, tuple) else (submatrix_size, submatrix_size)
    pixels = sy * sx

    if exclude_boundaries:
        h_ii, h_jj = np.mgrid[1:sy - 1, 1:sx - 2]
        v_ii, v_jj = np.mgrid[1:sy - 2, 1:sx - 1]
        d_iter = np.array([
            (r, c)
            for r in range(sy - 1)
            for c in range(sx - 1)
            if not ((r == 0 and c == 0) or (r == sy - 2 and c == sx - 2))
        ], dtype=int)
    else:
        h_ii, h_jj = np.mgrid[0:sy, 0:sx - 1]
        v_ii, v_jj = np.mgrid[0:sy - 1, 0:sx]
        d_iter = np.array([(r, c) for r in range(sy - 1) for c in range(sx - 1)], dtype=int)

    h_ii = h_ii.ravel(); h_jj = h_jj.ravel()
    v_ii = v_ii.ravel(); v_jj = v_jj.ravel()
    n_h = h_ii.size
    n_v = v_ii.size
    n_d = d_iter.shape[0] if d_iter.size else 0
    n_rows = n_h + n_v + 2 * n_d

    row_h = np.arange(n_h)
    row_v = np.arange(n_h, n_h + n_v)
    row_d1 = np.arange(n_h + n_v, n_h + n_v + n_d)
    row_d2 = np.arange(n_h + n_v + n_d, n_h + n_v + 2 * n_d)

    rows_parts = [row_h, row_h, row_v, row_v]
    cols_parts = [
        h_ii * sx + h_jj,
        h_ii * sx + h_jj + 1,
        pixels + v_ii * sx + v_jj,
        pixels + (v_ii + 1) * sx + v_jj,
    ]
    vals_parts = [
        np.full(n_h, -1.0), np.full(n_h, 1.0),
        np.full(n_v, -1.0), np.full(n_v, 1.0),
    ]

    if n_d > 0:
        dr = d_iter[:, 0]; dc = d_iter[:, 1]
        rows_parts += [row_d1, row_d1, row_d2, row_d2]
        cols_parts += [
            dr * sx + (dc + 1), (dr + 1) * sx + dc,
            pixels + (dr + 1) * sx + dc, pixels + dr * sx + (dc + 1),
        ]
        vals_parts += [
            np.full(n_d, 1.0), np.full(n_d, -1.0),
            np.full(n_d, 1.0), np.full(n_d, -1.0),
        ]

    rows = np.concatenate(rows_parts)
    cols = np.concatenate(cols_parts)
    vals = np.concatenate(vals_parts)
    return scipy.sparse.csr_matrix((vals, (rows, cols)),
                                   shape=(n_rows, 2 * pixels))
