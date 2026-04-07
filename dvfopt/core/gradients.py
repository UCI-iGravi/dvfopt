"""Analytical gradient (Jacobian matrix) of the 2D Jdet constraint.

The 2D Jacobian determinant at interior pixel (i,j) using central differences:

    J(i,j) = (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx

where ddx_dx = np.gradient(dx, axis=1), etc.

np.gradient stencils (unit spacing):
  - Interior:  central  (f[k+1] - f[k-1]) / 2
  - j=0:       forward  f[1] - f[0]
  - j=n-1:     backward f[-1] - f[-2]

Let a = 1+ddx_dx, b = 1+ddy_dy, c = ddx_dy, d = ddy_dx.
    J = a*b - c*d

    dJ/d(dx) = b * d(ddx_dx)/d(dx) - d * d(ddx_dy)/d(dx)
    dJ/d(dy) = a * d(ddy_dy)/d(dy) - c * d(ddy_dx)/d(dy)
"""

import numpy as np
import scipy.sparse


def _gradient_stencil_axis1(j, sx):
    """Return (indices, coefficients) for np.gradient along axis=1 at column j."""
    if sx == 1:
        return [j], [0.0]
    if j == 0:
        return [0, 1], [-1.0, 1.0]
    if j == sx - 1:
        return [sx - 2, sx - 1], [-1.0, 1.0]
    return [j - 1, j + 1], [-0.5, 0.5]


def _gradient_stencil_axis0(i, sy):
    """Return (row_indices, coefficients) for np.gradient along axis=0 at row i."""
    if sy == 1:
        return [i], [0.0]
    if i == 0:
        return [0, 1], [-1.0, 1.0]
    if i == sy - 1:
        return [sy - 2, sy - 1], [-1.0, 1.0]
    return [i - 1, i + 1], [-0.5, 0.5]


def jdet_constraint_jacobian_2d(phi_flat, submatrix_size, exclude_boundaries=True):
    """Sparse Jacobian matrix of the Jdet constraint w.r.t. phi_flat.

    Parameters
    ----------
    phi_flat : 1-D array
        Packed as ``[dx_flat, dy_flat]``.
    submatrix_size : int or tuple
        ``(sy, sx)`` sub-window size.
    exclude_boundaries : bool
        When True, the constraint covers only interior pixels (1:-1, 1:-1).

    Returns
    -------
    scipy.sparse.csr_matrix, shape ``(n_constraints, len(phi_flat))``
    """
    sy, sx = submatrix_size if isinstance(submatrix_size, tuple) else (submatrix_size, submatrix_size)
    pixels = sy * sx

    dx = phi_flat[:pixels].reshape(sy, sx)
    dy = phi_flat[pixels:].reshape(sy, sx)

    # Compute the 4 gradient components (matches _numpy_jdet_2d exactly)
    ddx_dx = np.gradient(dx, axis=1)
    ddy_dy = np.gradient(dy, axis=0)
    ddx_dy = np.gradient(dx, axis=0)
    ddy_dx = np.gradient(dy, axis=1)

    a = 1 + ddx_dx  # (sy, sx)
    b = 1 + ddy_dy
    c = ddx_dy
    d = ddy_dx

    if exclude_boundaries:
        i_range = range(1, sy - 1)
        j_range_fn = lambda i: range(1, sx - 1)
        n_rows = (sy - 2) * (sx - 2)
    else:
        i_range = range(sy)
        j_range_fn = lambda i: range(sx)
        n_rows = sy * sx

    rows = []
    cols = []
    vals = []

    row_idx = 0
    for i in i_range:
        for j in j_range_fn(i):
            a_ij = a[i, j]
            b_ij = b[i, j]
            c_ij = c[i, j]
            d_ij = d[i, j]

            # --- dJ/d(dx[m,n]) = b * d(ddx_dx)/d(dx) - d * d(ddx_dy)/d(dx) ---

            # ddx_dx: gradient of dx along axis=1 at (i,j)
            js, jc = _gradient_stencil_axis1(j, sx)
            for jj, coeff in zip(js, jc):
                if coeff != 0.0:
                    rows.append(row_idx)
                    cols.append(i * sx + jj)
                    vals.append(b_ij * coeff)

            # ddx_dy: gradient of dx along axis=0 at (i,j)
            # contribution: -d * d(ddx_dy)/d(dx)
            is_, ic = _gradient_stencil_axis0(i, sy)
            for ii, coeff in zip(is_, ic):
                if coeff != 0.0:
                    rows.append(row_idx)
                    cols.append(ii * sx + j)
                    vals.append(-d_ij * coeff)

            # --- dJ/d(dy[m,n]) = a * d(ddy_dy)/d(dy) - c * d(ddy_dx)/d(dy) ---

            # ddy_dy: gradient of dy along axis=0 at (i,j)
            is_, ic = _gradient_stencil_axis0(i, sy)
            for ii, coeff in zip(is_, ic):
                if coeff != 0.0:
                    rows.append(row_idx)
                    cols.append(pixels + ii * sx + j)
                    vals.append(a_ij * coeff)

            # ddy_dx: gradient of dy along axis=1 at (i,j)
            # contribution: -c * d(ddy_dx)/d(dy)
            js, jc = _gradient_stencil_axis1(j, sx)
            for jj, coeff in zip(js, jc):
                if coeff != 0.0:
                    rows.append(row_idx)
                    cols.append(pixels + i * sx + jj)
                    vals.append(-c_ij * coeff)

            row_idx += 1

    n_cols = 2 * pixels
    return scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_rows, n_cols))


def shoelace_constraint_jacobian_2d(phi_flat, submatrix_size, exclude_boundaries=True):
    """Sparse Jacobian of the shoelace quad-area constraint w.r.t. phi_flat.

    Each quad cell (r, c) has area depending on its 4 corner vertices.
    The gradient has 8 nonzeros per row (4 from dx, 4 from dy).
    """
    sy, sx = submatrix_size if isinstance(submatrix_size, tuple) else (submatrix_size, submatrix_size)
    pixels = sy * sx

    dx = phi_flat[:pixels].reshape(sy, sx)
    dy = phi_flat[pixels:].reshape(sy, sx)

    ref_y, ref_x = np.mgrid[:sy, :sx]
    def_x = ref_x + dx
    def_y = ref_y + dy

    if exclude_boundaries:
        r_range = range(1, sy - 2)
        c_range_fn = lambda r: range(1, sx - 2)
        n_rows = (sy - 3) * (sx - 3)
    else:
        r_range = range(sy - 1)
        c_range_fn = lambda r: range(sx - 1)
        n_rows = (sy - 1) * (sx - 1)

    rows = []
    cols = []
    vals = []

    row_idx = 0
    for r in r_range:
        for c in c_range_fn(r):
            # Deformed coordinates of the 4 corners
            x0, y0 = def_x[r, c], def_y[r, c]          # TL
            x1, y1 = def_x[r, c+1], def_y[r, c+1]      # TR
            x2, y2 = def_x[r+1, c+1], def_y[r+1, c+1]  # BR
            x3, y3 = def_x[r+1, c], def_y[r+1, c]      # BL

            # dx variable indices
            dx_tl = r * sx + c
            dx_tr = r * sx + c + 1
            dx_br = (r + 1) * sx + c + 1
            dx_bl = (r + 1) * sx + c

            # ∂Area/∂(dx) at each corner
            rows.extend([row_idx] * 4)
            cols.extend([dx_tl, dx_tr, dx_br, dx_bl])
            vals.extend([0.5 * (y1 - y3), 0.5 * (y2 - y0),
                         0.5 * (y3 - y1), 0.5 * (y0 - y2)])

            # dy variable indices (offset by pixels)
            dy_tl = pixels + dx_tl
            dy_tr = pixels + dx_tr
            dy_br = pixels + dx_br
            dy_bl = pixels + dx_bl

            # ∂Area/∂(dy) at each corner
            rows.extend([row_idx] * 4)
            cols.extend([dy_tl, dy_tr, dy_br, dy_bl])
            vals.extend([0.5 * (x3 - x1), 0.5 * (x0 - x2),
                         0.5 * (x1 - x3), 0.5 * (x2 - x0)])

            row_idx += 1

    n_cols = 2 * pixels
    return scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_rows, n_cols))


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

    # h_mono shape: (sy, sx-1); v_mono shape: (sy-1, sx)
    # d1, d2 shape: (sy-1, sx-1)
    if exclude_boundaries:
        h_i_range = range(1, sy - 1)
        h_j_range = range(1, sx - 2)
        n_h = (sy - 2) * (sx - 3)
        v_i_range = range(1, sy - 2)
        v_j_range = range(1, sx - 1)
        n_v = (sy - 3) * (sx - 2)
        # Diagonal: all cells except the two all-frozen corners.
        # Must match the keep-mask ordering in injectivity_constraint.
        n_diag = (sy - 1) * (sx - 1)
        n_d = max(0, n_diag - 2)
        d_iter = [
            (r, c)
            for r in range(sy - 1)
            for c in range(sx - 1)
            if not ((r == 0 and c == 0) or (r == sy - 2 and c == sx - 2))
        ]
    else:
        h_i_range = range(sy)
        h_j_range = range(sx - 1)
        n_h = sy * (sx - 1)
        v_i_range = range(sy - 1)
        v_j_range = range(sx)
        n_v = (sy - 1) * sx
        n_d = (sy - 1) * (sx - 1)
        d_iter = [(r, c) for r in range(sy - 1) for c in range(sx - 1)]

    n_rows = n_h + n_v + 2 * n_d
    rows = []
    cols = []
    vals = []

    row_idx = 0
    for i in h_i_range:
        for j in h_j_range:
            rows.extend([row_idx, row_idx])
            cols.extend([i * sx + j, i * sx + j + 1])
            vals.extend([-1.0, 1.0])
            row_idx += 1

    for i in v_i_range:
        for j in v_j_range:
            rows.extend([row_idx, row_idx])
            cols.extend([pixels + i * sx + j, pixels + (i + 1) * sx + j])
            vals.extend([-1.0, 1.0])
            row_idx += 1

    # d1[r,c] = 1 + dx[r, c+1] - dx[r+1, c]
    for r, c in d_iter:
        rows.extend([row_idx, row_idx])
        cols.extend([r * sx + (c + 1), (r + 1) * sx + c])
        vals.extend([1.0, -1.0])
        row_idx += 1

    # d2[r,c] = 1 + dy[r+1, c] - dy[r, c+1]
    for r, c in d_iter:
        rows.extend([row_idx, row_idx])
        cols.extend([pixels + (r + 1) * sx + c, pixels + r * sx + (c + 1)])
        vals.extend([1.0, -1.0])
        row_idx += 1

    n_cols = 2 * pixels
    return scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_rows, n_cols))
