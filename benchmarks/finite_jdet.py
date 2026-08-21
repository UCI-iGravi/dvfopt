"""Finite (forward-difference) Jacobian-determinant constraint — a third fold
metric between central-diff Jdet (blind to the checkerboard) and 2-tri (exact area).

Per cell (i,j), with forward differences
    a = dx[i,j+1]-dx[i,j]   b = dx[i+1,j]-dx[i,j]
    c = dy[i,j+1]-dy[i,j]   d = dy[i+1,j]-dy[i,j]
the forward-diff determinant is  J = (1+a)(1+d) - b*c  (the deformed-parallelogram
area from the two forward edges — a local 2-pixel stencil, so unlike central diff it
is NOT blind to high-frequency modes). Cells: (H-1)x(W-1).

Duck-typed to match dvfopt's Constraint interface (values / adjoint / jacobian /
flatten / unflatten / n_constraints / n_variables / shape) so it drops into the same
solver harness as JdetConstraint2D and TriConstraint2D. Phi pack: [dx, dy] (x-first),
matching JdetConstraint2D.
"""

import numpy as np
from scipy import sparse


class FiniteJdetConstraint2D:
    dim = 2

    def __init__(self, shape):
        self.shape = tuple(shape)

    @property
    def n_variables(self):
        h, w = self.shape
        return 2 * h * w

    @property
    def n_constraints(self):
        h, w = self.shape
        return (h - 1) * (w - 1)

    def flatten(self, phi):  # phi (2,H,W) = [dy, dx]
        phi = np.asarray(phi, dtype=np.float64)
        return np.concatenate([phi[1].ravel(), phi[0].ravel()])  # [dx, dy]

    def unflatten(self, flat):
        h, w = self.shape
        n = h * w
        return np.stack([flat[n:].reshape(h, w), flat[:n].reshape(h, w)])  # [dy, dx]

    def _grids(self, flat):
        h, w = self.shape
        n = h * w
        dx = flat[:n].reshape(h, w)
        dy = flat[n:].reshape(h, w)
        a = dx[:-1, 1:] - dx[:-1, :-1]
        b = dx[1:, :-1] - dx[:-1, :-1]
        c = dy[:-1, 1:] - dy[:-1, :-1]
        d = dy[1:, :-1] - dy[:-1, :-1]
        return dx, dy, a, b, c, d

    def values(self, flat):
        _, _, a, b, c, d = self._grids(np.asarray(flat, dtype=np.float64))
        return ((1 + a) * (1 + d) - b * c).ravel()

    def jacobian(self, flat):
        """Analytic sparse (m, n) Jacobian. 6 nonzeros per cell (3 dx, 3 dy)."""
        h, w = self.shape
        n = h * w
        _, _, a, b, c, d = self._grids(np.asarray(flat, dtype=np.float64))
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
        return sparse.csr_matrix((val, (r, cidx)), shape=(hc * wc, self.n_variables))

    def adjoint(self, flat, v):
        return self.jacobian(flat).T @ np.asarray(v)


def _min_finite_jdet(phi_dydx, shape=None):
    """Convenience: min forward-diff Jdet of a (2,H,W) [dy,dx] field."""
    h, w = phi_dydx.shape[1:]
    c = FiniteJdetConstraint2D((h, w))
    return float(np.asarray(c.values(c.flatten(phi_dydx))).min())


if __name__ == "__main__":  # self-check: analytic jacobian == numerical, values sane
    rng = np.random.default_rng(0)
    for h, w in [(6, 7), (9, 5)]:
        c = FiniteJdetConstraint2D((h, w))
        f = rng.normal(0, 0.4, c.n_variables)
        Ja = c.jacobian(f).toarray()
        # numerical jacobian
        eps = 1e-6
        Jn = np.zeros_like(Ja)
        v0 = np.asarray(c.values(f))
        for k in range(f.size):
            fp = f.copy()
            fp[k] += eps
            Jn[:, k] = (np.asarray(c.values(fp)) - v0) / eps
        err = np.abs(Ja - Jn).max()
        # identity field -> all determinants exactly 1
        ident = np.zeros(c.n_variables)
        v_id = np.asarray(c.values(ident))
        print(f"shape {(h, w)}: max|Jac_analytic - Jac_numeric|={err:.2e} | "
              f"identity min/max det = {v_id.min():.3f}/{v_id.max():.3f}")
        assert err < 1e-5 and np.allclose(v_id, 1.0)
    print("finite-jdet self-check OK")
