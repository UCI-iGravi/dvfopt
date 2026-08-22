import numpy as np
import scipy.sparse as sp

from dvfopt.core.primitives.tri import tri_areas_flat, tri_grad_T_v
from research.strict_feasibility_2d.algorithms.orientation_fix import n_triangles
from research.strict_feasibility_2d.algorithms.tri_linearize import (
    build_sparse_jacobian_T,
    linearize_T_2tri,
)


def _flat_zeros(H, W):
    return np.zeros(2 * H * W, dtype=np.float64)


def test_T_values_match_tri_areas_flat_at_zero():
    H, W = 7, 7
    phi_flat = _flat_zeros(H, W)
    T_vals, _ = linearize_T_2tri(phi_flat, H, W)
    assert T_vals.shape == (n_triangles(H, W),)
    np.testing.assert_allclose(T_vals, tri_areas_flat(phi_flat, H, W))


def test_jacobian_shape_and_sparsity():
    H, W = 7, 7
    K = n_triangles(H, W)
    J = build_sparse_jacobian_T(_flat_zeros(H, W), H, W)
    assert isinstance(J, sp.spmatrix)
    assert J.shape == (K, 2 * H * W)
    # 6 nonzeros per row.
    assert J.nnz == 6 * K


def test_jacobian_matches_adjoint_via_random_probes():
    """J @ e_col matches the adjoint-implied column, for a random sample."""
    rng = np.random.default_rng(0)
    H, W = 10, 10
    phi_flat = rng.uniform(-0.3, 0.3, size=2 * H * W)
    J = build_sparse_jacobian_T(phi_flat, H, W).tocsr()
    K = n_triangles(H, W)
    # For row k: J[k, :] == tri_grad_T_v(phi, H, W, e_k)
    for k in rng.integers(0, K, size=20):
        e_k = np.zeros(K)
        e_k[k] = 1.0
        row_expected = tri_grad_T_v(phi_flat, H, W, e_k)
        row_actual = J[k].toarray().ravel()
        np.testing.assert_allclose(row_actual, row_expected, atol=1e-12)


def test_linearization_first_order_accuracy():
    """T(phi + dphi) - T(phi) approx J @ dphi to O(||dphi||^2)."""
    rng = np.random.default_rng(1)
    H, W = 10, 10
    phi_flat = rng.uniform(-0.2, 0.2, size=2 * H * W)
    T0, J = linearize_T_2tri(phi_flat, H, W)
    errs_lin = []
    errs_const = []
    for _ in range(10):
        dphi = rng.normal(scale=1e-4, size=2 * H * W)
        T1 = tri_areas_flat(phi_flat + dphi, H, W)
        lin = T0 + J @ dphi
        errs_lin.append(np.max(np.abs(T1 - lin)))
        errs_const.append(np.max(np.abs(T1 - T0)))
    # Linearisation error must be at least 100x smaller than constant
    # prediction for these displacements (quadratic vs linear in dphi).
    assert np.mean(errs_lin) < np.mean(errs_const) / 100.0
