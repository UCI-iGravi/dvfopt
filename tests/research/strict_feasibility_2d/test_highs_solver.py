import numpy as np
import scipy.sparse as sp

from research.strict_feasibility_2d.algorithms.highs_solver import solve_l1_lp_step


def test_no_constraint_returns_input():
    """With no triangle constraints, solution = phi_in (L1 = 0)."""
    H, W = 5, 5
    phi_in = np.zeros(2 * H * W)
    T_lin = np.array([])
    J = sp.csr_matrix((0, 2 * H * W))
    phi_out, status = solve_l1_lp_step(
        phi_in_flat=phi_in,
        phi_lin_flat=phi_in,
        T_lin=T_lin,
        J_sparse=J,
        threshold=0.01,
    )
    assert status['success']
    np.testing.assert_allclose(phi_out, phi_in, atol=1e-9)


def test_lp_satisfies_linearized_constraint():
    """LP solution satisfies T_lin + J @ (phi - phi_lin) >= threshold."""
    rng = np.random.default_rng(0)
    H, W = 7, 7
    phi_in = rng.normal(scale=0.1, size=2 * H * W)
    phi_lin = phi_in.copy()

    from research.strict_feasibility_2d.algorithms.tri_linearize import linearize_T_2tri
    T_lin, J = linearize_T_2tri(phi_lin, H, W)
    phi_out, status = solve_l1_lp_step(
        phi_in_flat=phi_in,
        phi_lin_flat=phi_lin,
        T_lin=T_lin,
        J_sparse=J,
        threshold=0.01,
    )
    assert status['success']
    T_pred = T_lin + J @ (phi_out - phi_lin)
    assert np.all(T_pred >= 0.01 - 1e-6), f'worst slack = {(T_pred - 0.01).min():.4e}'


def test_trust_region_bounds_l_inf_step():
    """With trust_radius, ||phi_out - phi_lin||_inf <= trust_radius."""
    H, W = 5, 5
    phi_in = np.zeros(2 * H * W)
    phi_lin = phi_in.copy()
    from research.strict_feasibility_2d.algorithms.tri_linearize import linearize_T_2tri
    T_lin, J = linearize_T_2tri(phi_lin, H, W)
    phi_out, status = solve_l1_lp_step(
        phi_in_flat=phi_in,
        phi_lin_flat=phi_lin,
        T_lin=T_lin,
        J_sparse=J,
        threshold=0.01,
        trust_radius=0.1,
    )
    assert status['success']
    assert np.max(np.abs(phi_out - phi_lin)) <= 0.1 + 1e-6
