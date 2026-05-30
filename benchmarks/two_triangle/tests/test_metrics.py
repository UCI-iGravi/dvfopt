import numpy as np

from benchmarks.two_triangle import metrics


def _identity_phi_2d(H=4, W=4):
    return np.zeros((2, H, W), dtype=np.float64)


def _identity_phi_3d(D=3, H=4, W=4):
    return np.zeros((3, D, H, W), dtype=np.float64)


def test_fold_counts_clean_2d():
    phi = _identity_phi_2d()
    out = metrics.fold_counts(phi, threshold=0.01)
    assert out["fold_count_jdet"] == 0
    assert out["fold_count_tri"] == 0
    assert out["max_violation"] > 0  # min Jdet of identity ~ 1, then minus threshold


def test_fold_counts_clean_3d():
    phi = _identity_phi_3d()
    out = metrics.fold_counts(phi, threshold=0.01)
    assert out["fold_count_jdet"] == 0
    assert out["fold_count_tri"] == 0


def test_max_violation_reflects_threshold_feasibility():
    phi = _identity_phi_2d()
    out = metrics.fold_counts(phi, threshold=1.1)
    assert out["max_violation"] < 0


def test_l2_displacement_zero_when_unchanged():
    phi = _identity_phi_2d()
    assert metrics.l2_displacement(phi, phi) == 0.0


def test_l2_displacement_positive_when_changed():
    phi0 = _identity_phi_2d()
    phi1 = phi0.copy()
    phi1[0, 1, 1] = 0.5
    assert metrics.l2_displacement(phi1, phi0) == 0.5


def test_smoothness_zero_for_identity():
    phi = _identity_phi_2d(8, 8)
    assert metrics.smoothness(phi) == 0.0


def test_smoothness_positive_for_bumpy_field():
    phi = _identity_phi_2d(8, 8)
    phi[0, 4, 4] = 1.0
    assert metrics.smoothness(phi) > 0.0
