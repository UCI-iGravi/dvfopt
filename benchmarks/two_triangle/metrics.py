"""Metrics computed from phi arrays. Shared between variants and reporting.

Reuses dvfopt primitives (jacobian_det2D/3D, triangle_sign_count_negatives) by
import — never modifies them.
"""

import numpy as np

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.jacobian.triangle_sign import (
    triangle_sign_areas2D,
    triangle_sign_count_negatives,
)


def _is_3d(phi: np.ndarray) -> bool:
    """phi is (3, D, H, W) for 3D, (2, H, W) or (3, 1, H, W) for 2D."""
    return phi.ndim == 4 and phi.shape[0] == 3 and phi.shape[1] > 1


def fold_counts(phi: np.ndarray, *, threshold: float = 0.01) -> dict:
    """Return {fold_count_jdet, fold_count_tri, max_violation} for 2D or 3D phi.

    `max_violation` is min(active_constraint_value - threshold), so negative
    indicates infeasibility under the configured threshold.
    """
    if _is_3d(phi):
        jdet = jacobian_det3D(phi)
        n_jdet = int((jdet < threshold).sum())
        # 3D triangle (= tetrahedron) sign check uses the sitk_jdet path;
        # for 3D we approximate fold_count_tri as fold_count_jdet here.
        # When the strict 6-tet count helper lands in dvfopt this can be
        # swapped in; for now report jdet as the proxy.
        n_tri = n_jdet
        max_viol = float(jdet.min() - threshold)
    else:
        # Convert (3, 1, H, W) to (2, H, W) if needed
        if phi.ndim == 4 and phi.shape[0] == 3 and phi.shape[1] == 1:
            phi2 = np.stack([phi[1, 0], phi[2, 0]])
        else:
            phi2 = phi
        jdet = jacobian_det2D(phi2)
        n_jdet = int((jdet < threshold).sum())
        n_tri = triangle_sign_count_negatives(phi2)
        tri_areas = triangle_sign_areas2D(phi2)
        max_viol = float(min(jdet.min(), tri_areas.min()) - threshold)
    return {
        "fold_count_jdet": n_jdet,
        "fold_count_tri": n_tri,
        "max_violation": max_viol,
    }


def l2_displacement(phi: np.ndarray, phi_initial: np.ndarray) -> float:
    return float(np.sqrt(np.sum((phi - phi_initial) ** 2)))


def smoothness(phi: np.ndarray) -> float:
    """Frobenius norm of the discrete Laplacian of phi."""
    # Five-point Laplacian along each spatial axis, summed.
    if _is_3d(phi):
        # phi shape (3, D, H, W) -> Laplacian over last 3 axes
        lap = np.zeros_like(phi)
        for ax in (1, 2, 3):
            lap += np.gradient(np.gradient(phi, axis=ax), axis=ax)
    else:
        if phi.ndim == 4 and phi.shape[1] == 1:
            phi = np.stack([phi[1, 0], phi[2, 0]])
        lap = np.zeros_like(phi)
        for ax in (1, 2):
            lap += np.gradient(np.gradient(phi, axis=ax), axis=ax)
    return float(np.sqrt(np.sum(lap**2)))
