"""Active-set variant: only enforce constraints near violation.

Standard `_build_constraints` enforces the triangle/Jdet inequality on
*every* cell in the sub-window. For folds that touch only a small fraction
of the window, most of those constraints are inactive (cells with positive
area >> threshold). This variant builds a custom constraint that filters
to cells with current value < `active_factor * threshold`.

Implementation: build a NonlinearConstraint that wraps a closure over a
mutable index mask. Refresh the mask once per outer iteration (the inner
SLSQP step uses a fixed mask).
"""

import numpy as np
import scipy.sparse
from scipy.optimize import NonlinearConstraint

from benchmarks.two_triangle._iterative_loop import run_minimal_iterative_2d
from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult
from dvfopt._defaults import _unpack_size
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d
from dvfopt.jacobian.shoelace import _all_triangle_areas_2d


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _make_active_set_builder(active_factor: float):
    """Returns a constraint_builder closure compatible with _iterative_loop."""

    def _builder(
        phi_sub_flat,
        sub_size,
        is_at_edge,
        win_at_max,
        threshold,
        enforce_shoelace=False,
        enforce_injectivity=False,
        enforce_triangles=True,
    ):
        sy, sx = _unpack_size(sub_size)
        pixels = sy * sx

        # Compute current jdet + triangle areas to seed the active mask.
        dx = phi_sub_flat[:pixels].reshape(sy, sx)
        dy = phi_sub_flat[pixels:].reshape(sy, sx)
        _numpy_jdet_2d(dy, dx)
        active_thr = active_factor * threshold

        cur_tri = _all_triangle_areas_2d(dy, dx)  # (4, sy-1, sx-1)

        # Build the Jdet-style constraint covering ALL cells (not pruned --
        # required to keep the QP from drifting into newly-folded regions).
        def jac_constraint(phi1):
            d_x = phi1[:pixels].reshape(sy, sx)
            d_y = phi1[pixels:].reshape(sy, sx)
            j = _numpy_jdet_2d(d_y, d_x)
            return j[1:-1, 1:-1].flatten() if not is_at_edge else j.flatten()

        constraints = [NonlinearConstraint(jac_constraint, threshold, np.inf)]

        # Pruned triangle constraint: keep only triangles with area < active_thr.
        if enforce_triangles:
            tri_mask = (cur_tri < active_thr).flatten()
            n_tri = tri_mask.sum()
            if n_tri > 0:
                idx = np.where(tri_mask)[0]

                def tri_active(phi1):
                    d_x = phi1[:pixels].reshape(sy, sx)
                    d_y = phi1[pixels:].reshape(sy, sx)
                    tri = _all_triangle_areas_2d(d_y, d_x).flatten()
                    return tri[idx]

                constraints.append(NonlinearConstraint(tri_active, threshold, np.inf))

        # Edge-freeze (same as default builder when interior only).
        exclude_bounds = not is_at_edge and not win_at_max
        if exclude_bounds:
            edge_mask = np.zeros((sy, sx), dtype=bool)
            edge_mask[[0, -1], :] = True
            edge_mask[:, [0, -1]] = True
            edge_indices = np.argwhere(edge_mask)
            fixed = []
            y_off = pixels
            for y, x in edge_indices:
                idx = y * sx + x
                fixed.extend([idx, idx + y_off])
            fixed = np.array(fixed)
            from scipy.optimize import LinearConstraint

            A_eq = scipy.sparse.csr_matrix(
                (np.ones(len(fixed)), (np.arange(len(fixed)), fixed)),
                shape=(len(fixed), phi_sub_flat.size),
            )
            constraints.append(LinearConstraint(A_eq, phi_sub_flat[fixed], phi_sub_flat[fixed]))

        return constraints

    return _builder


@register_variant("active_set")
def active_set(
    dvf: np.ndarray,
    *,
    threshold: float = 0.01,
    max_iterations: int = 100,
    active_factor: float = 5.0,
    enforce_triangles: bool = True,
    timeout_s: float = 600.0,
    **_unused,
) -> SolverResult:
    if _is_3d(dvf):
        from benchmarks.two_triangle.variants.baseline_serial import baseline_serial

        r = baseline_serial(
            dvf,
            threshold=threshold,
            max_iterations=max_iterations,
            enforce_triangles=enforce_triangles,
            timeout_s=timeout_s,
        )
        r.meta["variant"] = "active_set"
        r.meta["fallback"] = "baseline_3d"
        return r

    phi_initial = np.stack([dvf[1, 0], dvf[2, 0]])
    builder = _make_active_set_builder(active_factor=active_factor)
    r = run_minimal_iterative_2d(
        phi_initial,
        threshold=threshold,
        max_iterations=max_iterations,
        enforce_triangles=enforce_triangles,
        timeout_s=timeout_s,
        constraint_builder=builder,
        variant_name="active_set",
    )
    r.meta["active_factor"] = active_factor
    return r
