"""trust_constr backend variant.

Identical to soft_margin in spirit but uses scipy's trust-constr method
in the inner optimizer. trust-constr accepts sparse analytic constraint
Jacobians and scales better than SLSQP on larger windows.

Wires trust-constr by passing method="trust-constr" to the inner minimize
call inside the re-implemented loop. The constraint builder is the
default _build_constraints (no soft-margin, no active-set).
"""

import numpy as np

from benchmarks.two_triangle._iterative_loop import run_minimal_iterative_2d
from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


@register_variant("trust_constr")
def trust_constr(
    dvf: np.ndarray,
    *,
    threshold: float = 0.01,
    max_iterations: int = 100,
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
        r.meta["variant"] = "trust_constr"
        r.meta["fallback"] = "baseline_3d"
        return r

    phi_initial = np.stack([dvf[1, 0], dvf[2, 0]])
    r = run_minimal_iterative_2d(
        phi_initial,
        threshold=threshold,
        max_iterations=max_iterations,
        enforce_triangles=enforce_triangles,
        timeout_s=timeout_s,
        method="trust-constr",
        variant_name="trust_constr",
    )
    return r
