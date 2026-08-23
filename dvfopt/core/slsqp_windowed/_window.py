"""Per-window SLSQP runners shared by the 2D solver.

* :func:`_window_minimize` — routes a subproblem through the traced SLSQP
  driver or scipy, shared by every window solve (2D and, via import, 3D).
* :func:`_full_grid_step` — fallback that optimises the entire H×W grid.
* :func:`_optimize_single_window` — SLSQP on one sub-window.
* :func:`_apply_result` — write a sub-window result back into ``phi``.

Originally bundled in ``dvfopt/core/solver.py`` — kept re-exported there
for backward compatibility.
"""

import time

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from dvfopt._defaults import _log, _unpack_size
from dvfopt.core.primitives.slsqp import ineq_dict, minimize_slsqp_traced
from dvfopt.core.slsqp_windowed._objective import objective_euc
from dvfopt.core.slsqp_windowed.constraints import _build_constraints
from dvfopt.jacobian.monotonicity import injectivity_constraint
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d
from dvfopt.jacobian.shoelace import _all_triangle_areas_2d, _shoelace_areas_2d


def _window_minimize(obj, x0, constraints, maxiter, method_name, disp=False):
    """Route a window subproblem: traced C-SLSQP when possible, scipy otherwise.

    The traced driver (:func:`minimize_slsqp_traced`) only understands
    old-style inequality dicts shaped ``fun(x) - lb >= 0`` with a scalar
    ``lb`` and analytic ``jac`` — it cannot represent a ``LinearConstraint``
    (the frozen-edge/no-damage rows most sub-windows carry), a per-row
    ``lb`` (the max-window halo constraint in
    ``_build_constraints_3d_maxwindow``), a finite ``ub``/equality
    constraint, or a jac-less constraint (``_full_grid_step``'s fallback
    constraints have none). Any of those, or a non-``SLSQP``
    ``method_name`` (the public knob), fall back to
    ``scipy.optimize.minimize`` unchanged. On the qualifying path the
    result is numerically identical to scipy's own SLSQP (same C core).
    """
    cons = []
    if method_name == "SLSQP":
        for c in constraints:
            if isinstance(c, NonlinearConstraint) and callable(c.jac):
                lb = np.atleast_1d(c.lb).astype(np.float64)
                ub_unbounded = np.all(np.atleast_1d(c.ub) == np.inf)
                if not ub_unbounded or not np.all(lb == lb[0]):
                    cons = None
                    break
                cons.append(ineq_dict(c.fun, c.jac, lb=float(lb[0])))
            elif (
                isinstance(c, dict)
                and callable(c.get("jac"))
                and c.get("type", "").lower() == "ineq"
            ):
                cons.append(c)
            else:
                cons = None
                break
        if cons is not None:
            # ponytail: per-window traces omitted; add if the GUI grows a
            # per-window inspector.
            return minimize_slsqp_traced(
                lambda z: obj(z)[0],
                x0,
                jac=lambda z: obj(z)[1],
                constraints=cons,
                maxiter=maxiter,
            )
    return minimize(
        obj,
        x0,
        jac=True,
        constraints=constraints,
        options={"maxiter": maxiter, "disp": disp},
        method=method_name,
    )


def _full_grid_step(
    phi,
    phi_init,
    H,
    W,
    threshold,
    max_minimize_iter,
    method_name,
    verbose,
    enforce_shoelace,
    enforce_injectivity,
    injectivity_threshold=None,
    enforce_triangles=False,
):
    """Optimize the entire H×W grid at once.

    Used as a fallback when the square sub-window (capped at
    ``min(H, W)``) cannot cover the full grid.  Constraints are applied
    to **all** pixels (including boundary), matching the behaviour of
    windowed optimisations whose windows touch the grid edge.
    """
    pixels = H * W
    inj_lb = threshold if injectivity_threshold is None else injectivity_threshold
    phi_flat = np.concatenate([phi[1].flatten(), phi[0].flatten()])
    phi_init_flat = np.concatenate([phi_init[1].flatten(), phi_init[0].flatten()])

    def jac_con(phi_xy):
        dx = phi_xy[:pixels].reshape(H, W)
        dy = phi_xy[pixels:].reshape(H, W)
        return _numpy_jdet_2d(dy, dx).flatten()

    constraints = [NonlinearConstraint(jac_con, threshold, np.inf)]

    if enforce_shoelace:

        def shoe_con(phi_xy):
            dx = phi_xy[:pixels].reshape(H, W)
            dy = phi_xy[pixels:].reshape(H, W)
            return _shoelace_areas_2d(dy, dx).flatten()

        constraints.append(NonlinearConstraint(shoe_con, threshold, np.inf))

    if enforce_triangles:

        def tri_con(phi_xy):
            dx = phi_xy[:pixels].reshape(H, W)
            dy = phi_xy[pixels:].reshape(H, W)
            A = _all_triangle_areas_2d(dy, dx)
            return A.reshape(A.shape[0], -1).ravel()

        constraints.append(NonlinearConstraint(tri_con, threshold, np.inf))

    if enforce_injectivity:
        constraints.append(
            NonlinearConstraint(
                lambda phi_xy: injectivity_constraint(phi_xy, (H, W), exclude_boundaries=False),
                inj_lb,
                np.inf,
            )
        )

    _log(verbose, 1, f"  [full-grid] Optimizing entire {H}x{W} grid ({2 * pixels} variables)")

    result = _window_minimize(
        lambda phi1: objective_euc(phi1, phi_init_flat),
        phi_flat,
        constraints,
        max_minimize_iter,
        method_name,
        disp=verbose >= 2,
    )

    phi[1] = result.x[:pixels].reshape(H, W)
    phi[0] = result.x[pixels:].reshape(H, W)


def _optimize_single_window(
    phi_sub_flat,
    phi_init_sub_flat,
    submatrix_size,
    is_at_edge,
    window_reached_max,
    threshold,
    max_minimize_iter,
    method_name,
    enforce_shoelace=False,
    enforce_injectivity=False,
    injectivity_threshold=None,
    enforce_triangles=False,
):
    """Run SLSQP on one sub-window.  Returns ``(result_x, elapsed, success)``."""
    constraints = _build_constraints(
        phi_sub_flat,
        submatrix_size,
        is_at_edge,
        window_reached_max,
        threshold,
        enforce_shoelace=enforce_shoelace,
        enforce_injectivity=enforce_injectivity,
        injectivity_threshold=injectivity_threshold,
        enforce_triangles=enforce_triangles,
    )

    t0 = time.time()
    result = _window_minimize(
        lambda phi1: objective_euc(phi1, phi_init_sub_flat),
        phi_sub_flat,
        constraints,
        max_minimize_iter,
        method_name,
        disp=False,
    )
    elapsed = time.time() - t0
    if not np.all(np.isfinite(result.x)):
        return phi_sub_flat, elapsed, False
    return result.x, elapsed, result.success


def _apply_result(phi, result_x, cy, cx, sub_size, write_size=None):
    """Write optimised sub-window back into *phi*.

    Parameters
    ----------
    sub_size : tuple
        Size of the optimised window (i.e., shape of ``result_x``).  When
        padded extraction was used this is ``(sy+2, sx+2)``.
    write_size : tuple or None
        Original unpadded window size ``(sy, sx)``.  When provided, only the
        inner ``write_size`` region of ``result_x`` (stripping the 1-pixel
        padding on each side) is written back.  ``None`` writes the full
        ``result_x`` (no padding).
    """
    opt_sy, opt_sx = _unpack_size(sub_size)
    pixels = opt_sy * opt_sx

    if write_size is not None:
        wr_sy, wr_sx = _unpack_size(write_size)
        hy, hx = wr_sy // 2, wr_sx // 2
        hy_hi, hx_hi = wr_sy - hy, wr_sx - hx
        phi[1, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi] = result_x[:pixels].reshape(
            opt_sy, opt_sx
        )[1:-1, 1:-1]
        phi[0, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi] = result_x[pixels:].reshape(
            opt_sy, opt_sx
        )[1:-1, 1:-1]
    else:
        hy, hx = opt_sy // 2, opt_sx // 2
        hy_hi, hx_hi = opt_sy - hy, opt_sx - hx
        phi[1, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi] = result_x[:pixels].reshape(
            opt_sy, opt_sx
        )
        phi[0, cy - hy : cy + hy_hi, cx - hx : cx + hx_hi] = result_x[pixels:].reshape(
            opt_sy, opt_sx
        )
