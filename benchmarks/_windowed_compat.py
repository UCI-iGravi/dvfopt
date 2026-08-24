"""Family-string compatibility adapter over the promoted windowed engine.

The windowed fold-corrector (``benchmarks/windowed_isqp.py``, PRs #61-64) was
promoted into the library as :mod:`dvfopt.core.windowed`. The engine's API takes
:mod:`dvfopt.constraints` instances and :mod:`dvfopt.objectives` objects; the
benchmark harnesses (``fullslice_bench`` / ``windowed_bench`` /
``comprehensive_bench``) still speak the historical string API
(``family='jdet'``, ``objective='l2'``, ``inner='isqp-osqp'``, per-slice ``z=``).
This module is the single translation layer: :func:`windowed_correct_compat`
maps the strings at the boundary and delegates to
:func:`dvfopt.core.windowed.windowed_correct` (behaviour identical — the L1
smoothing eps keeps the old ``1e-2`` default, and the engine accepts the old
inner labels as aliases anyway).
"""

from dvfopt.constraints import FiniteJdetConstraint2D, JdetConstraint2D, SimplexConstraint2D
from dvfopt.core.windowed import min_field as _engine_min_field
from dvfopt.core.windowed import windowed_correct
from dvfopt.objectives import L1Objective, L2Objective, NoneObjective

# Historical family-string API -> registered constraint types.
FAMILY = {"jdet": JdetConstraint2D, "2tri": SimplexConstraint2D, "finite": FiniteJdetConstraint2D}

# Old inner-solver labels -> the engine's canonical labels (the engine also
# accepts the old names as aliases; mapping keeps the canonical spelling).
INNER = {
    "isqp-osqp": "isqp",
    "scipy-slsqp": "slsqp",
    "scipy-slsqp+trust-constr": "slsqp+trust-constr",
}


def constraint_of(family, shape):
    """A shape-bound constraint instance for a family string."""
    try:
        ctype = FAMILY[family]
    except KeyError:
        raise ValueError(f"unknown family {family!r} (choose from {tuple(FAMILY)})") from None
    return ctype(shape=tuple(shape))


def objective_of(objective, eps=1e-2):
    """'l2' / 'l1' / 'none' string -> Objective (l1 keeps the old eps=1e-2 default)."""
    if objective == "l2":
        return L2Objective()
    if objective == "l1":
        return L1Objective(eps=eps)
    if objective == "none":
        return NoneObjective()
    raise ValueError(f"unknown objective {objective!r}")


def min_field(family, phi_dydx):
    """Family-string adapter over :func:`dvfopt.core.windowed.min_field`."""
    return _engine_min_field(constraint_of(family, phi_dydx.shape[1:]), phi_dydx)


def pixel_fold_mask(family, phi_dydx, threshold):
    """Boolean ``(H, W)`` pixel mask of folds (constraint value < threshold)."""
    return min_field(family, phi_dydx) < threshold


def windowed_correct_compat(
    phi_dydx,
    family="jdet",
    objective="l2",
    inner="isqp-osqp",
    threshold=0.01,
    maxiter=400,
    z=-1,
    eps=1e-2,
    **kw,
):
    """Old ``windowed_isqp.windowed_correct`` call shape -> the library engine.

    Returns ``(phi_out, SliceReport)`` exactly as before (the report fields the
    benches consume — folds_before/after, damage, giant_regions, mop_windows,
    rounds, time_s, ... — are unchanged). ``z`` is accepted and ignored: the
    engine has no per-slice tag (the benches carry ``z`` in their own records).
    Remaining kwargs (``margin``/``max_rounds``/``margin_delta``/
    ``max_window_area``/``mop_margin``/...) pass straight through with the same
    defaults as before.
    """
    del z  # accepted for back-compat; the engine has no per-slice tag
    return windowed_correct(
        phi_dydx,
        INNER.get(inner, inner),
        constraint=constraint_of(family, phi_dydx.shape[1:]),
        objective=objective_of(objective, eps=eps),
        threshold=threshold,
        maxiter=maxiter,
        **kw,
    )
