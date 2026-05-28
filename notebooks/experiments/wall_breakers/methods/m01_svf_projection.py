"""SVF (stationary velocity field) projection via scaling-and-squaring.

Idea: any DVF :math:`\\phi` is approximated by the flow of a velocity field
:math:`v` at time t=1:  :math:`\\phi = \\exp(v)`. Computed by scaling-and-
squaring (the standard SVF/LDDMM-cheap trick):

1.  set :math:`v_0 = \\phi / 2^N` for N large enough that
    :math:`\\|J(v_0)\\|_\\infty < 1` (i.e. exp(v_0) is one Newton step from
    identity and trivially diffeomorphic);
2.  apply :math:`\\phi_{k+1}(x) = \\phi_k(x + \\phi_k(x))` (composition with
    itself), N times. The result is exp(v_0).

The output is a discrete sampling of a smooth diffeomorphism -- if the
warping and the squaring are done carefully (clip out-of-bounds, use
linear interpolation), the result is *guaranteed* to satisfy
``Jdet(phi_out) > 0`` everywhere up to discretisation. The 2-tri metric
is satisfied as soon as the central-diff Jdet is positive *and* the
field is sampled densely enough that the triangulated quadrant agrees
with the continuous one.

Tradeoff: ``phi_out`` is NOT the L2-closest fold-free correction; it is
the closest *diffeomorphism in the exp(v) family*. The L2 distance from
the input scales with the input's deviation from being itself an
exp(v) field. For dense fold cores that deviation is large -- so this
method's value is "guaranteed feasibility" with measured L2 cost.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

NAME = 'svf_squaring'
DESCRIPTION = 'Stationary velocity field via scaling-and-squaring (guaranteed Jdet > 0 in the continuum)'


def _warp_field(phi: np.ndarray, sample_phi: np.ndarray) -> np.ndarray:
    """Resample ``phi`` (2,H,W) at warped locations ``id + sample_phi``.

    Used by the squaring step: phi_next(x) = sample_phi(x) + phi(x + sample_phi(x)).
    Linear interpolation, edge-extended.
    """
    H, W = phi.shape[1], phi.shape[2]
    ys = np.arange(H, dtype=np.float64)[:, None] + sample_phi[0]
    xs = np.arange(W, dtype=np.float64)[None, :] + sample_phi[1]
    coords = np.stack([ys.ravel(), xs.ravel()])
    out0 = map_coordinates(phi[0], coords, order=1, mode='nearest').reshape(H, W)
    out1 = map_coordinates(phi[1], coords, order=1, mode='nearest').reshape(H, W)
    return np.stack([out0, out1])


def _squaring_compose(v: np.ndarray, n_squarings: int) -> np.ndarray:
    """phi_0 = v;  phi_{k+1}(x) = phi_k(x) + phi_k(x + phi_k(x))."""
    phi = v.copy()
    for _ in range(n_squarings):
        warped = _warp_field(phi, phi)
        phi = phi + warped
    return phi


def _pick_n_squarings(phi: np.ndarray, target_velocity_max: float = 0.4) -> int:
    """Pick N so v_0 = phi / 2^N has max corner-to-corner step < target."""
    H, W = phi.shape[1], phi.shape[2]
    max_disp = max(np.max(np.abs(phi[0])), np.max(np.abs(phi[1])), 1e-9)
    n = max(1, int(np.ceil(np.log2(max_disp / target_velocity_max))))
    return min(n, 16)


def _fit_svf_simple(phi_target: np.ndarray, n_squarings: int) -> np.ndarray:
    """Trivial initial guess: v = phi_target / 2^N (the input itself in the
    log domain). For smooth diffeomorphisms this is essentially exact; for
    inputs that are not exp(v)-representable it is the best initial value
    for an outer Gauss-Newton fit -- but the fit is expensive and in our
    experience does not help much over the trivial guess plus more squarings.
    """
    return phi_target / (2 ** n_squarings)


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          n_squarings: int | None = None,
          target_velocity_max: float = 0.3,
          refine_iters: int = 0) -> dict:
    """Project phi_in onto an exp(v) diffeomorphism by scaling-and-squaring.

    refine_iters > 0 runs a fixed-point refinement: after producing phi_out,
    compute the residual ``phi_in - phi_out`` and add a fraction of it back
    to v, re-square. This lowers L2 cost at the price of risking the
    diffeomorphism guarantee on pathological cases (we check before
    accepting).
    """
    H, W = phi_in.shape[1], phi_in.shape[2]
    if n_squarings is None:
        n_squarings = _pick_n_squarings(phi_in, target_velocity_max)

    v = _fit_svf_simple(phi_in, n_squarings)
    phi_out = _squaring_compose(v, n_squarings)

    info = {
        'n_squarings': n_squarings,
        'init_v_max': float(np.max(np.abs(v))),
        'refine_iters_used': 0,
    }

    if refine_iters > 0:
        # Fixed-point: try to push closer to phi_in without losing feasibility.
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
        best = phi_out.copy()
        best_l2 = float(np.linalg.norm((phi_in - best).ravel()))
        for it in range(refine_iters):
            resid = phi_in - phi_out
            v_cand = v + 0.5 * resid / (2 ** n_squarings)
            phi_cand = _squaring_compose(v_cand, n_squarings)
            T1, T2 = _triangle_areas_2d(phi_cand[0], phi_cand[1])
            if min(T1.min(), T2.min()) >= threshold - 1e-4:
                l2 = float(np.linalg.norm((phi_in - phi_cand).ravel()))
                if l2 < best_l2:
                    best = phi_cand
                    best_l2 = l2
                    v = v_cand
                    phi_out = phi_cand
                    info['refine_iters_used'] = it + 1
                    continue
            break
        phi_out = best

    return {'phi_out': phi_out, 'info': info}
