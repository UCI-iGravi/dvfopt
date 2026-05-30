"""Neighborhood Mean Vector Filter (NMVF) — heuristic fold correction.

A fast, lossy, **non-optimisation** heuristic for clearing negative
Jacobian determinants in 2D deformation fields. The original method
this package was built around (see README); kept as a Strategy for
comparison and for cases where speed dominates accuracy.

Algorithm
---------

Iterate until ``min(J) > 0`` or ``max_iter`` is reached:

1. Find every pixel ``(y, x)`` with ``J[y, x] < 0`` (a "fold core").
2. For each fold core, look at the 3x3 neighborhood of pixels around
   it (including the core itself and any in-bounds neighbours).
3. For each neighbourhood pixel ``(py, px)``, replace its displacement
   vector with the **mean** of its own 3x3 patch (excluding the centre).
4. After all replacements, recompute J and loop.

Why it works (sometimes)
------------------------

A negative Jdet at ``(y, x)`` means the deformation field reversed the
sense of local rotation there — neighbouring vectors are pulling toward
the centre or past each other. Averaging the surrounding vectors
smooths the field locally; if the fold was caused by a single outlier
(noise, a stray correspondence), smoothing recovers feasibility in
one or two iterations.

Why it doesn't (other times)
----------------------------

NMVF doesn't minimise anything — it only smooths. For dense folds
where the entire neighbourhood is folded, the smoothing pulls vectors
toward the local mean which is itself near-zero (so the whole patch
collapses to identity), wiping out the registration result. The L2
distance from input can grow without bound. Even on cases where it
*does* clear all folds, the L2 / L1 deviation is typically much larger
than what the SLSQP/barrier/wallbreaker methods produce.

Use NMVF when:
* you want a fast first-pass smoother and don't care about minimum
  displacement,
* the input has only sparse, isolated folds,
* you need a feasibility-only result and the heuristic happens to
  converge.

Use the parameterized solvers (``BarrierStrategy``,
``HarmonicALMBarrierStrategy``, ``HarmonicALMRefineRepairStrategy``,
...) for everything else.
"""

from __future__ import annotations

import time

import numpy as np

from dvfopt.jacobian.numpy_jdet import jacobian_det2D


def _average_vector_excluding_center(patch: np.ndarray, center_idx: int) -> np.ndarray:
    """Average the 8 non-centre vectors of a 3x3 patch.

    Parameters
    ----------
    patch : ndarray, shape ``(C, 3, 3)`` where ``C`` is the channel count.
    center_idx : int
        Flat index of the centre in row-major order (i.e. ``0..8``).

    Returns
    -------
    ndarray, shape ``(C,)``.
    """
    patch_flat = patch.reshape(patch.shape[0], -1)
    patch_wo_centre = np.delete(patch_flat, center_idx, axis=1)
    return np.mean(patch_wo_centre, axis=1)


def _get_3x3_vector_patch(deformation: np.ndarray, y: int, x: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract a clipped 3x3 patch around ``(y, x)`` + its avg-excluding-centre.

    The patch slides off the edge if ``(y, x)`` is on the boundary so the
    return is always a full ``(C, 3, 3)`` block.

    Parameters
    ----------
    deformation : ndarray, shape ``(C, 1, H, W)`` or ``(C, H, W)``.
    y, x : int

    Returns
    -------
    patch : ndarray, shape ``(C, 3, 3)``.
    avg_vec : ndarray, shape ``(C,)``.
    """
    if deformation.ndim == 4:
        # Assumes the singleton-D axis is at position 1.
        H, W = deformation.shape[-2:]
        view = deformation[:, 0]
    else:
        _, H, W = deformation.shape
        view = deformation

    x_start = min(max(x - 1, 0), W - 3)
    y_start = min(max(y - 1, 0), H - 3)
    patch = view[:, y_start : y_start + 3, x_start : x_start + 3]
    idx_centre = (y - y_start) * 3 + (x - x_start)
    avg = _average_vector_excluding_center(patch, idx_centre)
    return patch, avg


def nmvf_correct_2d(
    deformation: np.ndarray,
    *,
    max_iter: int = 1000,
    record_history: bool = False,
    verbose: int = 0,
):
    """NMVF heuristic correction on a 2D deformation field.

    Parameters
    ----------
    deformation : ndarray
        Input field. Either ``(3, 1, H, W)`` (legacy layout, channels
        ``[dz=0, dy, dx]`` with dz ignored) or ``(2, H, W)``
        (channels ``[dy, dx]``). Always processed as a single 2D slice.
    max_iter : int
        Maximum smoothing iterations. The original notebook used 1000.
        Reduce for faster failure when the algorithm doesn't converge.
    record_history : bool
        If True, returns ``(phi, info)`` where ``info`` tracks per-iter
        ``num_neg_J``, ``min_J``, ``iter_wall_s`` stats.
    verbose : int
        ``0`` silent, ``1`` per-iter log line.

    Returns
    -------
    phi_corrected : ndarray, same layout as input.
    info : dict, only if ``record_history=True``.

    Notes
    -----
    Does NOT minimise any objective. The corrected field can deviate
    arbitrarily far from the input on dense folds; this method only
    chases feasibility via local smoothing. For minimum-displacement
    feasibility use the parameterized solvers (BarrierStrategy, etc).
    """
    # Coerce to (3, 1, H, W) — the legacy shape — and remember the original
    # layout to restore on return.
    orig_ndim = deformation.ndim
    if deformation.ndim == 3:
        if deformation.shape[0] != 2:
            raise ValueError(
                f'2D NMVF expects (2, H, W) or (3, 1, H, W); got shape {deformation.shape}'
            )
        H, W = deformation.shape[1], deformation.shape[2]
        phi = np.zeros((3, 1, H, W), dtype=np.float64)
        phi[1, 0] = deformation[0]
        phi[2, 0] = deformation[1]
    elif deformation.ndim == 4 and deformation.shape[0] == 3:
        H, W = deformation.shape[-2:]
        phi = deformation.astype(np.float64, copy=True)
    else:
        raise ValueError(
            f'2D NMVF expects (2, H, W) or (3, 1, H, W); got shape {deformation.shape}'
        )

    def _jdet(p):
        # jacobian_det2D returns (1, H, W) from a (2, H, W) input.
        return jacobian_det2D(p[1:, 0])[0]

    J = _jdet(phi)
    num_neg = int((J <= 0).sum())
    min_J = float(J.min())

    # Use ``history`` (not ``iters``) so :func:`_build_solve_info` picks
    # up the per-iteration list via its ``from_legacy_history`` path. The
    # entries below use ``n_neg``/``min_T``/``wall_s`` to match the keys
    # that adapter expects (instead of ``num_neg``/``min_J``/``iter_wall_s``).
    info: dict = {
        'init_neg': num_neg,
        'init_min_J': min_J,
        'history': [],
    }
    t0 = time.time()

    cur_iter = 0
    while cur_iter < max_iter and num_neg > 0:
        cur_iter += 1
        iter_t0 = time.time()
        neg_coords = np.argwhere(J < 0)  # (n_neg, 2) in (y, x)

        # Snapshot before in-place update so each replacement reads from a
        # consistent state. Matches the notebook's deformation_before.
        phi_prev = phi.copy()

        for y, x in neg_coords:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    py, px = y + dy, x + dx
                    if not (0 <= py < H and 0 <= px < W):
                        continue
                    _, avg = _get_3x3_vector_patch(phi_prev, int(py), int(px))
                    phi[:, 0, py, px] = avg

        J = _jdet(phi)
        num_neg = int((J <= 0).sum())
        min_J = float(J.min())
        iter_wall = time.time() - iter_t0
        if verbose >= 1:
            print(
                f'  [nmvf iter {cur_iter:3d}] num_neg={num_neg:4d}  '
                f'min_J={min_J:+.4f}  wall={iter_wall:.3f}s',
                flush=True,
            )
        if record_history:
            info['history'].append(
                dict(
                    phase=f'nmvf_iter_{cur_iter}',
                    nit=cur_iter,
                    n_neg=num_neg,
                    min_T=min_J,
                    wall_s=iter_wall,
                )
            )

    info['final_neg'] = num_neg
    info['final_min_J'] = min_J
    info['converged'] = num_neg == 0
    info['n_iter'] = cur_iter
    info['total_wall_s'] = time.time() - t0

    # Restore original layout.
    if orig_ndim == 3:
        out = np.stack([phi[1, 0], phi[2, 0]])
    else:
        out = phi
    return (out, info) if record_history else out


__all__ = ['nmvf_correct_2d']
