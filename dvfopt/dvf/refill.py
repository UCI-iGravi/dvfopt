"""Harmonic re-fill of a Laplacian-interpolated DVF from a subset of its own pins.

The field is re-solved as the discrete-harmonic interpolant (the stencil of
:func:`dvfopt.laplacian.utils.laplacianA3D`) of its values at the ``dirichlet``
voxels — typically the pins :mod:`dvfopt.dvf.pins` read out of the field, minus
the contradictory ones the pairwise cover dropped. Nothing but the field is read.

Solver: CG warm-started at the input channel. With ``pyamg`` importable the
preconditioner is smoothed-aggregation AMG (one hierarchy shared by both
channels); measured on the full B0039 volume (77M DOFs) 7 CG iterations / 211 s
+ 451 s setup vs 769 iterations / 2368 s for Jacobi-PCG. Without ``pyamg`` it
falls back to Jacobi-PCG. The two channels are solved one after the other on
purpose: two concurrent solves measured SLOWER (the SpMV is memory-bandwidth
bound).
"""

from __future__ import annotations

import time

import numpy as np

from dvfopt._logging import log_info, log_warning, vlog

_warned_no_amg = False


def harmonic_refill(
    phi: np.ndarray, dirichlet: np.ndarray, *, rtol: float = 1e-4, verbose: int = 0
) -> np.ndarray:
    """Re-solve the in-plane channels of ``phi`` as the harmonic interpolant of
    their values at ``dirichlet``.

    Parameters
    ----------
    phi : ndarray ``(3, D, H, W)`` ``[dz, dy, dx]`` or ``(2, H, W)`` ``[dy, dx]``
        The last two channels (in-plane) are re-solved; ``dz`` passes through.
    dirichlet : bool ndarray, the spatial shape of ``phi``
        Voxels whose value is kept and imposed; must be non-empty.
    rtol : float
        Relative CG tolerance.

    Returns
    -------
    ndarray, same shape and dtype as ``phi`` (a new array; ``phi`` is not
    mutated). ``out[..., dirichlet]`` equals ``phi[..., dirichlet]`` exactly.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg

    from dvfopt.laplacian.utils import laplacianA3D, propagate_dirichlet_rhs

    phi = np.asarray(phi)
    dirichlet = np.asarray(dirichlet, dtype=bool)
    if phi.ndim == 3 and phi.shape[0] == 2:
        shape3 = (1, *phi.shape[1:])  # a 2D grid is a one-slice volume for this stencil
    elif phi.ndim == 4 and phi.shape[0] == 3:
        shape3 = phi.shape[1:]
    else:
        raise ValueError(f'phi must be (3, D, H, W) or (2, H, W), got {phi.shape}')
    if dirichlet.shape != phi.shape[1:]:
        raise ValueError(f'dirichlet shape {dirichlet.shape} != spatial shape {phi.shape[1:]}')
    if not dirichlet.any():
        raise ValueError('dirichlet is empty: a harmonic re-fill needs at least one kept pin')

    bidx = np.flatnonzero(dirichlet.ravel())
    t = time.time()
    A = laplacianA3D(shape3, bidx, log_fn=lambda _m: None).tocsr()
    maxiter = 200
    try:
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(
            A, max_coarse=2000, strength=('symmetric', {'theta': 0.0})
        )
        M = ml.aspreconditioner(cycle='V')
        vlog(verbose, 1, f'[refill] matrix + AMG setup {time.time() - t:.1f} s')
    except ImportError:
        global _warned_no_amg
        if not _warned_no_amg:
            _warned_no_amg = True
            log_info(
                '[refill] pyamg not installed: Jacobi-PCG fallback (pyamg would be ~3-5x '
                'faster on a full volume; pip install dvfopt[solvers])'
            )
        M = diags(1.0 / A.diagonal())
        maxiter = 20000

    out = phi.copy()
    for ch in range(phi.shape[0] - 2, phi.shape[0]):
        u = phi[ch].astype(np.float64).ravel()
        rhs = np.zeros(A.shape[0])
        rhs[bidx] = u[bidx]
        propagate_dirichlet_rhs(shape3, bidx, rhs)
        n_it = [0]
        t = time.time()
        x, info = cg(
            A,
            rhs,
            x0=u,
            rtol=rtol,
            maxiter=maxiter,
            M=M,
            callback=lambda _x: n_it.__setitem__(0, n_it[0] + 1),
        )
        if info != 0:
            log_warning(
                f'[refill] ch{ch}: CG did not reach rtol={rtol} ({n_it[0]} it, info {info})'
            )
        x[bidx] = u[bidx]
        out[ch] = x.reshape(phi.shape[1:])
        vlog(verbose, 1, f'[refill] ch{ch}: cg info {info}, {n_it[0]} it, {time.time() - t:.1f} s')
    return out


__all__ = ['harmonic_refill']
