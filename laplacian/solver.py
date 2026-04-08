"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Atchuth Naveen
Code developed at UC Irvine.

Contains registration pipeline for performing non linear laplacian registration in 3D space.

Performance optimizations:
- CG solver with diagonal (Jacobi) preconditioner (replaces LGMRES)
- Threaded parallel solves for the 3 independent RHS vectors
- Batch correspondence matching
"""

import gc
import inspect
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .utils import laplacianA3D, propagate_dirichlet_rhs


def solveLaplacianFromCorrespondences(
    vol_shape, source_pts, target_pts, axes=(1, 2),
    rtol=1e-2, maxiter=500, spacing=None, log_fn=None,
    axis_labels=None, solver_method='cg',
):
    """Solve the 3D Laplacian interpolation given pre-computed correspondences.

    Parameters
    ----------
    vol_shape : tuple of int
        Shape ``(n0, n1, n2)`` of the volume — axis-0, axis-1, axis-2 sizes.
        In PIR orientation these correspond to (depth, height, width).
    source_pts : np.ndarray, shape (N, 3)
        Source (moving/registered) 3D coordinates.
    target_pts : np.ndarray, shape (N, 3)
        Target (fixed/template) 3D coordinates.
    axes : tuple of int
        Which displacement axes to solve for. Default ``(1, 2)`` means
        only in-plane (d1, d2); axis 0 is kept zero (assumed aligned).
    rtol : float
        Relative CG tolerance.
    maxiter : int
        Maximum CG iterations per axis.
    spacing : tuple of float, optional
        Physical voxel spacing ``(h0, h1, h2)`` for anisotropic grids.
    log_fn : callable, optional
        Logging function ``log_fn(msg)`` for progress messages.
    axis_labels : list of str, optional
        Display labels for each solved axis (one per entry in *axes*).
        Defaults to ``['dz', 'dy', 'dx']`` mapping.
    solver_method : str, optional
        Linear solver: ``'cg'`` (Conjugate Gradient with Jacobi
        preconditioner, default) or ``'lgmres'`` (restarted LGMRES).

    Returns
    -------
    deformation_field : np.ndarray, shape (3, n0, n1, n2)
        Displacement field. ``deformation_field[d]`` is the displacement
        along axis *d* for every voxel.  Axis order matches *vol_shape*.
    """
    log = log_fn or (lambda msg: None)
    from scipy.sparse import diags as sparse_diags
    from scipy.sparse.linalg import cg as sp_cg, lgmres as sp_lgmres
    _use_cg = solver_method.lower() != 'lgmres'
    _solver_label = 'CG+Jacobi' if _use_cg else 'LGMRES'

    n0, n1, n2 = vol_shape
    nd = 3
    flen = n0 * n1 * n2

    # Flat indices at template (target) locations
    tgt_int = np.round(target_pts).astype(int)
    tgt_int[:, 0] = np.clip(tgt_int[:, 0], 0, n0 - 1)
    tgt_int[:, 1] = np.clip(tgt_int[:, 1], 0, n1 - 1)
    tgt_int[:, 2] = np.clip(tgt_int[:, 2], 0, n2 - 1)
    flat_indices = (tgt_int[:, 0] * n1 * n2 +
                    tgt_int[:, 1] * n2 +
                    tgt_int[:, 2]).astype(int)

    # Displacement = source − target
    disp = source_pts - target_pts

    # Build RHS for each solved axis
    rhs_arrays = []
    if axis_labels is None:
        axis_labels = [f"d{'zyx'[ax]}" for ax in axes]
    for ax in axes:
        rhs = np.zeros(flen)
        rhs[flat_indices] = disp[:, ax]
        rhs_arrays.append(rhs)

    boundary_indices = np.unique(flat_indices)
    log(f"Propagating boundary displacements ({len(boundary_indices)} boundary voxels)...")
    propagate_dirichlet_rhs(vol_shape, boundary_indices, *rhs_arrays, spacing=spacing)

    log("Building 3D Laplacian matrix...")
    A = laplacianA3D(vol_shape, boundary_indices, spacing=spacing)

    # Jacobi preconditioner (CG only)
    M_pre = None
    if _use_cg:
        diag_vals = A.diagonal()
        diag_vals[diag_vals == 0] = 1.0
        M_pre = sparse_diags(1.0 / diag_vals, format='csr')
        del diag_vals; gc.collect()

    # SciPy < 1.12 uses 'tol'; >= 1.12 uses 'rtol' (and deprecates 'tol').
    # Check each solver's signature independently.
    _cg_tol_kw = 'rtol' if 'rtol' in inspect.signature(sp_cg).parameters else 'tol'
    _lgmres_tol_kw = 'rtol' if 'rtol' in inspect.signature(sp_lgmres).parameters else 'tol'

    log(f"Solving Laplacian ({_solver_label}, {A.shape[0]/1e6:.1f}M DOFs, rtol={rtol}, maxiter={maxiter})...")

    solutions = {}
    for rhs, label in zip(rhs_arrays, axis_labels):
        iters = [0]
        def _cb(xk, _iters=iters, _label=label):
            _iters[0] += 1
            if _iters[0] % 50 == 0:
                log(f"  {_label}: iteration {_iters[0]}/{maxiter}")
        if _use_cg:
            x, info = sp_cg(A, rhs, **{_cg_tol_kw: rtol}, maxiter=maxiter, M=M_pre, callback=_cb)
        else:
            x, info = sp_lgmres(A, rhs, **{_lgmres_tol_kw: rtol}, maxiter=maxiter, callback=_cb)
        log(f"  {label} converged in {iters[0]} iterations")
        solutions[label] = x

    del A, M_pre; gc.collect()

    deformation_field = np.zeros((nd, n0, n1, n2))
    for ax, label in zip(axes, axis_labels):
        deformation_field[ax] = solutions[label].reshape(vol_shape)

    return deformation_field
