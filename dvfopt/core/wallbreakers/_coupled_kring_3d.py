"""Coupled k-ring SLSQP for breaking 3D fold attractors.

The escape mechanism that broke through the 1-fold M10Tet attractor
in the strict_feasibility_3d research (REPORT Parts XI–XIV). Given a
residual fold cube ``(cz, cy, cx)`` after M10Tet has converged,
construct a coupled-corner problem over the cube and its k-ring of
neighbour cubes, and minimise the L2 corner shift subject to every
involved cube having its six tetrahedron volumes above a feasibility
threshold (typically a Lagrangian-relaxed threshold like 1e-3).

The constraint Jacobian is analytical (not finite-difference) — each
tet volume's gradient is the cross-product of two of its edges, so
the (6 ⋅ n_cubes, 3 ⋅ n_free_corners) Jacobian is built in one
numpy-vectorised pass per call. This is the difference between SLSQP
running at k=2 (~5 sec) vs k=3 (~80 sec) vs being stuck at k=4.

Typical pipeline use::

    from dvfopt.core.wallbreakers._coupled_kring_3d import (
        coupled_kring_slsqp_3d, find_worst_fold_cube,
    )
    fold = find_worst_fold_cube(phi)
    if fold is not None:
        cz, cy, cx = fold
        phi_out, info = coupled_kring_slsqp_3d(
            phi, cz, cy, cx, k_ring=2, feasibility_thr=1e-3,
        )

The strategy wrapper :class:`dvfopt.strategies.CoupledKRing3DStrategy`
is the recommended high-level API.

Solver-choice rationale (why scipy SLSQP)
-----------------------------------------
scipy's SLSQP is a 1994-era Fortran implementation (Kraft). It has
known limitations on this problem class:

* Dense QP subproblem, ``O(m³)`` per iter — slow above ~5 000
  constraints.
* No sparsity exploitation — our constraint Jacobian is ~95% sparse.
* QP rejects user-supplied Jacobians on tight problems (the
  "Positive directional derivative for linesearch" failure documented
  on :func:`coupled_kring_slsqp_3d`). Forces fallback to the built-in
  finite-difference path.
* Practical scale ceiling around ``k_ring=3`` (~1 500 constraints).

For our actual production pipeline the SLSQP step takes ~5 s while
the M10Tet recovery step takes ~60 min, so SLSQP is not the
bottleneck. The right architectural moves, in order of payoff:

1. **Parallel cluster SLSQP** (implemented here — see
   :func:`coupled_kring_slsqp_3d_parallel`). Multi-cluster fields
   parallelise across cores; well-separated clusters get full
   speedup.
2. **Local M10Tet recovery** restricted to the SLSQP-modified halo
   (not yet implemented; biggest pipeline win, ~30× faster recovery).
3. **OSQP-based SQP** when ``k_ring ≥ 4`` is needed (custom outer
   loop around the OSQP sparse-QP solver). ~2-3 weeks of work; not
   yet implemented.
4. **CyIpopt** for >10⁴ constraints (industrial sparse interior-point);
   requires Fortran toolchain.

Writing our own SLSQP from scratch was considered and rejected: it
would be 2-3 months of engineering to match scipy's robustness on a
step that's already 2 orders of magnitude faster than the dominant
M10Tet recovery cost.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from dvfopt._logging import log_info, log_warning
from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    six_tet_min_volume_3d,
)

_TET_VERTICES_ARR = np.array(_TET_VERTICES, dtype=np.int64)
_TET_SIGN_ARR = np.array(_TET_SIGN, dtype=np.float64)


def find_worst_fold_cube(phi):
    """Locate the cube with the lowest minimum tet volume.

    Returns ``(cz, cy, cx)`` of the cube whose worst-of-six tet volume
    is the most negative (or smallest), or ``None`` if every cube is
    strictly feasible (all six tets above zero).
    """
    min_per_cube = six_tet_min_volume_3d(phi)  # fused parallel volume+min
    if (min_per_cube > 0).all():
        return None
    idx = int(min_per_cube.argmin())
    cz, cy, cx = np.unravel_index(idx, min_per_cube.shape)
    return (int(cz), int(cy), int(cx))


def _cube_corners(cz, cy, cx):
    """Return the 8 lattice corners of cube ``(cz, cy, cx)``."""
    return [(cz + ((i >> 2) & 1), cy + ((i >> 1) & 1), cx + (i & 1)) for i in range(8)]


def _build_problem(phi, cz, cy, cx, k_ring):
    """Enumerate cubes in the k-ring halo + their corners; build x0."""
    D, H, W = phi.shape[1:]
    cube_max = (D - 1, H - 1, W - 1)
    cubes = []
    for dz in range(-k_ring, k_ring + 1):
        for dy in range(-k_ring, k_ring + 1):
            for dx in range(-k_ring, k_ring + 1):
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < cube_max[0] and 0 <= ny < cube_max[1] and 0 <= nx < cube_max[2]:
                    cubes.append((nz, ny, nx))
    corner_set = set()
    for cube in cubes:
        corner_set.update(_cube_corners(*cube))
    free_corners = sorted(corner_set)
    corner_idx = {c: i for i, c in enumerate(free_corners)}
    n_dof = 3 * len(free_corners)
    x0 = np.zeros(n_dof)
    for ci, (z, y, x) in enumerate(free_corners):
        x0[3 * ci + 0] = phi[0, z, y, x]
        x0[3 * ci + 1] = phi[1, z, y, x]
        x0[3 * ci + 2] = phi[2, z, y, x]
    return cubes, free_corners, corner_idx, x0


def _make_index_tables(cubes, corner_idx):
    n_cubes = len(cubes)
    cube_corner_x_idx = np.zeros((n_cubes, 8, 3), dtype=np.int64)
    cube_corner_base = np.zeros((n_cubes, 8, 3))
    for ci, (cz, cy, cx) in enumerate(cubes):
        for k in range(8):
            iz = (k >> 2) & 1
            iy = (k >> 1) & 1
            ix = k & 1
            corner = (cz + iz, cy + iy, cx + ix)
            i = corner_idx[corner]
            cube_corner_x_idx[ci, k, 0] = 3 * i + 0
            cube_corner_x_idx[ci, k, 1] = 3 * i + 1
            cube_corner_x_idx[ci, k, 2] = 3 * i + 2
            cube_corner_base[ci, k] = [cz + iz, cy + iy, cx + ix]
    return cube_corner_x_idx, cube_corner_base


def _make_constraint_fn(cube_corner_x_idx, cube_corner_base, feasibility_thr):
    """Vectorised constraint values ``g(x) = V - feasibility_thr``."""
    tets = _TET_VERTICES_ARR
    signs = _TET_SIGN_ARR

    def constraint(x):
        ds = x[cube_corner_x_idx]
        pos = cube_corner_base + ds
        A = pos[:, tets[:, 0], :]
        B = pos[:, tets[:, 1], :]
        C = pos[:, tets[:, 2], :]
        D = pos[:, tets[:, 3], :]
        AB = B - A
        AC = C - A
        AD = D - A
        vols = (
            AB[..., 0] * (AC[..., 1] * AD[..., 2] - AC[..., 2] * AD[..., 1])
            - AB[..., 1] * (AC[..., 0] * AD[..., 2] - AC[..., 2] * AD[..., 0])
            + AB[..., 2] * (AC[..., 0] * AD[..., 1] - AC[..., 1] * AD[..., 0])
        ) / 6.0
        vols = vols * signs[None, :]
        return (vols - feasibility_thr).reshape(-1)

    return constraint


def _make_constraint_jacobian(cube_corner_x_idx, cube_corner_base, n_dof):
    """Analytical Jacobian of ``g(x)`` w.r.t. ``x``.

    For each tet ``(A, B, C, D)`` with edges ``AB = B-A``, ``AC``,
    ``AD``, the signed volume is ``V = (AB · (AC × AD)) / 6``. The
    gradients are::

        ∂V/∂B = (AC × AD) / 6
        ∂V/∂C = (AD × AB) / 6
        ∂V/∂D = (AB × AC) / 6
        ∂V/∂A = -∂V/∂B - ∂V/∂C - ∂V/∂D

    Each gradient is a 3-vector applied to the corresponding corner's
    three DOF; the sign-of-identity factor from ``_TET_SIGN`` flips
    cube-canonical tets that integrate to a negative reference
    volume.
    """
    tets = _TET_VERTICES_ARR
    signs = _TET_SIGN_ARR
    n_cubes = cube_corner_x_idx.shape[0]

    def jacobian(x):
        ds = x[cube_corner_x_idx]
        pos = cube_corner_base + ds
        A = pos[:, tets[:, 0], :]
        B = pos[:, tets[:, 1], :]
        C = pos[:, tets[:, 2], :]
        D = pos[:, tets[:, 3], :]
        AB = B - A
        AC = C - A
        AD = D - A
        cross_CD = np.cross(AC, AD)
        cross_DB = np.cross(AD, AB)
        cross_BC = np.cross(AB, AC)
        sgn = signs[None, :, None] / 6.0  # (1, 6, 1)
        grad_A = -(cross_CD + cross_DB + cross_BC) * sgn
        grad_B = cross_CD * sgn
        grad_C = cross_DB * sgn
        grad_D = cross_BC * sgn
        J = np.zeros((n_cubes * 6, n_dof))
        rows = (np.arange(n_cubes)[:, None] * 6 + np.arange(6)[None, :]).reshape(-1)
        for _k_local, (grad_corner, corner_axis) in enumerate(
            zip((grad_A, grad_B, grad_C, grad_D), (0, 1, 2, 3))
        ):
            corner_indices = tets[:, corner_axis]
            for k_dim in range(3):
                col_idx = cube_corner_x_idx[
                    np.arange(n_cubes)[:, None], corner_indices[None, :], k_dim
                ].reshape(-1)
                vals = grad_corner[:, :, k_dim].reshape(-1)
                J[rows, col_idx] += vals
        return J

    return jacobian


def _make_objective(x0_anchor):
    def obj(x):
        d = x - x0_anchor
        return 0.5 * float(np.dot(d, d))

    def grad(x):
        return x - x0_anchor

    return obj, grad


def _apply_x_to_phi(phi, x, free_corners):
    out = phi.copy()
    for ci, (z, y, x_lat) in enumerate(free_corners):
        out[0, z, y, x_lat] = x[3 * ci + 0]
        out[1, z, y, x_lat] = x[3 * ci + 1]
        out[2, z, y, x_lat] = x[3 * ci + 2]
    return out


def coupled_kring_slsqp_3d(
    phi,
    cz: int,
    cy: int,
    cx: int,
    *,
    k_ring: int = 2,
    feasibility_thr: float = 1e-3,
    maxiter: int = 200,
    ftol: float = 1e-9,
    use_analytical_jacobian: bool = False,
):
    """Run the coupled k-ring SLSQP centred on cube ``(cz, cy, cx)``.

    Minimises the L2 displacement-shift from the input field subject
    to every cube in the k-ring halo having its six tetrahedron
    volumes above ``feasibility_thr``. Setting ``feasibility_thr``
    below the canonical 0.005 (e.g., 1e-3) is the "Lagrangian
    relaxation" that lets the L1 anchor breathe more — empirically
    the fastest pipeline variant from REPORT Part XIV.

    Parameters
    ----------
    phi : ndarray of shape ``(3, D, H, W)``
        Displacement field.
    cz, cy, cx : int
        Cube indices to centre the halo on (typically the worst fold
        cube). Get one via :func:`find_worst_fold_cube`.
    k_ring : int, default 2
        Halo radius in cube units. ``k_ring=2`` gives ~5×5×5 cubes.
    feasibility_thr : float, default 1e-3
        Lower bound the SLSQP enforces on every tet volume.
    maxiter : int, default 200
    ftol : float, default 1e-9
    use_analytical_jacobian : bool, default False
        Pass ``True`` to use the closed-form constraint Jacobian
        (verified correct to 1e-11 against FD). Note: scipy SLSQP's
        QP subproblem empirically rejects user-supplied Jacobians on
        this problem ("Positive directional derivative" failure),
        while its built-in FD path converges cleanly. The analytical
        path is kept for non-SLSQP solvers and as a research
        artefact. Leave at ``False`` for production SLSQP use.

    Returns
    -------
    phi_out : ndarray
        Corrected displacement field.
    info : dict
        Diagnostic info: ``success``, ``wall_s``, ``n_iter``,
        ``n_cubes``, ``n_dof``, ``n_constraints``, ``fun``,
        ``message``.
    """
    if phi.shape[0] != 3 or phi.ndim != 4:
        raise ValueError(f'phi must have shape (3, D, H, W), got {phi.shape}')

    cubes, free_corners, corner_idx, x0 = _build_problem(phi, cz, cy, cx, k_ring)
    n_dof = 3 * len(free_corners)
    n_cubes = len(cubes)
    n_constraints = 6 * n_cubes

    cube_corner_x_idx, cube_corner_base = _make_index_tables(cubes, corner_idx)
    constraint_fn = _make_constraint_fn(cube_corner_x_idx, cube_corner_base, feasibility_thr)
    obj, obj_grad = _make_objective(x0.copy())

    constraint_dict = {'type': 'ineq', 'fun': constraint_fn}
    if use_analytical_jacobian:
        constraint_dict['jac'] = _make_constraint_jacobian(
            cube_corner_x_idx, cube_corner_base, n_dof
        )

    t0 = time.time()
    if use_analytical_jacobian:
        from dvfopt.core.primitives.slsqp import minimize_slsqp_traced

        res = minimize_slsqp_traced(
            obj,
            x0,
            jac=obj_grad,
            constraints=[constraint_dict],
            maxiter=maxiter,
            ftol=ftol,
        )
    else:
        # ponytail: FD-jacobian path stays on scipy — the traced driver
        # deliberately requires analytic jacs; upgrade when kring defaults flip.
        res = minimize(
            obj,
            x0,
            jac=obj_grad,
            constraints=[constraint_dict],
            method='SLSQP',
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False},
        )
    wall = time.time() - t0

    phi_out = _apply_x_to_phi(phi, res.x, free_corners)
    info = {
        'success': bool(res.success),
        'wall_s': float(wall),
        'n_iter': int(res.nit),
        'n_cubes': int(n_cubes),
        'n_dof': int(n_dof),
        'n_constraints': int(n_constraints),
        'fun': float(res.fun),
        'message': str(res.message),
        'fold_center': (int(cz), int(cy), int(cx)),
        'k_ring': int(k_ring),
        'feasibility_thr': float(feasibility_thr),
        'analytical_jac': bool(use_analytical_jacobian),
    }
    return phi_out, info


def _default_m10tet_inner(threshold):
    """Build a callable that runs M10Tet (HarmonicALMBarrier3D) on a crop.

    Lazily imports the Solver/Strategy layer (which imports core, so a
    top-level import here would be circular). Safe at call time.
    """

    def inner(crop, time_budget_s=600.0):
        from dvfopt import (  # local import to avoid import cycle
            HarmonicALMBarrier3DStrategy,
            L1Objective,
            Solver,
            Tet6Constraint3D,
        )

        solver = Solver(
            constraint=Tet6Constraint3D(shape=crop.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=threshold,
        )
        return solver.fit(crop).corrected

    return inner


def local_alm_recovery_3d(
    phi,
    *,
    center=None,
    k_ring=2,
    pad=4,
    threshold=0.012,
    inner_solve=None,
    max_widen=2,
    verbose=0,
):
    """Run M10Tet recovery on ONLY a crop around the perturbed region.

    This is the highest-ROI pipeline optimisation (REPORT Part XV "open
    directions" / ``_coupled_kring_3d`` docstring item #2). A coupled
    k-ring SLSQP escape (:func:`coupled_kring_slsqp_3d`) modifies only a
    tiny halo of corners, but the conventional recovery step re-runs the
    barrier/ALM pipeline over the WHOLE chunk (~60 min on B0039). Since
    only the halo changed, the recovery need only touch the halo's
    bounding box plus a padding ring — typically a 10×14×14 crop that
    M10Tet clears in seconds-to-minutes.

    The crop is solved with the full M10Tet pipeline (harmonic seed +
    PHR-ALM + log-barrier polish) and pasted back. Correctness against
    the shared-corner topology is handled the same way the Schwarz
    decomposition handles it: a generous ``pad`` ring absorbs boundary
    effects, and the global fold count is re-checked after pasting. If
    the paste increased the global fold count, the pad is widened and
    the crop re-solved (up to ``max_widen`` times); if it still
    regresses, the original field is returned unchanged.

    Parameters
    ----------
    phi : ndarray of shape (3, D, H, W)
    center : tuple[int, int, int] | None
        Cube to centre the recovery crop on (typically the cube the
        SLSQP escape was centred on). If ``None``, the bounding box of
        ALL current fold cubes is used.
    k_ring : int, default 2
        Half-width (in cubes) of the perturbed region to recover around
        ``center``. Ignored when ``center is None``.
    pad : int, default 4
        Padding ring (in cubes) added around the perturbed region. The
        ring is re-solved too (overlap), then global-verified.
    threshold : float, default 0.012
        Feasibility threshold for the inner M10Tet.
    inner_solve : callable | None
        ``(crop, time_budget_s=...) -> crop_out``. If ``None``, the
        default M10Tet inner is built via :func:`_default_m10tet_inner`.
    max_widen : int, default 2
        How many times to widen ``pad`` (by +pad each time) if a paste
        regresses the global fold count.
    verbose : int, default 0

    Returns
    -------
    phi_out : ndarray
    info : dict
        ``crop_bbox``, ``crop_shape``, ``wall_s``, ``n_neg_before``,
        ``n_neg_after``, ``widen_used``, ``accepted``. The fold counts are
        per-CUBE (a cube with any of its six tets folded counts once), via
        the fused ``six_tet_min_volume_3d`` kernel — not per-TET as the
        old materialised ``six_tet_volumes_3d`` count was.
    """
    if phi.shape[0] != 3 or phi.ndim != 4:
        raise ValueError(f'phi must have shape (3, D, H, W), got {phi.shape}')

    D, H, W = phi.shape[1:]
    if inner_solve is None:
        inner_solve = _default_m10tet_inner(threshold)

    # Fused per-cube min kernel: never materialises the (6, D-1, H-1, W-1)
    # volume array. Counts are per-cube; before/after share the semantics.
    min0 = six_tet_min_volume_3d(phi)
    n_neg_before = int((min0 <= 0).sum())

    # Determine the cube bounding box of the region to recover.
    if center is None:
        fold_mask = min0 <= 0
        if not fold_mask.any():
            return phi.copy(), {
                'crop_bbox': None,
                'crop_shape': None,
                'wall_s': 0.0,
                'n_neg_before': n_neg_before,
                'n_neg_after': n_neg_before,
                'widen_used': 0,
                'accepted': True,
            }
        cz, cy, cx = np.where(fold_mask)
        cube_lo = (int(cz.min()), int(cy.min()), int(cx.min()))
        cube_hi = (int(cz.max()), int(cy.max()), int(cx.max()))
    else:
        ccz, ccy, ccx = center
        cube_lo = (ccz - k_ring, ccy - k_ring, ccx - k_ring)
        cube_hi = (ccz + k_ring, ccy + k_ring, ccx + k_ring)

    t0 = time.time()
    best = phi
    accepted = False
    widen_used = 0
    bbox = None
    crop_shape = None
    n_neg_after = n_neg_before

    for widen in range(max_widen + 1):
        p = pad + widen * pad
        # Cube range -> corner range. A cube (c) spans corners [c, c+1];
        # padding by p cubes, then convert to corner indices and clip.
        z0 = max(0, cube_lo[0] - p)
        z1 = min(D - 1, cube_hi[0] + 1 + p)
        y0 = max(0, cube_lo[1] - p)
        y1 = min(H - 1, cube_hi[1] + 1 + p)
        x0 = max(0, cube_lo[2] - p)
        x1 = min(W - 1, cube_hi[2] + 1 + p)
        if z1 - z0 < 2 or y1 - y0 < 2 or x1 - x0 < 2:
            # Crop too small to contain a cube; nothing to do.
            break
        crop = phi[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1].copy()
        try:
            crop_out = inner_solve(crop)
        except Exception as exc:
            log_warning(f'local recovery inner_solve failed: {type(exc).__name__}: {exc}')
            break
        trial = phi.copy()
        trial[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = crop_out
        # Fused per-cube count (same semantics as n_neg_before above).
        n_trial = int((six_tet_min_volume_3d(trial) <= 0).sum())
        if verbose:
            log_info(
                f'  local recovery widen={widen} pad={p} crop='
                f'{crop.shape[1:]} n_neg {n_neg_before}->{n_trial}',
            )
        if n_trial <= n_neg_before:
            best = trial
            accepted = True
            widen_used = widen
            bbox = (z0, z1, y0, y1, x0, x1)
            crop_shape = tuple(int(s) for s in crop.shape[1:])
            n_neg_after = n_trial
            break
        # Regressed — remember the last attempt's stats for diagnostics,
        # then widen and retry.
        n_neg_after = n_trial
        bbox = (z0, z1, y0, y1, x0, x1)
        crop_shape = tuple(int(s) for s in crop.shape[1:])

    info = {
        'crop_bbox': bbox,
        'crop_shape': crop_shape,
        'wall_s': float(time.time() - t0),
        'n_neg_before': n_neg_before,
        'n_neg_after': int(n_neg_after if accepted else n_neg_before),
        'widen_used': int(widen_used),
        'accepted': bool(accepted),
    }
    return (best if accepted else phi.copy()), info


def _solve_band_crop(args):
    """Worker: run M10Tet on a single crop. Picklable (crop ndarray +
    threshold float only), so it ships small over the process pool."""
    crop, threshold = args
    inner = _default_m10tet_inner(threshold)
    try:
        return inner(crop)
    except Exception:
        return None


def _padded_box(bbox, pad, shape):
    """Inclusive padded corner-index box for a cube bbox, clipped to grid."""
    cz0, cz1, cy0, cy1, cx0, cx1 = bbox
    D, H, W = shape
    return (
        max(0, cz0 - pad),
        min(D - 1, cz1 + 1 + pad),
        max(0, cy0 - pad),
        min(H - 1, cy1 + 1 + pad),
        max(0, cx0 - pad),
        min(W - 1, cx1 + 1 + pad),
    )


def _boxes_separated(a, b, gap=2):
    """True if padded corner-boxes a, b are at least ``gap`` apart on some
    axis (so they share no corners and can be solved/pasted concurrently)."""
    az0, az1, ay0, ay1, ax0, ax1 = a
    bz0, bz1, by0, by1, bx0, bx1 = b
    return (
        az1 + gap < bz0
        or bz1 + gap < az0
        or ay1 + gap < by0
        or by1 + gap < ay0
        or ax1 + gap < bx0
        or bx1 + gap < ax0
    )


def _batch_nonoverlapping_boxes(pboxes):
    """Greedy partition of padded boxes into batches with no intra-batch
    overlap, so each batch's crops are independent (parallel-safe)."""
    batches = []  # each: list of indices into pboxes
    for i, box in enumerate(pboxes):
        placed = False
        for batch in batches:
            if all(_boxes_separated(box, pboxes[j]) for j in batch):
                batch.append(i)
                placed = True
                break
        if not placed:
            batches.append([i])
    return batches


def _fold_cluster_bboxes(min_per_cube, threshold, merge_dilation=2):
    """Connected-component bounding boxes of folded cubes (scalable).

    Uses binary dilation + 26-connectivity labelling (same as the Schwarz
    decomposition) so cost is O(n_cells), not O(n_folds^2). Returns a list
    of (cz0, cz1, cy0, cy1, cx0, cx1) inclusive cube-index bounding boxes,
    one per fold cluster.
    """
    from scipy.ndimage import (
        binary_dilation,
        generate_binary_structure,
    )
    from scipy.ndimage import (
        label as cc_label,
    )

    fold = min_per_cube < threshold
    if not fold.any():
        return []
    if merge_dilation < 0:
        raise ValueError(f'merge_dilation must be >= 0, got {merge_dilation}')
    # scipy treats iterations < 1 as "repeat until convergence", which would
    # dilate the mask to the whole grid — only dilate for a positive count.
    grouped = binary_dilation(fold, iterations=merge_dilation) if merge_dilation >= 1 else fold
    labels, n = cc_label(grouped, structure=generate_binary_structure(3, 3))
    bboxes = []
    for cid in range(1, n + 1):
        comp = (labels == cid) & fold
        if not comp.any():
            continue
        cz, cy, cx = np.where(comp)
        bboxes.append(
            (
                int(cz.min()),
                int(cz.max()),
                int(cy.min()),
                int(cy.max()),
                int(cx.min()),
                int(cx.max()),
            )
        )
    return bboxes


def _tile_bbox(bbox, max_box):
    """Split an oversized cube bbox into bounded sub-bboxes.

    ``bbox`` is an inclusive cube-index box ``(cz0, cz1, cy0, cy1, cx0,
    cx1)``. Any axis spanning more than ``max_box`` cubes is tiled into
    pieces of at most ``max_box`` cubes each (all three axes z, y, x),
    generalising the 2D per-slice ``range(start, stop, MAX_BOX)`` tiling in
    ``research/strict_feasibility_3d/runners/_marching_full_volume.py``.
    A box already within ``max_box`` on every axis yields itself unchanged.
    Tiles are disjoint on cube indices and abut exactly (each tile starts
    one cube after its predecessor's inclusive end, so no cube is solved
    twice); the padded crops still overlap at the seams, and the verify
    after each paste keeps the seams honest.
    """
    cz0, cz1, cy0, cy1, cx0, cx1 = bbox
    # Inclusive bounds: a tile starting at t covers cubes [t, t+max_box-1],
    # i.e. exactly max_box cubes, and the next tile starts at t+max_box.
    zs = list(range(cz0, cz1 + 1, max_box)) or [cz0]
    ys = list(range(cy0, cy1 + 1, max_box)) or [cy0]
    xs = list(range(cx0, cx1 + 1, max_box)) or [cx0]
    tiles = []
    for tz in zs:
        for ty in ys:
            for tx in xs:
                tiles.append(
                    (
                        tz,
                        min(cz1, tz + max_box - 1),
                        ty,
                        min(cy1, ty + max_box - 1),
                        tx,
                        min(cx1, tx + max_box - 1),
                    )
                )
    return tiles


def _freeze_rim(crop, crop_out):
    """Return ``crop_out`` with its six outer boundary faces reset to
    ``crop``'s originals.

    The inner solve is free to move interior corners, but its rim must not
    move: a moved rim would introduce a boundary-discontinuity fold at the
    paste seam (the crop's rim corners are shared with cells outside the
    crop). Restoring the rim guarantees the paste cannot create a seam fold;
    the post-paste global verify still rejects any interior regression.
    """
    out = crop_out.copy()
    out[:, 0, :, :] = crop[:, 0, :, :]
    out[:, -1, :, :] = crop[:, -1, :, :]
    out[:, :, 0, :] = crop[:, :, 0, :]
    out[:, :, -1, :] = crop[:, :, -1, :]
    out[:, :, :, 0] = crop[:, :, :, 0]
    out[:, :, :, -1] = crop[:, :, :, -1]
    return out


def active_band_alm_recovery_3d(
    phi,
    *,
    threshold=0.012,
    pad=4,
    merge_dilation=2,
    inner_solve=None,
    max_widen=1,
    max_box=48,
    n_workers=1,
    verbose=0,
):
    """Active-band M10Tet: solve only folded-cell regions, not the field.

    The bulk M10Tet recovery normally optimises the WHOLE field even
    though most cells are already feasible — so every L-BFGS forward +
    adjoint kernel evaluation scans all cells. This wrapper restricts the
    work to the active band: it finds the connected fold clusters, crops a
    padded box around EACH, runs M10Tet only on that crop, pastes it back,
    and accepts only if the global fold count did not increase. Cells
    outside every crop are untouched (and were already feasible), so the
    strict 6-tet guarantee is preserved and re-verified globally.

    This is the crop-based realisation of the audit's "active-band
    restriction" — same wall-clock benefit as kernel-level DOF masking
    (the kernels only run on crop cells) but with the low risk of the
    verified crop+paste+verify pattern already proven in
    :func:`local_alm_recovery_3d`.

    A cluster whose bounding box spans more than ``max_box`` cubes on any
    axis is split into bounded, padded tiles (:func:`_tile_bbox`) and each
    tile goes through the same crop->solve->paste->verify loop. The full
    field is never solved as one crop (which on a large volume builds a
    multi-million-column sparse tet system and OOM-segfaults SuperLU).

    Parameters
    ----------
    phi : ndarray (3, D, H, W)
    threshold : float, default 0.012   feasibility threshold for inner M10Tet
    pad : int, default 4               padding ring (cubes) around each cluster
    merge_dilation : int, default 2    cluster-merge dilation for CC labelling
    inner_solve : callable | None      (crop, time_budget_s=...) -> crop_out;
                                       defaults to M10Tet via _default_m10tet_inner
    max_widen : int, default 1         pad-widen retries on a regressing paste
    max_box : int, default 48          per-axis cube cap; larger clusters are
                                       tiled into bounded boxes (never global)
    n_workers : int, default 1
        Process-pool workers for solving non-overlapping cluster crops
        concurrently. ``1`` = sequential (default). ``None`` = cpu_count.
        NOTE the Windows process-spawn + per-worker Numba recompile tax
        (~2-6 s/worker) means parallelism only pays off with MANY large
        clusters; for a few small crops it is slower than sequential. The
        coarse-grained win (parallelise whole z-bands) belongs in the
        orchestrator, where spawn cost is amortised over big jobs.
    verbose : int

    Returns
    -------
    phi_out : ndarray
    info : dict  (n_neg_before, n_neg_after, n_clusters, accepted, wall_s, per_cluster)
    """
    if phi.shape[0] != 3 or phi.ndim != 4:
        raise ValueError(f'phi must have shape (3, D, H, W), got {phi.shape}')
    D, H, W = phi.shape[1:]
    if inner_solve is None:
        inner_solve = _default_m10tet_inner(threshold)

    t0 = time.time()
    cur = phi.copy()
    min0 = six_tet_min_volume_3d(cur)
    n_neg_before = int((min0 <= 0).sum())
    bboxes = _fold_cluster_bboxes(min0, threshold, merge_dilation)
    n_clusters_total = len(bboxes)
    # Tile oversized clusters up front so BOTH the parallel and sequential
    # paths only ever see bounded boxes — the full field is never solved as
    # one crop.
    bboxes = [tile for bb in bboxes for tile in _tile_bbox(bb, max_box)]
    per_cluster = []

    # Parallel path: solve non-overlapping cluster crops concurrently. All
    # boxes are already tiled to <= max_box per axis, so every crop is
    # bounded. Each batch is intra-disjoint, so crops paste independently;
    # batches are pasted+verified sequentially.
    if n_workers != 1 and len(bboxes) > 1:
        import os

        from dvfopt.core._pool import pool_map

        if n_workers is None:
            n_workers = max(1, os.cpu_count() or 1)
        pboxes = [_padded_box(bb, pad, (D, H, W)) for bb in bboxes]
        retry_bboxes = []  # per-crop-rejected tiles -> sequential pad-widen path
        for batch in _batch_nonoverlapping_boxes(pboxes):
            crops = [
                cur[
                    :,
                    pboxes[i][0] : pboxes[i][1] + 1,
                    pboxes[i][2] : pboxes[i][3] + 1,
                    pboxes[i][4] : pboxes[i][5] + 1,
                ].copy()
                for i in batch
            ]
            if len(batch) == 1:
                results = [_solve_band_crop((crops[0], threshold))]
            else:
                # Constant pool size: get_pool() tears down and respawns the
                # whole warm pool whenever the requested size changes
                # (~5-10 s/worker), and batch sizes vary between iterations.
                # pool_map only dispatches len(args) tasks, so idle workers
                # are free — always ask for the same n_workers.
                results = pool_map(
                    _solve_band_crop,
                    [(c, threshold) for c in crops],
                    n_workers,
                )
            # Frozen-rim candidates + CROP-LOCAL fold recounts. _freeze_rim
            # restores all six crop faces, so only cubes strictly inside a
            # crop's own cube range can change; cubes straddling the crop
            # boundary depend only on rim + outside nodes (both unchanged).
            # Counting on the crop arrays is therefore EXACT — no full-volume
            # trial copy or full-volume recount is needed per batch.
            n_before = int((six_tet_min_volume_3d(cur) <= 0).sum())
            cands = []  # aligned with batch: (candidate, local_before, local_after) | None
            for crop, crop_out in zip(crops, results):
                if crop_out is None:
                    cands.append(None)
                    continue
                cand = _freeze_rim(crop, crop_out)
                local_before = int((six_tet_min_volume_3d(crop) <= 0).sum())
                local_after = int((six_tet_min_volume_3d(cand) <= 0).sum())
                cands.append((cand, local_before, local_after))
            batch_delta = sum(c[2] - c[1] for c in cands if c is not None)
            n_after = n_before + batch_delta
            if verbose:
                log_info(
                    f'  active-band parallel batch ({len(batch)} crops): '
                    f'n_neg {n_before}->{n_after}',
                )
            if batch_delta <= 0:
                # Batch-global accept: paste every solved crop.
                for i, c in zip(batch, cands):
                    if c is None:
                        continue
                    z0, z1, y0, y1, x0, x1 = pboxes[i]
                    cur[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = c[0]
                    pb = pboxes[i]
                    per_cluster.append(
                        dict(
                            bbox=pb,
                            crop_shape=(pb[1] - pb[0] + 1, pb[3] - pb[2] + 1, pb[5] - pb[4] + 1),
                            n_before=n_before,
                            n_after=n_after,
                            parallel=True,
                        )
                    )
            else:
                # Batch regressed: fall back to pasting crops ONE AT A TIME
                # with per-crop verify, so a single bad crop cannot discard
                # its siblings' fixes. Individually rejected crops are handed
                # to the sequential path below for the pad-widen retries.
                n_run = n_before
                for i, c in zip(batch, cands):
                    if c is None:
                        continue
                    cand, local_before, local_after = c
                    pb = pboxes[i]
                    if local_after <= local_before:
                        z0, z1, y0, y1, x0, x1 = pb
                        cur[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = cand
                        per_cluster.append(
                            dict(
                                bbox=pb,
                                crop_shape=(
                                    pb[1] - pb[0] + 1,
                                    pb[3] - pb[2] + 1,
                                    pb[5] - pb[4] + 1,
                                ),
                                n_before=n_run,
                                n_after=n_run + (local_after - local_before),
                                parallel=True,
                                per_crop=True,
                            )
                        )
                        n_run += local_after - local_before
                    else:
                        if verbose:
                            log_info(
                                f'  active-band per-crop reject bbox={pb} '
                                f'local n_neg {local_before}->{local_after}; '
                                f'queued for sequential retry',
                            )
                        retry_bboxes.append(bboxes[i])
        # Accepted boxes were solved above; individually rejected ones fall
        # through to the sequential path (with its pad-widen retries).
        bboxes = retry_bboxes

    for cz0, cz1, cy0, cy1, cx0, cx1 in bboxes:
        # ``cur`` is unchanged across the pad-widen retries (a paste only
        # happens on accept, which exits the loop), so the global count is
        # computed ONCE per cluster, not once per attempt.
        n_before = int((six_tet_min_volume_3d(cur) <= 0).sum())
        for widen in range(max_widen + 1):
            p = pad + widen * pad
            z0 = max(0, cz0 - p)
            z1 = min(D - 1, cz1 + 1 + p)
            y0 = max(0, cy0 - p)
            y1 = min(H - 1, cy1 + 1 + p)
            x0 = max(0, cx0 - p)
            x1 = min(W - 1, cx1 + 1 + p)
            if z1 - z0 < 2 or y1 - y0 < 2 or x1 - x0 < 2:
                break
            crop = cur[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1].copy()
            try:
                crop_out = inner_solve(crop)
            except Exception as exc:
                log_warning(f'active-band cluster solve failed: {type(exc).__name__}: {exc}')
                break
            # CROP-LOCAL verify: _freeze_rim restores all six crop faces, so
            # only cubes strictly inside the crop's cube range [z0, z1-1] x
            # [y0, y1-1] x [x0, x1-1] can change. Cubes straddling the crop
            # boundary depend only on rim + outside nodes (both unchanged),
            # so recounting on the crop arrays is EXACT — no full-volume
            # copy or full-volume recount per attempt.
            candidate = _freeze_rim(crop, crop_out)
            local_before = int((six_tet_min_volume_3d(crop) <= 0).sum())
            local_after = int((six_tet_min_volume_3d(candidate) <= 0).sum())
            n_after = n_before - local_before + local_after
            if verbose:
                log_info(
                    f'  active-band bbox z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}] '
                    f'crop={crop.shape[1:]} n_neg {n_before}->{n_after}',
                )
            if local_after <= local_before:
                # Paste directly into cur only after the local check passes;
                # cur was untouched until now, so no rollback is ever needed.
                cur[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = candidate
                per_cluster.append(
                    dict(
                        bbox=(z0, z1, y0, y1, x0, x1),
                        crop_shape=tuple(int(s) for s in crop.shape[1:]),
                        n_before=n_before,
                        n_after=n_after,
                    )
                )
                break
    # Final full-volume verify — the single safety net for the crop-local
    # accounting above.
    min_f = six_tet_min_volume_3d(cur)
    n_neg_after = int((min_f <= 0).sum())
    info = {
        'n_neg_before': n_neg_before,
        'n_neg_after': n_neg_after,
        'n_clusters': n_clusters_total,
        'accepted': n_neg_after <= n_neg_before,
        'wall_s': float(time.time() - t0),
        'per_cluster': per_cluster,
    }
    if n_neg_after > n_neg_before:
        info['warning'] = (
            f'final full-volume verify found n_neg {n_neg_after} > initial '
            f'{n_neg_before} despite per-crop accepts; no rollback attempted'
        )
        if verbose:
            log_info(f'  active-band WARNING: {info["warning"]}')
    return cur, info


def _solve_zband_worker(args):
    """Worker: active-band M10Tet on one z-band crop (full y, x). Picklable
    (band ndarray + scalars). Runs sequentially inside the worker
    (n_workers=1) — no nested pools."""
    band, threshold, pad = args
    try:
        out, _ = active_band_alm_recovery_3d(
            band, threshold=threshold, pad=pad, n_workers=1, verbose=0
        )
        return out
    except Exception:
        return None


def parallel_zband_solve(
    phi,
    *,
    threshold=0.012,
    band_size=24,
    overlap=4,
    pad=4,
    n_workers=None,
    seam_cleanup=True,
    verbose=0,
):
    """Coarse z-band parallel bulk solve — the orchestrator's big lever.

    Splits the volume into z-bands (each full in y, x), solves each band's
    folds with active-band M10Tet IN PARALLEL across a process pool, pastes
    each band's interior z-planes back (overlap planes are owned by the
    neighbour), then runs ONE active-band cleanup pass on any residual
    seam folds. Because each band is a big job, the Windows process-spawn
    + Numba-recompile tax is amortised — unlike fine-grained cluster
    parallelism (which is spawn-dominated). This is the coarse parallelism
    the audit recommended for whole-volume passes.

    Correctness: each band solve is internally active-band (per-cluster
    crop + global-verify-within-band), so it never increases that band's
    folds. Seams between bands (shared corner planes) can introduce a few
    folds; ``seam_cleanup`` runs a final ``active_band_alm_recovery_3d`` on
    the assembled field to repair them. A global six-tet re-check is the
    ground truth in ``info``.

    Parameters
    ----------
    phi : ndarray (3, D, H, W)
    threshold : float, default 0.012
    band_size : int, default 24       interior cube-depth of each band
    overlap : int, default 4          halo cube-depth shared with neighbours
    pad : int, default 4              active-band crop pad inside each band
    n_workers : int | None            pool size (None = cpu_count)
    seam_cleanup : bool, default True run a final active-band pass on seams
    verbose : int

    Returns
    -------
    phi_out : ndarray
    info : dict (n_neg_before, n_neg_after, n_bands, n_workers, wall_s,
                 seam_cleanup_ran)
    """
    import os

    from dvfopt.core._pool import pool_map

    if phi.shape[0] != 3 or phi.ndim != 4:
        raise ValueError(f'phi must have shape (3, D, H, W), got {phi.shape}')
    D, H, W = phi.shape[1:]
    Dc = D - 1  # number of cube layers
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 1)

    t0 = time.time()
    n_neg_before = int((six_tet_min_volume_3d(phi) <= 0).sum())

    # Build z-bands over cube layers [0, Dc): interior [s, e), halo +/- overlap.
    bands = []  # (corner_z0, corner_z1, interior_cube_s, interior_cube_e)
    s = 0
    while s < Dc:
        e = min(s + band_size, Dc)
        cz0 = max(0, s - overlap)
        cz1 = min(Dc, e + overlap)  # cube range [cz0, cz1)
        # corner range covering cubes [cz0, cz1): corners [cz0, cz1] inclusive
        bands.append((cz0, cz1 + 0, s, e))
        s = e
    n_bands = len(bands)

    cur = phi.copy()
    if n_bands == 1:
        # Single band — no parallelism to gain; just active-band it.
        cur, _ = active_band_alm_recovery_3d(
            cur, threshold=threshold, pad=pad, n_workers=1, verbose=verbose
        )
    else:
        # Extract band crops (full y, x; corner z-range [cz0, cz1]).
        crops = [phi[:, cz0 : cz1 + 1, :, :].copy() for (cz0, cz1, _, _) in bands]
        if n_workers == 1:
            results = [_solve_zband_worker((c, threshold, pad)) for c in crops]
        else:
            results = pool_map(
                _solve_zband_worker,
                [(c, threshold, pad) for c in crops],
                min(n_workers, n_bands),
            )
        # Paste each band's INTERIOR corner planes (cubes [s, e) -> corner
        # planes [s, e); the last band also writes its top corner plane e).
        for (cz0, _cz1, si, ei), solved in zip(bands, results):
            if solved is None:
                continue
            # corner planes to write: [si, ei] inclusive for the last band,
            # else [si, ei).  Map to the crop's local index (offset cz0).
            w_lo = si
            w_hi = ei + 1 if ei == Dc else ei  # inclusive top only at volume top
            loc_lo = w_lo - cz0
            loc_hi = w_hi - cz0
            cur[:, w_lo:w_hi, :, :] = solved[:, loc_lo:loc_hi, :, :]
        if verbose:
            n_mid = int((six_tet_min_volume_3d(cur) <= 0).sum())
            log_info(f'  z-band paste: n_neg={n_mid} (bands={n_bands})')

    seam_ran = False
    if seam_cleanup and int((six_tet_min_volume_3d(cur) <= 0).sum()) > 0:
        cur, _ = active_band_alm_recovery_3d(
            cur, threshold=threshold, pad=pad, n_workers=1, verbose=verbose
        )
        seam_ran = True

    n_neg_after = int((six_tet_min_volume_3d(cur) <= 0).sum())
    info = {
        'n_neg_before': n_neg_before,
        'n_neg_after': n_neg_after,
        'n_bands': n_bands,
        'n_workers': n_workers,
        'wall_s': float(time.time() - t0),
        'seam_cleanup_ran': seam_ran,
    }
    return cur, info


def cluster_fold_cubes(fold_cells, radius=2):
    """Group fold cubes into spatially-connected clusters.

    Two cubes belong to the same cluster if their Chebyshev distance
    (max-axis lattice distance) is ``<= radius``. Returns the cluster
    centroids and member lists.

    Parameters
    ----------
    fold_cells : list of (cz, cy, cx)
    radius : int, default 2

    Returns
    -------
    centroids : list of (cz, cy, cx)
        Centroid of each cluster (integer-rounded).
    members : list of list of (cz, cy, cx)
        Cluster membership.
    radii : list of int
        Max-axis radius of each cluster around its centroid.
    """
    if not fold_cells:
        return [], [], []
    pts = np.array(fold_cells, dtype=int)
    n = len(pts)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(pts[i] - pts[j]).max() <= radius:
                adj[i, j] = adj[j, i] = True
    labels = [-1] * n
    visited = [False] * n
    cl = 0
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        labels[i] = cl
        while stack:
            v = stack.pop()
            for j in range(n):
                if adj[v, j] and not visited[j]:
                    visited[j] = True
                    labels[j] = cl
                    stack.append(j)
        cl += 1
    members = [[] for _ in range(cl)]
    for i, lbl in enumerate(labels):
        members[lbl].append(fold_cells[i])
    centroids = []
    radii = []
    for mem in members:
        m_pts = np.array(mem)
        c = m_pts.mean(axis=0).astype(int)
        r = int(np.max(np.abs(m_pts - c)))
        centroids.append(tuple(int(v) for v in c))
        radii.append(r)
    return centroids, members, radii


def _run_one_cluster(args):
    """Worker entry-point for parallel cluster SLSQP.

    args = (phi, cz, cy, cx, k_ring, feasibility_thr, maxiter, ftol,
            use_analytical_jacobian).
    Returns (phi_modified_in_local_region, free_corners, info).
    We return just the free-corner displacements + index list rather
    than the whole phi to keep IPC payload small.
    """
    (phi, cz, cy, cx, k_ring, feasibility_thr, maxiter, ftol, use_analytical_jacobian) = args
    phi_out, info = coupled_kring_slsqp_3d(
        phi,
        cz,
        cy,
        cx,
        k_ring=k_ring,
        feasibility_thr=feasibility_thr,
        maxiter=maxiter,
        ftol=ftol,
        use_analytical_jacobian=use_analytical_jacobian,
    )
    # Extract only the changed corner displacements so we can apply
    # them back in the parent without shipping the entire phi.
    _, free_corners, _, _ = _build_problem(phi, cz, cy, cx, k_ring)
    deltas = []
    for z, y, x in free_corners:
        if not np.array_equal(phi_out[:, z, y, x], phi[:, z, y, x]):
            deltas.append(
                (
                    (int(z), int(y), int(x)),
                    (
                        float(phi_out[0, z, y, x]),
                        float(phi_out[1, z, y, x]),
                        float(phi_out[2, z, y, x]),
                    ),
                )
            )
    return deltas, info


def coupled_kring_slsqp_3d_parallel(
    phi,
    fold_centers,
    *,
    k_ring=2,
    feasibility_thr=1e-3,
    maxiter=200,
    ftol=1e-9,
    use_analytical_jacobian=False,
    n_workers=None,
    overlap_check=True,
    accept_check=True,
):
    """Parallel coupled k-ring SLSQP across multiple fold-cluster centres.

    Solves a separate SLSQP per centre concurrently using
    ``concurrent.futures.ProcessPoolExecutor``. Each worker receives a
    READ-ONLY copy of ``phi`` and returns only the displacements of
    the corners it modified. The parent merges these back into a
    single output field.

    .. warning::
        For correctness when k-ring halos overlap between two centres,
        the workers' moves can conflict at shared corners. Set
        ``overlap_check=True`` (the default) to skip centres whose
        halo intersects an already-processed centre's halo; the
        skipped centres run sequentially after the parallel batch.
        With well-separated clusters this still parallelises 90%+ of
        the work.

    Parameters
    ----------
    phi : ndarray of shape (3, D, H, W)
    fold_centers : list of (cz, cy, cx)
        Cubes to centre each SLSQP halo on, typically cluster centroids
        from :func:`cluster_fold_cubes`.
    k_ring, feasibility_thr, maxiter, ftol, use_analytical_jacobian
        Per-SLSQP-call options; see :func:`coupled_kring_slsqp_3d`.
    n_workers : int | None
        Worker count for the ProcessPoolExecutor. ``None`` uses
        ``os.cpu_count()``. Set to 1 to force sequential execution.
    overlap_check : bool, default True
        Partition centres into non-overlapping halo batches; centres in
        later batches still run but sequentially after their batch.
    accept_check : bool, default True
        After applying each worker's deltas, verify that global
        ``n_neg`` does not increase by more than the local fold count
        (rejects pathological worker outputs).

    Returns
    -------
    phi_out : ndarray
    infos : list of dict
        Per-centre diagnostic dicts.
    """
    import os

    from dvfopt.core._pool import pool_map

    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 1)

    # Partition centres into non-overlapping batches if requested.
    if overlap_check and len(fold_centers) > 1:
        batches = _partition_non_overlapping(fold_centers, k_ring)
    else:
        batches = [list(fold_centers)]

    cur = phi.copy()
    infos = []
    for batch in batches:
        if n_workers == 1 or len(batch) == 1:
            # Sequential path.
            for c in batch:
                phi_new, info = coupled_kring_slsqp_3d(
                    cur,
                    c[0],
                    c[1],
                    c[2],
                    k_ring=k_ring,
                    feasibility_thr=feasibility_thr,
                    maxiter=maxiter,
                    ftol=ftol,
                    use_analytical_jacobian=use_analytical_jacobian,
                )
                infos.append(info)
                if accept_check:
                    cur = _accept_or_reject(cur, phi_new)
                else:
                    cur = phi_new
            continue
        # Parallel path: each worker gets its own copy of `cur`.
        args = [
            (cur, c[0], c[1], c[2], k_ring, feasibility_thr, maxiter, ftol, use_analytical_jacobian)
            for c in batch
        ]
        # Constant pool size: get_pool() respawns the warm pool whenever the
        # requested size changes, and batch sizes vary — pool_map only
        # dispatches len(args) tasks, so idle workers are free.
        results = pool_map(_run_one_cluster, args, n_workers)
        for deltas, info in results:
            infos.append(info)
            trial = cur.copy()
            for (z, y, x), (dz, dy, dx) in deltas:
                trial[0, z, y, x] = dz
                trial[1, z, y, x] = dy
                trial[2, z, y, x] = dx
            if accept_check:
                cur = _accept_or_reject(cur, trial)
            else:
                cur = trial
    return cur, infos


def _partition_non_overlapping(centres, k_ring):
    """Greedy partition: build batches whose halos don't overlap.

    Two centres' halos overlap if their Chebyshev distance is
    ``<= 2 * k_ring`` (each halo extends ``k_ring`` cubes around its
    centre). The greedy partition assigns each centre to the lowest-
    indexed batch with no overlapping centre.
    """
    batches: list[list[tuple[int, int, int]]] = []
    overlap_radius = 2 * k_ring
    for c in centres:
        placed = False
        for batch in batches:
            ok = True
            for other in batch:
                if (
                    max(abs(c[0] - other[0]), abs(c[1] - other[1]), abs(c[2] - other[2]))
                    <= overlap_radius
                ):
                    ok = False
                    break
            if ok:
                batch.append(c)
                placed = True
                break
        if not placed:
            batches.append([c])
    return batches


def _accept_or_reject(cur, trial):
    """Accept `trial` only if it doesn't increase global n_neg more than
    the local SLSQP can plausibly fix. Otherwise keep `cur`.

    Counts use the fused per-CUBE ``six_tet_min_volume_3d`` kernel (a cube
    with any of its six tets folded counts once) instead of materialising
    the (6, D-1, H-1, W-1) volume array and counting per-TET. Both sides of
    the comparison changed semantics consistently; the +5 slack is a
    heuristic and is kept as-is.
    """
    n_cur = int((six_tet_min_volume_3d(cur) <= 0).sum())
    n_trial = int((six_tet_min_volume_3d(trial) <= 0).sum())
    # Allow small regressions (SLSQP can introduce small boundary leaks
    # that subsequent recovery cleans up); reject only large ones.
    if n_trial <= n_cur + 5:
        return trial
    return cur
