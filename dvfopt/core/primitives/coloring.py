"""CPR (Curtis–Powell–Reid) coloring helpers over a constraint's adjoint —
recover the full sparse constraint Jacobian in one adjoint call per COLOUR
instead of one per row. Shared by the isqp solver, the windowed engine, and
the benchmarks harnesses. Duck-typed on ``.adjoint`` / ``.n_constraints`` /
``.n_variables`` / ``.shape`` (no ``dvfopt.constraints`` import)."""

import numpy as np


def dense_jacobian(constraint, flat):
    """Dense (m, n) constraint Jacobian at *flat*: row i = adjoint(flat, e_i).

    dvfopt exposes the constraint's adjoint (Jᵀv) but not J itself; applying it to
    each unit constraint vector recovers the exact rows. Only tractable for small
    constraint counts (crops / windows).
    """
    m = constraint.n_constraints
    eye = np.eye(m)
    return np.stack([constraint.adjoint(flat, eye[i]) for i in range(m)])


def colored_jacobian(constraint, flat, pattern, colors, stride=None):
    """Sparse (m, n) constraint Jacobian via CPR coloring — one adjoint call per
    COLOUR instead of one per row.

    Constraints sharing a colour have DISJOINT variable supports, so probing the
    adjoint with their indicator returns each of their rows superposed on
    non-overlapping columns, and ``colvals[pattern[r]]`` recovers row ``r`` exactly.
    ``pattern``/``colors`` are precomputed once per grid shape (Jdet: pixel grid,
    ``(i%3)*3+j%3``; 2-tri: cell grid, ``triangle*4 + (i%2)*2 + j%2``). ``stride`` is
    accepted for back-compat but ignored — the colour count is ``colors.max()+1``.
    Returns a ``scipy.sparse`` CSC matrix.
    """
    from scipy import sparse

    m, n = constraint.n_constraints, constraint.n_variables
    rows, cols, vals = [], [], []
    for cid in range(int(colors.max()) + 1):
        grp = np.nonzero(colors == cid)[0]
        if grp.size == 0:
            continue
        v = np.zeros(m)
        v[grp] = 1.0
        colvals = constraint.adjoint(flat, v)  # length n; disjoint supports per grp
        for r in grp:
            pr = pattern[r]
            rows.append(np.full(pr.size, r))
            cols.append(pr)
            vals.append(colvals[pr])
    return sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, n)
    )


def jacobian_coloring(constraint, flat0, stride=3, probes=4, seed=0):
    """Precompute (pattern, colors, stride) for :func:`colored_jacobian`.

    ``pattern[r]`` = the nonzero column indices of Jacobian row ``r``, taken as the
    UNION of nonzeros over ``probes`` random perturbations of ``flat0`` (a single
    point can accidentally zero a structurally-nonzero entry, which then corrupts
    the coloring). ``colors[r]`` = ``(i%stride)*stride + j%stride`` over the
    ``H*W`` constraint grid. stride 3 is exact for the radius-1 Jdet stencil.
    """
    h, w = constraint.shape
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(probes):
        b = np.abs(dense_jacobian(constraint, flat0 + rng.normal(0, 0.4, flat0.size))) > 0
        acc = b if acc is None else (acc | b)
    pattern = [np.nonzero(acc[r])[0] for r in range(acc.shape[0])]
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    colors = ((ii % stride) * stride + (jj % stride)).ravel()
    return pattern, colors, stride


__all__ = ['colored_jacobian', 'dense_jacobian', 'jacobian_coloring']
