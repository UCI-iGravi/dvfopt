"""Plot helpers for :class:`dvfopt.unified.Result`.

These were inline methods on ``Result`` (~150 LOC) — extracted here so
``unified.py`` focuses on dispatch/result composition and the plotting
machinery (which pulls in matplotlib) lives in its own module.

Each function takes a ``Result`` and returns nothing — they call
``plt.show()`` internally for inline notebook display.

The functions are wired back onto ``Result`` as methods in
``unified.py``; users continue to call ``result.plot_convergence(z=0)``.
"""

from __future__ import annotations

import numpy as np

from dvfopt.jacobian.shoelace import _ref_grid
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _extract_2d_slice(deformation, z):
    """Return a (2, H, W) [dy, dx] slice from any of the supported shapes."""
    if deformation.ndim == 2:
        raise ValueError('input must be at least 3D (channels + spatial)')
    if deformation.ndim == 3:
        if deformation.shape[0] == 2:  # (2, H, W)
            return deformation
        if deformation.shape[0] == 3:  # (3, H, W)
            return np.stack([deformation[1], deformation[2]])
    if deformation.ndim == 4 and deformation.shape[0] == 3:  # (3, D, H, W)
        return np.stack([deformation[1, z], deformation[2, z]])
    raise ValueError(f'unsupported deformation shape {deformation.shape}')


def _compute_constraint_2d(phi2, kind):
    """Constraint values as a (n_constraints,) ndarray for plotting.

    No corner patches — plot code reshapes T1/T2 onto an (H-1, W-1)
    cell grid, which the patches would break.
    """
    from dvfopt.core.primitives.constraint_values import compute_constraint_values_2d

    return compute_constraint_values_2d(phi2, kind, include_patches=False)


def plot_convergence(result, z=None, ax=None):
    """Plot n_neg and min_tri vs iteration for one slice (or all)."""
    import matplotlib.pyplot as plt

    slices = result.slice_results if z is None else [s for s in result.slice_results if s.z == z]
    if not slices:
        raise ValueError(f'no slice with z={z}')
    _fig, (a1, a2) = (
        plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True) if ax is None else (None, ax)
    )
    for s in slices:
        if not s.history:
            continue
        it = list(range(len(s.history)))
        n = [h.get('n_neg', np.nan) for h in s.history]
        m = [h.get('min_tri', h.get('min_J', h.get('min_T', np.nan))) for h in s.history]
        a1.plot(it, n, marker='o', label=f'z={s.z}')
        a2.plot(it, m, marker='o', label=f'z={s.z}')
    a1.set_yscale('symlog', linthresh=1)
    a1.set_xlabel('step')
    a1.set_ylabel('n_neg (symlog)')
    a1.set_title('folds vs iteration')
    a1.axhline(0, color='k', lw=0.5)
    a1.grid(alpha=0.3)
    a2.set_xlabel('step')
    a2.set_ylabel('min constraint value')
    a2.set_title('min constraint vs iteration')
    a2.axhline(
        result.config.threshold,
        color='#1b8a3a',
        ls='--',
        label=f'threshold {result.config.threshold}',
    )
    a2.axhline(0, color='k', lw=0.5)
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    if ax is None:
        plt.show()


def plot_feasibility(result, z=0, snapshot=-1, ax=None):
    """Visualize the constraint field (T or Jdet) for slice z."""
    import matplotlib.pyplot as plt

    sr = next((s for s in result.slice_results if s.z == z), None)
    if sr is None:
        raise ValueError(f'no slice with z={z}')
    if sr.snapshots:
        snap = sr.snapshots[snapshot]
        T = snap['T']
        tag = snap['tag']
    else:
        phi2 = _extract_2d_slice(result.corrected, z)
        T = _compute_constraint_2d(phi2, result.config.constraint)
        tag = 'final'
    thr = result.config.threshold
    n_neg = int((T <= 0).sum())
    min_val = float(T.min())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    if result.config.constraint in ('2tri', '2tri_standard') and T.ndim == 1:
        n_cells = T.size // 2
        T1 = T[:n_cells]
        T2 = T[n_cells:]
        phi2 = _extract_2d_slice(result.corrected, z)
        H, W = phi2.shape[1], phi2.shape[2]
        T1 = T1.reshape(H - 1, W - 1)
        T2 = T2.reshape(H - 1, W - 1)
        tmap = np.minimum(T1, T2)
    else:
        phi2 = _extract_2d_slice(result.corrected, z)
        H, W = phi2.shape[1], phi2.shape[2]
        tmap = T.reshape(H - 1, W - 1) if T.ndim == 1 else T
    vmax = max(abs(tmap.min()), 1.5 * thr, 0.05)
    im = a1.imshow(tmap, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    a1.set_title(f'z={z} {tag}: min_constraint={min_val:+.4f}  n_below_0={n_neg}')
    a1.set_xticks([])
    a1.set_yticks([])
    fig.colorbar(im, ax=a1, shrink=0.85)
    flat = T.ravel()
    bins = np.linspace(min(flat.min(), -0.05), max(0.05, thr * 2), 80)
    a2.hist(flat, bins=bins, color='#5b7fb5', edgecolor='none')
    a2.axvline(0, color='k', lw=0.6, label='T = 0 (fold boundary)')
    a2.axvline(thr, color='#1b8a3a', lw=1.2, ls='--', label=f'threshold ({thr})')
    a2.set_xlabel('constraint value')
    a2.set_ylabel('# cells')
    a2.set_yscale('log')
    a2.set_title('distribution + feasibility wall')
    a2.legend(fontsize=9)
    plt.show()


def plot_gradient_region(result, z=0, ax=None):
    """Visualize the constraint-gradient *magnitude* at each cell.

    Shows where the optimization landscape is fragile (small ``||grad T||``
    indicates near-degenerate triangles, where SLSQP's active-set line
    search degenerates).
    """
    import matplotlib.pyplot as plt

    phi2 = _extract_2d_slice(result.corrected, z)
    H, W = phi2.shape[1], phi2.shape[2]
    if result.config.constraint not in ('2tri', '2tri_standard'):
        raise ValueError('plot_gradient_region only implemented for 2tri')
    ref_y, ref_x = _ref_grid(H, W)
    dy, dx = phi2[0], phi2[1]
    def_x = ref_x + dx
    def_y = ref_y + dy
    x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
    x_tr, y_tr = def_x[:-1, 1:], def_y[:-1, 1:]
    x_bl, y_bl = def_x[1:, :-1], def_y[1:, :-1]
    x_br, y_br = def_x[1:, 1:], def_y[1:, 1:]

    bc_sq = (x_br - x_bl) ** 2 + (y_br - y_bl) ** 2
    ca_sq = (x_tr - x_br) ** 2 + (y_tr - y_br) ** 2
    ab_sq = (x_bl - x_tr) ** 2 + (y_bl - y_tr) ** 2
    norm_T1 = 0.5 * np.sqrt(bc_sq + ca_sq + ab_sq)
    bc_sq = (x_tr - x_bl) ** 2 + (y_tr - y_bl) ** 2
    ca_sq = (x_tl - x_tr) ** 2 + (y_tl - y_tr) ** 2
    ab_sq = (x_bl - x_tl) ** 2 + (y_bl - y_tl) ** 2
    norm_T2 = 0.5 * np.sqrt(bc_sq + ca_sq + ab_sq)

    T1, T2 = _triangle_areas_2d(dy, dx)
    with np.errstate(divide='ignore', invalid='ignore'):
        stiffness = np.minimum(
            np.where(norm_T1 > 1e-9, np.abs(T1) / norm_T1, 0.0),
            np.where(norm_T2 > 1e-9, np.abs(T2) / norm_T2, 0.0),
        )

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    cell_min = np.minimum(T1, T2)
    vm = max(abs(cell_min.min()), 0.05)
    im0 = a1.imshow(cell_min, cmap='RdBu_r', vmin=-vm, vmax=vm)
    a1.set_title(f'z={z}: min(T1, T2)')
    a1.set_xticks([])
    a1.set_yticks([])
    fig.colorbar(im0, ax=a1, shrink=0.85)
    im1 = a2.imshow(stiffness, cmap='magma')
    a2.set_title('|T| / ||grad T||  (Newton step magnitude)\nhigher = stiffer / harder for SLSQP')
    a2.set_xticks([])
    a2.set_yticks([])
    fig.colorbar(im1, ax=a2, shrink=0.85)
    plt.show()


__all__ = [
    'plot_convergence',
    'plot_feasibility',
    'plot_gradient_region',
]
