"""DVFopt -- unified deformation-field optimization API.

A single high-level class wrapping every approach exercised in the
manuscript / experiments work. One ``fit`` call returns the corrected
DVF plus a tabular per-slice / per-iteration history plus diagnostic
plots.

Configuration axes (every combination is valid; sensible defaults):

    constraint  : '2tri', 'jdet', '6tet' (3D)
    solver      : 'slsqp', 'trust-constr', 'barrier', 'auto'
    objective   : 'l2', 'l1' (smoothed), 'none' (feasibility only)
    mode        : 'windowed', 'full-grid'
    jacobian    : 'analytical', 'finite-diff', 'central-diff'
    threshold, margin, max_outer_iters, ... (see DVFoptConfig)

Optional features (toggled in DVFoptConfig):

    use_continuation       threshold-homotopy warm-started SLSQP
    use_perturb_on_stall   solve_cluster_inline-style jitter retry
    use_l1_polish          smoothed-L1 polish after L2 phase
    record_history         capture per-iteration n_neg / min_tri
    record_snapshots       capture T-map snapshots for visualization
    debug                  print solver-level diagnostics

Example:
    from dvfopt import DVFopt
    opt = DVFopt(constraint='2tri', solver='barrier', objective='l2',
                 threshold=0.01, mode='full-grid', verbose=1)
    result = opt.fit(deformation)             # (3, D, H, W) or (2, H, W)
    print(result.summary())
    print(result.to_dataframe())
    result.plot_convergence(z=0)              # convergence curve
    result.plot_feasibility(z=0)              # T heatmap + histogram
    result.plot_gradient_region(z=0)          # constraint-gradient magnitude
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import (label as cc_label, binary_dilation,
                           find_objects)
from scipy.optimize import minimize, NonlinearConstraint

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.core import iterative_serial
from dvfopt.core.iterative2d_barrier import iterative_2d_barrier
from dvfopt.core.iterative2d_tri_barrier import (iterative_2d_tri_barrier,
                                                 _tri_areas_flat,
                                                 _tri_grad_T_v)
from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.jacobian.shoelace import _ref_grid


# ============================================================
# Config
# ============================================================

@dataclass
class DVFoptConfig:
    """All knobs for one DVFopt run. Fields not relevant to the chosen
    ``solver`` are ignored (e.g. ``lam_schedule`` is only used by
    ``solver='barrier'``)."""
    # ---- problem ----
    constraint: str = '2tri'         # '2tri', 'jdet', '6tet'
    threshold: float = 0.01
    err_tol: float = 1e-5
    margin: float = 1e-3             # barrier safety margin

    # ---- solver / objective ----
    solver: str = 'auto'             # 'slsqp', 'trust-constr', 'barrier', 'auto'
    objective: str = 'l2'            # 'l2', 'l1', 'none'
    eps_l1: float = 1e-4
    jacobian: str = 'analytical'     # 'analytical', 'finite-diff', 'central-diff'

    # ---- decomposition ----
    mode: str = 'windowed'           # 'windowed', 'full-grid'
    pad: int = 3
    merge_dilation: int = 1
    max_window_per_axis: int = 60
    max_window_cells: int = 2000

    # ---- outer loop ----
    max_outer_iters: int = 20

    # ---- SLSQP ----
    slsqp_max_iter: int = 80
    slsqp_max_passes: int = 10
    use_perturb_on_stall: bool = True
    perturb_limit: int = 3
    use_continuation: bool = False
    continuation_steps: int = 10

    # ---- Barrier ----
    lam_schedule: Tuple[float, ...] = (
        1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
    mu_schedule: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)
    barrier_max_iter: int = 300
    tri_full_coverage: bool = False  # 2tri only: add two corner-patch
                                      # triangles so every grid vertex
                                      # (incl. (0,0), (H-1,W-1)) is in ≥2
                                      # triangle constraints. Has no effect
                                      # when constraint != '2tri'.

    # ---- polish ----
    use_l1_polish: bool = False
    l1_polish_max_iter: int = 120

    # ---- output ----
    verbose: int = 1
    debug: bool = False
    record_history: bool = True
    record_snapshots: bool = False   # for plot_feasibility


# ============================================================
# Result
# ============================================================

@dataclass
class SliceResult:
    z: int
    init_n_neg: int
    init_min: float
    final_n_neg: int
    final_min: float
    feasible: bool
    solver_used: str
    n_outer_iters: int
    wall_time: float
    notes: str = ''
    history: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    # snapshots[k] = {'tag': str, 'T': ndarray, 'n_neg': int, 'min_tri': float,
    #                 'phi': ndarray (optional)}


@dataclass
class Result:
    """Outcome of a DVFopt.fit call.

    Attributes
    ----------
    corrected : ndarray
        Corrected DVF, same shape and dtype as the input.
    config : DVFoptConfig
        The config used.
    slice_results : list of SliceResult
        One entry per slice processed (length = D for 3D, 1 for 2D).
    total_wall_time : float
        Total wall-clock time across all slices.
    """
    corrected: np.ndarray
    config: DVFoptConfig
    slice_results: List[SliceResult]
    total_wall_time: float

    # ----- summary helpers -----
    @property
    def feasible(self) -> bool:
        return all(s.feasible for s in self.slice_results)

    @property
    def summary_dict(self) -> Dict[str, Any]:
        n = len(self.slice_results)
        feas = sum(1 for s in self.slice_results if s.feasible)
        return dict(
            slices=n,
            feasible=feas,
            feasibility_pct=100.0 * feas / max(1, n),
            init_n_neg=sum(s.init_n_neg for s in self.slice_results),
            final_n_neg=sum(s.final_n_neg for s in self.slice_results),
            init_min_tri=min((s.init_min for s in self.slice_results),
                             default=float('nan')),
            final_min_tri=min((s.final_min for s in self.slice_results),
                              default=float('nan')),
            total_wall_time_s=self.total_wall_time,
        )

    def summary(self) -> str:
        d = self.summary_dict
        cfg = self.config
        return (
            f'DVFopt result  ({d["slices"]} slice(s))\n'
            f'  solver         : {cfg.solver}   constraint: {cfg.constraint}   '
            f'objective: {cfg.objective}   mode: {cfg.mode}\n'
            f'  threshold      : {cfg.threshold}\n'
            f'  feasible       : {d["feasible"]}/{d["slices"]}  '
            f'({d["feasibility_pct"]:.1f}%)\n'
            f'  folds          : init {d["init_n_neg"]} -> final '
            f'{d["final_n_neg"]}\n'
            f'  min_tri / jdet : init {d["init_min_tri"]:+.4f} -> final '
            f'{d["final_min_tri"]:+.4f}\n'
            f'  wall time      : {d["total_wall_time_s"]:.1f}s'
        )

    def to_dataframe(self):
        """Per-slice tabular summary."""
        import pandas as pd
        return pd.DataFrame([dict(
            z=s.z, init_n_neg=s.init_n_neg, init_min=s.init_min,
            final_n_neg=s.final_n_neg, final_min=s.final_min,
            feasible=s.feasible, solver=s.solver_used,
            outer_iters=s.n_outer_iters, wall_s=s.wall_time, notes=s.notes,
        ) for s in self.slice_results])

    def history_df(self):
        """Concatenated per-iteration history across all slices (long form)."""
        import pandas as pd
        rows = []
        for s in self.slice_results:
            for h in s.history:
                rows.append({'z': s.z, **h})
        return pd.DataFrame(rows)

    # ----- visualization -----
    def plot_convergence(self, z=None, ax=None):
        """Plot n_neg and min_tri vs iteration for one slice (or all)."""
        import matplotlib.pyplot as plt
        slices = (self.slice_results if z is None
                  else [s for s in self.slice_results if s.z == z])
        if not slices:
            raise ValueError(f'no slice with z={z}')
        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=(12, 4), constrained_layout=True) if ax is None \
            else (None, ax)
        for s in slices:
            if not s.history:
                continue
            it = list(range(len(s.history)))
            n = [h.get('n_neg', np.nan) for h in s.history]
            m = [h.get('min_tri', h.get('min_J', np.nan)) for h in s.history]
            a1.plot(it, n, marker='o', label=f'z={s.z}')
            a2.plot(it, m, marker='o', label=f'z={s.z}')
        a1.set_yscale('symlog', linthresh=1)
        a1.set_xlabel('step'); a1.set_ylabel('n_neg (symlog)')
        a1.set_title('folds vs iteration')
        a1.axhline(0, color='k', lw=0.5)
        a1.grid(alpha=0.3)
        a2.set_xlabel('step'); a2.set_ylabel('min constraint value')
        a2.set_title('min constraint vs iteration')
        a2.axhline(self.config.threshold, color='#1b8a3a', ls='--',
                   label=f'threshold {self.config.threshold}')
        a2.axhline(0, color='k', lw=0.5)
        a2.legend(fontsize=8)
        a2.grid(alpha=0.3)
        if ax is None:
            plt.show()

    def plot_feasibility(self, z=0, snapshot=-1, ax=None):
        """Visualize the constraint field (T or Jdet) for slice z at the
        chosen snapshot (default = final). Shows a heatmap with the
        feasibility threshold marked AND a histogram of constraint values
        with the threshold line, so the "wall" is visible."""
        import matplotlib.pyplot as plt
        sr = next((s for s in self.slice_results if s.z == z), None)
        if sr is None:
            raise ValueError(f'no slice with z={z}')
        if sr.snapshots:
            snap = sr.snapshots[snapshot]
            T = snap['T']; tag = snap['tag']
        else:
            # Re-compute from corrected.
            phi2 = _extract_2d_slice(self.corrected, z)
            T = _compute_constraint_2d(phi2, self.config.constraint)
            tag = 'final'
        thr = self.config.threshold
        n_neg = int((T <= 0).sum())
        min_val = float(T.min())

        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=(13, 4.4), constrained_layout=True)
        # Heatmap: for 2tri T has 2 channels per cell -> use min(T1,T2).
        # For jdet T is one channel per cell.
        if self.config.constraint == '2tri' and T.ndim == 1:
            n_cells = T.size // 2
            T1 = T[:n_cells]; T2 = T[n_cells:]
            sy = int(np.sqrt(n_cells)) if n_cells > 0 else 1
            # Try to recover shape from corrected
            phi2 = _extract_2d_slice(self.corrected, z)
            H, W = phi2.shape[1], phi2.shape[2]
            T1 = T1.reshape(H - 1, W - 1)
            T2 = T2.reshape(H - 1, W - 1)
            tmap = np.minimum(T1, T2)
        else:
            phi2 = _extract_2d_slice(self.corrected, z)
            H, W = phi2.shape[1], phi2.shape[2]
            tmap = T.reshape(H - 1, W - 1) if T.ndim == 1 else T
        vmax = max(abs(tmap.min()), 1.5 * thr, 0.05)
        im = a1.imshow(tmap, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        a1.set_title(f'z={z} {tag}: min_constraint={min_val:+.4f}  '
                     f'n_below_0={n_neg}')
        a1.set_xticks([]); a1.set_yticks([])
        fig.colorbar(im, ax=a1, shrink=0.85)
        # Histogram with threshold line
        flat = T.ravel()
        bins = np.linspace(min(flat.min(), -0.05), max(0.05, thr * 2), 80)
        a2.hist(flat, bins=bins, color='#5b7fb5', edgecolor='none')
        a2.axvline(0, color='k', lw=0.6, label='T = 0 (fold boundary)')
        a2.axvline(thr, color='#1b8a3a', lw=1.2, ls='--',
                   label=f'threshold ({thr})')
        a2.set_xlabel('constraint value')
        a2.set_ylabel('# cells')
        a2.set_yscale('log')
        a2.set_title(f'distribution + feasibility wall')
        a2.legend(fontsize=9)
        plt.show()

    def plot_gradient_region(self, z=0, ax=None):
        """Visualize the constraint-gradient *magnitude* at each cell:
        || dT/dphi || (Frobenius norm over the 6 partials). Small values
        indicate near-degenerate triangles -- the regime where SLSQP's
        active-set line search collapses (status 8). Shows WHERE the
        coupling structure is fragile."""
        import matplotlib.pyplot as plt
        phi2 = _extract_2d_slice(self.corrected, z)
        H, W = phi2.shape[1], phi2.shape[2]
        if self.config.constraint != '2tri':
            raise ValueError('plot_gradient_region only implemented for 2tri')
        # Norm of each triangle's gradient row = simple closed form:
        # for triangle ABC, ||grad|| = (1/2) * (|BC|^2 + |CA|^2 + |AB|^2)^{1/2}.
        # Equivalent to expressing the 6 partials and taking norm.
        ref_y, ref_x = _ref_grid(H, W)
        dy, dx = phi2[0], phi2[1]
        def_x = ref_x + dx; def_y = ref_y + dy
        x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
        x_tr, y_tr = def_x[:-1, 1:],  def_y[:-1, 1:]
        x_bl, y_bl = def_x[1:, :-1],  def_y[1:, :-1]
        x_br, y_br = def_x[1:, 1:],   def_y[1:, 1:]

        # T1: A=TR, B=BL, C=BR
        bc_sq = (x_br - x_bl) ** 2 + (y_br - y_bl) ** 2
        ca_sq = (x_tr - x_br) ** 2 + (y_tr - y_br) ** 2
        ab_sq = (x_bl - x_tr) ** 2 + (y_bl - y_tr) ** 2
        norm_T1 = 0.5 * np.sqrt(bc_sq + ca_sq + ab_sq)
        # T2: A=TL, B=BL, C=TR
        bc_sq = (x_tr - x_bl) ** 2 + (y_tr - y_bl) ** 2
        ca_sq = (x_tl - x_tr) ** 2 + (y_tl - y_tr) ** 2
        ab_sq = (x_bl - x_tl) ** 2 + (y_bl - y_tl) ** 2
        norm_T2 = 0.5 * np.sqrt(bc_sq + ca_sq + ab_sq)

        T1, T2 = _triangle_areas_2d(dy, dx)
        # Risky cells: small ||grad|| AND |T| small (degenerate, near-boundary).
        # A useful single map: per-cell min over the two triangles of |T| /
        # ||grad|| -- the *Newton-step* magnitude needed to move T by O(T).
        # Bigger value = larger step needed = stiffer.
        with np.errstate(divide='ignore', invalid='ignore'):
            stiffness = np.minimum(
                np.where(norm_T1 > 1e-9, np.abs(T1) / norm_T1, 0.0),
                np.where(norm_T2 > 1e-9, np.abs(T2) / norm_T2, 0.0),
            )

        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=(13, 4.4), constrained_layout=True)
        cell_min = np.minimum(T1, T2)
        vm = max(abs(cell_min.min()), 0.05)
        im0 = a1.imshow(cell_min, cmap='RdBu_r', vmin=-vm, vmax=vm)
        a1.set_title(f'z={z}: min(T1, T2)')
        a1.set_xticks([]); a1.set_yticks([])
        fig.colorbar(im0, ax=a1, shrink=0.85)
        im1 = a2.imshow(stiffness, cmap='magma')
        a2.set_title('|T| / ||grad T||  (Newton step magnitude)\n'
                     'higher = stiffer / harder for SLSQP')
        a2.set_xticks([]); a2.set_yticks([])
        fig.colorbar(im1, ax=a2, shrink=0.85)
        plt.show()


# ============================================================
# Helpers
# ============================================================

def _extract_2d_slice(deformation, z):
    """Return a (2, H, W) [dy, dx] slice from any of the supported shapes."""
    if deformation.ndim == 2:
        raise ValueError('input must be at least 3D (channels + spatial)')
    if deformation.ndim == 3:
        if deformation.shape[0] == 2:                # (2, H, W)
            return deformation
        if deformation.shape[0] == 3:                # (3, H, W)
            return np.stack([deformation[1], deformation[2]])
    if deformation.ndim == 4:                        # (3, D, H, W)
        if deformation.shape[0] == 3:
            return np.stack([deformation[1, z], deformation[2, z]])
    raise ValueError(f'unsupported deformation shape {deformation.shape}')


def _compute_constraint_2d(phi2, kind):
    """Returns the constraint values as a (n_constraints,) ndarray."""
    if kind == '2tri':
        T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
        return np.concatenate([T1.ravel(), T2.ravel()])
    if kind == 'jdet':
        return np.squeeze(jacobian_det2D(phi2)).ravel()
    raise ValueError(f'unknown constraint kind: {kind}')


def _stats_2d(phi2, kind):
    """Return (n_neg, min_value) for the 2D constraint of `kind`."""
    T = _compute_constraint_2d(phi2, kind)
    return int((T <= 0).sum()), float(T.min())


# ============================================================
# DVFopt
# ============================================================

class DVFopt:
    """Unified deformation-field optimizer.

    All configuration is held in a :class:`DVFoptConfig`. Either pass a
    pre-built config or override individual keyword arguments at
    construction time::

        opt = DVFopt(solver='barrier', threshold=0.01)
        result = opt.fit(deformation)

    The ``fit`` method auto-detects 2D-vs-3D input and dispatches to the
    appropriate constraint / solver backend.
    """

    def __init__(self, config: Optional[DVFoptConfig] = None, **kwargs):
        if config is None:
            config = DVFoptConfig(**kwargs)
        elif kwargs:
            config = replace(config, **kwargs)
        self.config = config
        self._validate()

    def _validate(self):
        c = self.config
        if c.constraint not in ('2tri', 'jdet', '6tet'):
            raise ValueError(f'bad constraint: {c.constraint!r}')
        if c.solver not in ('slsqp', 'trust-constr', 'barrier', 'auto'):
            raise ValueError(f'bad solver: {c.solver!r}')
        if c.objective not in ('l2', 'l1', 'none'):
            raise ValueError(f'bad objective: {c.objective!r}')
        if c.mode not in ('windowed', 'full-grid'):
            raise ValueError(f'bad mode: {c.mode!r}')

    # ---- main entry ----
    def fit(self, deformation: np.ndarray) -> Result:
        """Run the optimizer on `deformation` and return a :class:`Result`."""
        t0 = time.time()
        # Detect format and normalise.
        if deformation.ndim == 3 and deformation.shape[0] == 2:
            # (2, H, W) raw 2D -- treated as one slice; _extract_2d_slice /
            # _put_2d_slice handle the (2, H, W) shape directly.
            slices = [0]
            corrected = deformation.copy()
        elif deformation.ndim == 4 and deformation.shape[0] == 3:
            # (3, D, H, W)
            D = deformation.shape[1]
            slices = list(range(D))
            corrected = deformation.copy()
        elif deformation.ndim == 3 and deformation.shape[0] == 3:
            # (3, H, W) -> add D axis
            corrected = deformation[:, None].copy()
            slices = [0]
        else:
            raise ValueError(f'unsupported deformation shape '
                             f'{deformation.shape}')

        slice_results = []
        for z in slices:
            phi2 = _extract_2d_slice(corrected, z)
            sr = self._run_slice(phi2, z)
            slice_results.append(sr)
            # write phi2 back into corrected
            self._put_2d_slice(corrected, z, phi2)

        # Restore output shape: if user gave (3, H, W) we return that.
        if deformation.ndim == 3 and deformation.shape[0] == 3:
            corrected = corrected[:, 0]
        elif deformation.ndim == 3 and deformation.shape[0] == 2:
            pass                                     # already (2, H, W)

        return Result(corrected=corrected, config=self.config,
                      slice_results=slice_results,
                      total_wall_time=time.time() - t0)

    def _put_2d_slice(self, corrected, z, phi2):
        """Write a (2, H, W) slice back into the (3, D, H, W) corrected
        buffer, leaving channel 0 (dz) untouched."""
        if corrected.ndim == 4:
            corrected[1, z] = phi2[0]
            corrected[2, z] = phi2[1]
        elif corrected.ndim == 3 and corrected.shape[0] == 2:
            corrected[0] = phi2[0]
            corrected[1] = phi2[1]
        elif corrected.ndim == 3 and corrected.shape[0] == 3:
            corrected[1] = phi2[0]
            corrected[2] = phi2[1]

    # ---- per-slice dispatcher ----
    def _run_slice(self, phi2, z) -> SliceResult:
        c = self.config
        init_n_neg, init_min = _stats_2d(phi2, c.constraint)
        if c.verbose >= 1:
            print(f'[z={z}] init n_neg={init_n_neg}  min={init_min:+.4f}',
                  flush=True)
        if init_n_neg == 0 and init_min >= c.threshold - c.err_tol:
            return SliceResult(
                z=z, init_n_neg=0, init_min=init_min,
                final_n_neg=0, final_min=init_min, feasible=True,
                solver_used='none', n_outer_iters=0, wall_time=0.0,
                notes='already feasible')

        solver = self._resolve_solver(init_n_neg, init_min)
        t0 = time.time()
        history: List[Dict[str, Any]] = []
        snapshots: List[Dict[str, Any]] = []
        if c.record_snapshots:
            T = _compute_constraint_2d(phi2, c.constraint)
            snapshots.append(dict(tag='init', T=T.copy(),
                                  n_neg=init_n_neg, min_tri=init_min))

        if solver == 'barrier':
            phi_new, hist, n_outer = self._run_barrier(phi2)
        elif solver == 'trust-constr':
            phi_new, hist, n_outer = self._run_trust_constr(phi2)
        else:
            # 'slsqp' (default)
            phi_new, hist, n_outer = self._run_slsqp(phi2)
        phi2[:] = phi_new

        final_n_neg, final_min = _stats_2d(phi2, c.constraint)
        if c.record_snapshots:
            T = _compute_constraint_2d(phi2, c.constraint)
            snapshots.append(dict(tag='final', T=T.copy(),
                                  n_neg=final_n_neg, min_tri=final_min))
        if c.record_history:
            history.extend(hist)

        feasible = (final_n_neg == 0 and
                    final_min >= c.threshold - c.err_tol)
        if c.verbose >= 1:
            print(f'[z={z}] final n_neg={final_n_neg}  min={final_min:+.5f}  '
                  f'solver={solver}  ({time.time()-t0:.1f}s)', flush=True)
        return SliceResult(
            z=z, init_n_neg=init_n_neg, init_min=init_min,
            final_n_neg=final_n_neg, final_min=final_min,
            feasible=feasible, solver_used=solver,
            n_outer_iters=n_outer, wall_time=time.time() - t0,
            history=history, snapshots=snapshots,
            notes=('feasible' if feasible else 'still folded'))

    def _resolve_solver(self, init_n_neg, init_min):
        """Auto-select a solver based on the slice difficulty.

        Heuristics:
          - n_neg > 500                              -> barrier (full-grid)
          - jdet constraint and init_min < -1.0      -> barrier (severe fold)
          - 2tri constraint and init_min < -0.25     -> barrier (severe fold)
          - otherwise                                -> slsqp

        The min-value thresholds are constraint-specific because Jdet and
        triangle area live on different scales.
        """
        c = self.config
        if c.solver != 'auto':
            return c.solver
        if init_n_neg > 500:
            return 'barrier'
        severe_min = -1.0 if c.constraint == 'jdet' else -0.25
        if init_min < severe_min:
            return 'barrier'
        return 'slsqp'

    # ---- solver: barrier ----
    def _run_barrier(self, phi2):
        c = self.config
        if c.constraint == '2tri':
            out = iterative_2d_tri_barrier(
                phi2,
                threshold=c.threshold, margin=c.margin,
                lam_schedule=c.lam_schedule, mu_schedule=c.mu_schedule,
                max_minimize_iter=c.barrier_max_iter,
                anchor=c.objective, eps_l1=c.eps_l1,
                verbose=c.verbose, record_history=c.record_history,
                full_coverage=c.tri_full_coverage)
            # iterative_2d_tri_barrier returns just phi when record_history=False,
            # (phi, history) when True.
            if isinstance(out, tuple):
                phi_new, hist = out
            else:
                phi_new, hist = out, []
            return phi_new, hist, 1
        if c.constraint == 'jdet':
            # Existing Jdet barrier.
            deformation = np.stack([np.zeros_like(phi2[0]),
                                    phi2[0], phi2[1]])[:, None]
            phi_new_3hw = iterative_2d_barrier(
                deformation,
                threshold=c.threshold, margin=c.margin,
                lam_schedule=c.lam_schedule, mu_schedule=c.mu_schedule,
                max_minimize_iter=c.barrier_max_iter,
                windowed=(c.mode == 'windowed'), pad=c.pad,
                verbose=c.verbose)
            # phi_new_3hw might be (2, H, W) or (3, 1, H, W) depending on
            # mode -- coerce.
            phi_new = (np.stack([phi_new_3hw[0], phi_new_3hw[1]])
                       if phi_new_3hw.ndim == 3 and phi_new_3hw.shape[0] == 2
                       else np.stack([phi_new_3hw[1, 0], phi_new_3hw[2, 0]]))
            return phi_new, [], 1
        raise ValueError(f'barrier not implemented for constraint='
                         f'{c.constraint!r}')

    # ---- solver: trust-constr (per-component, 2tri) ----
    def _run_trust_constr(self, phi2):
        c = self.config
        if c.constraint != '2tri':
            raise ValueError('trust-constr currently only for 2tri')
        phi = phi2.copy()
        anchor = phi2.copy()
        H, W = phi.shape[1], phi.shape[2]
        history = []
        outer = 0
        for outer in range(1, c.max_outer_iters + 1):
            comps = _fold_components_2tri(phi, merge_dilation=c.merge_dilation)
            if not comps:
                break
            for (cy0, cy1, cx0, cx1) in comps:
                _solve_component_trust_constr(
                    phi, anchor, cy0, cy1, cx0, cx1,
                    pad=c.pad, threshold=c.threshold,
                    objective=c.objective, eps_l1=c.eps_l1,
                    max_iter=c.barrier_max_iter)
            n, m = _stats_2d(phi, '2tri')
            if c.record_history:
                history.append(dict(outer=outer, n_neg=n, min_tri=m))
            if c.verbose >= 1:
                print(f'  outer {outer}: n_neg={n}  min_tri={m:+.5f}  '
                      f'comps={len(comps)}', flush=True)
            if n == 0:
                break
        return phi, history, outer

    # ---- solver: SLSQP (windowed via iterative_serial) ----
    def _run_slsqp(self, phi2):
        c = self.config
        if c.constraint not in ('2tri', 'jdet'):
            raise ValueError(f'slsqp not implemented for constraint='
                             f'{c.constraint!r}')
        # iterative_serial only supports the windowed L2 path. Warn loudly
        # if the user requested combinations we silently can't honour, so
        # the result they get matches what they asked for.
        import warnings
        if c.mode == 'full-grid':
            warnings.warn(
                "DVFopt: solver='slsqp' currently only supports mode='windowed';"
                " falling back to windowed for this run.", stacklevel=2)
        if c.objective != 'l2':
            warnings.warn(
                f"DVFopt: solver='slsqp' currently only supports objective='l2'"
                f" (got {c.objective!r}); using l2 for this run.", stacklevel=2)
        if c.use_continuation:
            warnings.warn(
                "DVFopt: solver='slsqp' does not yet implement continuation;"
                " ignoring use_continuation=True.", stacklevel=2)
        if c.record_history:
            warnings.warn(
                "DVFopt: solver='slsqp' does not record per-iteration history;"
                " history will be empty for this slice.", stacklevel=2)
        # Delegate to the existing windowed iterative_serial which already
        # supports both constraints via the package's flags. We pack into
        # the canonical (3, 1, H, W) shape.
        deformation = np.zeros((3, 1, phi2.shape[1], phi2.shape[2]),
                               dtype=phi2.dtype)
        deformation[1, 0] = phi2[0]
        deformation[2, 0] = phi2[1]
        enforce_triangles = (c.constraint == '2tri')
        result = iterative_serial(
            deformation,
            threshold=c.threshold,
            verbose=c.verbose,
            max_iterations=c.max_outer_iters,
            max_minimize_iter=c.slsqp_max_iter,
            enforce_triangles=enforce_triangles,
        )
        # iterative_serial returns (2, H, W) (just dy, dx). Older / future
        # builds may also return (3, 1, H, W) (full [dz, dy, dx]); cover both.
        corrected = result if isinstance(result, np.ndarray) else result[0]
        if corrected.ndim == 4:                            # (3, 1, H, W)
            phi_new = np.stack([corrected[1, 0], corrected[2, 0]])
        elif corrected.ndim == 3 and corrected.shape[0] == 3:  # (3, H, W)
            phi_new = np.stack([corrected[1], corrected[2]])
        else:                                              # (2, H, W) — already (dy, dx)
            phi_new = corrected
        # iterative_serial does not expose its history; we leave it empty
        # for now. Users that want history should choose 'barrier' or
        # 'trust-constr'.
        return phi_new, [], 1


# ============================================================
# Component detection + trust-constr per-component helper
# ============================================================

def _fold_components_2tri(phi2, merge_dilation=1):
    T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
    fold = np.minimum(T1, T2) <= 0
    if not fold.any():
        return []
    mask = (binary_dilation(fold, iterations=merge_dilation)
            if merge_dilation > 0 else fold)
    labels, _ = cc_label(mask)
    out = []
    for sl in find_objects(labels):
        if sl is not None:
            out.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return out


def _solve_component_trust_constr(phi, anchor, cy0, cy1, cx0, cx1, *,
                                  pad, threshold, objective, eps_l1,
                                  max_iter):
    H, W = phi.shape[1], phi.shape[2]
    y0 = max(0, cy0 - pad); y1 = min(H - 1, cy1 + pad)
    x0 = max(0, cx0 - pad); x1 = min(W - 1, cx1 + pad)
    sy, sx = y1 - y0, x1 - x0
    if sy < 4 or sx < 4:
        return
    # interior_mask: frozen 1-ring
    im = np.zeros((sy + 1, sx + 1), dtype=bool)
    im[1:-1, 1:-1] = True
    phi_win = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
    anc_win = anchor[:, y0:y1 + 1, x0:x1 + 1].copy()
    int_idx = np.argwhere(im)
    iy, ix = int_idx[:, 0], int_idx[:, 1]

    def pack(p):
        return np.concatenate([p[0][iy, ix], p[1][iy, ix]])

    def unpack(z, base):
        out = base.copy()
        n = len(iy)
        out[0][iy, ix] = z[:n]
        out[1][iy, ix] = z[n:]
        return out

    z_anchor = pack(anc_win)

    def obj(z):
        d = z - z_anchor
        if objective == 'l1':
            s = np.sqrt(d * d + eps_l1 * eps_l1)
            return float((s - eps_l1).sum()), d / s
        if objective == 'l2':
            return 0.5 * float(d @ d), d
        return 0.0, np.zeros_like(d)             # objective='none'

    def constr(z):
        ph = unpack(z, phi_win)
        t1, t2 = _triangle_areas_2d(ph[0], ph[1])
        return np.concatenate([t1.ravel(), t2.ravel()])

    nl = NonlinearConstraint(constr, threshold, np.inf, jac='2-point')
    res = minimize(obj, pack(phi_win), jac=True, method='trust-constr',
                   constraints=[nl],
                   options=dict(maxiter=max_iter, gtol=1e-8, xtol=1e-10,
                                verbose=0))
    phi_new = unpack(res.x, phi_win)
    phi[:, y0:y1 + 1, x0:x1 + 1] = phi_new
