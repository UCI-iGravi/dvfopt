"""Convergence plot for a :class:`~dvfopt.solver.SolveInfo`.

Every Strategy returns a ``SolveInfo`` whose ``phases`` carry the
per-phase feasibility trace (``n_neg``, ``min_T``, ``wall_s``). This
module renders that trace uniformly across strategies, so any
``record_history=True`` run is visualizable in one line::

    result = Solver(...).fit(phi, record_history=True)
    from dvfopt.viz import plot_solve_info
    plot_solve_info(result.info, threshold=0.01)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.figure import Figure

from dvfopt.viz.theme import PALETTE, apply_theme


def plot_solve_info(
    info,
    *,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
    figsize: tuple = (9, 5.5),
    save_path: Optional[str] = None,
) -> Figure:
    """Two-panel convergence plot of a strategy run.

    Top panel: ``min_T`` vs cumulative wall time (with the feasibility
    ``threshold`` as a horizontal line when given). Bottom panel:
    ``n_neg`` on a symlog axis. Phase boundaries are marked and labeled
    with the phase names, so multi-stage pipelines (harmonic → ALM →
    polish, penalty → barrier, ...) read as segmented curves.

    Parameters
    ----------
    info : SolveInfo
        As returned by ``Solver.fit(..., record_history=True)`` (via
        ``result.info``) or any Strategy's ``solve``.
    threshold : float, optional
        Feasibility threshold to draw on the ``min_T`` panel.
    title : str, optional
        Suptitle; defaults to the strategy name.
    figsize : tuple
    save_path : str, optional
        When given, the figure is also saved here.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    apply_theme()

    phases = [p for p in getattr(info, 'phases', []) if p is not None]
    fig, (ax_t, ax_n) = plt.subplots(2, 1, sharex=True, figsize=figsize)

    if not phases:
        ax_t.text(
            0.5,
            0.5,
            'no phase history recorded\n(run with record_history=True)',
            ha='center',
            va='center',
            transform=ax_t.transAxes,
        )
        ax_n.set_xlabel('phase')
        if title or getattr(info, 'strategy_name', ''):
            fig.suptitle(title or info.strategy_name)
        return fig

    # X-axis is the PHASE INDEX, not accumulated wall_s: producers are
    # inconsistent about wall_s semantics (the barrier core records
    # per-step durations, several wallbreaker stage dicts record
    # cumulative elapsed-since-start), so summing them would inflate the
    # axis for some strategies. Each phase's own wall_s is shown in its
    # tick label instead.
    x = np.arange(len(phases))
    min_t = np.array([float(p.min_T) for p in phases])
    n_neg = np.array([float(p.n_neg) for p in phases])
    labels = [f'{p.name}\n{float(p.wall_s):.2g}s' for p in phases]

    # ---- min_T panel -------------------------------------------------
    ax_t.plot(x, min_t, marker='o', color=PALETTE.blue, label='min T')
    if threshold is not None:
        ax_t.axhline(threshold, color=PALETTE.feasible, linestyle='--', label=f'thr={threshold:g}')
    ax_t.axhline(0.0, color=PALETTE.gray, linewidth=0.6)
    ax_t.set_ylabel('min constraint value')
    ax_t.grid(True, axis='y')
    ax_t.legend(loc='lower right')

    # ---- n_neg panel -------------------------------------------------
    known = n_neg >= 0  # -1 = "not recorded for this phase"
    ax_n.plot(x[known], n_neg[known], marker='o', color=PALETTE.red, label='n_neg')
    ax_n.set_yscale('symlog', linthresh=1)
    ax_n.set_ylim(bottom=-0.5)
    ax_n.set_ylabel('violated cells')
    ax_n.set_xlabel('phase')
    ax_n.set_xticks(x)
    ax_n.set_xticklabels(labels, fontsize='xx-small', rotation=30, ha='right')
    ax_n.grid(True, axis='y')

    feas_idx = getattr(info, 'feasible_after_phase', -1)
    if 0 <= feas_idx < len(x):
        ax_n.axvline(x[feas_idx], color=PALETTE.feasible, linewidth=1.2, label='first feasible')
        ax_n.legend(loc='upper right')

    fig.suptitle(title or getattr(info, 'strategy_name', ''))
    if save_path is not None:
        fig.savefig(save_path)
    return fig


__all__ = ['plot_solve_info']
