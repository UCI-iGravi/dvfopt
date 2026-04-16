"""Shared plotting, metric, and output utilities for benchmark notebooks."""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from dvfopt import jacobian_det2D, jacobian_det3D


# ---------------------------------------------------------------------------
# Output directory management
# ---------------------------------------------------------------------------

def get_output_dir(method, notebook_name, base="output"):
    """Create and return an output directory for a benchmark run.

    Structure: ``output/<method>/<notebook_name>/``

    Parameters
    ----------
    method : str
        Solver category, e.g. ``"slsqp"``, ``"barrier"``, ``"barrier-gpu"``.
    notebook_name : str
        Short identifier for the notebook, e.g. ``"scalability"``.
    base : str
        Root output folder (relative to cwd or absolute).

    Returns
    -------
    pathlib.Path
    """
    out = Path(base) / method / notebook_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig, output_dir, name, dpi=150, close=False):
    """Save a matplotlib figure as PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    output_dir : str or Path
    name : str
        Filename without extension.
    dpi : int
    close : bool
        If True, close the figure after saving.
    """
    path = Path(output_dir) / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  [saved] {path}")
    if close:
        plt.close(fig)


def save_results_csv(rows, columns, output_dir, name="results"):
    """Save tabular results as CSV.

    Parameters
    ----------
    rows : list[dict] or list[list]
        Each row is a dict (keys = column names) or a list of values.
    columns : list[str]
        Column headers.
    output_dir : str or Path
    name : str
        Filename without extension.
    """
    path = Path(output_dir) / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns) if isinstance(rows[0], dict) else csv.writer(f)
        if isinstance(rows[0], dict):
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer.writerow(columns)
            writer.writerows(rows)
    print(f"  [saved] {path}")


def save_summary_json(data, output_dir, name="summary"):
    """Save a JSON summary of the benchmark run.

    Parameters
    ----------
    data : dict
        Arbitrary JSON-serialisable data.
    output_dir : str or Path
    name : str
        Filename without extension.
    """
    path = Path(output_dir) / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [saved] {path}")


def log_run_header(method, notebook_name, output_dir, extra=None):
    """Print a standardised run header and return a dict for the summary.

    Parameters
    ----------
    method : str
    notebook_name : str
    output_dir : Path or str
    extra : dict, optional
        Additional key-value pairs to include.

    Returns
    -------
    dict  — summary skeleton (fill in results later).
    """
    ts = datetime.now().isoformat(timespec="seconds")
    print("=" * 72)
    print(f"  Benchmark  : {notebook_name}")
    print(f"  Method     : {method}")
    print(f"  Timestamp  : {ts}")
    print(f"  Output dir : {output_dir}")
    if extra:
        for k, v in extra.items():
            print(f"  {k:<11s}: {v}")
    print("=" * 72)
    return {"benchmark": notebook_name, "method": method, "timestamp": ts,
            "output_dir": str(output_dir), **(extra or {})}


def log_run_footer(summary, results):
    """Print a standardised footer and return the completed summary.

    Parameters
    ----------
    summary : dict
        From ``log_run_header``.
    results : dict
        Keyed by case label; each value should have at least
        ``n_neg_init``, ``n_neg_final``, ``min_jdet``, ``l2_err``, ``time``.
    """
    total_cases = len(results)
    converged = sum(1 for r in results.values() if r.get("n_neg_final", 1) == 0)
    total_time = sum(r.get("time", 0) for r in results.values())
    print()
    print("-" * 72)
    print(f"  Cases: {total_cases}  |  Converged: {converged}/{total_cases}  "
          f"|  Total time: {total_time:.2f}s")
    print("-" * 72)
    summary["total_cases"] = total_cases
    summary["converged"] = converged
    summary["total_time_s"] = round(total_time, 2)
    return summary


def show_and_save(output_dir, name=None, fig=None, dpi=150):
    """Save the current figure as PNG and then call ``plt.show()``.

    Drop-in replacement for ``plt.show()`` that also persists the figure.

    Parameters
    ----------
    output_dir : str or Path
        Where to save the PNG.
    name : str, optional
        Filename stem.  Auto-increments ``figure_01``, ``figure_02``, ...
        if omitted.
    fig : Figure, optional
        Defaults to ``plt.gcf()``.
    dpi : int
    """
    if fig is None:
        fig = plt.gcf()
    if name is None:
        show_and_save._counter = getattr(show_and_save, "_counter", 0) + 1
        name = f"figure_{show_and_save._counter:02d}"
    save_figure(fig, output_dir, name, dpi=dpi)
    plt.show()


def reset_figure_counter():
    """Reset the auto-increment counter used by ``show_and_save``."""
    show_and_save._counter = 0


def results_to_rows(results, extra_cols=None):
    """Convert a results dict to a list of flat dicts for CSV export.

    Parameters
    ----------
    results : dict[str, dict]
        Keyed by label. Values must contain the standard metric keys.
    extra_cols : list[str], optional
        Additional keys to extract from each result dict.

    Returns
    -------
    rows : list[dict]
    columns : list[str]
    """
    base_cols = ["case", "n_neg_init", "n_neg_final", "min_jdet_init",
                 "min_jdet", "l2_err", "time"]
    extra = extra_cols or []
    columns = base_cols + extra
    rows = []
    for label, r in results.items():
        row = {"case": label}
        for c in base_cols[1:] + extra:
            val = r.get(c)
            if isinstance(val, float):
                val = round(val, 6)
            row[c] = val
        rows.append(row)
    return rows, columns


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def run_correction(dvf, solver, verbose=0, **solver_kwargs):
    """Run a Jacobian correction solver and collect standard metrics.

    Parameters
    ----------
    dvf : ndarray, shape (3, 1, H, W)
        Deformation field (channels [dz, dy, dx]).
    solver : callable
        Solver function, e.g. ``iterative_parallel`` or
        ``iterative_serial``.  Called as
        ``solver(dvf, verbose=verbose, **solver_kwargs)``.
    verbose : int
        Passed to the solver.
    **solver_kwargs
        Extra keyword arguments forwarded to the solver.

    Returns
    -------
    dict with keys:
        phi_init, phi          – (2, H, W) arrays
        jac_init, jac_final    – (1, H, W) Jacobian determinant arrays
        time                   – wall-clock seconds
        n_neg_init, n_neg_final – negative-Jdet pixel counts
        min_jdet_init, min_jdet – worst Jacobian determinant values
        l2_err                 – L2 norm of (phi - phi_init)
    """
    phi_init = np.stack([dvf[-2, 0], dvf[-1, 0]])
    jac_init = jacobian_det2D(phi_init)

    t0 = time.perf_counter()
    phi = solver(dvf.copy(), verbose=verbose, **solver_kwargs)
    elapsed = time.perf_counter() - t0

    jac_final = jacobian_det2D(phi)

    return {
        "phi_init": phi_init,
        "phi": phi,
        "jac_init": jac_init,
        "jac_final": jac_final,
        "time": elapsed,
        "n_neg_init": int((jac_init <= 0).sum()),
        "n_neg_final": int((jac_final <= 0).sum()),
        "min_jdet_init": float(jac_init.min()),
        "min_jdet": float(jac_final.min()),
        "l2_err": float(np.sqrt(np.sum((phi - phi_init) ** 2))),
    }


def run_correction_3d(dvf, solver, verbose=0, **solver_kwargs):
    """Run a 3D Jacobian correction solver and collect standard metrics.

    Parameters
    ----------
    dvf : ndarray, shape (3, D, H, W)
        3D deformation field with channels ``[dz, dy, dx]``.
    solver : callable
        3D solver, e.g. ``iterative_3d``.  Called as
        ``solver(dvf, verbose=verbose, **solver_kwargs)``.

    Returns
    -------
    dict with keys:
        phi_init, phi          – (3, D, H, W) arrays
        jac_init, jac_final    – (D, H, W) Jacobian determinant arrays
        time                   – wall-clock seconds
        n_neg_init, n_neg_final – negative-Jdet voxel counts
        min_jdet_init, min_jdet – worst Jacobian determinant values
        l2_err                 – L2 norm of (phi - phi_init)
    """
    phi_init = dvf.copy().astype(np.float64)
    jac_init = jacobian_det3D(phi_init)

    t0 = time.perf_counter()
    phi = solver(dvf.copy(), verbose=verbose, **solver_kwargs)
    elapsed = time.perf_counter() - t0

    jac_final = jacobian_det3D(phi)

    return {
        "phi_init": phi_init,
        "phi": phi,
        "jac_init": jac_init,
        "jac_final": jac_final,
        "time": elapsed,
        "n_neg_init": int((jac_init <= 0).sum()),
        "n_neg_final": int((jac_final <= 0).sum()),
        "min_jdet_init": float(jac_init.min()),
        "min_jdet": float(jac_final.min()),
        "l2_err": float(np.sqrt(np.sum((phi - phi_init) ** 2))),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_jac_heatmaps(jac_grid, col_labels, row_labels=("Before", "After"),
                      title=None, figscale=2.5):
    """Grid of Jacobian determinant heatmaps with diverging colormap.

    Parameters
    ----------
    jac_grid : list[list[ndarray]]
        ``jac_grid[row][col]`` is a **2-D** Jacobian determinant array.
    col_labels : list[str]
        Column header for each test case.
    row_labels : list[str]
        Row label (y-axis) for each condition (default Before / After).
    title : str, optional
        Figure suptitle.
    figscale : float
        Approximate inches per subplot side.
    """
    n_rows = len(jac_grid)
    n_cols = len(jac_grid[0])

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figscale * n_cols, figscale * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    all_vals = np.concatenate([jac_grid[r][c].ravel()
                               for r in range(n_rows) for c in range(n_cols)])
    vmin = min(float(all_vals.min()), -0.01)
    vmax = max(float(all_vals.max()), 0.01)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    font = min(11, max(8, 120 // max(n_cols, 1)))
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            jac = jac_grid[r][c]
            im = ax.imshow(jac, cmap="RdBu_r", norm=norm, origin="upper")
            if (jac <= 0).any():
                ax.contour(jac, levels=[0], colors="black", linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=font)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=11)

    fig.colorbar(im, ax=axes, label="Jacobian determinant", shrink=0.8)
    if title:
        plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def plot_correction_magnitude(phi_pairs, labels, title=None, figscale=2.5):
    """Heatmaps of per-pixel correction magnitude ``|phi - phi_init|``.

    Parameters
    ----------
    phi_pairs : list[tuple[ndarray, ndarray]]
        Each element is ``(phi_corrected, phi_init)`` with shape ``(2, H, W)``.
    labels : list[str]
        Title for each subplot.
    title : str, optional
        Figure suptitle.
    figscale : float
        Approximate inches per subplot width.
    """
    n = len(phi_pairs)
    fig, axes = plt.subplots(1, n, figsize=(figscale * n, 3))
    if n == 1:
        axes = [axes]

    font = min(10, max(8, 100 // max(n, 1)))
    for i, (phi, phi_init) in enumerate(phi_pairs):
        ax = axes[i]
        diff = np.sqrt(((phi - phi_init) ** 2).sum(axis=0))
        im = ax.imshow(diff, cmap="hot", origin="upper")
        ax.set_title(labels[i], fontsize=font)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def plot_jdet_histograms(jac_groups, labels, title=None, figscale=2.5,
                         colors=None):
    """Overlaid Jacobian determinant distribution histograms.

    Parameters
    ----------
    jac_groups : list[list[tuple[str, ndarray]]]
        For each subplot, a list of ``(series_name, jac_2d_array)`` tuples.
    labels : list[str]
        Title for each subplot.
    title : str, optional
        Figure suptitle.
    figscale : float
        Approximate inches per subplot width.
    colors : list[str], optional
        Colours for each series (cycles if shorter than series count).
        Defaults to ``["tab:red", "tab:blue", ...]``.
    """
    default_colors = ["tab:red", "tab:blue", "tab:gray", "tab:green",
                      "tab:orange"]
    if colors is None:
        colors = default_colors

    n = len(jac_groups)
    fig, axes = plt.subplots(1, n, figsize=(figscale * n, 3), sharey=True)
    if n == 1:
        axes = [axes]

    font = min(10, max(8, 100 // max(n, 1)))
    for i, group in enumerate(jac_groups):
        ax = axes[i]
        all_vals = np.concatenate([j.ravel() for _, j in group])
        lo = float(all_vals.min()) - 0.1
        hi = float(all_vals.max()) + 0.1
        bins = np.linspace(lo, hi, 40)
        for j, (name, jac) in enumerate(group):
            ax.hist(jac.ravel(), bins=bins, alpha=0.5, label=name,
                    color=colors[j % len(colors)])
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(labels[i], fontsize=font)
        ax.set_xlabel("Jdet")
        if i == 0:
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

    if title:
        plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig
