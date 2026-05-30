"""Shared plotting, metric, and output utilities for benchmark notebooks."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from dvfopt import jacobian_det2D, jacobian_det3D
from dvfopt._defaults import DEFAULT_PARAMS

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
        if not rows:
            writer = csv.writer(f)
            writer.writerow(columns)
        elif isinstance(rows[0], dict):
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
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
    return {
        "benchmark": notebook_name,
        "method": method,
        "timestamp": ts,
        "output_dir": str(output_dir),
        **(extra or {}),
    }


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
    print(
        f"  Cases: {total_cases}  |  Converged: {converged}/{total_cases}  "
        f"|  Total time: {total_time:.2f}s"
    )
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
    base_cols = ["case", "n_neg_init", "n_neg_final", "min_jdet_init", "min_jdet", "l2_err", "time"]
    extra = extra_cols or []
    # Some notebooks store the standard CSV metrics under shorter keys such as
    # ``neg`` and ``l2`` inside per-method result dicts.
    aliases = {
        "n_neg_final": ("n_neg_final", "neg"),
        "l2_err": ("l2_err", "l2"),
    }
    # These keys identify nested method result payloads (for example
    # ``windowed``/``fullgrid``) rather than shared metadata like ``jac_init``.
    metric_keys = {"min_jdet", "time", "n_neg_final", "neg", "l2_err", "l2"}

    def _round_if_float(value):
        if isinstance(value, float):
            return round(value, 6)
        return value

    def _is_method_result(payload):
        return isinstance(payload, dict) and any(key in payload for key in metric_keys)

    def _get_value(payload, key):
        for candidate in aliases.get(key, (key,)):
            if candidate in payload:
                return payload[candidate]
        if key == "min_jdet_init" and "jac_init" in payload:
            return float(np.min(payload["jac_init"]))
        if key == "n_neg_init" and "jac_init" in payload:
            return int(np.sum(payload["jac_init"] < DEFAULT_PARAMS["threshold"]))
        return None

    include_method = False
    rows = []
    for label, r in results.items():
        # Split method-specific result dicts from shared case-level metadata.
        method_results = {name: value for name, value in r.items() if _is_method_result(value)}
        if method_results:
            include_method = True
            # Shared metadata is copied into every per-method CSV row.
            common_data = {name: value for name, value in r.items() if name not in method_results}
            for method, payload in method_results.items():
                merged = {**common_data, **payload}
                row = {"case": label, "method": method}
                for c in base_cols[1:] + extra:
                    row[c] = _round_if_float(_get_value(merged, c))
                rows.append(row)
            continue

        row = {"case": label}
        for c in base_cols[1:] + extra:
            row[c] = _round_if_float(_get_value(r, c))
        rows.append(row)

    columns = ["case", "method", *base_cols[1:], *extra] if include_method else base_cols + extra
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


def plot_jac_heatmaps(
    jac_grid, col_labels, row_labels=("Before", "After"), title=None, figscale=2.5
):
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figscale * n_cols, figscale * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    all_vals = np.concatenate(
        [jac_grid[r][c].ravel() for r in range(n_rows) for c in range(n_cols)]
    )
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


def plot_jdet_histograms(jac_groups, labels, title=None, figscale=2.5, colors=None):
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
    default_colors = ["tab:red", "tab:blue", "tab:gray", "tab:green", "tab:orange"]
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
            ax.hist(jac.ravel(), bins=bins, alpha=0.5, label=name, color=colors[j % len(colors)])
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


# ---------------------------------------------------------------------------
# Canonical 2-triangle benchmark harness
# ---------------------------------------------------------------------------


def benchmark_canonical_2tri_2d(
    method_fn, *, label=None, threshold=None, err_tol=1e-5, verbose=True
):
    """Run ``method_fn`` over every canonical 2D 2-tri case and report results.

    Parameters
    ----------
    method_fn : callable
        ``phi_2hw -> phi_corrected_2hw``. Must accept and return arrays of
        shape ``(2, H, W)`` with channels ``[dy, dx]``. The function may
        raise; failures are caught and recorded in the row.
    label : str, optional
        Method label used in printed output. Inferred from ``method_fn``
        if not provided.
    threshold : float, optional
        Lower bound used for the feasibility check. Defaults to
        ``DEFAULT_PARAMS['threshold']``.
    err_tol : float, optional
        Tolerance for the feasibility check
        (``min_T >= threshold - err_tol``).
    verbose : bool
        If True, print a per-case progress line.

    Returns
    -------
    rows : list of dict
        One dict per case with keys:
        ``case, shape, init_n_neg, init_min_T,
         final_n_neg, final_min_T, feasible,
         wall_s, l1, l2, error``.

    Example
    -------
    >>> from dvfopt import iterative_2d_tri_barrier
    >>> rows = benchmark_canonical_2tri_2d(
    ...     lambda p: iterative_2d_tri_barrier(p, verbose=0),
    ...     label='barrier_l2')
    """
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
    from test_cases import canonical_2tri_2d

    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    if label is None:
        label = getattr(method_fn, '__name__', 'method')

    rows = []
    for name, phi_in, meta in canonical_2tri_2d():
        t0 = time.perf_counter()
        err = ''
        try:
            phi_out = method_fn(phi_in.copy())
        except Exception as exc:
            wall = time.perf_counter() - t0
            err = f'{type(exc).__name__}: {exc}'
            phi_out = None
        else:
            wall = time.perf_counter() - t0

        row = dict(
            case=name,
            shape=tuple(meta['shape']),
            method=label,
            init_n_neg=meta['init_n_neg'],
            init_min_T=meta['init_min_T'],
            wall_s=wall,
            error=err,
        )
        if phi_out is None:
            row.update(
                final_n_neg=-1,
                final_min_T=float('nan'),
                feasible=False,
                l1=float('nan'),
                l2=float('nan'),
            )
        else:
            T1, T2 = _triangle_areas_2d(phi_out[0], phi_out[1])
            n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
            min_T = float(min(T1.min(), T2.min()))
            diff = (phi_out - phi_in).ravel()
            row.update(
                final_n_neg=n_neg,
                final_min_T=min_T,
                feasible=(n_neg == 0 and min_T >= threshold - err_tol),
                l1=float(np.abs(diff).sum()),
                l2=float(np.sqrt(np.dot(diff, diff))),
            )
        rows.append(row)
        if verbose:
            tag = 'OK' if row.get('feasible') else ('ERR' if err else 'FAIL')
            print(
                f'  [{tag:>4}] {name:<22} {label:<22} '
                f'wall={row["wall_s"]:6.2f}s  '
                f'n_neg={row["final_n_neg"]:>4}  '
                f'min_T={row["final_min_T"]:+.4f}  '
                f'L1={row["l1"]:>8.3f}  L2={row["l2"]:>7.3f}' + (f'  {err}' if err else '')
            )
    return rows


def benchmark_methods_table(methods, *, threshold=None, verbose=True):
    """Convenience: run several methods over the canonical suite and return
    a single flat list of result rows.

    Parameters
    ----------
    methods : list of (str, callable)
        ``(label, method_fn)`` pairs.
    threshold, verbose
        Forwarded to :func:`benchmark_canonical_2tri_2d`.

    Returns
    -------
    list of dict
        Concatenated rows from every method.
    """
    out = []
    for label, fn in methods:
        if verbose:
            print(f'=== {label} ===')
        out.extend(
            benchmark_canonical_2tri_2d(fn, label=label, threshold=threshold, verbose=verbose)
        )
    return out
