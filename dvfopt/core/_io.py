"""IO / setup helpers shared by every iterative solver.

* :func:`_setup_accumulators` — five-tuple of empty containers a loop fills.
* :func:`_init_phi` — pull (dy, dx) channels out of a ``(3, 1, H, W)`` deformation.
* :func:`_print_summary` — end-of-run human-readable block.
* :func:`_save_results` — checkpoint a run to disk for later inspection.

These were originally bundled in ``dvfopt/core/solver.py`` — they stayed
re-exported from there for backward compatibility with existing imports.
"""

import os
from collections import defaultdict

import numpy as np

from dvfopt._defaults import _log


def _setup_accumulators():
    """Return the five tracking structures used by every iterative loop."""
    return [], [], [], [], defaultdict(int)
    # error_list, num_neg_jac, iter_times, min_jdet_list, window_counts


def _init_phi(deformation_i):
    """Create the initial ``phi`` working array from a ``(3, 1, H, W)`` deformation.

    Returns ``(phi, phi_init, H, W)``.
    """
    H, W = deformation_i.shape[-2:]
    phi = np.zeros((2, H, W))
    phi[1] = deformation_i[-1]
    phi[0] = deformation_i[-2]
    phi_init = phi.copy()
    return phi, phi_init, H, W


def _print_summary(verbose, method_label, grid_shape, iteration,
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed, extra_lines=""):
    """Print the end-of-run summary block shared by all iterative solvers."""
    grid_str = " x ".join(str(d) for d in grid_shape)
    _log(verbose, 1, "")
    _log(verbose, 1, "=" * 60)
    _log(verbose, 1, f"  SUMMARY  ({method_label})")
    _log(verbose, 1, "-" * 60)
    _log(verbose, 1, f"  Grid size        : {grid_str}")
    iter_line = f"  Iterations       : {iteration}"
    if extra_lines:
        iter_line += f"  {extra_lines}"
    _log(verbose, 1, iter_line)
    _log(verbose, 1, f"  Neg-Jdet  {init_neg:>5d} -> {final_neg:>5d}")
    _log(verbose, 1, f"  Min Jdet  {init_min:+.6f} -> {final_min:+.6f}")
    _log(verbose, 1, f"  L2 error         : {final_err:.6f}")
    _log(verbose, 1, f"  Time             : {elapsed:.2f}s")
    _log(verbose, 1, "=" * 60)


def _save_results(save_path, *, method, threshold, err_tol, max_iterations,
                  max_per_index_iter, max_minimize_iter,
                  grid_shape, elapsed, final_err, init_neg, final_neg,
                  init_min, final_min, iteration, phi, error_list,
                  num_neg_jac, iter_times, min_jdet_list, window_counts,
                  extra_settings="", extra_results=""):
    """Write correction results to *save_path*.

    Parameters
    ----------
    grid_shape : tuple
        ``(H, W)`` for 2D or ``(D, H, W)`` for 3D.
    """
    os.makedirs(save_path, exist_ok=True)

    ndim = len(grid_shape)
    if ndim == 2:
        res_label = "height x width"
        dim_names = ["window_height", "window_width"]
    else:
        res_label = "D x H x W"
        dim_names = ["window_depth", "window_height", "window_width"]
    res_str = " x ".join(str(d) for d in grid_shape)

    output_text = "Settings:\n"
    output_text += f"\tMethod: {method}\n"
    output_text += f"\tThreshold: {threshold}\n"
    output_text += f"\tError tolerance: {err_tol}\n"
    output_text += f"\tMax iterations: {max_iterations}\n"
    output_text += f"\tMax per index iterations: {max_per_index_iter}\n"
    output_text += f"\tMax minimize iterations: {max_minimize_iter}\n"
    if extra_settings:
        output_text += extra_settings
    output_text += "\nResults:\n"
    output_text += f"\tInput deformation field resolution ({res_label}): {res_str}\n"
    output_text += f"\tTotal run-time: {elapsed} seconds\n"
    output_text += f"\tFinal L2 error: {final_err}\n"
    output_text += f"\tStarting number of non-positive Jacobian determinants: {init_neg}\n"
    output_text += f"\tFinal number of non-positive Jacobian determinants: {final_neg}\n"
    output_text += f"\tStarting Jacobian determinant minimum value: {init_min}\n"
    output_text += f"\tFinal Jacobian determinant minimum value: {final_min}\n"
    output_text += f"\tNumber of index iterations: {iteration}"
    if extra_results:
        output_text += "\n" + extra_results

    with open(os.path.join(save_path, "results.txt"), "w") as f:
        f.write(output_text)

    np.save(os.path.join(save_path, "phi.npy"), phi)
    np.save(os.path.join(save_path, "error_list.npy"), error_list)
    np.save(os.path.join(save_path, "num_neg_jac.npy"), num_neg_jac)
    np.save(os.path.join(save_path, "iter_times.npy"), iter_times)
    np.save(os.path.join(save_path, "min_jdet_list.npy"), min_jdet_list)

    csv_header = ",".join(dim_names) + ",count\n"
    with open(os.path.join(save_path, "window_counts.csv"), "w") as f:
        f.write(csv_header)
        for ws in sorted(window_counts):
            dims = ws if isinstance(ws, tuple) else (ws,)
            f.write(",".join(str(d) for d in dims) + f",{window_counts[ws]}\n")
