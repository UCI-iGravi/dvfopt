"""
Iterative SLSQP optimisation for correcting negative Jacobian determinants
in 2D deformation (displacement) fields.

Usage:
    from modules.dvfopt import iterative_with_jacobians2, plot_deformations, run_lapl_and_correction
"""

import time
from collections import defaultdict
from pprint import pprint

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

import modules.jacobian as jacobian
import modules.laplacian as laplacian


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "threshold": 0.01,
    "err_tol": 1e-5,
    "max_iterations": 10000,
    "max_per_index_iter": 50,
    "max_minimize_iter": 1000,
    "starting_window_size": 7,
}


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------
def objectiveEuc(phi, phi_init):
    """L2 norm objective function."""
    return np.linalg.norm(phi - phi_init)


def objectiveManh(phi, phi_init):
    """L1 norm objective function."""
    return np.linalg.norm(phi - phi_init, ord=1)


# ---------------------------------------------------------------------------
# Jacobian helpers
# ---------------------------------------------------------------------------
def compute_jacobian_determinant(deformation):
    """
    Compute the Jacobian determinant of a 2D deformation field (z,y,x) using:
    - Central differences for internal pixels
    - Forward/backward differences at edges
    Returns: (1, H, W) Jacobian determinant

    NOTE: This is a reference implementation. The main code uses
    ``jacobian.sitk_jacobian_determinant`` instead.
    """
    _, _, H, W = deformation.shape
    dy = deformation[1, 0]
    dx = deformation[2, 0]

    def gradient_central_with_fallback(f):
        df_dx = np.zeros_like(f)
        df_dy = np.zeros_like(f)
        # Central
        df_dx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0
        df_dy[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0
        # Forward (left/top edge)
        df_dx[:, 0] = f[:, 1] - f[:, 0]
        df_dy[0, :] = f[1, :] - f[0, :]
        # Backward (right/bottom edge)
        df_dx[:, -1] = f[:, -1] - f[:, -2]
        df_dy[-1, :] = f[-1, :] - f[-2, :]
        return df_dx, df_dy

    dy_dx, dy_dy = gradient_central_with_fallback(dy)
    dx_dx, dx_dy = gradient_central_with_fallback(dx)

    det_J = (1 + dx_dx) * (1 + dy_dy) - dx_dy * dy_dx
    det_J = det_J[np.newaxis, :, :]
    return det_J


def jacobian_det2D(phi_xy):
    """Compute the Jacobian determinant from a ``(2, H, W)`` phi array."""
    deformation = np.zeros((3, 1, *phi_xy.shape[-2:]))
    deformation[2] = phi_xy[1]
    deformation[1] = phi_xy[0]
    return jacobian.sitk_jacobian_determinant(deformation)


def jacobian_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened Jacobian determinant values for optimiser constraints."""
    deformation = np.zeros((3, 1, submatrix_size, submatrix_size))
    pixels = submatrix_size * submatrix_size
    deformation[2] = phi_xy[:pixels].reshape((submatrix_size, submatrix_size))
    deformation[1] = phi_xy[pixels:].reshape((submatrix_size, submatrix_size))
    jacobian_mat = jacobian.sitk_jacobian_determinant(deformation)
    if exclude_boundaries:
        return jacobian_mat[0, 1:-1, 1:-1].flatten()
    else:
        return jacobian_mat.flatten()


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
def nearest_center(shape, n):
    """Build a dict mapping every (z,y,x) to the nearest valid sub-window centre."""
    near_cent = {}
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                y_cent = y
                x_cent = x
                if x - n < 0:
                    x_cent = n
                elif shape[2] - x < n + 1:
                    x_cent = shape[2] - n - 1
                if y - n < 0:
                    y_cent = n
                elif shape[1] - y < n + 1:
                    y_cent = shape[1] - n - 1
                near_cent[(z, y, x)] = [z, y_cent, x_cent]
    return near_cent


def get_nearest_center(neg_index, slice_shape, submatrix_size, near_cent_dict):
    """Look up (or compute) the nearest valid centre for *neg_index*."""
    if submatrix_size in near_cent_dict:
        return near_cent_dict[submatrix_size][(0, *neg_index)]
    else:
        near_cent = nearest_center(slice_shape, submatrix_size // 2)
        near_cent_dict[submatrix_size] = near_cent
        return near_cent[(0, *neg_index)]


def argmin_excluding_edges(jacobian_matrix):
    """Index of the pixel with the lowest Jacobian determinant, excluding edges."""
    inner = jacobian_matrix[0, 1:-1, 1:-1]
    flat_index = np.argmin(inner)
    inner_idx = np.unravel_index(flat_index, inner.shape)
    return (inner_idx[0] + 1, inner_idx[1] + 1)


def get_phi_sub_flat(phi, cz, cy, cx, shape, d):
    """Extract and flatten a square sub-window of *phi* around (cy, cx)."""
    phix = phi[1, cy - d:cy + d + 1, cx - d:cx + d + 1]
    phiy = phi[0, cy - d:cy + d + 1, cx - d:cx + d + 1]
    return np.concatenate([phix.flatten(), phiy.flatten()])


# ---------------------------------------------------------------------------
# Main iterative SLSQP algorithm
# ---------------------------------------------------------------------------
def iterative_with_jacobians2(
    deformation_i,
    methodName="SLSQP",
    print_results=True,
    save_path=None,
    plot_callback=None,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
    starting_window_size=None,
):
    """Perform SLSQP on submatrices iteratively to fix negative Jacobian determinants.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
        Input deformation field with channels ``[dz, dy, dx]``.
    methodName : str
        Optimiser method passed to ``scipy.optimize.minimize``.
    print_results : bool
        Print progress to stdout.
    save_path : str or None
        Directory to save results. ``None`` disables saving.
    plot_callback : callable or None
        Optional callback receiving ``(msample, fsample, deformation_i, phi)``
        after each sub-optimisation.
    threshold, err_tol, max_iterations, max_per_index_iter,
    max_minimize_iter, starting_window_size :
        Override the corresponding default parameters.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
        Corrected displacement field ``[dy, dx]``.
    """
    # Resolve parameters – use defaults where not overridden
    p = dict(DEFAULT_PARAMS)
    for name, val in [
        ("threshold", threshold),
        ("err_tol", err_tol),
        ("max_iterations", max_iterations),
        ("max_per_index_iter", max_per_index_iter),
        ("max_minimize_iter", max_minimize_iter),
        ("starting_window_size", starting_window_size),
    ]:
        if val is not None:
            p[name] = val
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]
    starting_window_size = p["starting_window_size"]

    # Accumulators
    error_list = []
    num_neg_jac = []
    iter_times = []
    min_jdet_list = []
    window_counts = defaultdict(int)

    start_time = time.time()
    slice_shape = (1, *deformation_i.shape[-2:])
    near_cent_dict = {}

    # Working phi – updated iteratively
    phi = np.zeros((2, *deformation_i.shape[-2:]))
    phi[1] = deformation_i[-1]
    phi[0] = deformation_i[-2]
    if print_results:
        print(f"deformation_i shape: {deformation_i.shape}, phi_init shape: {phi.shape}")
    phi_init = phi.copy()

    # Initial Jacobian
    jacobian_matrix = jacobian_det2D(phi)
    min_jdet_list.append(jacobian_matrix.min())
    num_neg_jac.append((jacobian_matrix <= 0).sum())

    iteration = 0
    while iteration < max_iterations and (jacobian_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any():
        iteration += 1
        window_reached_max = False

        neg_index_tuple = argmin_excluding_edges(jacobian_matrix)
        if print_results:
            print(f"\n\n\nIter {iteration}: Fixing index {neg_index_tuple}")

        submatrix_size = starting_window_size
        per_index_iter = 0

        while (
            submatrix_size == starting_window_size
            or (
                (not window_reached_max)
                and per_index_iter < max_per_index_iter
                and (jacobian_matrix[0, cy - center_distance:cy + center_distance + 1,
                                     cx - center_distance:cx + center_distance + 1] < threshold - err_tol).any()
            )
        ):
            per_index_iter += 1

            if submatrix_size < min(slice_shape[1:]) - 1:
                submatrix_size += 2
                window_counts[submatrix_size] += 1

                sub_shape = (1, submatrix_size, submatrix_size)
                cz, cy, cx = get_nearest_center(neg_index_tuple, slice_shape, submatrix_size, near_cent_dict)

                phi_init_sub_flat = get_phi_sub_flat(phi_init, cz, cy, cx, slice_shape, submatrix_size // 2)
                phi_sub_flat = get_phi_sub_flat(phi, cz, cy, cx, slice_shape, submatrix_size // 2)

                if submatrix_size > starting_window_size and print_results:
                    print(f"Iter {iteration}: For index {neg_index_tuple}, window size increased to {submatrix_size} at iter {per_index_iter}")

            # Build constraints
            if submatrix_size >= min(slice_shape[1:]) - 1:
                window_reached_max = True
                nonlinear_constraints = NonlinearConstraint(
                    lambda phi1: jacobian_constraint(phi1, submatrix_size, False), threshold, np.inf
                )
                constraints = [nonlinear_constraints]
            else:
                start_y = cy - submatrix_size // 2
                end_y = cy + submatrix_size // 2
                start_x = cx - submatrix_size // 2
                end_x = cx + submatrix_size // 2
                max_y, max_x = slice_shape[1:]

                is_at_edge = start_y == 0 or end_y >= max_y - 1 or start_x == 0 or end_x >= max_x - 1
                if print_results:
                    print(f"Is at edge: {is_at_edge}, start_y: {start_y}, end_y: {end_y}, start_x: {start_x}, end_x: {end_x}, max_y: {max_y - 1}, max_x: {max_x - 1}")

                nonlinear_constraints = NonlinearConstraint(
                    lambda phi1: jacobian_constraint(phi1, submatrix_size, not is_at_edge), threshold, np.inf
                )

                edge_mask = np.zeros((submatrix_size, submatrix_size), dtype=bool)
                edge_mask[[0, -1], :] = True
                edge_mask[:, [0, -1]] = True

                if is_at_edge:
                    if print_results:
                        print("AT EDGE")
                    edge_mask = np.zeros((submatrix_size, submatrix_size), dtype=bool)

                if print_results:
                    print(f"Edge mask for submatrix size {submatrix_size}:\n{edge_mask}")

                if not is_at_edge:
                    if print_results:
                        print("NOT AT EDGE")
                    edge_indices = np.argwhere(edge_mask)
                    fixed_indices = []
                    y_offset_sub = submatrix_size * submatrix_size
                    for y, x in edge_indices:
                        idx = y * submatrix_size + x
                        fixed_indices.extend([idx, idx + y_offset_sub])

                    if print_results:
                        print("Fixed indices")
                        print(phi_sub_flat.shape, fixed_indices)
                    fixed_values = phi_sub_flat[fixed_indices]

                    A_eq = np.zeros((len(fixed_indices), phi_sub_flat.size))
                    for row, idx in enumerate(fixed_indices):
                        A_eq[row, idx] = 1

                    linear_constraint = LinearConstraint(A_eq, fixed_values, fixed_values)
                    constraints = [nonlinear_constraints, linear_constraint]
                else:
                    constraints = [nonlinear_constraints]

                if print_results:
                    print(f"Constraints: {constraints}")

            # Run optimisation
            iter_start = time.time()
            result = minimize(
                lambda phi1: objectiveEuc(phi1, phi_init_sub_flat),
                phi_sub_flat,
                constraints=constraints,
                options={"maxiter": max_minimize_iter, "disp": True},
                method=methodName,
            )
            iter_end = time.time()
            iter_times.append(iter_end - iter_start)

            phi_x_res = result.x[: len(result.x) // 2].reshape(sub_shape[-2:])
            phi_y_res = result.x[len(result.x) // 2 :].reshape(sub_shape[-2:])

            center_distance = submatrix_size // 2

            # Update phi
            phi[1, cy - center_distance:cy + center_distance + 1, cx - center_distance:cx + center_distance + 1] = phi_x_res
            phi[0, cy - center_distance:cy + center_distance + 1, cx - center_distance:cx + center_distance + 1] = phi_y_res

            jacobian_matrix = jacobian_det2D(phi)
            num_neg_jac.append((jacobian_matrix <= 0).sum())
            min_jdet_list.append(jacobian_matrix.min())

            if print_results:
                pprint(jacobian_matrix[0, cy - center_distance:cy + center_distance + 1,
                                       cx - center_distance:cx + center_distance + 1])

            if plot_callback is not None:
                plot_callback(deformation_i, phi)

            continue_flag = submatrix_size == 3 or (
                (not window_reached_max)
                and per_index_iter < max_per_index_iter
                and (jacobian_matrix < threshold - 1e-5).any()
            )
            if print_results:
                print("\nContinue flags:")
                print(continue_flag)
                print(f"y_start: {cy - center_distance}, y_end: {cy + center_distance + 1}, x_start: {cx - center_distance}, x_end: {cx + center_distance + 1}")
                print(f"submatrix_size==3: {submatrix_size == 3} OR\nnot window_reached_max: {not window_reached_max}\nper_index_iter < max_per_index_iter: {per_index_iter < max_per_index_iter}\njacobian_submatrix < threshold - 1e-5: {(jacobian_matrix < threshold - 1e-5).any()}")
                print(f"Min jacobian value in full image: {jacobian_matrix.min()}")
                print(f"Number of total -ve jacobians: {(jacobian_matrix <= 0).sum()}")
                print()

            error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))

            if jacobian_matrix.min() > threshold - err_tol:
                if print_results:
                    print(f"All jacobians are positive, stopping at iter {iteration}")
                break

        if print_results:
            print(f"Iter {iteration} with {(jacobian_matrix <= 0).sum()} -ve jacs")

        if jacobian_matrix.min() > threshold - err_tol:
            if print_results:
                print(f"All jacobians are positive, stopping at iter {iteration}")
            break

    end_time = time.time()

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    if print_results:
        print(f"Final L2 error = {final_err}")
        print(f"Changed number of -ve jacobians from {num_neg_jac[0]} to {(jacobian_matrix <= 0).sum()}")
        print(f"Time taken for iter SLSQP optimisation: {end_time - start_time} seconds")

    num_neg_jac.append((jacobian_matrix <= 0).sum())

    # Save results
    if save_path is not None:
        output_text = "Settings:\n"
        output_text += f"\tMethod: {methodName}\n"
        output_text += f"\tThreshold: {threshold}\n"
        output_text += f"\tError tolerance: {err_tol}\n"
        output_text += f"\tMax iterations: {max_iterations}\n"
        output_text += f"\tMax per index iterations: {max_per_index_iter}\n"
        output_text += f"\tMax minimize iterations: {max_minimize_iter}\n"
        output_text += f"\tStarting window size: {starting_window_size + 2}\n\n"

        output_text += "Results:\n"
        output_text += f"\tInput deformation field resolution (height x width): {deformation_i.shape[2]} x {deformation_i.shape[3]}\n"
        output_text += f"\tTotal run-time: {end_time - start_time} seconds\n"
        output_text += f"\tFinal L2 error: {final_err}\n"
        output_text += f"\tStarting number of non-positive Jacobian determinants: {num_neg_jac[0]}\n"
        output_text += f"\tFinal number of non-positive Jacobian determinants: {(jacobian_matrix <= 0).sum()}\n"
        output_text += f"\tStarting Jacobian determinant minimum value: {min_jdet_list[0]}\n"
        output_text += f"\tFinal Jacobian determinant minimum value: {jacobian_matrix.min()}\n"
        output_text += f"\tNumber of index iterations: {iteration}"

        with open(save_path + "/results.txt", "w") as f:
            f.write(output_text)

        np.save(save_path + "/phi.npy", phi)
        np.save(save_path + "/error_list.npy", error_list)
        np.save(save_path + "/num_neg_jac.npy", num_neg_jac)
        np.save(save_path + "/iter_times.npy", iter_times)
        np.save(save_path + "/min_jdet_list.npy", min_jdet_list)

        window_counts_df = pd.DataFrame.from_dict(window_counts, orient="index", columns=["count"])
        window_counts_df.index.name = "window_size"
        window_counts_df.to_csv(save_path + "/window_counts.csv")

    return phi


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_deformations(
    msample, fsample, deformation_i, phi_corrected,
    figsize=(10, 10), save_path=None, title="", quiver_scale=1,
):
    """Plot initial vs corrected Jacobian determinants and deformation quiver fields."""
    jacobian_initial = jacobian_det2D(deformation_i[1:])
    jacobian_final = jacobian_det2D(phi_corrected)

    data = {
        ("x-def", "min"): [np.min(deformation_i[2, 0]), np.min(phi_corrected[1])],
        ("x-def", "max"): [np.max(deformation_i[2, 0]), np.max(phi_corrected[1])],
        ("y-def", "min"): [np.min(deformation_i[1, 0]), np.min(phi_corrected[0])],
        ("y-def", "max"): [np.max(deformation_i[1, 0]), np.max(phi_corrected[0])],
        ("jacobian", "min"): [np.min(jacobian_initial), np.min(jacobian_final)],
        ("jacobian", "max"): [np.max(jacobian_initial), np.max(jacobian_final)],
    }
    row_names = ["initial", "final"]
    df = pd.DataFrame(data, index=row_names)
    print(df)

    norm = mcolors.TwoSlopeNorm(
        vmin=min(jacobian_initial.min(), jacobian_final.min(), -1),
        vcenter=0,
        vmax=max(jacobian_initial.max(), jacobian_final.max(), 1),
    )

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    im0 = axs[0, 0].imshow(jacobian_initial[0], cmap="seismic", norm=norm, interpolation="nearest")
    im1 = axs[0, 1].imshow(jacobian_final[0], cmap="seismic", norm=norm, interpolation="nearest")

    x, y = np.meshgrid(range(deformation_i.shape[3]), range(deformation_i.shape[2]), indexing="xy")

    axs[0, 0].set_title("Initial J det")
    axs[0, 1].set_title("Final J det")

    axs[1, 0].set_title("Initial deformation")
    axs[1, 0].quiver(x, y, deformation_i[2, 0], -deformation_i[1, 0], scale=quiver_scale, scale_units="xy")

    axs[1, 1].set_title("Final deformation")
    axs[1, 1].quiver(x, y, phi_corrected[1], -phi_corrected[0], scale=quiver_scale, scale_units="xy")

    for i in range(2):
        axs[1, i].invert_yaxis()

    cax = fig.add_axes([0.95, 0.5, 0.02, 0.4])
    fig.colorbar(im1, cax=cax)

    if save_path is not None:
        plt.savefig(save_path + "/plot_final.png", bbox_inches="tight")
    plt.suptitle(title, fontsize=16)
    plt.show()


def plot_jacobians_iteratively(jacobians, msample, fsample, methodName="SLSQP"):
    """Plot a sequence of Jacobian determinant maps side-by-side."""
    num_jacobians = len(jacobians)
    ncols = min(2, num_jacobians)
    nrows = (num_jacobians + ncols - 1) // ncols

    all_vals = [j[0] for j in jacobians]
    vmin = min(j.min() for j in all_vals)
    vmax = max(j.max() for j in all_vals)
    norm = mcolors.TwoSlopeNorm(vmin=min(vmin, -1), vcenter=0, vmax=max(vmax, 1))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten()

    for i, jac in enumerate(jacobians):
        im = axs[i].imshow(jac[0], cmap="seismic", norm=norm, interpolation="nearest")
        num_negs = np.sum(jac <= 0)
        axs[i].set_title(f"Jacobian #{i}, {num_negs} -ves" if i > 0 else f"Initial J det: {num_negs} -ves")
        if i == 0:
            axs[i].scatter(msample[:, 2], msample[:, 1], c="g", label="Moving", s=10)
            axs[i].scatter(fsample[:, 2], fsample[:, 1], c="violet", label="Fixed", s=10)
            axs[i].legend()

    for i in range(len(msample)):
        axs[0].annotate(
            "",
            xy=(fsample[i][2], fsample[i][1]),
            xytext=(msample[i][2], msample[i][1]),
            arrowprops=dict(facecolor="black", shrink=0.1, headwidth=3, headlength=5, width=1),
        )

    for j in range(len(jacobians), len(axs)):
        axs[j].axis("off")

    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def run_lapl_and_correction(fixed_sample, msample, fsample, methodName="SLSQP", save_path=None, title="", **kwargs):
    """End-to-end: Laplacian interpolation → iterative SLSQP correction → plot.

    Extra ``**kwargs`` are forwarded to :func:`iterative_with_jacobians2`.
    """
    deformation_i, A, Zd, Yd, Xd = laplacian.sliceToSlice3DLaplacian(fixed_sample, msample, fsample)
    print(f"deformation shape: {deformation_i.shape}")
    phi_corrected = iterative_with_jacobians2(deformation_i, methodName, save_path=save_path, **kwargs)
    plot_deformations(msample, fsample, deformation_i, phi_corrected, figsize=(10, 10), save_path=save_path, title=title)
