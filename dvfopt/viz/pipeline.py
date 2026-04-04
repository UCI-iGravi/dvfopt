"""End-to-end convenience pipelines that combine Laplacian interpolation,
correction, and visualisation."""

from dvfopt.core import iterative_serial
from dvfopt.laplacian import sliceToSlice3DLaplacian
from dvfopt.viz.fields import plot_initial_deformation, plot_deformations
from dvfopt.viz.grids import plot_grid_before_after


def run_lapl_and_correction(fixed_sample, msample, fsample, methodName="SLSQP",
                            save_path=None, title="", **kwargs):
    """End-to-end: Laplacian interpolation -> iterative SLSQP correction -> plot.

    Extra ``**kwargs`` are forwarded to :func:`dvfopt.core.iterative_serial`.
    """
    deformation_i, A, Zd, Yd, Xd = sliceToSlice3DLaplacian(fixed_sample, msample, fsample)
    print(f"[Laplacian] deformation shape: {deformation_i.shape}")
    plot_initial_deformation(deformation_i, msample, fsample)
    phi_corrected = iterative_serial(deformation_i, methodName, save_path=save_path, **kwargs)
    plot_deformations(msample, fsample, deformation_i, phi_corrected,
                      figsize=(14, 12), save_path=save_path, title=title)
    plot_grid_before_after(deformation_i, phi_corrected, title=title)
    return deformation_i, phi_corrected
