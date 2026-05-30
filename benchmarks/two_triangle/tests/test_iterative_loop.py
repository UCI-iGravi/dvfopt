import numpy as np

from benchmarks.two_triangle._iterative_loop import run_minimal_iterative_2d
from benchmarks.two_triangle.trajectory import TrajectoryAccumulator


def test_trajectory_accumulator_basic():
    acc = TrajectoryAccumulator()
    phi0 = np.zeros((2, 4, 4))
    acc.record(
        outer_iter=0, phi=phi0, phi_initial=phi0, n_active_windows=0, inner_iters=0, t_elapsed=0.0
    )
    acc.record(
        outer_iter=1, phi=phi0, phi_initial=phi0, n_active_windows=1, inner_iters=12, t_elapsed=0.42
    )
    df = acc.to_frame()
    assert len(df) == 2
    assert list(df.columns)[:3] == ["outer_iter", "time_s", "fold_count_jdet"]


def _tiny_2d_fold():
    H, W = 6, 6
    dvf = np.zeros((3, 1, H, W))
    dvf[1, 0, 3, 3] = -2.0
    dvf[2, 0, 3, 3] = -2.0
    return dvf


def test_run_minimal_iterative_2d_smoke():
    dvf = _tiny_2d_fold()
    phi_init = np.stack([dvf[1, 0], dvf[2, 0]])
    result = run_minimal_iterative_2d(
        phi_init.copy(),
        threshold=0.01,
        max_iterations=20,
    )
    assert result.phi_final.shape == phi_init.shape
    assert len(result.trajectory) >= 1
    # Final fold count should be no worse than initial
    initial_folds = result.trajectory.iloc[0]["fold_count_tri"]
    final_folds = result.trajectory.iloc[-1]["fold_count_tri"]
    assert final_folds <= initial_folds
