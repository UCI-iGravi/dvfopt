import numpy as np
import pandas as pd
import pytest

from benchmarks.two_triangle.result import SolverResult


def test_construct_minimal():
    phi = np.zeros((2, 4, 4), dtype=np.float64)
    traj = pd.DataFrame(
        {
            "outer_iter": [0],
            "time_s": [0.0],
            "fold_count_jdet": [0],
            "fold_count_tri": [0],
            "max_violation": [0.0],
            "l2_disp": [0.0],
            "smoothness": [0.0],
            "n_active_windows": [0],
            "inner_iters": [0],
        }
    )
    r = SolverResult(
        phi_final=phi,
        trajectory=traj,
        converged=True,
        timed_out=False,
        error=None,
        meta={"variant": "baseline", "case": "demo"},
    )
    assert r.converged is True
    assert r.phi_final.shape == (2, 4, 4)
    assert len(r.trajectory) == 1


def test_required_trajectory_columns():
    phi = np.zeros((2, 4, 4))
    bad_traj = pd.DataFrame({"outer_iter": [0]})  # missing required columns
    with pytest.raises(ValueError, match="missing trajectory columns"):
        SolverResult(
            phi_final=phi,
            trajectory=bad_traj,
            converged=True,
            timed_out=False,
            error=None,
            meta={},
        )


def test_to_csv_roundtrip(tmp_path):
    phi = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    traj = pd.DataFrame(
        {
            "outer_iter": [0, 1],
            "time_s": [0.0, 0.5],
            "fold_count_jdet": [3, 0],
            "fold_count_tri": [4, 0],
            "max_violation": [-0.1, 0.05],
            "l2_disp": [0.0, 1.2],
            "smoothness": [0.0, 0.3],
            "n_active_windows": [1, 1],
            "inner_iters": [10, 5],
        }
    )
    r = SolverResult(
        phi_final=phi,
        trajectory=traj,
        converged=True,
        timed_out=False,
        error=None,
        meta={"variant": "soft_margin", "case": "synth2d_single_cell_flip"},
    )
    path = tmp_path / "result.csv"
    r.to_csv(path)
    r2 = SolverResult.from_csv(path)
    np.testing.assert_array_equal(r2.phi_final, phi)
    pd.testing.assert_frame_equal(r2.trajectory, traj)
    assert r2.converged == r.converged
    assert r2.meta["variant"] == "soft_margin"
