"""Per-iteration metric accumulator used by loop-owning variants."""

import numpy as np
import pandas as pd

from benchmarks.two_triangle.metrics import (
    fold_counts,
    l2_displacement,
    smoothness,
)


class TrajectoryAccumulator:
    """Captures per-outer-iter metrics into rows; emits a DataFrame at end."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(
        self,
        *,
        outer_iter: int,
        phi: np.ndarray,
        phi_initial: np.ndarray,
        n_active_windows: int,
        inner_iters: int,
        t_elapsed: float,
        threshold: float = 0.01,
    ) -> None:
        fc = fold_counts(phi, threshold=threshold)
        self._rows.append(
            {
                "outer_iter": outer_iter,
                "time_s": t_elapsed,
                "fold_count_jdet": fc["fold_count_jdet"],
                "fold_count_tri": fc["fold_count_tri"],
                "max_violation": fc["max_violation"],
                "l2_disp": l2_displacement(phi, phi_initial),
                "smoothness": smoothness(phi),
                "n_active_windows": n_active_windows,
                "inner_iters": inner_iters,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)
