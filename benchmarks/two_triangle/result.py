"""SolverResult dataclass + dependency-free bundle serialization."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_TRAJECTORY_COLS = (
    "outer_iter",
    "time_s",
    "fold_count_jdet",
    "fold_count_tri",
    "max_violation",
    "l2_disp",
    "smoothness",
    "n_active_windows",
    "inner_iters",
)


@dataclass
class SolverResult:
    phi_final: np.ndarray
    trajectory: pd.DataFrame
    converged: bool
    timed_out: bool
    error: Optional[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        missing = [c for c in REQUIRED_TRAJECTORY_COLS if c not in self.trajectory.columns]
        if missing:
            raise ValueError(f"missing trajectory columns: {missing}")

    def to_csv(self, path) -> None:
        path = Path(path)
        self.trajectory.to_csv(path, index=False)
        np.savez_compressed(path.with_suffix(".phi.npz"), phi=self.phi_final)
        sidecar = {
            "converged": self.converged,
            "timed_out": self.timed_out,
            "error": self.error,
            "meta": self.meta,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(sidecar, default=str))

    @classmethod
    def from_csv(cls, path) -> "SolverResult":
        path = Path(path)
        traj = pd.read_csv(path)
        phi = np.load(path.with_suffix(".phi.npz"))["phi"]
        sidecar = json.loads(path.with_suffix(".meta.json").read_text())
        return cls(
            phi_final=phi,
            trajectory=traj,
            converged=sidecar["converged"],
            timed_out=sidecar["timed_out"],
            error=sidecar["error"],
            meta=sidecar["meta"],
        )

    # Backward-compatible aliases with old API names.
    def to_parquet(self, path) -> None:
        self.to_csv(path)

    @classmethod
    def from_parquet(cls, path) -> "SolverResult":
        return cls.from_csv(path)
