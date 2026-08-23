"""Real-data slice cases.

2D: loaded via dvfopt.testdata.load_slice (correspondences -> Laplacian field).
3D: loaded directly from pre-saved .npy files in data/test_cases_3d/.

Data files are gitignored (*.npy is in .gitignore), so these cases will
raise FileNotFoundError if the user hasn't checked out the data dir.
The case-registry test handles that gracefully via pytest.skip.
"""

from pathlib import Path

import numpy as np

from benchmarks.two_triangle.registry import register_case
from dvfopt.testdata import load_slice

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_3D = REPO_ROOT / "data" / "test_cases_3d"


def _real_2d_factory(slice_idx: int, scale: float, label: str):
    def case_fn():
        dvf, _ms, _fs = load_slice(
            slice_idx,
            scale_factor=scale,
            mpoints_path=str(
                REPO_ROOT / "data" / "corrected_correspondences_count_touching" / "mpoints.npy"
            ),
            fpoints_path=str(
                REPO_ROOT / "data" / "corrected_correspondences_count_touching" / "fpoints.npy"
            ),
        )
        return dvf, {"title": label, "dim": 2, "slice_idx": slice_idx, "scale": scale}

    return case_fn


def _real_3d_factory(filename: str, label: str):
    def case_fn():
        path = DATA_3D / filename
        dvf = np.load(path)
        # Stored arrays may be (3, D, H, W) directly.
        return dvf, {"title": label, "dim": 3, "source": filename}

    return case_fn


register_case("real2d_slice90_64x91", category="real_2d", dim=2)(
    _real_2d_factory(90, 0.2, "Real 2D slice 90 @ 64x91")
)

register_case("real2d_slice200_64x91", category="real_2d", dim=2)(
    _real_2d_factory(200, 0.2, "Real 2D slice 200 @ 64x91")
)

register_case("real3d_slice090_5x10x10", category="real_3d", dim=3)(
    _real_3d_factory("slice090_5x10x10.npy", "Real 3D slice 090 @ 5x10x10")
)

register_case("real3d_slice200_5x10x10", category="real_3d", dim=3)(
    _real_3d_factory("slice200_5x10x10.npy", "Real 3D slice 200 @ 5x10x10")
)
