import numpy as np

from benchmarks.two_triangle import registry


def _make_tiny_2d_fold():
    """Construct a (3, 1, H, W) deformation with one folded pixel."""
    H, W = 6, 6
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    # Push pixel (3, 3) far enough to flip the local triangle
    dvf[1, 0, 3, 3] = -2.0  # dy
    dvf[2, 0, 3, 3] = -2.0  # dx
    return dvf


def test_baseline_variant_registered():
    assert "baseline_serial" in registry.list_variants()


def test_baseline_variant_runs_2d():
    dvf = _make_tiny_2d_fold()
    fn = registry.get_variant("baseline_serial")
    result = fn(dvf, threshold=0.01, max_iterations=20)
    assert result.phi_final.shape == (2, 6, 6)
    # Trajectory has at least one row (the final state)
    assert len(result.trajectory) >= 1
    # Either converged or hit max-iter — both fine for a smoke test
    assert isinstance(result.converged, bool)
    assert result.error is None
