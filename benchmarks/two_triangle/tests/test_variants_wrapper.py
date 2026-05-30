import numpy as np
import pytest

from benchmarks.two_triangle import registry


def _tiny_2d_fold():
    H, W = 6, 6
    dvf = np.zeros((3, 1, H, W))
    dvf[1, 0, 3, 3] = -2.0
    dvf[2, 0, 3, 3] = -2.0
    return dvf


@pytest.mark.parametrize("name", ["combinatorial_prefix"])
def test_wrapper_variant_runs(name):
    fn = registry.get_variant(name)
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    assert result.phi_final.shape == (2, 6, 6)
    assert len(result.trajectory) >= 1


def test_combinatorial_prefix_reduces_or_preserves_folds():
    """Prefix step alone should not increase folds beyond initial."""
    from benchmarks.two_triangle.metrics import fold_counts

    fn = registry.get_variant("combinatorial_prefix")
    dvf = _tiny_2d_fold()
    init_fc = fold_counts(dvf, threshold=0.01)
    result = fn(dvf, threshold=0.01, max_iterations=0)  # only prefix, no SLSQP
    final_fc = fold_counts(result.phi_final, threshold=0.01)
    assert final_fc["fold_count_tri"] <= init_fc["fold_count_tri"]


def test_multigrid_registered():
    assert "multigrid" in registry.list_variants()


def test_multigrid_runs_2d():
    fn = registry.get_variant("multigrid")
    dvf = _tiny_2d_fold()
    # 6x6 is too small for two-level cascade (downsampling to 3x3 wouldn't
    # give a meaningful warm-start), but the variant should still produce
    # a valid result by skipping levels that are too small.
    result = fn(dvf, threshold=0.01, max_iterations=10)
    assert result.phi_final.shape == (2, 6, 6)


def test_svf_warmstart_registered():
    assert "svf_warmstart" in registry.list_variants()


def test_svf_warmstart_runs_2d():
    fn = registry.get_variant("svf_warmstart")
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=10)
    assert result.phi_final.shape == (2, 6, 6)
    assert len(result.trajectory) >= 2  # init + post-SVF + post-baseline


def test_svf_warmstart_preserves_clean_fields():
    """If input has no folds, SVF projection should leave it nearly unchanged."""
    H, W = 8, 8
    dvf = np.zeros((3, 1, H, W))
    dvf[1, 0, 4, 4] = 0.1  # tiny perturbation, no fold
    dvf[2, 0, 4, 4] = 0.1
    fn = registry.get_variant("svf_warmstart")
    result = fn(dvf, threshold=0.01, max_iterations=5)
    # No folds before — there should be none after either
    from benchmarks.two_triangle.metrics import fold_counts

    fc = fold_counts(result.phi_final, threshold=0.01)
    assert fc["fold_count_jdet"] == 0
    assert fc["fold_count_tri"] == 0
