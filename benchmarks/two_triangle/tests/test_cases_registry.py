import pytest

from benchmarks.two_triangle import registry

SYNTHETIC_2D = (
    "synth2d_single_cell_flip",
    "synth2d_horizontal_bowtie",
    "synth2d_diagonal_bowtie",
    "synth2d_layered_bowtie_stack",
)


@pytest.mark.parametrize("name", SYNTHETIC_2D)
def test_synthetic_2d_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    dvf, meta = fn()
    assert dvf.ndim == 4
    assert dvf.shape[0] == 3
    assert dvf.shape[1] == 1
    assert "title" in meta
    assert meta.get("dim") in (2, None)


@pytest.mark.parametrize("name", SYNTHETIC_2D)
def test_synthetic_2d_case_has_initial_folds(name):
    """Every synthetic case must start with at least one fold to be useful."""
    from benchmarks.two_triangle.metrics import fold_counts

    fn = registry.get_case(name)
    dvf, _ = fn()
    fc = fold_counts(dvf, threshold=0.01)
    assert fc["fold_count_jdet"] > 0 or fc["fold_count_tri"] > 0, (
        f"Case {name} has no initial folds — useless as a benchmark"
    )


SYNTHETIC_3D = (
    "synth3d_single_tet_flip",
    "synth3d_6tet_bowtie",
)


@pytest.mark.parametrize("name", SYNTHETIC_3D)
def test_synthetic_3d_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    dvf, meta = fn()
    assert dvf.ndim == 4
    assert dvf.shape[0] == 3
    assert dvf.shape[1] >= 2  # 3D: D > 1
    assert meta.get("dim") == 3


@pytest.mark.parametrize("name", SYNTHETIC_3D)
def test_synthetic_3d_case_has_initial_folds(name):
    from benchmarks.two_triangle.metrics import fold_counts

    fn = registry.get_case(name)
    dvf, _ = fn()
    fc = fold_counts(dvf, threshold=0.01)
    assert fc["fold_count_jdet"] > 0


RANDOM_3D = (
    "rand3d_grid16_low",
    "rand3d_grid16_high",
    "rand3d_grid24_low",
    "rand3d_grid24_high",
    "rand3d_grid32_low",
    "rand3d_grid32_high",
)


@pytest.mark.parametrize("name", RANDOM_3D)
def test_random_3d_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    dvf, meta = fn()
    assert dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] >= 2
    assert meta.get("dim") == 3


@pytest.mark.parametrize("name", ("rand3d_grid16_high", "rand3d_grid32_high"))
def test_random_3d_high_severity_has_folds(name):
    from benchmarks.two_triangle.metrics import fold_counts

    fn = registry.get_case(name)
    dvf, _ = fn()
    fc = fold_counts(dvf, threshold=0.01)
    assert fc["fold_count_jdet"] > 0


REAL_CASES = (
    "real2d_slice90_64x91",
    "real2d_slice200_64x91",
    "real3d_slice090_5x10x10",
    "real3d_slice200_5x10x10",
)


@pytest.mark.parametrize("name", REAL_CASES)
def test_real_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    try:
        dvf, meta = fn()
    except FileNotFoundError:
        pytest.skip(f"{name}: data file not present (expected; .npy files are gitignored)")
    assert dvf.ndim == 4 and dvf.shape[0] == 3
    assert "title" in meta
