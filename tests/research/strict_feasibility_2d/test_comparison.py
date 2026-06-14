import numpy as np
import pytest

from research.strict_feasibility_2d.runners._compare import (
    METHOD_NAMES,
    run_method,
)


def _bowtie_phi():
    phi = np.zeros((2, 7, 7), dtype=np.float64)
    phi[1, 3, 3] = +1.2
    phi[1, 3, 4] = -1.2
    return phi


def test_method_names_include_all_seven():
    expected = {
        'harmonic_only', 'm10', 'm14', 'm14_schwarz',
        'cluster_pipeline', 'lp_oneshot', 'slp_iter',
    }
    assert set(METHOD_NAMES) >= expected


def test_run_method_lp_oneshot_returns_expected_keys():
    phi_in = _bowtie_phi()
    rec = run_method('lp_oneshot', phi_in)
    for k in (
        'method', 'phi_out', 'init_n_neg_2tri', 'init_min_T',
        'final_n_neg_2tri', 'final_min_T', 'feasible',
        'L1_dev', 'L2_dev', 'Linf_dev', 'wall_s',
    ):
        assert k in rec, f'missing key {k!r}'
    assert rec['method'] == 'lp_oneshot'
    assert rec['phi_out'].shape == phi_in.shape


def test_run_method_unknown_raises():
    with pytest.raises(ValueError):
        run_method('not_a_real_method', _bowtie_phi())


def test_run_method_cluster_pipeline_records_error_not_raises():
    """cluster_pipeline is not wired yet; the harness must record the
    failure as 'error' on the row, not raise."""
    rec = run_method('cluster_pipeline', _bowtie_phi())
    assert rec['method'] == 'cluster_pipeline'
    assert rec['error'] is not None
