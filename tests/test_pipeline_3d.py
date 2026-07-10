"""Tests for the end-to-end 3D orchestrator ``correct_dvf_3d``."""

import numpy as np
import pytest

from dvfopt import Correct3DReport, correct_dvf_3d
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


def _planted(seed=0, centers=((3, 8, 8), (6, 25, 28), (9, 32, 12))):
    rng = np.random.default_rng(seed)
    phi = rng.normal(0, 0.02, (3, 12, 40, 40)).astype(np.float64)
    for z, y, x in centers:
        phi[0, z, y : y + 2, x : x + 2] = 1.5
        phi[0, z + 1, y : y + 2, x : x + 2] = -1.5
    return phi


class TestNonFiniteInputRejected:
    """The entry point must reject non-finite fields up front: the fused
    min-volume kernel's ``<`` comparisons always fail on NaN, so a NaN cube
    gets min-volume 1e300 and would otherwise be reported as feasible
    (silent success on corrupted data)."""

    def test_nan_in_one_channel_raises(self):
        phi = _planted()
        phi[1, 4, 10, 10] = np.nan  # single NaN in dy
        with pytest.raises(ValueError, match='non-finite'):
            correct_dvf_3d(phi, threshold=0.01)

    def test_all_nan_raises(self):
        phi = np.full((3, 6, 8, 8), np.nan)
        with pytest.raises(ValueError, match='non-finite'):
            correct_dvf_3d(phi, threshold=0.01)

    def test_inf_raises(self):
        phi = np.zeros((3, 6, 8, 8))
        phi[2, 3, 4, 4] = np.inf
        with pytest.raises(ValueError, match='non-finite'):
            correct_dvf_3d(phi, threshold=0.01)

    def test_nan_field_not_reported_feasible(self):
        """Regression guard for the exact failure mode: an all-NaN field
        must never come back as Correct3DReport(feasible=True)."""
        phi = np.full((3, 6, 8, 8), np.nan)
        try:
            _, rep = correct_dvf_3d(phi, threshold=0.01)
        except ValueError:
            return  # expected path
        assert not rep.feasible  # unreachable if the guard works


def test_small_planted_dz_fold_smoke():
    """Small (3, 6, 20, 20) planted-dz-fold field reaches strict feasibility
    (planting in dz is correct for the true-3D pipeline)."""
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, 6, 20, 20)).astype(np.float64)
    phi[0, 2, 8:10, 8:10] = 1.5
    phi[0, 3, 8:10, 8:10] = -1.5
    n0 = int((six_tet_min_volume_3d(phi) <= 0).sum())
    assert n0 > 0
    out, rep = correct_dvf_3d(phi, threshold=0.01)
    assert rep.n_neg_in == n0
    assert rep.n_neg_out <= n0
    # report numbers match an independent re-measure of the output
    mv = six_tet_min_volume_3d(out)
    assert int((mv <= 0).sum()) == rep.n_neg_out
    assert mv.min() == pytest.approx(rep.min_T_out)


def test_already_feasible_noop():
    phi = np.zeros((3, 6, 8, 8))
    out, rep = correct_dvf_3d(phi, threshold=0.01)
    assert rep.feasible
    assert rep.n_neg_in == 0 and rep.n_neg_out == 0
    assert np.array_equal(out, phi)


def test_sparse_reaches_feasible():
    phi = _planted()
    n0 = int((six_tet_min_volume_3d(phi) <= 0).sum())
    assert n0 > 0
    out, rep = correct_dvf_3d(phi, threshold=0.01)
    assert isinstance(rep, Correct3DReport)
    assert rep.feasible
    assert rep.n_neg_out == 0 and rep.n_below_out == 0
    # output truly feasible at the strict threshold
    mv = six_tet_min_volume_3d(out)
    assert mv.min() >= 0.01 - 1e-9
    # routed through active-band (sparse), not a global solve
    assert any(s['stage'] == 'bulk:active_band' for s in rep.stages)


def test_report_fields_populated():
    phi = _planted(seed=1)
    _, rep = correct_dvf_3d(phi, threshold=0.01)
    assert rep.n_neg_in > 0
    assert rep.l1_from_input > 0
    assert rep.wall_s > 0
    assert rep.best_diag_floor_in >= 0
    # stages always include triage first
    assert rep.stages[0]['stage'] == 'triage'


def test_never_increases_folds():
    """Whatever happens, the orchestrator must not return a field with more
    folds than the input (it accepts only non-regressing moves internally)."""
    phi = _planted(seed=2)
    n0 = int((six_tet_min_volume_3d(phi) <= 0).sum())
    out, rep = correct_dvf_3d(phi, threshold=0.01)
    assert rep.n_neg_out <= n0
    assert int((six_tet_min_volume_3d(out) <= 0).sum()) == rep.n_neg_out


def test_explicit_global_route():
    phi = _planted(seed=3)
    out, rep = correct_dvf_3d(phi, threshold=0.01, bulk='global')
    assert any(s['stage'] == 'bulk:global' for s in rep.stages)
    assert rep.n_neg_out <= rep.n_neg_in
