"""Tests for testcases — test case builders."""

import numpy as np
import pytest

from testcases import (
    SYNTHETIC_CASES,
    RANDOM_DVF_CASES,
    make_deformation,
    make_random_dvf,
)


class TestMakeDeformation:
    @pytest.mark.parametrize("case_key", list(SYNTHETIC_CASES.keys()))
    def test_output_shape(self, case_key):
        deformation, msample, fsample = make_deformation(case_key)
        H, W = SYNTHETIC_CASES[case_key]["resolution"]
        assert deformation.shape == (3, 1, H, W)

    @pytest.mark.parametrize("case_key", list(SYNTHETIC_CASES.keys()))
    def test_dz_channel_zero(self, case_key):
        """For 2D slices, the dz channel should be zero."""
        deformation, _, _ = make_deformation(case_key)
        np.testing.assert_array_equal(deformation[0], 0.0)

    @pytest.mark.parametrize("case_key", list(SYNTHETIC_CASES.keys()))
    def test_returns_correspondences(self, case_key):
        deformation, msample, fsample = make_deformation(case_key)
        expected_ms = SYNTHETIC_CASES[case_key]["msample"]
        expected_fs = SYNTHETIC_CASES[case_key]["fsample"]
        np.testing.assert_array_equal(msample, expected_ms)
        np.testing.assert_array_equal(fsample, expected_fs)

    def test_crossing_case_has_negative_jdet(self):
        """Crossing correspondence cases should produce negative Jacobians."""
        from dvfopt.jacobian.numpy_jdet import jacobian_det2D

        deformation, _, _ = make_deformation("01a_10x10_crossing")
        jdet = jacobian_det2D(deformation[[1, 2], 0])
        assert jdet.min() < 0.5, "Crossing case should have low/negative Jdet"


class TestMakeRandomDvf:
    @pytest.mark.parametrize("case_key", list(RANDOM_DVF_CASES.keys()))
    def test_output_shape(self, case_key):
        dvf = make_random_dvf(case_key)
        case = RANDOM_DVF_CASES[case_key]
        if case["new_size"] is not None:
            H, W = case["new_size"]
        else:
            H, W = case["original_shape"][2], case["original_shape"][3]
        assert dvf.shape == (3, 1, H, W)

    @pytest.mark.parametrize("case_key", list(RANDOM_DVF_CASES.keys()))
    def test_reproducible(self, case_key):
        dvf1 = make_random_dvf(case_key)
        dvf2 = make_random_dvf(case_key)
        np.testing.assert_array_equal(dvf1, dvf2)

    def test_has_nonzero_displacement(self):
        dvf = make_random_dvf("01e_20x20_random_spirals")
        assert np.abs(dvf).max() > 0
