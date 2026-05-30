"""Parity: baseline_serial variant must match a direct iterative_serial call."""

import numpy as np
import pytest

from benchmarks.two_triangle import registry
from dvfopt.core.slsqp.iterative import iterative_serial


@pytest.mark.parametrize(
    "case_name",
    [
        "synth2d_single_cell_flip",
        "synth2d_horizontal_bowtie",
    ],
)
def test_baseline_matches_direct_solver(case_name):
    case_fn = registry.get_case(case_name)
    dvf, _ = case_fn()

    direct = iterative_serial(
        dvf.copy(), threshold=0.01, max_iterations=50, verbose=0, enforce_triangles=True
    )

    baseline_fn = registry.get_variant("baseline_serial")
    result = baseline_fn(dvf.copy(), threshold=0.01, max_iterations=50)

    # Direct call returns (2, H, W) [dy, dx]; baseline returns the same canonical
    # (2, H, W) format.
    np.testing.assert_allclose(result.phi_final, direct, rtol=1e-10, atol=1e-10)
