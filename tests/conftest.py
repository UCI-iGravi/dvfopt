"""Shared fixtures for dvfopt unit tests."""

import numpy as np
import pytest


@pytest.fixture
def identity_phi_2d():
    """A 10x10 zero-displacement field (identity deformation)."""
    return np.zeros((2, 10, 10), dtype=np.float64)


@pytest.fixture
def identity_deformation_2d():
    """A (3,1,10,10) zero-displacement deformation field."""
    return np.zeros((3, 1, 10, 10), dtype=np.float64)


@pytest.fixture
def small_deformation_2d():
    """A (3,1,8,8) deformation with mild smooth displacement (no negative Jdet)."""
    rng = np.random.default_rng(42)
    d = np.zeros((3, 1, 8, 8), dtype=np.float64)
    # Small smooth displacements that won't cause negative Jacobians
    d[1, 0] = rng.uniform(-0.1, 0.1, (8, 8))  # dy
    d[2, 0] = rng.uniform(-0.1, 0.1, (8, 8))  # dx
    return d


@pytest.fixture
def identity_phi_3d():
    """A 6x6x6 zero-displacement 3D field."""
    return np.zeros((3, 6, 6, 6), dtype=np.float64)


@pytest.fixture
def identity_deformation_3d():
    """A (3,6,6,6) zero-displacement 3D deformation field."""
    return np.zeros((3, 6, 6, 6), dtype=np.float64)
