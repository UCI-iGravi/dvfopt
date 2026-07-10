"""Numba-kernel vs pure-numpy fallback parity.

Both ``dvfopt.jacobian.triangle_sign`` and
``dvfopt.jacobian.tetrahedron_sign`` gate their Numba kernels behind a
module-level ``_HAVE_NUMBA`` flag that the public wrappers read at
**call time** (``if not _HAVE_NUMBA: return <numpy fallback>``), not at
import time. Monkeypatching the flag to ``False`` therefore genuinely
routes the same public call through the pure-numpy fallback path — no
``importlib.reload`` needed.

Each test evaluates the public function twice on the same random field:
once with the module untouched (Numba kernel when numba is installed)
and once under ``monkeypatch.setattr(mod, '_HAVE_NUMBA', False)``
(forced numpy fallback), and asserts max |diff| < 1e-12.

When numba is not installed both evaluations take the numpy path and
the parity assertion holds trivially; the numba-vs-numpy comparison is
only meaningful in environments with numba (the default dev setup).
"""

from __future__ import annotations

import numpy as np
import pytest

import dvfopt.jacobian.tetrahedron_sign as tet_mod
import dvfopt.jacobian.triangle_sign as tri_mod

ATOL = 1e-12


def _max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestTriangleAreasParity:
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_triangle_areas_2d(self, monkeypatch, seed):
        rng = np.random.default_rng(seed)
        H, W = 13, 17
        dy = rng.normal(0, 0.4, (H, W))
        dx = rng.normal(0, 0.4, (H, W))

        T1_default, T2_default = tri_mod._triangle_areas_2d(dy, dx)
        monkeypatch.setattr(tri_mod, '_HAVE_NUMBA', False)
        T1_numpy, T2_numpy = tri_mod._triangle_areas_2d(dy, dx)

        # The forced-fallback result must be the reference numpy impl.
        T1_ref, T2_ref = tri_mod._triangle_areas_2d_numpy(dy, dx)
        np.testing.assert_array_equal(T1_numpy, T1_ref)
        np.testing.assert_array_equal(T2_numpy, T2_ref)

        assert _max_abs_diff(T1_default, T1_numpy) < ATOL
        assert _max_abs_diff(T2_default, T2_numpy) < ATOL


class TestTetVolumesParity:
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_six_tet_volumes_3d(self, monkeypatch, seed):
        rng = np.random.default_rng(seed)
        phi = rng.normal(0, 0.4, (3, 5, 6, 7))

        V_default = tet_mod.six_tet_volumes_3d(phi)
        monkeypatch.setattr(tet_mod, '_HAVE_NUMBA', False)
        V_numpy = tet_mod.six_tet_volumes_3d(phi)

        # The forced-fallback result must be the reference numpy impl.
        np.testing.assert_array_equal(V_numpy, tet_mod._six_tet_volumes_3d_numpy(phi))

        assert V_default.shape == V_numpy.shape == (6, 4, 5, 6)
        assert _max_abs_diff(V_default, V_numpy) < ATOL

    def test_six_tet_min_volume_3d(self, monkeypatch, rng):
        phi = rng.normal(0, 0.4, (3, 5, 6, 7))

        mv_default = tet_mod.six_tet_min_volume_3d(phi)
        monkeypatch.setattr(tet_mod, '_HAVE_NUMBA', False)
        mv_numpy = tet_mod.six_tet_min_volume_3d(phi)

        assert mv_default.shape == mv_numpy.shape == (4, 5, 6)
        assert _max_abs_diff(mv_default, mv_numpy) < ATOL


class TestTetGradParity:
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_tet_grad_T_v(self, monkeypatch, seed):
        rng = np.random.default_rng(seed)
        D, H, W = 4, 5, 6
        phi_flat = rng.normal(0, 0.2, 3 * D * H * W)  # [dx, dy, dz] pack
        v = rng.normal(size=6 * (D - 1) * (H - 1) * (W - 1))

        g_default = tet_mod.tet_grad_T_v(phi_flat, D, H, W, v)
        monkeypatch.setattr(tet_mod, '_HAVE_NUMBA', False)
        g_numpy = tet_mod.tet_grad_T_v(phi_flat, D, H, W, v)

        # The forced-fallback result must be the reference numpy impl.
        np.testing.assert_array_equal(g_numpy, tet_mod._tet_grad_T_v_numpy(phi_flat, D, H, W, v))

        assert g_default.shape == g_numpy.shape == (3 * D * H * W,)
        assert _max_abs_diff(g_default, g_numpy) < ATOL

    def test_tet_grad_sparse_v(self, monkeypatch, rng):
        """The JIT kernel has a per-cell sparsity early-exit; verify
        parity holds on a mostly-zero co-vector too."""
        D, H, W = 4, 5, 6
        phi_flat = rng.normal(0, 0.2, 3 * D * H * W)
        v = np.zeros(6 * (D - 1) * (H - 1) * (W - 1))
        v[::17] = rng.normal(size=v[::17].shape)

        g_default = tet_mod.tet_grad_T_v(phi_flat, D, H, W, v)
        monkeypatch.setattr(tet_mod, '_HAVE_NUMBA', False)
        g_numpy = tet_mod.tet_grad_T_v(phi_flat, D, H, W, v)

        assert _max_abs_diff(g_default, g_numpy) < ATOL
