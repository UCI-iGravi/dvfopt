"""Tests for the persistent worker pool and the multi-scale 3D seed."""

import numpy as np
import pytest

from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


def _planted_cluster(shape=(3, 16, 16, 16), scale=1.2):
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.05, shape).astype(np.float64)
    z = shape[1] // 2
    phi[0, z - 1 : z + 2, 7:10, 7:10] = scale
    phi[0, z : z + 3, 7:10, 7:10] = -scale
    return phi


class TestPersistentPool:
    def test_get_pool_reuse_and_resize(self):
        """Grow-only respawn: the pool is reused whenever its size already
        covers the request (including DOWNSIZE requests — respawning on a
        downsize would re-pay the multi-second warmup for nothing); it is
        rebuilt only when the request grows past the current size."""
        from dvfopt.core._pool import get_pool, shutdown_pool

        try:
            p1 = get_pool(2)
            p2 = get_pool(2)
            assert p1 is p2  # same size -> same pool reused
            p3 = get_pool(3)
            assert p3 is not p1  # grew -> new pool
            p4 = get_pool(2)
            assert p4 is p3  # downsize request -> existing pool reused
            p5 = get_pool(1)
            assert p5 is p3  # further downsize -> still reused
            p6 = get_pool(4)
            assert p6 is not p3  # grew past current size -> rebuilt
        finally:
            shutdown_pool()

    def test_no_nested_pools_inside_a_worker(self, monkeypatch):
        """Inside a worker process (CLI --n-workers, DVFopt(n_workers>1)) a
        sub-pool would multiply processes — the request is capped to 1."""
        import dvfopt.core._pool as poolmod

        seen = {}

        class _StubExec:
            def __init__(self, max_workers=None, **kw):
                seen['max_workers'] = max_workers

            def shutdown(self, wait=False):
                pass

        monkeypatch.setattr(poolmod, 'ProcessPoolExecutor', _StubExec)
        monkeypatch.setattr(poolmod.multiprocessing, 'parent_process', lambda: object())
        monkeypatch.setattr(poolmod, '_POOL', None)
        monkeypatch.setattr(poolmod, '_POOL_N', None)
        poolmod.get_pool(8)
        assert seen['max_workers'] == 1

    def test_pool_executes_warm(self):
        """A trivial task runs on the warmed pool (workers imported + JIT'd)."""
        from dvfopt.core._pool import get_pool, shutdown_pool

        try:
            pool = get_pool(2)
            out = list(pool.map(abs, [-1, -2, -3]))
            assert out == [1, 2, 3]
        finally:
            shutdown_pool()

    def test_pool_map_results(self):
        """pool_map maps a worker over args and returns the result list."""
        from dvfopt.core._pool import pool_map, shutdown_pool

        try:
            out = pool_map(abs, [-1, -2, -3], 2)
            assert out == [1, 2, 3]
        finally:
            shutdown_pool()

    def test_pool_map_falls_back_to_serial_on_broken_pool(self, monkeypatch):
        """A dead worker (BrokenProcessPool) must not crash the caller:
        pool_map tears the pool down and completes the work serially."""
        from concurrent.futures.process import BrokenProcessPool

        import dvfopt.core._pool as poolmod

        teardowns = []

        class _BrokenPool:
            def map(self, *a, **k):
                raise BrokenProcessPool('worker died')

        broken = _BrokenPool()
        monkeypatch.setattr(poolmod, 'get_pool', lambda n: broken)
        monkeypatch.setattr(poolmod, '_shutdown_if_current', lambda ex: teardowns.append(ex))
        # Serial fallback still produces the correct answer...
        out = poolmod.pool_map(abs, [-5, -6], 2)
        assert out == [5, 6]
        # ...and recovery targeted the exact executor that failed.
        assert teardowns == [broken]

    def test_shutdown_if_current_is_identity_guarded(self, monkeypatch):
        """The recovery teardown must NOT destroy a pool that was rebuilt
        out from under the caller (stale-reference TOCTOU). It only tears
        down the module pool when the failed executor IS still installed."""
        import dvfopt.core._pool as poolmod

        class _FakeExec:
            def __init__(self):
                self.shut = False

            def shutdown(self, wait=False):
                self.shut = True

        stale, live = _FakeExec(), _FakeExec()
        # Simulate: a concurrent resize already installed `live` as _POOL,
        # while our caller still holds the now-stale `stale` reference.
        monkeypatch.setattr(poolmod, '_POOL', live)
        monkeypatch.setattr(poolmod, '_POOL_N', 3)
        poolmod._shutdown_if_current(stale)
        # The live pool another caller depends on is untouched...
        assert poolmod._POOL is live
        assert live.shut is False
        # ...but tearing down the pool that IS current does clear it.
        poolmod._shutdown_if_current(live)
        assert poolmod._POOL is None
        assert live.shut is True


class TestMultiscaleSeed:
    def test_shape_helpers_roundtrip(self):
        from dvfopt.core.wallbreakers._multiscale_3d import (
            _downsample_2x,
            _upsample_2x,
        )

        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.05, (3, 16, 18, 20)).astype(np.float64)
        coarse = _downsample_2x(phi)
        assert coarse.shape == (3, 8, 9, 10)
        up = _upsample_2x(coarse, phi.shape[1:])
        assert up.shape == (3, 16, 18, 20)

    def test_reduces_folds(self):
        from dvfopt.core.wallbreakers._multiscale_3d import multiscale_seed_3d

        phi = _planted_cluster()
        n0 = int((six_tet_min_volume_3d(phi) <= 0).sum())
        if n0 == 0:
            pytest.skip('no fold planted')
        out, info = multiscale_seed_3d(phi, threshold=0.012)
        n1 = int((six_tet_min_volume_3d(out) <= 0).sum())
        assert n1 <= n0
        assert info['used_multiscale'] is True

    def test_too_small_falls_back(self):
        from dvfopt.core.wallbreakers._multiscale_3d import multiscale_seed_3d

        phi = np.zeros((3, 3, 3, 3))  # halving -> 1 cube, too small
        out, info = multiscale_seed_3d(phi, threshold=0.012)
        assert info['used_multiscale'] is False
        assert out.shape == phi.shape


class TestMultiscaleRoute:
    def test_explicit_multiscale_route(self):
        from dvfopt import correct_dvf_3d

        phi = _planted_cluster(scale=1.2)
        n0 = int((six_tet_min_volume_3d(phi) <= 0).sum())
        if n0 == 0:
            pytest.skip('no fold planted')
        out, rep = correct_dvf_3d(phi, threshold=0.01, bulk='multiscale')
        assert any(s['stage'] == 'bulk:multiscale' for s in rep.stages)
        assert rep.n_neg_out <= n0
