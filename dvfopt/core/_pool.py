"""Persistent, pre-warmed process pool for the parallel 3D solvers.

Per-call ``ProcessPoolExecutor`` pays the Windows process-spawn cost AND
re-imports dvfopt + re-JITs the Numba kernels in every fresh worker on
its first task (~5-10 s/worker) — which made fine/coarse parallelism
*slower* than serial on the scales measured (REPORT Part XVII / the
z-band benchmark).

This module keeps ONE long-lived pool whose workers run a warmup
initializer once at startup (importing the kernels and JIT-compiling them
on a tiny field). Subsequent tasks hit warm, imported workers — so the
spawn + import + recompile tax is amortised across every band/cluster of
a run (and across runs in the same session). Callers get the pool via
:func:`get_pool`; it is reused while the existing pool already covers the
requested worker count (grow-only respawn) and torn down at interpreter
exit.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Optional

_POOL: Optional[ProcessPoolExecutor] = None
_POOL_N: Optional[int] = None
_LOCK = threading.Lock()

#: Every thread-count env var a dvfopt worker's dependencies honour. Measured
#: on a 24-logical-core Windows box: an UNPINNED worker carries 53 OS threads
#: (4 baseline + 23 from ``import numpy`` + 26 more from scipy) — i.e. numpy
#: and scipy each spin up a full-width OpenBLAS/OpenMP pool, so N pool workers
#: fight over N x 24 compute threads and per-solve time grows linearly with N.
#: numexpr / numba (``prange`` kernels) / rayon (Clarabel's Rust runtime) are
#: pinned defensively: measured here they add nothing (Clarabel's qdldl path
#: never starts a rayon pool), but they are all-cores-by-default too.
THREAD_ENV = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
    'RAYON_NUM_THREADS',
)


def pin_worker_threads() -> None:
    """Force the CURRENT process down to one compute thread per library.

    No-op outside a pool worker: a worker is one core's worth of work by
    contract, but the serial paths share the caller's process and must not have
    their environment rewritten underneath them.

    Call it at the top of a worker function. It is a belt-and-braces companion
    to :func:`pinned_thread_env` — OpenBLAS/MKL read the env once, at import,
    which in a spawned worker has usually already happened by the time the
    worker function runs; the env inherited from the parent is what pins those.
    This call still covers late imports, numba's runtime API, and any nested
    pool the worker itself creates.
    """
    if multiprocessing.parent_process() is None:
        return
    os.environ.update(dict.fromkeys(THREAD_ENV, '1'))
    try:
        import numba

        numba.set_num_threads(1)
    except Exception:  # pragma: no cover - numba optional / already pinned
        pass


@contextmanager
def pinned_thread_env() -> Iterator[None]:
    """Pin :data:`THREAD_ENV` to ``'1'`` for the duration, then restore.

    Wrap pool CREATION in this: spawned children inherit the parent's
    environment at interpreter start, which is the only point early enough for
    OpenBLAS/MKL to honour it.
    """
    saved = {k: os.environ.get(k) for k in THREAD_ENV}
    os.environ.update(dict.fromkeys(THREAD_ENV, '1'))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _warmup_worker():  # pragma: no cover - runs in subprocess
    """Run once per worker at pool start: pin each worker to a SINGLE
    thread, then import + JIT-compile the kernels.

    Pinning to one thread is essential: the tet kernels are themselves
    thread-parallel (``prange``), so without this every worker process
    would spawn its own pool of compute threads and N workers would
    oversubscribe the cores — making process-parallelism SLOWER than the
    serial-but-thread-parallel path. With one thread per worker, N
    workers use N cores with no oversubscription, so process-level
    parallelism (over independent z-bands / clusters) composes cleanly
    with the kernels."""
    pin_worker_threads()

    import numpy as np

    from dvfopt.jacobian.tetrahedron_sign import (
        six_tet_min_volume_3d,
        six_tet_volumes_3d,
        six_tet_volumes_all_diagonals,
        tet_grad_T_v,
    )

    phi = np.zeros((3, 4, 4, 4), dtype=np.float64)
    six_tet_volumes_3d(phi)
    six_tet_min_volume_3d(phi)
    six_tet_volumes_all_diagonals(phi)
    n = 4 * 4 * 4
    tet_grad_T_v(np.zeros(3 * n), 4, 4, 4, np.zeros(6 * 3 * 3 * 3))


def get_pool(n_workers: int) -> ProcessPoolExecutor:
    """Return the shared pre-warmed pool with capacity for ``n_workers``.

    Creates it (with the warmup initializer) on first use. The existing
    pool is REUSED whenever its size already covers the request
    (``current size >= n_workers``) — a downsize request does not
    respawn the pool, since respawning re-pays the spawn + import + JIT
    warmup tax (~5-10 s/worker) and no caller depends on an exact pool
    size (extra workers simply idle). The pool is rebuilt only when the
    request GROWS past the current size. Thread-safe.

    Never nests: inside a worker process (the CLI's ``--n-workers`` per-slice
    pool, ``DVFopt(n_workers>1)``, a caller's own multiprocessing children) the
    request is capped to 1 — that worker is already one core's worth of work, so
    a sub-pool per worker would just multiply processes and oversubscribe.
    """
    global _POOL, _POOL_N
    if multiprocessing.parent_process() is not None:
        n_workers = 1
    with _LOCK:
        if _POOL is None or _POOL_N is None or n_workers > _POOL_N:
            if _POOL is not None:
                _POOL.shutdown(wait=False)
            # spawn, EXPLICITLY: Linux's default is fork, and forking a
            # process whose OpenMP runtime is live (any prior numba prange
            # call) is undefined behavior — modern libgomp deliberately
            # terminates such children ("Terminating: fork() called from a
            # process already using GNU OpenMP, this is unsafe."), which
            # broke the pool on CI and silently degraded pool_map to its
            # serial fallback on strict-libgomp Linux. Spawned workers are
            # pristine interpreters, and this module's whole design already
            # amortises the spawn+import+JIT warm-up cost. (forkserver is
            # NOT safe here: the server itself forks from the live parent.)
            _POOL = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_warmup_worker,
                mp_context=multiprocessing.get_context('spawn'),
            )
            _POOL_N = n_workers
        return _POOL


def pool_map(worker, args, n_workers):
    """Map ``worker`` over ``args`` on the shared pool, falling back to a
    serial in-process map if the pool breaks (a worker OOMs / dies →
    ``BrokenProcessPool``) or any pool error occurs. A dead worker must
    never crash the caller; on failure the broken pool is torn down so the
    next call rebuilds a fresh one. Returns the list of results."""
    from concurrent.futures.process import BrokenProcessPool

    ex = get_pool(n_workers)
    try:
        # Workers spawn lazily, on the first submit — so the pinned env has to
        # be live across the map, not merely across pool construction.
        # list() forces the lazy map generator, surfacing worker deaths here.
        with pinned_thread_env():
            return list(ex.map(worker, args))
    except (BrokenProcessPool, OSError, RuntimeError):
        # Only tear down the shared pool if it is STILL the executor we
        # used. Another thread may have resized the pool out from under us
        # (get_pool installs a fresh executor), in which case ``ex`` is a
        # stale reference whose ``map`` failed — but the live module pool
        # is healthy and must not be destroyed by our recovery.
        _shutdown_if_current(ex)
        return [worker(a) for a in args]


def _shutdown_if_current(ex) -> None:
    """Tear the shared pool down only if it is still ``ex`` (identity).

    Used by :func:`pool_map`'s recovery path so a stale-reference failure
    in one caller never destroys a pool a concurrent caller just rebuilt.
    """
    global _POOL, _POOL_N
    with _LOCK:
        if _POOL is ex and _POOL is not None:
            _POOL.shutdown(wait=False)
            _POOL = None
            _POOL_N = None


def shutdown_pool() -> None:
    """Tear the shared pool down unconditionally (called at interpreter
    exit). Clears whatever pool currently exists."""
    global _POOL, _POOL_N
    with _LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
            _POOL = None
            _POOL_N = None


atexit.register(shutdown_pool)


__all__ = [
    'THREAD_ENV',
    'get_pool',
    'pin_worker_threads',
    'pinned_thread_env',
    'pool_map',
    'shutdown_pool',
]
