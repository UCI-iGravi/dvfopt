"""Logging setup for DVFopt.

Provides a package-level logger ``dvfopt`` plus a thin compatibility
shim so existing ``print(..., flush=True)`` call sites can migrate
without churning every call.

Convention
----------
* **All solver progress output** goes through the package logger
  (``dvfopt`` namespace). Levels:
    - ``INFO`` for top-level per-pass summaries (the lines you'd want
      to see in a notebook).
    - ``DEBUG`` for per-iteration detail.
    - ``WARNING`` for budget-exhaustion + recoverable failures.
    - ``ERROR`` for actual errors (typically before raising).
* **The package never installs a handler at import time** — that
  remains the caller's choice. Use :func:`enable_default_handler` to
  attach a simple stderr handler with sensible formatting; users with
  their own logging config simply don't call it.
* The ``verbose=`` kwargs on solver functions still work as before:
  ``verbose=0`` silences everything; ``verbose>=1`` raises the logger
  to INFO and (if not already configured) installs the default handler
  for the duration of the call.

This is opt-in: existing callers that pass ``verbose=0`` get no output;
callers that pass ``verbose>=1`` see the same lines as before but
routed through the logger.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager

logger = logging.getLogger('dvfopt')
logger.addHandler(logging.NullHandler())  # silence "no handler" warnings


def enable_default_handler(level: int = logging.INFO, stream=None) -> None:
    """Attach a simple stderr formatter to the ``dvfopt`` logger.

    Idempotent — repeated calls don't duplicate the handler.
    """
    # Check whether we already added our handler.
    for h in logger.handlers:
        if getattr(h, '_dvfopt_default', False):
            h.setLevel(level)
            logger.setLevel(level)
            return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter('%(message)s'))
    handler.setLevel(level)
    handler._dvfopt_default = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(level)


def _ensure_visible(want_level: int) -> None:
    """Make sure a record at ``want_level`` would actually be emitted.

    Installs the default stderr handler when the caller has no real
    handler configured on the ``dvfopt`` logger (preserving the old
    print-visibility). When OUR default handler is the one filtering,
    lower it to ``want_level``; a user-configured handler is left alone
    (their level wins).
    """
    has_real = any(not isinstance(h, logging.NullHandler) for h in logger.handlers)
    if not has_real:
        enable_default_handler(level=want_level)
        return
    if logger.isEnabledFor(want_level):
        return
    for h in logger.handlers:
        if getattr(h, '_dvfopt_default', False):
            enable_default_handler(level=want_level)
            break


def vlog(verbose: int, level: int, msg: str) -> None:
    """Verbosity-gated logging shim for solver internals.

    Drop-in replacement for the historical
    ``if verbose >= level: print(msg, flush=True)`` pattern: emits *msg*
    through the ``dvfopt`` logger at INFO (``level <= 1``) or DEBUG
    (``level >= 2``) iff ``verbose >= level``, with the auto-handler
    behavior of :func:`_ensure_visible`.
    """
    if verbose < level:
        return
    lvl = logging.INFO if level <= 1 else logging.DEBUG
    _ensure_visible(lvl)
    logger.log(lvl, msg)


def log_info(msg: str) -> None:
    """Unconditional INFO emit with the auto-handler behavior.

    For solver call sites already guarded by their own ``if verbose:`` —
    the historical ``print(msg, flush=True)`` drop-in.
    """
    vlog(1, 1, msg)


def log_warning(msg: str) -> None:
    """Unconditional WARNING emit with the auto-handler behavior.

    For recoverable solver failures (a cluster solve raising, a step
    callback blowing up) that must surface regardless of ``verbose``.
    """
    _ensure_visible(logging.WARNING)
    logger.warning(msg)


@contextmanager
def verbose_scope(verbose: int):
    """Context manager: raise/lower the package log level for the body.

    ``verbose>=1`` enables INFO; ``verbose=0`` silences the package
    logger. On exit, restores the previous level.
    """
    prev = logger.level
    try:
        if verbose >= 1:
            enable_default_handler(level=logging.DEBUG if verbose >= 2 else logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        yield logger
    finally:
        logger.setLevel(prev)


__all__ = [
    'enable_default_handler',
    'log_info',
    'log_warning',
    'logger',
    'verbose_scope',
    'vlog',
]
