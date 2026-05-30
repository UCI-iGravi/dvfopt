"""Tests for the dvfopt logging setup."""

from __future__ import annotations

import io
import logging

import pytest

import dvfopt
from dvfopt._logging import enable_default_handler, logger, verbose_scope


def test_package_logger_exists_and_is_namespaced():
    assert logger.name == 'dvfopt'
    # Even before enable_default_handler, the package must have a
    # NullHandler attached so logger.info() doesn't print "No handler".
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_enable_default_handler_is_idempotent():
    n0 = len(logger.handlers)
    enable_default_handler()
    n1 = len(logger.handlers)
    enable_default_handler()
    n2 = len(logger.handlers)
    assert n1 > n0  # added on first call
    assert n2 == n1  # second call no-op


def test_verbose_scope_raises_then_restores_level():
    """verbose=1 raises to INFO; verbose=0 silences (WARNING+); the
    context manager restores the prior level."""
    prev = logger.level
    with verbose_scope(0):
        assert logger.level == logging.WARNING
    assert logger.level == prev
    with verbose_scope(1):
        assert logger.level == logging.INFO
    assert logger.level == prev


def test_logger_routes_to_attached_handler():
    """A handler attached to ``dvfopt`` logger should receive package
    log messages."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter('%(message)s'))
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info('hello from dvfopt')
        out = buf.getvalue()
        assert 'hello from dvfopt' in out
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def test_top_level_exposes_logger_helpers():
    """``logger`` and ``enable_default_handler`` should be importable
    from the package root."""
    assert dvfopt.logger is logger
    assert callable(dvfopt.enable_default_handler)
