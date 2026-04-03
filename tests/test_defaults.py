"""Tests for dvfopt._defaults — parameter resolution and size unpacking."""

import pytest
from dvfopt._defaults import (
    DEFAULT_PARAMS,
    _resolve_params,
    _unpack_size,
    _unpack_size_3d,
)


class TestResolveParams:
    def test_no_overrides_returns_defaults(self):
        p = _resolve_params()
        assert p == DEFAULT_PARAMS
        assert p is not DEFAULT_PARAMS  # must be a copy

    def test_override_replaces_value(self):
        p = _resolve_params(threshold=0.05)
        assert p["threshold"] == 0.05
        assert p["err_tol"] == DEFAULT_PARAMS["err_tol"]

    def test_none_override_keeps_default(self):
        p = _resolve_params(threshold=None)
        assert p["threshold"] == DEFAULT_PARAMS["threshold"]

    def test_multiple_overrides(self):
        p = _resolve_params(threshold=0.1, max_iterations=500)
        assert p["threshold"] == 0.1
        assert p["max_iterations"] == 500

    def test_extra_keys_added(self):
        p = _resolve_params(custom_key=99)
        assert p["custom_key"] == 99


class TestUnpackSize:
    def test_int_becomes_square_tuple(self):
        assert _unpack_size(5) == (5, 5)

    def test_tuple_passthrough(self):
        assert _unpack_size((3, 7)) == (3, 7)

    def test_list_passthrough(self):
        assert _unpack_size([4, 6]) == (4, 6)

    def test_float_int_truncated(self):
        assert _unpack_size(3.9) == (3, 3)

    def test_tuple_floats_truncated(self):
        assert _unpack_size((3.9, 5.1)) == (3, 5)


class TestUnpackSize3D:
    def test_int_becomes_cube(self):
        assert _unpack_size_3d(4) == (4, 4, 4)

    def test_triple_tuple(self):
        assert _unpack_size_3d((2, 3, 5)) == (2, 3, 5)

    def test_list_triple(self):
        assert _unpack_size_3d([2, 4, 6]) == (2, 4, 6)

    def test_non_triple_tuple_raises(self):
        # A 2-element tuple doesn't match len==3, falls through to int() on tuple → TypeError
        with pytest.raises(TypeError):
            _unpack_size_3d((3, 5))
