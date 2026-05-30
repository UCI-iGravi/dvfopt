"""Decorator-based registries for benchmark variants and test cases."""

from typing import Callable

_VARIANTS: dict[str, Callable] = {}
_CASES: dict[str, Callable] = {}
_CASE_META: dict[str, dict] = {}


def register_variant(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        if name in _VARIANTS:
            raise ValueError(f"Variant {name!r} already registered")
        _VARIANTS[name] = fn
        return fn

    return deco


def register_case(name: str, *, category: str, dim: int, **extra) -> Callable:
    def deco(fn: Callable) -> Callable:
        if name in _CASES:
            raise ValueError(f"Case {name!r} already registered")
        _CASES[name] = fn
        _CASE_META[name] = {"category": category, "dim": dim, **extra}
        return fn

    return deco


def get_variant(name: str) -> Callable:
    return _VARIANTS[name]


def get_case(name: str) -> Callable:
    return _CASES[name]


def case_metadata(name: str) -> dict:
    return dict(_CASE_META[name])


def list_variants() -> list:
    return sorted(_VARIANTS)


def list_cases() -> list:
    return sorted(_CASES)


def clear() -> None:
    """Test-only: wipe the registries."""
    _VARIANTS.clear()
    _CASES.clear()
    _CASE_META.clear()
