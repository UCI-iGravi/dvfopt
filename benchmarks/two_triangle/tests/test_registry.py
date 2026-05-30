import pytest

from benchmarks.two_triangle import registry


def test_register_and_get_variant():
    @registry.register_variant("test_variant_1")
    def _v(phi, **kw):
        return None

    assert registry.get_variant("test_variant_1") is _v
    assert "test_variant_1" in registry.list_variants()


def test_register_and_get_case():
    @registry.register_case("test_case_1", category="synthetic_2d", dim=2)
    def _c():
        return None, {}

    assert registry.get_case("test_case_1") is _c
    meta = registry.case_metadata("test_case_1")
    assert meta["category"] == "synthetic_2d"
    assert meta["dim"] == 2


def test_duplicate_variant_raises():
    @registry.register_variant("dup_variant")
    def _a(phi, **kw):
        return None

    with pytest.raises(ValueError, match="already registered"):

        @registry.register_variant("dup_variant")
        def _b(phi, **kw):
            return None


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        registry.get_variant("nonexistent")
