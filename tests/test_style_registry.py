import pytest

from app.styles.registry import StyleRegistry


def test_style_registry_switches_styles_correctly():
    registry = StyleRegistry()

    assert registry.names() == ["risograph", "cyanotype", "stippling"]
    assert registry.current().name == "risograph"
    assert registry.next().name == "cyanotype"
    assert registry.set("stippling").name == "stippling"
    assert registry.next().name == "risograph"


def test_style_registry_rejects_unknown_style():
    registry = StyleRegistry()

    with pytest.raises(KeyError):
        registry.set("unknown")
