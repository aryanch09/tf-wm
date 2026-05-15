"""Tests for component registry."""

import pytest
from tfwm.utils.registry import Registry


def test_registry_register_and_get():
    """Test registering and getting components."""
    registry = Registry()

    @registry.register("test_component")
    class TestComponent:
        pass

    cls = registry.get("test_component")
    assert cls is TestComponent


def test_registry_instantiate():
    """Test instantiating registered components."""
    registry = Registry()

    @registry.register("test_component")
    class TestComponent:
        def __init__(self, value):
            self.value = value

    instance = registry.instantiate("test_component", value=42)
    assert instance.value == 42


def test_registry_unknown_component():
    """Test error for unknown component."""
    registry = Registry()

    with pytest.raises(KeyError, match="Unknown component"):
        registry.get("unknown")


def test_registry_list_components():
    """Test listing registered components."""
    registry = Registry()

    @registry.register("comp1")
    class Comp1:
        pass

    @registry.register("comp2")
    class Comp2:
        pass

    components = registry.list_components()
    assert set(components) == {"comp1", "comp2"}
