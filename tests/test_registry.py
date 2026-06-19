"""Unit tests for the registry system."""

import pytest
from flashdet.registry import Registry


def test_register_and_build():
    reg = Registry("test")

    @reg.register("MyClass")
    class MyClass:
        def __init__(self, value=10):
            self.value = value

    obj = reg.build("MyClass", value=42)
    assert obj.value == 42


def test_register_without_name():
    reg = Registry("test")

    @reg.register()
    class AnotherClass:
        pass

    assert "AnotherClass" in reg


def test_duplicate_raises():
    reg = Registry("test")

    @reg.register("Dup")
    class Dup1:
        pass

    with pytest.raises(KeyError):
        @reg.register("Dup")
        class Dup2:
            pass


def test_build_missing_raises():
    reg = Registry("test")
    with pytest.raises(KeyError, match="not found"):
        reg.build("NonExistent")


def test_list():
    reg = Registry("test")

    @reg.register("B")
    class B:
        pass

    @reg.register("A")
    class A:
        pass

    assert reg.list() == ["A", "B"]
