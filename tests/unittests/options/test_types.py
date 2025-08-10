from collections.abc import Mapping

import pytest

from twrapform.options import FrozenDict


def test_basic_access():
    v = FrozenDict({"key": "value", "num": 42})
    assert isinstance(v, Mapping)
    assert v["key"] == "value"
    assert v["num"] == 42


def test_nested_freeze():
    v = FrozenDict({"nested": {"a": 1, "b": [2, 3], "c": {"d": "x"}}})
    assert isinstance(v["nested"], FrozenDict)
    assert v["nested"]["b"] == (2, 3)
    assert isinstance(v["nested"]["c"], FrozenDict)


def test_export_returns_deep_copy():
    v = FrozenDict({"x": [1, 2], "y": {"z": "ok"}})
    exported = v.export()
    assert isinstance(exported["x"], tuple)  # [1, 2] → (1, 2)
    assert exported["x"] == (1, 2)
    assert exported["y"] == {"z": "ok"}
    exported["x"] = "modified"  # confirm it's a mutable copy
    assert v["x"] == (1, 2)


def test_attribute_immutability():
    v = FrozenDict({"a": "b"})

    with pytest.raises(TypeError):
        v["a"] = "new"  # even if you reach inside

    with pytest.raises(AttributeError):
        v.new_attribute = 123  # not allowed due to __slots__

    with pytest.raises(AttributeError):
        del v._data


def test_repr_and_len():
    v = FrozenDict({"a": 1, "b": 2})
    assert len(v) == 2
    assert list(v) == ["a", "b"]


def test_deeply_nested_structure():
    v = FrozenDict(
        {
            "level1": {
                "level2": {
                    "level3": {
                        "value": [1, {"inner": "x"}],
                        "flags": {"a", "b"},
                    }
                }
            },
            "simple": True,
        }
    )

    # check structure
    assert isinstance(v["level1"], FrozenDict)
    assert isinstance(v["level1"]["level2"], FrozenDict)
    assert isinstance(v["level1"]["level2"]["level3"], FrozenDict)

    # nested list to tuple
    assert isinstance(v["level1"]["level2"]["level3"]["value"], tuple)
    assert v["level1"]["level2"]["level3"]["value"][0] == 1

    # tuple in dict to FrozenDict
    inner = v["level1"]["level2"]["level3"]["value"][1]
    assert isinstance(inner, FrozenDict)
    assert inner["inner"] == "x"

    # set to frozenset
    flags = v["level1"]["level2"]["level3"]["flags"]
    assert isinstance(flags, frozenset)
    assert "a" in flags and "b" in flags


def test_export_deep_nesting():
    v = FrozenDict(
        {"nested": {"sub": {"data": [10, {"key": "val"}], "enabled": False}}}
    )

    exported = v.export()
    assert exported == {
        "nested": {"sub": {"data": (10, {"key": "val"}), "enabled": False}}
    }

    assert isinstance(exported["nested"], dict)
    assert isinstance(exported["nested"]["sub"]["data"][1], dict)
