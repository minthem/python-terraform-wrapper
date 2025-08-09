from collections.abc import Mapping
from typing import Any, Iterator


class FrozenDict(Mapping):
    """
    Immutable, recursively frozen mapping for safe configuration or structured data.

    This class wraps a dictionary-like structure and recursively freezes nested
    dicts, lists, and sets to ensure immutability. Once initialized, neither the
    top-level mapping nor any nested structure can be modified.

    - dicts are wrapped as Vars recursively
    - lists are converted to tuples
    - sets are converted to frozensets
    - arbitrary values are preserved as-is

    Attributes are locked using __slots__, and attempts to overwrite or delete them
    raise AttributeError. Useful in contexts where accidental mutation must be
    prevented (e.g., configuration, caching, snapshotting state).

    Example:
        >>> config = FrozenDict({
        ...     "db": {"host": "localhost", "ports": [5432, 5433]},
        ...     "debug": True,
        ...     "tags": {"alpha", "beta"}
        ... })
        >>> config["db"]["host"]
        'localhost'
        >>> config.export()
        {'db': {'host': 'localhost', 'ports': (5432, 5433)}, 'debug': True, 'tags': frozenset({'alpha', 'beta'})}
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]):
        self._data = {k: self._freeze(v) for k, v in data.items()}

    def __getitem__(self, key):
        return self._data[key]

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(
                f"{self.__class__.__name__} is immutable: cannot overwrite {name}"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        raise AttributeError(
            f"{self.__class__.__name__} is immutable: cannot delete attributes"
        )

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _freeze(self, value):
        if isinstance(value, dict):
            return FrozenDict(value)
        elif isinstance(value, list):
            return tuple(self._freeze(v) for v in value)
        elif isinstance(value, set):
            return frozenset(self._freeze(v) for v in value)
        else:
            return value

    def _unfreeze(self, value):
        if isinstance(value, FrozenDict):
            return value.export()
        elif isinstance(value, tuple):
            return tuple([self._unfreeze(v) for v in value])
        elif isinstance(value, frozenset):
            return set([self._unfreeze(v) for v in value])
        else:
            return value

    def export(self) -> dict:
        return {k: self._unfreeze(v) for k, v in self._data.items()}
