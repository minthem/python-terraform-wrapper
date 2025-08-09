import os
from collections.abc import Mapping
from json import dumps
from typing import Any

from .types import FrozenDict


def bool_option(opt_flag: str, value: bool | None) -> tuple[str, ...]:
    if value is not None:
        return (f"{opt_flag}={str(value).lower()}",)
    else:
        return ()


def bool_flag_option(opt_flag: str, value: bool | None) -> tuple[str, ...]:
    return (opt_flag,) if value is not None and value else ()


def backend_config_option(
    opt_flag: str, value: dict[str, str] | FrozenDict | str | None
) -> tuple[str, ...]:
    if value is None:
        return ()
    elif isinstance(value, str):
        return (f"{opt_flag}={value}",)
    elif isinstance(value, Mapping):
        return tuple(f"{opt_flag}='{k}={v}'" for k, v in sorted(value.items()))
    else:
        raise TypeError(f"{opt_flag} Unsupported type {type(value)}")


def list_str_option(opt_flag: str, value: tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()

    return tuple(f"{opt_flag}={v}" for v in value)


def var_option(
    opt_flag: str,
    value: FrozenDict | dict[str, Any] | None,
) -> tuple[str, ...]:
    if value is None:
        return tuple()

    result = tuple()

    for key in sorted(value.keys()):
        val = value[key]
        if isinstance(val, bool):
            result += (opt_flag, f"{key}={str(val).lower()}")
        elif isinstance(val, (int, float, str)):
            result += (
                opt_flag,
                f"{key}={str(val)}",
            )
        elif isinstance(val, (list, tuple, set, frozenset)):
            if isinstance(val, set):
                v_list = sorted(list(val))
            else:
                v_list = list(val)
            result += (
                opt_flag,
                f"{key}={dumps(v_list)}",
            )
        elif isinstance(val, Mapping) or val is None:
            result += (
                opt_flag,
                f"{key}={dumps(val, sort_keys=True, default=_json_serializer)}",
            )
        else:
            raise TypeError(f"Unsupported type {type(val)} (key: {key}, value: {val})")

    return result


def _json_serializer(obj):
    if isinstance(obj, FrozenDict):
        return obj.export()
    elif isinstance(obj, frozenset | set | tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
