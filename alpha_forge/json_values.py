"""Immutable JSON values used inside durable transcript events."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

type JsonScalar = None | bool | int | float | str
type FrozenJsonValue = JsonScalar | tuple["FrozenJsonValue", ...] | FrozenJsonObject


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FrozenJsonObject(Mapping[str, FrozenJsonValue]):
    _data: Mapping[str, FrozenJsonValue]
    __hash__ = None

    def __init__(self, values: Mapping[str, object]) -> None:
        if any(not isinstance(key, str) for key in values):
            raise TypeError("JSON object keys must be strings")
        object.__setattr__(
            self,
            "_data",
            MappingProxyType(
                {key: freeze_json(value) for key, value in values.items()}
            ),
        )

    def __getitem__(self, key: str) -> FrozenJsonValue:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenJsonObject):
            return self._data == other._data
        if isinstance(other, Mapping):
            try:
                return self._data == FrozenJsonObject(other)._data
            except (TypeError, ValueError):
                return False
        return False

    def __deepcopy__(self, _memo: dict[int, object]) -> FrozenJsonObject:
        return self


def freeze_json(value: object) -> FrozenJsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return FrozenJsonObject(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def thaw_json(value: FrozenJsonValue) -> object:
    if isinstance(value, FrozenJsonObject):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


__all__ = [
    "FrozenJsonObject",
    "FrozenJsonValue",
    "JsonScalar",
    "freeze_json",
    "thaw_json",
]
