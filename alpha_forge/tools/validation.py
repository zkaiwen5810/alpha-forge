"""JSON Schema validation for provider-produced tool inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from alpha_forge.json_values import FrozenJsonObject, thaw_json


class ToolInputValidationError(ValueError):
    """Raised when decoded tool arguments violate their declared schema."""


def check_input_schema(schema: FrozenJsonObject) -> None:
    """Fail early when a tool definition contains an invalid schema."""

    try:
        Draft202012Validator.check_schema(thaw_json(schema))
    except SchemaError as exc:
        raise ValueError(f"invalid tool input schema: {exc.message}") from exc


class ToolInputValidator:
    """Compiled Draft 2020-12 validator for one registered tool."""

    def __init__(self, schema: FrozenJsonObject) -> None:
        self._validator = Draft202012Validator(
            thaw_json(schema),
            format_checker=FormatChecker(),
        )

    def validate(self, arguments: Mapping[str, Any]) -> None:
        errors = sorted(
            self._validator.iter_errors(dict(arguments)),
            key=lambda error: (
                tuple(str(item) for item in error.absolute_path),
                error.message,
            ),
        )
        if not errors:
            return
        error = errors[0]
        path = "$" + "".join(
            f"[{item}]" if isinstance(item, int) else f".{item}"
            for item in error.absolute_path
        )
        raise ToolInputValidationError(
            f"tool input does not match schema at {path}: {error.message}"
        )


__all__ = [
    "ToolInputValidationError",
    "ToolInputValidator",
    "check_input_schema",
]
