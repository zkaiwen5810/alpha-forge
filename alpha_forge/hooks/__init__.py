"""Public lifecycle hook API."""

from alpha_forge.hooks.core import Hook, HookAction, HookRegistry
from alpha_forge.hooks.events import LifecycleEvent, PreToolExecution
from alpha_forge.hooks.matcher import HookMatcher, match_lifecycle, match_tool_names
from alpha_forge.hooks.permission import (
    PermissionAction,
    PermissionDeniedError,
    PermissionRequester,
)

__all__ = [
    "Hook",
    "HookAction",
    "HookMatcher",
    "HookRegistry",
    "LifecycleEvent",
    "PermissionAction",
    "PermissionDeniedError",
    "PermissionRequester",
    "PreToolExecution",
    "match_lifecycle",
    "match_tool_names",
]
