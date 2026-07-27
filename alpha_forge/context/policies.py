"""Context policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from alpha_forge.context.models import ModelContextSnapshot
from alpha_forge.transcript.events import ContextOperation, PolicyInvocation


@dataclass(frozen=True, slots=True)
class ContextPolicyDecision:
    policy: PolicyInvocation
    operations: tuple[ContextOperation, ...] = ()


class ContextPolicy(Protocol):
    def evaluate(
        self,
        snapshot: ModelContextSnapshot,
    ) -> ContextPolicyDecision:
        """Return declarative operations for one committed context revision."""


__all__ = ["ContextPolicy", "ContextPolicyDecision"]
