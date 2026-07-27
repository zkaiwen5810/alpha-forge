"""Ordered context-policy composition."""

from __future__ import annotations

from collections.abc import Callable

from alpha_forge.context.models import ModelContextSnapshot
from alpha_forge.context.policies import ContextPolicy
from alpha_forge.transcript.events import ContextEdited


class ContextPipeline:
    """Evaluate configured policies serially with commit feedback between them."""

    def __init__(self, policies: tuple[ContextPolicy, ...] = ()) -> None:
        self.policies = policies

    def prepare(
        self,
        *,
        project: Callable[[], ModelContextSnapshot],
        commit: Callable[[ContextEdited], object],
    ) -> ModelContextSnapshot:
        """Apply policies serially, reprojecting only after a durable edit.

        A policy that has nothing to change emits no transcript event. The
        callback keeps transcript ownership outside the context subsystem.
        """

        snapshot = project()
        for policy in self.policies:
            decision = policy.evaluate(snapshot)
            if not decision.operations:
                continue
            commit(ContextEdited(decision.policy, decision.operations))
            snapshot = project()
        return snapshot


__all__ = ["ContextPipeline"]
