"""Budget, preview, and persist tool results before model submission."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from alpha_forge.conversation import (
    PreviewReason,
    ToolMessage,
    ToolResultPreview,
)

MAX_TOOL_RESULT_CHARS = 16_000
# MAX_TOOL_RESULT_CHARS = 512
MAX_TOOL_RESULTS_CHARS = 32_000
_SAFE_CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")


class ToolResultBudgetError(RuntimeError):
    """Raised when required preview metadata cannot fit in its budget."""


class ToolResultPersistenceError(RuntimeError):
    """Raised when a previewed result cannot be saved safely."""


@dataclass(frozen=True)
class RawToolResult:
    """One complete result collected before iteration-level budgeting."""

    tool_call_id: str
    content: str
    failed: bool = False


def default_persist_directory() -> Path:
    """Return Alpha Forge's XDG data directory for persisted results."""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "alpha-forge").resolve()


class ToolResultManager:
    """Apply result budgets and persist previews for a supplied session."""

    def __init__(
        self,
        *,
        persist_directory: Path | None = None,
        individual_limit: int = MAX_TOOL_RESULT_CHARS,
        aggregate_limit: int = MAX_TOOL_RESULTS_CHARS,
    ) -> None:
        if individual_limit <= 0:
            raise ValueError("individual tool-result limit must be positive")
        if aggregate_limit <= 0:
            raise ValueError("aggregate tool-result limit must be positive")
        self.persist_directory = (
            persist_directory or default_persist_directory()
        ).expanduser().resolve()
        self.individual_limit = individual_limit
        self.aggregate_limit = aggregate_limit

    def process(
        self,
        results: tuple[RawToolResult, ...],
        *,
        session_id: str,
    ) -> tuple[ToolMessage, ...]:
        """Return model-facing results bounded individually and in aggregate."""
        if not _SAFE_SESSION_ID.fullmatch(session_id):
            raise ValueError(
                "session ID must contain only letters, digits, _ or -"
            )
        if not results:
            return ()

        # Apply the per-result cap first, then water-fill those desired sizes
        # against the iteration cap. This makes aggregate pressure fair without
        # taking space away from results that are already below the fair share.
        desired_lengths = [
            min(len(result.content), self.individual_limit)
            for result in results
        ]
        allocated_lengths = self._water_fill(
            desired_lengths,
            self.aggregate_limit,
        )

        messages: list[ToolMessage] = []
        pending_writes: list[tuple[Path, str]] = []
        seen_paths: set[Path] = set()

        # Build every preview and validate all metadata overhead before writing
        # anything. The later persistence phase can therefore roll back only
        # I/O failures; budget failures never leave orphaned files.
        for result, desired, allocated in zip(
            results,
            desired_lengths,
            allocated_lengths,
            strict=True,
        ):
            if len(result.content) <= allocated:
                messages.append(
                    ToolMessage(
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                        failed=result.failed,
                    )
                )
                continue

            reason = self._preview_reason(
                original_length=len(result.content),
                desired_length=desired,
                allocated_length=allocated,
            )
            path = self._result_path(
                session_id,
                result.tool_call_id,
                result.content,
            )
            if path in seen_paths:
                raise ToolResultPersistenceError(
                    f"duplicate persisted tool-result path: {path}"
                )
            seen_paths.add(path)
            preview_content = self._build_preview(
                result.content,
                path=path,
                reason=reason,
                max_chars=allocated,
            )
            metadata = ToolResultPreview(
                persisted_path=path,
                original_chars=len(result.content),
                reason=reason,
            )
            messages.append(
                ToolMessage(
                    content=preview_content,
                    tool_call_id=result.tool_call_id,
                    failed=result.failed,
                    preview=metadata,
                )
            )
            pending_writes.append((path, result.content))

        if any(len(message.content) > self.individual_limit for message in messages):
            raise AssertionError("individual tool-result budget was exceeded")
        if sum(len(message.content) for message in messages) > self.aggregate_limit:
            raise AssertionError("aggregate tool-result budget was exceeded")
        self._persist(pending_writes)
        return tuple(messages)

    @staticmethod
    def _water_fill(caps: list[int], total: int) -> list[int]:
        """Allocate ``total`` fairly without exceeding any result's cap."""
        if sum(caps) <= total:
            return list(caps)

        allocations = [0] * len(caps)
        remaining = total
        pending = sorted(range(len(caps)), key=caps.__getitem__)
        while pending:
            share, remainder = divmod(remaining, len(pending))
            smallest = pending[0]
            if caps[smallest] <= share:
                allocations[smallest] = caps[smallest]
                remaining -= caps[smallest]
                pending.pop(0)
                continue
            for position, index in enumerate(pending):
                allocations[index] = share + (1 if position < remainder else 0)
            break
        return allocations

    @staticmethod
    def _preview_reason(
        *,
        original_length: int,
        desired_length: int,
        allocated_length: int,
    ) -> PreviewReason:
        individual = original_length > desired_length
        aggregate = allocated_length < desired_length
        if individual and aggregate:
            return "individual_and_aggregate_limits"
        if individual:
            return "individual_limit"
        return "aggregate_limit"

    @staticmethod
    def _build_preview(
        content: str,
        *,
        path: Path,
        reason: PreviewReason,
        max_chars: int,
    ) -> str:
        path_literal = json.dumps(str(path), ensure_ascii=False)
        prefix = (
            "[alpha-forge tool-result-preview]\n"
            "truncated: true\n"
            f"reason: {reason}\n"
            f"original_chars: {len(content)}\n"
            f"persisted_path: {path_literal}\n"
            "--- preview head ---\n"
        )
        middle = "\n--- content omitted ---\n"
        suffix = "\n--- preview tail ---\n"
        overhead = len(prefix) + len(middle) + len(suffix)
        if overhead > max_chars:
            raise ToolResultBudgetError(
                "tool-result budget is too small for required preview metadata "
                f"({max_chars} available, {overhead} required)"
            )
        excerpt_chars = max_chars - overhead
        head_chars = (excerpt_chars + 1) // 2
        tail_chars = excerpt_chars // 2
        head = content[:head_chars]
        tail = content[-tail_chars:] if tail_chars else ""
        return prefix + head + middle + suffix + tail

    def _result_path(
        self,
        session_id: str,
        tool_call_id: str,
        content: str,
    ) -> Path:
        extension = "json" if self._is_json(content) else "txt"
        filename = f"{self._safe_call_id(tool_call_id)}.{extension}"
        return (
            self.persist_directory
            / session_id
            / "tool-results"
            / filename
        )

    @staticmethod
    def _safe_call_id(tool_call_id: str) -> str:
        if _SAFE_CALL_ID.fullmatch(tool_call_id):
            return tool_call_id
        digest = hashlib.sha256(
            tool_call_id.encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:12]
        readable = re.sub(r"[^A-Za-z0-9_-]+", "_", tool_call_id).strip("_")
        return f"{(readable[:80] or 'tool-call')}-{digest}"

    @staticmethod
    def _is_json(content: str) -> bool:
        try:
            json.loads(content)
        except (json.JSONDecodeError, ValueError, RecursionError):
            return False
        return True

    @classmethod
    def _persist(cls, writes: list[tuple[Path, str]]) -> None:
        written: list[Path] = []
        try:
            for path, content in writes:
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                cls._write_exclusive(path, content)
                written.append(path)
        except (OSError, UnicodeError) as exc:
            for path in written:
                path.unlink(missing_ok=True)
            raise ToolResultPersistenceError(
                f"cannot persist full tool result at {path}: {exc}"
            ) from exc

    @staticmethod
    def _write_exclusive(path: Path, content: str) -> None:
        created = False
        try:
            # Tool-call IDs should be unique within a session. Refusing to
            # overwrite turns a provider ID collision into a visible failure
            # instead of silently replacing an earlier complete result.
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            created = True
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
        except (OSError, UnicodeError):
            if created:
                path.unlink(missing_ok=True)
            raise
