"""Transcript record envelope and storage errors."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_forge.transcript.events import TranscriptEvent

SCHEMA_VERSION = 1


class TranscriptError(RuntimeError):
    """Base exception for transcript operations."""


class TranscriptCorruptError(TranscriptError):
    """Raised when a completed transcript record or protocol is invalid."""


class TranscriptPersistenceError(TranscriptError):
    """Raised when a transcript cannot be durably updated."""


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    sequence: int
    event_id: str
    recorded_at: str
    event: TranscriptEvent


def default_transcript_directory() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "alpha-forge" / "transcripts").resolve()


__all__ = [
    "SCHEMA_VERSION",
    "TranscriptCorruptError",
    "TranscriptError",
    "TranscriptPersistenceError",
    "TranscriptRecord",
    "default_transcript_directory",
]
