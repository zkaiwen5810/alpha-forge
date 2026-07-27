"""Exclusive JSONL write-ahead transcript storage."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import fcntl

from alpha_forge.transcript.codec import decode_record, encode_record
from alpha_forge.transcript.events import (
    SessionOpened,
    ToolResult,
    TranscriptEvent,
)
from alpha_forge.transcript.records import (
    TranscriptCorruptError,
    TranscriptError,
    TranscriptPersistenceError,
    TranscriptRecord,
    default_transcript_directory,
)
from alpha_forge.transcript.validation import TranscriptState, validate_records


class TranscriptStore:
    """One linear transcript with an exclusive append entry point."""

    def __init__(
        self,
        *,
        path: Path | None,
        records: list[TranscriptRecord],
        state: TranscriptState,
        fd: int | None,
    ) -> None:
        self.path = path
        self._records = records
        self._state = state
        self._fd = fd
        self._poisoned = False
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        instructions: str | None,
        session_id: str | None = None,
        path: Path | None = None,
    ) -> TranscriptStore:
        resolved_id = session_id or uuid4().hex
        resolved_path = (
            path.expanduser().resolve()
            if path is not None
            else default_transcript_directory() / f"{resolved_id}.jsonl"
        )
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(resolved_path.parent, 0o700)
            fd = os.open(
                resolved_path,
                os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            _lock(fd, resolved_path)
            _sync_directory(resolved_path.parent)
        except (OSError, TranscriptPersistenceError) as exc:
            if isinstance(exc, TranscriptPersistenceError):
                raise
            raise TranscriptPersistenceError(
                f"cannot create transcript at {resolved_path}: {exc}"
            ) from exc
        store = cls(
            path=resolved_path,
            records=[],
            state=TranscriptState(),
            fd=fd,
        )
        try:
            store.append(
                SessionOpened(resolved_id, instructions),
                expected_revision=0,
            )
        except Exception:
            store.close()
            raise
        return store

    @classmethod
    def in_memory(
        cls,
        *,
        instructions: str | None = None,
        session_id: str | None = None,
    ) -> TranscriptStore:
        store = cls(
            path=None,
            records=[],
            state=TranscriptState(),
            fd=None,
        )
        store.append(
            SessionOpened(session_id or uuid4().hex, instructions),
            expected_revision=0,
        )
        return store

    @classmethod
    def resume(cls, path: Path) -> TranscriptStore:
        resolved = path.expanduser().resolve()
        try:
            fd = os.open(resolved, os.O_RDWR | os.O_APPEND)
            _lock(fd, resolved)
            raw = _read_all(fd)
        except (OSError, TranscriptPersistenceError) as exc:
            if isinstance(exc, TranscriptPersistenceError):
                raise
            raise TranscriptError(
                f"cannot read transcript at {resolved}: {exc}"
            ) from exc
        try:
            raw = _repair_tail(fd, raw, resolved)
            records = _decode_records(raw)
            state = validate_records(records)
        except Exception:
            _unlock_and_close(fd)
            raise
        return cls(
            path=resolved,
            records=records,
            state=state,
            fd=fd,
        )

    @property
    def records(self) -> tuple[TranscriptRecord, ...]:
        return tuple(self._records)

    @property
    def events(self) -> tuple[TranscriptEvent, ...]:
        return tuple(record.event for record in self._records)

    @property
    def revision(self) -> int:
        return len(self._records)

    @property
    def state(self) -> TranscriptState:
        # Expose a read snapshot of replay indexes. Events and contents remain
        # shared immutable references; mutable index containers do not.
        return self._state.clone()

    @property
    def session_id(self) -> str:
        assert self._state.session is not None
        return self._state.session.session_id

    @property
    def instructions(self) -> str | None:
        assert self._state.session is not None
        return self._state.session.instructions

    def append(
        self,
        event: TranscriptEvent,
        *,
        expected_revision: int,
    ) -> TranscriptRecord:
        if self._closed:
            raise TranscriptPersistenceError("transcript writer is closed")
        if self._poisoned:
            raise TranscriptPersistenceError(
                "transcript writer is unavailable after an earlier write failure"
            )
        if expected_revision != self.revision:
            raise TranscriptPersistenceError(
                f"stale transcript revision: expected {expected_revision}, "
                f"current {self.revision}"
            )
        record = TranscriptRecord(
            sequence=self.revision,
            event_id=uuid4().hex,
            recorded_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            event=event,
        )
        candidate_state = self._state.clone()
        candidate_state.apply(record)

        try:
            data = (
                json.dumps(
                    encode_record(record),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise TranscriptCorruptError(
                f"transcript event cannot be encoded: {exc}"
            ) from exc

        if self._fd is not None:
            try:
                view = memoryview(data)
                while view:
                    written = os.write(self._fd, view)
                    if written <= 0:
                        raise OSError("zero-byte transcript write")
                    view = view[written:]
                os.fsync(self._fd)
            except OSError as exc:
                self._poisoned = True
                raise TranscriptPersistenceError(
                    f"cannot append transcript at {self.path}: {exc}"
                ) from exc

        # The candidate was validated before the WAL write. It becomes visible
        # to in-process readers only after the durable append succeeds.
        self._state = candidate_state
        self._records.append(record)
        return record

    def result(self, result_event_id: str) -> tuple[TranscriptRecord, ToolResult]:
        for record in self._records:
            if record.event_id == result_event_id and isinstance(
                record.event,
                ToolResult,
            ):
                return record, record.event
        raise KeyError(result_event_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fd is not None:
            _unlock_and_close(self._fd)
            self._fd = None

    def __enter__(self) -> TranscriptStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _lock(fd: int, path: Path) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        raise TranscriptPersistenceError(
            f"transcript is already open for writing: {path}"
        ) from exc


def _unlock_and_close(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_all(fd: int) -> bytes:
    size = os.fstat(fd).st_size
    data = bytearray()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            break
        data.extend(chunk)
        offset += len(chunk)
    return bytes(data)


def _repair_tail(fd: int, raw: bytes, path: Path) -> bytes:
    if not raw:
        raise TranscriptCorruptError("transcript is empty")
    if raw.endswith(b"\n"):
        return raw
    newline = raw.rfind(b"\n")
    if newline < 0:
        raise TranscriptCorruptError("transcript contains no completed records")
    repaired = raw[: newline + 1]
    try:
        os.ftruncate(fd, len(repaired))
        os.fsync(fd)
    except OSError as exc:
        raise TranscriptPersistenceError(
            f"cannot repair transcript tail at {path}: {exc}"
        ) from exc
    return repaired


def _decode_records(raw: bytes) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            decoded = json.loads(line)
            records.append(decode_record(decoded))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TranscriptCorruptError(
                f"invalid transcript record on line {line_number}: {exc}"
            ) from exc
    return records


__all__ = ["TranscriptStore"]
