"""Persistence errors shared by storage and execution."""


class DatabaseBusyError(RuntimeError):
    """Another application owns this database."""


class PersistenceError(RuntimeError):
    """Persistence failed or a committed checkpoint is missing."""


class ThreadNotFoundError(ValueError):
    """No thread with the requested ID exists."""
