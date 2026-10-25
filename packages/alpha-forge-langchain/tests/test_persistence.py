"""Upgrade existing thread metadata without losing native conversation state."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import aiosqlite
from alpha_forge_langchain import AgentApplication
from alpha_forge_langchain.store import ThreadStore
from langchain_core.messages import AIMessage
from test_agent_application import ScriptedModel, aconsume


class PersistenceMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.database = Path(temporary.name) / "migration.sqlite3"

    async def test_startup_upgrades_payload_schema_and_preserves_checkpoints(self):
        model = ScriptedModel(
            responses=[
                AIMessage(content="first answer"),
                AIMessage(content="second answer"),
                RuntimeError("unfinished turn"),
            ]
        )
        async with AgentApplication(self.database, model=model) as app:
            first = await app.threads.acreate()
            second = await app.threads.acreate()
            await aconsume(app, first, "first")
            await aconsume(app, second, "second")
            with self.assertRaisesRegex(RuntimeError, "unfinished turn"):
                await aconsume(app, second, "abandoned")
            original = [await app.threads.aget(tid) for tid in (first, second)]

        with closing(sqlite3.connect(self.database)) as connection:
            # Reproduce the former NOT NULL constraints, including an obsolete
            # payload which must never be needed to restore native history.
            connection.execute(
                "ALTER TABLE threads ADD COLUMN state_type TEXT NOT NULL DEFAULT 'msgpack'"
            )
            connection.execute(
                "ALTER TABLE threads ADD COLUMN state BLOB NOT NULL DEFAULT X'00'"
            )
            checkpoints = connection.execute(
                "SELECT * FROM checkpoints ORDER BY thread_id, checkpoint_ns, checkpoint_id"
            ).fetchall()
            writes = connection.execute(
                "SELECT * FROM writes ORDER BY thread_id, checkpoint_ns, checkpoint_id, task_id, idx"
            ).fetchall()

        async with AgentApplication(
            self.database,
            model=ScriptedModel(responses=[AIMessage(content="after upgrade")]),
        ) as app:
            self.assertEqual(
                [await app.threads.aget(tid) for tid in (first, second)], original
            )
            self.assertEqual(
                [m.text for m in await app.threads.ahistory(first)],
                ["first", "first answer"],
            )
            self.assertEqual(
                [m.text for m in await app.threads.ahistory(second)],
                ["second", "second answer"],
            )
            with closing(sqlite3.connect(self.database)) as connection:
                self.assertEqual(
                    [
                        row[1]
                        for row in connection.execute("PRAGMA table_info(threads)")
                    ],
                    ["thread_id", "checkpoint_id", "unfinished"],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT * FROM checkpoints ORDER BY thread_id, checkpoint_ns, checkpoint_id"
                    ).fetchall(),
                    checkpoints,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT * FROM writes ORDER BY thread_id, checkpoint_ns, checkpoint_id, task_id, idx"
                    ).fetchall(),
                    writes,
                )
            created = await app.threads.acreate()  # Previously raised NOT NULL.
            result = await aconsume(app, first, "continue")
            self.assertEqual(
                [m.text for m in result["messages"]],
                ["first", "first answer", "continue", "after upgrade"],
            )

        # Reopening an already migrated database is harmless and remains keyless.
        async with AgentApplication(self.database) as app:
            self.assertEqual(await app.threads.ahistory(created), [])
            self.assertEqual((await app.threads.aget(second)), original[1])
            self.assertEqual(len(await app.threads.ahistory(first)), 4)

    async def test_failed_upgrade_rolls_back_column_removal(self):
        async with aiosqlite.connect(self.database) as connection:
            await connection.execute("""
                CREATE TABLE threads (
                    thread_id TEXT PRIMARY KEY, checkpoint_id TEXT NOT NULL,
                    state_type TEXT NOT NULL, state BLOB NOT NULL,
                    unfinished INTEGER NOT NULL DEFAULT 0
                )
            """)
            await connection.execute(
                "INSERT INTO threads VALUES ('thread', 'head', 'msgpack', X'00', 1)"
            )
            await connection.commit()
            execute = connection.execute

            def fail_second_drop(sql, *args, **kwargs):
                if sql == "ALTER TABLE threads DROP COLUMN state":
                    raise sqlite3.OperationalError("migration failed")
                return execute(sql, *args, **kwargs)

            with (
                patch.object(connection, "execute", side_effect=fail_second_drop),
                self.assertRaisesRegex(sqlite3.OperationalError, "migration failed"),
            ):
                await ThreadStore(connection).asetup()

            async with connection.execute("PRAGMA table_info(threads)") as cursor:
                self.assertEqual(
                    [row[1] for row in await cursor.fetchall()],
                    ["thread_id", "checkpoint_id", "state_type", "state", "unfinished"],
                )
            async with connection.execute("SELECT * FROM threads") as cursor:
                self.assertEqual(
                    await cursor.fetchall(), [("thread", "head", "msgpack", b"\x00", 1)]
                )
            await ThreadStore(connection).asetup()
            info = await ThreadStore(connection).aget("thread")
            self.assertEqual(info.checkpoint_id, "head")
            self.assertTrue(info.unfinished_work_excluded)
