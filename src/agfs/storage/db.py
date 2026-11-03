"""Layer 1: SQLite 3.54+ WAL connection engine and atomic transaction manager."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agfs.storage.schema import PRAGMAS_SQL, SCHEMA_DDL


class SQLiteWALEngine:
    """Manages a single-file `.agdb` SQLite 3 database in WAL mode."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,  # Explicit transaction control
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.executescript(PRAGMAS_SQL)
        self.conn.executescript(SCHEMA_DDL)

    @contextmanager
    def transaction(self, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Execute an atomic SQLite transaction (nested-safe via savepoints)."""
        if self.conn.in_transaction:
            sp_name = f"sp_{ time.time_ns() }"
            self.conn.execute(f"SAVEPOINT {sp_name}")
            try:
                yield self.conn
                self.conn.execute(f"RELEASE SAVEPOINT {sp_name}")
            except Exception:
                self.conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
                self.conn.execute(f"RELEASE SAVEPOINT {sp_name}")
                raise
        else:
            mode = "BEGIN IMMEDIATE" if immediate else "BEGIN"
            self.conn.execute(mode)
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except Exception:
                if self.conn.in_transaction:
                    self.conn.execute("ROLLBACK")
                raise

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value_json FROM kernel_meta WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row["value_json"])

    def set_meta(self, key: str, value: Any) -> None:
        now_ns = time.time_ns()
        payload = json.dumps(value, sort_keys=True)
        self.conn.execute(
            """
            INSERT INTO kernel_meta(key, value_json, updated_at_ns)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at_ns = excluded.updated_at_ns
            """,
            (key, payload, now_ns),
        )

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
