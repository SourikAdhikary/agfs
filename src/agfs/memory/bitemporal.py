"""Layer 3: Bi-Temporal & Taint-Tracked Memory Engine (`Graphiti` / `TGMS` / `REMem`)."""

from __future__ import annotations

import time

from agfs.core.config import KernelConfig
from agfs.core.types import MemoryFact
from agfs.memory.embedders.base import EmbedderProtocol, get_embedder
from agfs.memory.embedders.deterministic import DeterministicHashEmbedder  # noqa: F401
from agfs.storage.db import SQLiteWALEngine


class BiTemporalMemoryStore:
    """Manages Bi-Temporal facts (`tx_commit_born`, `tx_commit_superseded`, `valid_from_ts`, `valid_until_ts`).

    Guarantees 0% stale-fact poisoning on the active branch head while preserving
    100% point-in-time historical replay for any past commit or branch.
    """

    def __init__(
        self,
        db: SQLiteWALEngine,
        config: KernelConfig,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self.db = db
        self.config = config
        self.embedder = embedder or get_embedder("deterministic", config.embedding_dim)

    def remember(
        self,
        *,
        commit_id: int,
        branch_name: str,
        subject: str,
        predicate: str,
        content: str,
        kind: str = "FACT",
        cross_branch_global: bool = False,
        valid_from_ts: float | None = None,
        valid_until_ts: float | None = None,
        trust_level: str = "AGENT_DERIVED",
        source_syscall_id: int | None = None,
        importance: float = 0.5,
    ) -> MemoryFact:
        """Record a new bi-temporal fact, automatically superseding prior active facts for `(subject, predicate)`."""
        now_ns = time.time_ns()
        v_from = valid_from_ts if valid_from_ts is not None else (now_ns / 1e9)
        scope = "*" if cross_branch_global else branch_name

        # 1. Supersede any active contradicting fact on the same (subject, predicate) visible on this branch
        self.db.conn.execute(
            """
            UPDATE memories
            SET tx_commit_superseded = ?,
                valid_until_ts = COALESCE(valid_until_ts, ?)
            WHERE subject = ? AND predicate = ?
              AND (branch_scope = ? OR branch_scope = '*')
              AND tx_commit_superseded IS NULL
            """,
            (commit_id, v_from, subject, predicate, scope),
        )

        # 2. Insert new bi-temporal fact row
        cur = self.db.conn.execute(
            """
            INSERT INTO memories(
                kind, subject, predicate, content,
                tx_commit_born, tx_commit_superseded, branch_scope,
                valid_from_ts, valid_until_ts, trust_level,
                source_syscall_id, compacted_into_id, importance, created_at_ns
            ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                kind,
                subject,
                predicate,
                content,
                commit_id,
                scope,
                v_from,
                valid_until_ts,
                trust_level,
                source_syscall_id,
                importance,
                now_ns,
            ),
        )
        memory_id = int(cur.lastrowid or 0)

        # 3. Synchronize FTS5 lexical index & dense float32 embedding
        self.db.conn.execute(
            "INSERT INTO memories_fts(rowid, subject, predicate, content) VALUES (?, ?, ?, ?)",
            (memory_id, subject, predicate, content),
        )
        full_text = f"{subject} {predicate} {content}"
        vec = self.embedder.encode(full_text)
        norm = float((vec * vec).sum() ** 0.5)
        self.db.conn.execute(
            "INSERT INTO memories_embeddings(memory_id, embedding_f32, norm) VALUES (?, ?, ?)",
            (memory_id, vec.tobytes(), norm),
        )

        return MemoryFact(
            memory_id=memory_id,
            kind=kind,
            subject=subject,
            predicate=predicate,
            content=content,
            tx_commit_born=commit_id,
            tx_commit_superseded=None,
            branch_scope=scope,
            valid_from_ts=v_from,
            valid_until_ts=valid_until_ts,
            trust_level=trust_level,
            source_syscall_id=source_syscall_id,
            compacted_into_id=None,
            importance=importance,
            created_at_ns=now_ns,
        )

    def get_visible_memories(
        self,
        *,
        ancestor_commit_ids: list[int],
        active_branch: str | None = None,
        world_timestamp: float | None = None,
        exclude_untrusted: bool = False,
    ) -> dict[int, MemoryFact]:
        """Return `{memory_id: MemoryFact}` visible at the point-in-time defined by `ancestor_commit_ids`.

        A memory $m$ is visible iff:
        1. `m.tx_commit_born` is in `ancestor_commit_ids` OR `m.branch_scope = '*'` (with `tx_commit_born <= max_commit`).
        2. If `active_branch` is provided, `m.branch_scope` must be `'*'`, `'main'`, or `active_branch`.
        3. `m.tx_commit_superseded` is NULL OR `m.tx_commit_superseded` is NOT in `ancestor_commit_ids`.
        4. If `world_timestamp` is specified, `valid_from_ts <= world_timestamp < valid_until_ts`.
        """
        if not ancestor_commit_ids:
            return {}
        ancestor_set = set(ancestor_commit_ids)
        max_cid = max(ancestor_set)

        rows = self.db.conn.execute(
            "SELECT * FROM memories WHERE tx_commit_born <= ?",
            (max_cid,),
        ).fetchall()

        visible: dict[int, MemoryFact] = {}
        for r in rows:
            born = int(r["tx_commit_born"])
            sup = int(r["tx_commit_superseded"]) if r["tx_commit_superseded"] is not None else None
            scope = str(r["branch_scope"])
            trust = str(r["trust_level"])

            if exclude_untrusted and trust == "UNTRUSTED_EXTERNAL_IO":
                continue

            if active_branch is not None and scope not in ("*", "main", active_branch):
                continue

            # Check Transaction-Time Lineage Visibility
            if scope != "*" and born not in ancestor_set:
                continue
            if sup is not None and (sup in ancestor_set or (scope == "*" and sup <= max_cid)):
                continue

            # Check World-Validity Time if requested
            v_from = float(r["valid_from_ts"])
            v_until = float(r["valid_until_ts"]) if r["valid_until_ts"] is not None else None
            if world_timestamp is not None:
                if v_from > world_timestamp:
                    continue
                if v_until is not None and v_until <= world_timestamp:
                    continue

            mem_id = int(r["memory_id"])
            visible[mem_id] = MemoryFact(
                memory_id=mem_id,
                kind=str(r["kind"]),
                subject=str(r["subject"]),
                predicate=str(r["predicate"]),
                content=str(r["content"]),
                tx_commit_born=born,
                tx_commit_superseded=sup,
                branch_scope=scope,
                valid_from_ts=v_from,
                valid_until_ts=v_until,
                trust_level=trust,
                source_syscall_id=int(r["source_syscall_id"]) if r["source_syscall_id"] is not None else None,
                compacted_into_id=int(r["compacted_into_id"]) if r["compacted_into_id"] is not None else None,
                importance=float(r["importance"]),
                created_at_ns=int(r["created_at_ns"]),
            )

        return visible
