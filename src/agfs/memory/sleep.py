"""Layer 3: Sleep-Time Compute (Letta / UC Berkeley, 2025) Idle Memory Compactor."""

from __future__ import annotations

from agfs.memory.bitemporal import BiTemporalMemoryStore
from agfs.storage.db import SQLiteWALEngine


class SleepCompactor:
    """Consolidates uncompacted episodic traces into compact semantic facts during idle windows."""

    def __init__(self, db: SQLiteWALEngine, bitemporal: BiTemporalMemoryStore) -> None:
        self.db = db
        self.bitemporal = bitemporal

    def sleep_compact(self, *, branch_name: str, commit_id: int) -> int:
        """Scan uncompacted `EPISODE_GIST` rows on `branch_name` and consolidate duplicates."""
        rows = self.db.conn.execute(
            """
            SELECT memory_id, subject, predicate, content
            FROM memories
            WHERE branch_scope = ?
              AND kind = 'EPISODE_GIST'
              AND tx_commit_superseded IS NULL
              AND compacted_into_id IS NULL
            ORDER BY memory_id ASC
            """,
            (branch_name,),
        ).fetchall()

        if len(rows) < 2:
            return 0

        grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for r in rows:
            key = (str(r["subject"]), str(r["predicate"]))
            grouped.setdefault(key, []).append((int(r["memory_id"]), str(r["content"])))

        compacted_count = 0
        for (subj, pred), items in grouped.items():
            if len(items) < 2:
                continue
            merged_text = " | ".join(dict.fromkeys(text for _, text in items))
            new_fact = self.bitemporal.remember(
                commit_id=commit_id,
                branch_name=branch_name,
                subject=subj,
                predicate=pred,
                content=f"[Consolidated {len(items)} episodes]: {merged_text}",
                kind="FACT",
            )
            old_ids = [mid for mid, _ in items]
            placeholders = ",".join("?" for _ in old_ids)
            self.db.conn.execute(
                f"""
                UPDATE memories
                SET compacted_into_id = ?, tx_commit_superseded = ?
                WHERE memory_id IN ({placeholders})
                """,
                (new_fact.memory_id, commit_id, *old_ids),
            )
            compacted_count += len(old_ids)

        self.db.conn.execute("PRAGMA optimize")
        return compacted_count
