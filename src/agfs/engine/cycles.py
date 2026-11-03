"""Layer 2: SentinelAgent (2025) Merkle State-Cycle Trap (`<0.1 ms` loop detection)."""

from __future__ import annotations

from agfs.core.errors import StateCycleTrap
from agfs.storage.db import SQLiteWALEngine


class CycleDetector:
    """Detects deterministic state oscillations using `(vfs_root_hash, last_error_hash)`."""

    def __init__(self, db: SQLiteWALEngine) -> None:
        self.db = db

    def verify_no_state_cycle(
        self,
        *,
        branch_name: str,
        vfs_root_hash: str,
        last_error_hash: str,
        check_across_branches: bool = True,
    ) -> None:
        """Raise `StateCycleTrap` if `(vfs_root_hash, last_error_hash)` was already visited.

        Only traps when `last_error_hash` is non-empty (i.e. returning to a state that
        previously produced the exact same failure), or when returning to an identical
        broken VFS state on the same branch.
        """
        if not last_error_hash:
            return

        if check_across_branches:
            row = self.db.conn.execute(
                """
                SELECT commit_id, step_index, branch_name
                FROM commits
                WHERE vfs_root_hash = ? AND last_error_hash = ?
                ORDER BY commit_id ASC
                LIMIT 1
                """,
                (vfs_root_hash, last_error_hash),
            ).fetchone()
        else:
            row = self.db.conn.execute(
                """
                SELECT commit_id, step_index, branch_name
                FROM commits
                WHERE branch_name = ? AND vfs_root_hash = ? AND last_error_hash = ?
                ORDER BY commit_id ASC
                LIMIT 1
                """,
                (branch_name, vfs_root_hash, last_error_hash),
            ).fetchone()

        if row is not None:
            raise StateCycleTrap(
                branch_name=branch_name,
                prior_commit_id=int(row["commit_id"]),
                prior_step_index=int(row["step_index"]),
                vfs_root_hash=vfs_root_hash,
                last_error_hash=last_error_hash,
            )
