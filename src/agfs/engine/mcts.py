"""Layer 2: SWE-Search (ICLR 2025) MCTS State-Tree Controller & Recursive CTE Backpropagation."""

from __future__ import annotations

import math

from agfs.core.config import KernelConfig
from agfs.core.types import CommitNode, UCTCandidate
from agfs.storage.db import SQLiteWALEngine


class MCTSEngine:
    """First-class Monte Carlo Tree Search over `.agdb` state commits."""

    def __init__(self, db: SQLiteWALEngine, config: KernelConfig) -> None:
        self.db = db
        self.config = config

    def get_commit(self, commit_id: int) -> CommitNode:
        row = self.db.conn.execute(
            "SELECT * FROM commits WHERE commit_id = ?", (commit_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Commit #{commit_id} not found")
        return self._row_to_commit(row)

    def get_ancestors(self, commit_id: int | None) -> list[int]:
        """Return ordered list of ancestor commit_ids (including `commit_id` itself)."""
        if commit_id is None:
            return []
        rows = self.db.conn.execute(
            """
            WITH RECURSIVE ancestors(cid, pid, depth) AS (
                SELECT commit_id, parent_id, 0 FROM commits WHERE commit_id = ?
                UNION ALL
                SELECT c.commit_id, c.parent_id, a.depth + 1
                FROM commits c
                JOIN ancestors a ON c.commit_id = a.pid
            )
            SELECT cid FROM ancestors ORDER BY depth DESC
            """,
            (commit_id,),
        ).fetchall()
        return [int(r["cid"]) for r in rows]

    def backpropagate(self, commit_id: int, reward: float) -> int:
        """Backpropagate `reward` up the commit ancestor chain using a single recursive CTE.

        Returns number of ancestor commit nodes updated.
        """
        self.db.conn.execute(
            "UPDATE commits SET immediate_reward = ? WHERE commit_id = ? AND immediate_reward IS NULL",
            (reward, commit_id),
        )
        cur = self.db.conn.execute(
            """
            WITH RECURSIVE ancestors(cid) AS (
                SELECT ?
                UNION ALL
                SELECT c.parent_id
                FROM commits c
                JOIN ancestors a ON c.commit_id = a.cid
                WHERE c.parent_id IS NOT NULL
            )
            UPDATE commits
            SET mcts_visits = mcts_visits + 1,
                mcts_value_sum = mcts_value_sum + ?,
                mcts_value_sq_sum = mcts_value_sq_sum + (? * ?)
            WHERE commit_id IN (SELECT cid FROM ancestors)
            """,
            (commit_id, reward, reward, reward),
        )
        return cur.rowcount

    def select_uct(self, exploration_c: float | None = None) -> list[UCTCandidate]:
        """Rank all ACTIVE branch head commits by Upper Confidence Bound for Trees (UCT).

        UCT(i) = V_bar(i) + c * sqrt(ln(max(1, N_parent)) / (1 + n_i))
        """
        c = exploration_c if exploration_c is not None else self.config.uct_exploration_c
        rows = self.db.conn.execute(
            """
            SELECT
                c.commit_id,
                c.branch_name,
                c.step_index,
                c.mcts_visits,
                c.mcts_value_sum,
                c.vfs_root_hash,
                COALESCE(p.mcts_visits, c.mcts_visits, 1) AS parent_visits
            FROM branches b
            JOIN commits c ON c.commit_id = b.head_commit_id
            LEFT JOIN commits p ON p.commit_id = c.parent_id
            WHERE b.status = 'ACTIVE'
            """
        ).fetchall()

        total_root_visits = max(
            1,
            sum(int(r["mcts_visits"]) for r in rows),
        )

        candidates: list[UCTCandidate] = []
        for r in rows:
            n_i = int(r["mcts_visits"])
            v_sum = float(r["mcts_value_sum"])
            mean_val = (v_sum / n_i) if n_i > 0 else 0.5  # Optimistic prior for unvisited frontier
            n_parent = max(total_root_visits, int(r["parent_visits"]), 1)
            bonus = c * math.sqrt(math.log(max(2, n_parent)) / (1.0 + n_i))
            uct = mean_val + bonus
            candidates.append(
                UCTCandidate(
                    commit_id=int(r["commit_id"]),
                    branch_name=str(r["branch_name"]),
                    step_index=int(r["step_index"]),
                    mcts_visits=n_i,
                    mean_value=mean_val,
                    uct_score=uct,
                    vfs_root_hash=str(r["vfs_root_hash"]),
                )
            )
        candidates.sort(key=lambda item: (item.uct_score, -item.commit_id), reverse=True)
        return candidates

    @staticmethod
    def compute_hybrid_reward(
        *,
        test_pass_rate: float,
        lines_changed: int = 0,
        is_cycle: bool = False,
        critic_score: float | None = None,
    ) -> float:
        """Compute SWE-Search hybrid numerical + qualitative state value in [0.0, 1.0]."""
        if is_cycle:
            return 0.0
        diff_penalty = 0.15 * min(1.0, max(0, lines_changed) / 500.0)
        base = 0.85 * max(0.0, min(1.0, test_pass_rate)) + (0.15 - diff_penalty)
        if critic_score is not None:
            base = 0.75 * base + 0.25 * max(0.0, min(1.0, critic_score))
        return round(max(0.0, min(1.0, base)), 6)

    @staticmethod
    def _row_to_commit(row: object) -> CommitNode:
        r = row  # type: ignore[assignment]
        return CommitNode(
            commit_id=int(r["commit_id"]),
            parent_id=int(r["parent_id"]) if r["parent_id"] is not None else None,
            branch_name=str(r["branch_name"]),
            step_index=int(r["step_index"]),
            message=str(r["message"]),
            merkle_hash=str(r["merkle_hash"]),
            vfs_root_hash=str(r["vfs_root_hash"]),
            last_error_hash=str(r["last_error_hash"]),
            mcts_visits=int(r["mcts_visits"]),
            mcts_value_sum=float(r["mcts_value_sum"]),
            mcts_value_sq_sum=float(r["mcts_value_sq_sum"]),
            immediate_reward=float(r["immediate_reward"]) if r["immediate_reward"] is not None else None,
            test_pass_rate=float(r["test_pass_rate"]) if r["test_pass_rate"] is not None else None,
            step_cost_usd=float(r["step_cost_usd"]),
            created_at_ns=int(r["created_at_ns"]),
        )
