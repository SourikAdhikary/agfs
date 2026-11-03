"""Layer 2: AIOS (COLM 2025) Preemptible Process Control Block & Capability Sandbox."""

from __future__ import annotations

import fnmatch
import json
import time
from typing import Any

from agfs.core.errors import CapabilityDeniedTrap
from agfs.core.types import BranchInfo
from agfs.storage.db import SQLiteWALEngine


class PCBManager:
    """Manages AIOS `agent_pcb` snapshots and enforces branch-level capability masks."""

    def __init__(self, db: SQLiteWALEngine) -> None:
        self.db = db

    def get_branch_info(self, branch_name: str) -> BranchInfo:
        row = self.db.conn.execute(
            "SELECT * FROM branches WHERE branch_name = ?", (branch_name,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Branch '{branch_name}' does not exist")
        return BranchInfo(
            branch_name=str(row["branch_name"]),
            head_commit_id=int(row["head_commit_id"]) if row["head_commit_id"] is not None else None,
            parent_branch=str(row["parent_branch"]) if row["parent_branch"] is not None else None,
            fork_commit_id=int(row["fork_commit_id"]) if row["fork_commit_id"] is not None else None,
            status=str(row["status"]),
            read_only_globs=json.loads(row["read_only_globs"]),
            allow_tools=json.loads(row["allow_tools_json"]),
            max_budget_usd=float(row["max_budget_usd"]),
            spent_budget_usd=float(row["spent_budget_usd"]),
            created_at_ns=int(row["created_at_ns"]),
        )

    def check_capabilities(
        self,
        *,
        branch_name: str,
        tool_name: str | None = None,
        mutated_paths: list[str] | None = None,
        step_cost_usd: float = 0.0,
    ) -> None:
        """Verify tool permissions, read-only path globs, and budget cap for `branch_name`."""
        b = self.get_branch_info(branch_name)

        if tool_name and "*" not in b.allow_tools and tool_name not in b.allow_tools:
            raise CapabilityDeniedTrap(
                branch_name=branch_name,
                capability="allow_tools",
                details=f"Tool '{tool_name}' is not in branch allowlist {b.allow_tools}",
            )

        if mutated_paths and b.read_only_globs:
            for p in mutated_paths:
                norm_p = p.lstrip("./")
                for glob_pat in b.read_only_globs:
                    if fnmatch.fnmatch(norm_p, glob_pat) or (
                        glob_pat.endswith("/**") and norm_p.startswith(glob_pat[:-2])
                    ):
                        raise CapabilityDeniedTrap(
                            branch_name=branch_name,
                            capability="read_only_globs",
                            details=f"Path '{norm_p}' matches read-only capability glob '{glob_pat}'",
                        )

        if b.spent_budget_usd + step_cost_usd > b.max_budget_usd + 1e-9:
            raise CapabilityDeniedTrap(
                branch_name=branch_name,
                capability="max_budget_usd",
                details=(
                    f"Step cost ${step_cost_usd:.4f} would exceed branch budget cap "
                    f"(${b.spent_budget_usd:.4f} + ${step_cost_usd:.4f} > ${b.max_budget_usd:.4f})"
                ),
            )

    def suspend_pcb(
        self,
        *,
        pcb_id: str,
        branch_name: str,
        commit_id: int,
        active_messages: list[dict[str, Any]],
        prefix_cache_hash: str,
        rng_seed: int = 42,
    ) -> None:
        """Serialize an agent Process Control Block (`SIGSTOP` equivalent) into `.agdb`."""
        now_ns = time.time_ns()
        self.db.conn.execute(
            """
            INSERT INTO agent_pcb(
                pcb_id, branch_name, commit_id, active_messages_json,
                prefix_cache_hash, rng_seed, state_status, updated_at_ns
            ) VALUES (?, ?, ?, ?, ?, ?, 'SUSPENDED', ?)
            ON CONFLICT(pcb_id) DO UPDATE SET
                branch_name = excluded.branch_name,
                commit_id = excluded.commit_id,
                active_messages_json = excluded.active_messages_json,
                prefix_cache_hash = excluded.prefix_cache_hash,
                rng_seed = excluded.rng_seed,
                state_status = 'SUSPENDED',
                updated_at_ns = excluded.updated_at_ns
            """,
            (
                pcb_id,
                branch_name,
                commit_id,
                json.dumps(active_messages),
                prefix_cache_hash,
                rng_seed,
                now_ns,
            ),
        )

    def resume_pcb(self, pcb_id: str) -> dict[str, Any]:
        """Restore a suspended Process Control Block (`SIGCONT` equivalent)."""
        row = self.db.conn.execute(
            "SELECT * FROM agent_pcb WHERE pcb_id = ?", (pcb_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"PCB '{pcb_id}' not found")
        self.db.conn.execute(
            "UPDATE agent_pcb SET state_status = 'RUNNING', updated_at_ns = ? WHERE pcb_id = ?",
            (time.time_ns(), pcb_id),
        )
        return {
            "pcb_id": str(row["pcb_id"]),
            "branch_name": str(row["branch_name"]),
            "commit_id": int(row["commit_id"]),
            "active_messages": json.loads(row["active_messages_json"]),
            "prefix_cache_hash": str(row["prefix_cache_hash"]),
            "rng_seed": int(row["rng_seed"]),
            "state_status": "RUNNING",
        }
