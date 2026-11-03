"""Layer 5: Unified `AgFSKernel` orchestrating VFS, MCTS, Bi-Temporal Memory, Context, Invariants & Saga."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agfs.core.config import KernelConfig
from agfs.core.errors import (
    CapabilityDeniedTrap,
    InvariantViolationTrap,
    StateCycleTrap,
)
from agfs.core.hashing import compute_commit_merkle_hash, hash_stderr
from agfs.core.types import (
    BranchInfo,
    CommitNode,
    DoctorReport,
    MemoryFact,
    RecallHit,
    SagaEffect,
    SyscallRecord,
    UCTCandidate,
)
from agfs.engine.context import ContextVirtualizer
from agfs.engine.cycles import CycleDetector
from agfs.engine.doctor import run_doctor_checks
from agfs.engine.mcts import MCTSEngine
from agfs.engine.paste import PASTEPrefetcher
from agfs.engine.pcb import PCBManager
from agfs.memory.bitemporal import BiTemporalMemoryStore
from agfs.memory.hybrid import HybridMemoryEngine
from agfs.memory.ppr import HippoRAGGraph
from agfs.memory.sleep import SleepCompactor
from agfs.safety.invariants import InvariantVerifier
from agfs.safety.saga import SagaOutboxManager
from agfs.storage.db import SQLiteWALEngine
from agfs.storage.vfs import CoWVFS


class AgFSKernel:
    """Single-file `.agdb` SQLite Operating System & Transactional Kernel for AI Agents."""

    def __init__(
        self,
        db_path: Path | str,
        workspace_root: Path | str | None = None,
        *,
        protected_globs: list[str] | None = None,
        block_secrets: bool = True,
        block_test_skip: bool = True,
        auto_init: bool = True,
    ) -> None:
        ws = Path(workspace_root).resolve() if workspace_root is not None else None
        self.config = KernelConfig(db_path=Path(db_path).resolve(), workspace_root=ws)
        self.db = SQLiteWALEngine(self.config.db_path)
        self.vfs = CoWVFS(self.db, self.config)
        self.mcts = MCTSEngine(self.db, self.config)
        self.cycles = CycleDetector(self.db)
        self.context = ContextVirtualizer(self.db, self.config)
        self.paste = PASTEPrefetcher(self.db)
        self.pcb = PCBManager(self.db)
        self.bitemporal = BiTemporalMemoryStore(self.db, self.config)
        self.graph = HippoRAGGraph(self.db, self.config)
        self.memory = HybridMemoryEngine(self.db, self.config, self.bitemporal, self.graph)
        self.sleep = SleepCompactor(self.db, self.bitemporal)
        self.invariants = InvariantVerifier(self.db)
        self.saga = SagaOutboxManager(self.db)

        if auto_init:
            self._ensure_initialized(
                protected_globs=protected_globs,
                block_secrets=block_secrets,
                block_test_skip=block_test_skip,
            )

    def _ensure_initialized(
        self,
        *,
        protected_globs: list[str] | None,
        block_secrets: bool,
        block_test_skip: bool,
    ) -> None:
        row = self.db.conn.execute(
            "SELECT head_commit_id FROM branches WHERE branch_name = 'main'"
        ).fetchone()
        if row is None:
            with self.db.transaction():
                now_ns = time.time_ns()
                self.db.conn.execute(
                    """
                    INSERT INTO branches(
                        branch_name, head_commit_id, parent_branch, fork_commit_id,
                        status, read_only_globs, allow_tools_json, max_budget_usd,
                        spent_budget_usd, created_at_ns
                    ) VALUES ('main', NULL, NULL, NULL, 'ACTIVE', ?, '["*"]', 10.0, 0.0, ?)
                    """,
                    (json.dumps(protected_globs or []), now_ns),
                )
                self.db.set_meta("active_branch", "main")
                self.invariants.register_default_guardrails(
                    protected_globs=protected_globs,
                    block_secrets=block_secrets,
                    block_test_skip=block_test_skip,
                )
            # Create initial root commit #1 capturing initial workspace without triggering protected_globs on genesis
            self._create_genesis_commit()
        elif protected_globs is not None:
            with self.db.transaction():
                self.invariants.register_default_guardrails(
                    protected_globs=protected_globs,
                    block_secrets=block_secrets,
                    block_test_skip=block_test_skip,
                )

    def _create_genesis_commit(self) -> CommitNode:
        with self.db.transaction():
            manifest, vfs_root_hash, _ = self.vfs.scan_workspace_changes(
                parent_commit_id=None,
                workspace_root=self.config.workspace_root,
            )
            merkle = compute_commit_merkle_hash(
                parent_merkle="GENESIS",
                branch_name="main",
                step_index=0,
                vfs_root_hash=vfs_root_hash,
                last_error_hash="",
                message="Initial workspace genesis snapshot",
            )
            now_ns = time.time_ns()
            cur = self.db.conn.execute(
                """
                INSERT INTO commits(
                    parent_id, branch_name, step_index, message,
                    merkle_hash, vfs_root_hash, last_error_hash,
                    mcts_visits, mcts_value_sum, mcts_value_sq_sum,
                    immediate_reward, test_pass_rate, step_cost_usd, created_at_ns
                ) VALUES (NULL, 'main', 0, ?, ?, ?, '', 1, 0.5, 0.25, 0.5, NULL, 0.0, ?)
                """,
                ("Initial workspace genesis snapshot", merkle, vfs_root_hash, now_ns),
            )
            cid = int(cur.lastrowid or 1)
            self.vfs.persist_manifest(cid, manifest)
            self.db.conn.execute(
                "UPDATE branches SET head_commit_id = ? WHERE branch_name = 'main'",
                (cid,),
            )
            self.db.set_meta("active_commit_id", cid)
            return self.mcts.get_commit(cid)

    @property
    def active_branch(self) -> str:
        return str(self.db.get_meta("active_branch", "main"))

    @property
    def current_commit_id(self) -> int:
        cid = self.db.get_meta("active_commit_id")
        if cid is not None:
            return int(cid)
        row = self.db.conn.execute(
            "SELECT head_commit_id FROM branches WHERE branch_name = ?",
            (self.active_branch,),
        ).fetchone()
        return int(row["head_commit_id"])

    def fork(
        self,
        branch_name: str,
        *,
        from_commit: int | None = None,
        read_only_globs: list[str] | None = None,
        allow_tools: list[str] | None = None,
        max_budget_usd: float = 5.0,
        checkout: bool = True,
    ) -> BranchInfo:
        """Create a `<1.5 ms` Copy-on-Write speculative branch from `from_commit` (default HEAD)."""
        base_cid = from_commit if from_commit is not None else self.current_commit_id
        parent_branch = self.active_branch
        parent_info = self.pcb.get_branch_info(parent_branch)
        ro_globs = read_only_globs if read_only_globs is not None else parent_info.read_only_globs
        tools = allow_tools if allow_tools is not None else parent_info.allow_tools
        now_ns = time.time_ns()

        with self.db.transaction():
            self.db.conn.execute(
                """
                INSERT INTO branches(
                    branch_name, head_commit_id, parent_branch, fork_commit_id,
                    status, read_only_globs, allow_tools_json, max_budget_usd,
                    spent_budget_usd, created_at_ns
                ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, ?, 0.0, ?)
                ON CONFLICT(branch_name) DO UPDATE SET
                    head_commit_id = excluded.head_commit_id,
                    fork_commit_id = excluded.fork_commit_id,
                    status = 'ACTIVE'
                """,
                (
                    branch_name,
                    base_cid,
                    parent_branch,
                    base_cid,
                    json.dumps(ro_globs),
                    json.dumps(tools),
                    max_budget_usd,
                    now_ns,
                ),
            )
            if checkout:
                prev_cid = self.current_commit_id
                self.db.set_meta("active_branch", branch_name)
                self.db.set_meta("active_commit_id", base_cid)
                if prev_cid != base_cid and self.config.workspace_root is not None:
                    self.vfs.materialize_delta_to_disk(prev_cid, base_cid)

        return self.pcb.get_branch_info(branch_name)

    def checkout(
        self,
        target: str | int,
        *,
        materialize_to_disk: bool = True,
    ) -> CommitNode:
        """Atomically switch both workspace files and active Bi-Temporal memory lineage in `<2 ms`."""
        prev_cid = self.current_commit_id
        with self.db.transaction():
            if isinstance(target, int) or (isinstance(target, str) and target.isdigit()):
                target_cid = int(target)
                node = self.mcts.get_commit(target_cid)
                target_branch = node.branch_name
            else:
                target_branch = str(target)
                b_info = self.pcb.get_branch_info(target_branch)
                if b_info.head_commit_id is None:
                    raise ValueError(f"Branch '{target_branch}' has no head commit")
                target_cid = b_info.head_commit_id
                node = self.mcts.get_commit(target_cid)

            self.db.set_meta("active_branch", target_branch)
            self.db.set_meta("active_commit_id", target_cid)

            if materialize_to_disk and self.config.workspace_root is not None and prev_cid != target_cid:
                self.vfs.materialize_delta_to_disk(prev_cid, target_cid)

        return node

    def commit(
        self,
        message: str,
        *,
        virtual_overlays: dict[str, bytes | str | None] | None = None,
        stderr_text: str = "",
        test_pass_rate: float | None = None,
        step_cost_usd: float = 0.0,
        tool_name: str | None = None,
        check_cycles: bool = True,
        check_across_branches: bool = True,
    ) -> CommitNode:
        """Capture workspace/virtual deltas via FastCDC, verify ePCA invariants & StateCycleTrap, and commit."""
        branch = self.active_branch
        parent_cid = self.current_commit_id
        parent_node = self.mcts.get_commit(parent_cid)

        norm_overlays: dict[str, bytes | None] | None = None
        if virtual_overlays:
            norm_overlays = {
                k: (v.encode("utf-8") if isinstance(v, str) else v)
                for k, v in virtual_overlays.items()
            }

        try:
            with self.db.transaction():
                new_manifest, vfs_root_hash, changed_paths = self.vfs.scan_workspace_changes(
                    parent_commit_id=parent_cid,
                    workspace_root=self.config.workspace_root,
                    virtual_overlays=norm_overlays,
                )

                # 1. Check AIOS Branch Capabilities (tool allowlist, read-only globs, budget)
                self.pcb.check_capabilities(
                    branch_name=branch,
                    tool_name=tool_name,
                    mutated_paths=changed_paths,
                    step_cost_usd=step_cost_usd,
                )

                # 2. Check ePCA Executable Commit Invariants before committing
                changed_text_map: dict[str, str] = {}
                for cp in changed_paths:
                    if cp in new_manifest:
                        raw_b = self.vfs.reconstruct_file_bytes(new_manifest[cp].chunks)
                        changed_text_map[cp] = raw_b.decode("utf-8", errors="ignore")

                self.invariants.verify_before_commit(
                    branch_name=branch,
                    changed_paths=changed_paths,
                    changed_file_contents=changed_text_map,
                    step_cost_usd=step_cost_usd,
                )

                # 3. Check SentinelAgent Merkle State-Cycle Trap
                err_hash = hash_stderr(stderr_text)
                if check_cycles and err_hash:
                    self.cycles.verify_no_state_cycle(
                        branch_name=branch,
                        vfs_root_hash=vfs_root_hash,
                        last_error_hash=err_hash,
                        check_across_branches=check_across_branches,
                    )

                # 4. Append new Merkle-chained commit node
                step_idx = parent_node.step_index + 1
                merkle = compute_commit_merkle_hash(
                    parent_merkle=parent_node.merkle_hash,
                    branch_name=branch,
                    step_index=step_idx,
                    vfs_root_hash=vfs_root_hash,
                    last_error_hash=err_hash,
                    message=message,
                )
                now_ns = time.time_ns()
                cur = self.db.conn.execute(
                    """
                    INSERT INTO commits(
                        parent_id, branch_name, step_index, message,
                        merkle_hash, vfs_root_hash, last_error_hash,
                        mcts_visits, mcts_value_sum, mcts_value_sq_sum,
                        immediate_reward, test_pass_rate, step_cost_usd, created_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0.0, 0.0, NULL, ?, ?, ?)
                    """,
                    (
                        parent_cid,
                        branch,
                        step_idx,
                        message,
                        merkle,
                        vfs_root_hash,
                        err_hash,
                        test_pass_rate,
                        step_cost_usd,
                        now_ns,
                    ),
                )
                new_cid = int(cur.lastrowid or 0)
                self.vfs.persist_manifest(new_cid, new_manifest)
                self.db.conn.execute(
                    """
                    UPDATE branches
                    SET head_commit_id = ?,
                        spent_budget_usd = spent_budget_usd + ?
                    WHERE branch_name = ?
                    """,
                    (new_cid, step_cost_usd, branch),
                )
                self.db.set_meta("active_commit_id", new_cid)

                for cp in changed_paths:
                    self.paste.invalidate_path(cp)

                if test_pass_rate is not None:
                    reward = self.mcts.compute_hybrid_reward(
                        test_pass_rate=test_pass_rate,
                        lines_changed=len(changed_paths) * 10,
                    )
                    self.mcts.backpropagate(new_cid, reward)

                return self.mcts.get_commit(new_cid)

        except (InvariantViolationTrap, CapabilityDeniedTrap, StateCycleTrap):
            # If on-disk workspace was dirtied before the rejected commit, restore clean parent state!
            if self.config.workspace_root is not None:
                self.vfs.materialize_delta_to_disk(None, parent_cid, self.config.workspace_root)
            raise

    def select_uct(self, exploration_c: float | None = None) -> list[UCTCandidate]:
        """Return active branch frontier nodes ranked by SWE-Search UCT."""
        return self.mcts.select_uct(exploration_c=exploration_c)

    def backpropagate(
        self,
        reward: float,
        *,
        commit_id: int | None = None,
        cross_branch_lesson: tuple[str, str, str] | None = None,
    ) -> int:
        """Backpropagate `reward` up the MCTS tree and optionally store a global cross-branch lesson."""
        target_cid = commit_id if commit_id is not None else self.current_commit_id
        with self.db.transaction():
            updated = self.mcts.backpropagate(target_cid, reward)
            if cross_branch_lesson is not None:
                subj, pred, content = cross_branch_lesson
                self.bitemporal.remember(
                    commit_id=target_cid,
                    branch_name=self.active_branch,
                    subject=subj,
                    predicate=pred,
                    content=content,
                    kind="LESSON_CROSS_BRANCH",
                    cross_branch_global=True,
                    importance=0.95,
                )
            return updated

    def remember(
        self,
        subject: str,
        predicate: str,
        content: str,
        *,
        kind: str = "FACT",
        cross_branch_global: bool = False,
        valid_from_ts: float | None = None,
        valid_until_ts: float | None = None,
        trust_level: str = "AGENT_DERIVED",
        edges: list[tuple[str, str, str]] | None = None,
    ) -> MemoryFact:
        """Record a Bi-Temporal fact and optional HippoRAG 2 entity-relation triples `(src, rel, dst)`."""
        cid = self.current_commit_id
        with self.db.transaction():
            fact = self.bitemporal.remember(
                commit_id=cid,
                branch_name=self.active_branch,
                subject=subject,
                predicate=predicate,
                content=content,
                kind=kind,
                cross_branch_global=cross_branch_global,
                valid_from_ts=valid_from_ts,
                valid_until_ts=valid_until_ts,
                trust_level=trust_level,
            )
            # Automatically link (subject -> predicate -> object/entity) in HippoRAG 2 graph
            self.graph.add_edge(
                src_name=subject,
                dst_name=f"{subject}:{predicate}",
                relation=predicate,
                commit_id=cid,
                memory_id=fact.memory_id,
            )
            if edges:
                for src_e, rel_e, dst_e in edges:
                    self.graph.add_edge(
                        src_name=src_e,
                        dst_name=dst_e,
                        relation=rel_e,
                        commit_id=cid,
                        memory_id=fact.memory_id,
                    )
            return fact

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        as_of_commit: int | None = None,
        world_timestamp: float | None = None,
        exclude_untrusted: bool = False,
    ) -> list[RecallHit]:
        """Execute 3-Way Bi-Temporal Hybrid Recall (Dense Vector + FTS5 BM25 + HippoRAG 2 PPR)."""
        cid = as_of_commit if as_of_commit is not None else self.current_commit_id
        ancestors = self.mcts.get_ancestors(cid)
        target_branch = self.mcts.get_commit(cid).branch_name
        return self.memory.recall(
            query=query,
            ancestor_commit_ids=ancestors,
            active_branch=target_branch,
            top_k=top_k,
            world_timestamp=world_timestamp,
            exclude_untrusted=exclude_untrusted,
        )

    def record_tool_observation(
        self,
        tool_name: str,
        args: dict[str, Any],
        output_text: str,
        *,
        stderr_text: str = "",
        exit_code: int = 0,
        duration_ms: float = 0.0,
        mutability_class: str = "PURE_READ",
    ) -> SyscallRecord:
        """Record a tool execution into `syscall_trace` and generate its `/.proc/obs/<id>.txt` pointer."""
        with self.db.transaction():
            return self.context.record_syscall(
                commit_id=self.current_commit_id,
                branch_name=self.active_branch,
                tool_name=tool_name,
                mutability_class=mutability_class,
                args=args,
                output_text=output_text,
                stderr_text=stderr_text,
                exit_code=exit_code,
                duration_ms=duration_ms,
            )

    def read_obs(
        self,
        syscall_id: int,
        *,
        start_line: int = 1,
        end_line: int | None = 100,
        grep_pattern: str | None = None,
    ) -> str:
        """Read verbatim line slice or regex match from `/.proc/obs/<syscall_id>.txt`."""
        return self.context.read_virtual_inode(
            syscall_id,
            start_line=start_line,
            end_line=end_line,
            grep_pattern=grep_pattern,
        )

    def build_context(
        self,
        *,
        as_of_commit: int | None = None,
        mask_after_turns: int | None = None,
        mask_min_tokens: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, str]:
        """Build prefix-stable context messages with Virtual Inode Observation Masking."""
        cid = as_of_commit if as_of_commit is not None else self.current_commit_id
        ancestors = self.mcts.get_ancestors(cid)
        return self.context.build_context_messages(
            ancestor_commit_ids=ancestors,
            mask_after_turns=mask_after_turns,
            mask_min_tokens=mask_min_tokens,
        )

    def read_file_paste(
        self,
        rel_path: str,
        *,
        streaming_thought_prefix: str = "",
    ) -> tuple[bytes, bool]:
        """Read a file from the VFS while updating and consulting the PASTE speculative prefetcher."""
        norm = rel_path.lstrip("./")
        sig = f"vfs_read:{norm}"
        cid = self.current_commit_id

        def _loader(s: str) -> bytes:
            target_p = s.split("vfs_read:", 1)[-1]
            return self.vfs.read_file(cid, target_p)

        data, was_hit = self.paste.fetch_or_load(sig, _loader)
        # Immediately pre-warm predicted follow-up files in RAM
        self.paste.speculate_and_warm(
            current_signature=sig,
            streaming_thought_prefix=streaming_thought_prefix,
            loader_fn=lambda s: self._safe_load_for_prefetch(cid, s),
            top_k=2,
        )
        return (data if isinstance(data, bytes) else data.encode("utf-8")), was_hit

    def _safe_load_for_prefetch(self, commit_id: int, signature: str) -> bytes | None:
        if not signature.startswith("vfs_read:"):
            return None
        path = signature.split("vfs_read:", 1)[-1]
        try:
            return self.vfs.read_file(commit_id, path)
        except Exception:
            return None

    def saga_execute(
        self,
        *,
        target_service: str,
        forward_action: dict[str, Any],
        compensate_action: dict[str, Any],
        gate_policy: str = "DEFER_UNTIL_MERGE",
        custom_idempotency_key: str | None = None,
    ) -> SagaEffect:
        """Stage or execute an external side effect in the SagaLLM + Atomix outbox."""
        with self.db.transaction():
            return self.saga.stage_or_execute(
                commit_id=self.current_commit_id,
                branch_name=self.active_branch,
                target_service=target_service,
                forward_action=forward_action,
                compensate_action=compensate_action,
                gate_policy=gate_policy,
                custom_idempotency_key=custom_idempotency_key,
            )

    def merge_to_main(self, source_branch: str) -> tuple[CommitNode, list[SagaEffect]]:
        """Merge winning `source_branch` into `main` and flush all staged Atomix outbox effects."""
        if source_branch == "main":
            return self.mcts.get_commit(self.current_commit_id), []

        src_info = self.pcb.get_branch_info(source_branch)
        if src_info.head_commit_id is None:
            raise ValueError(f"Source branch '{source_branch}' has no commits")
        src_head = src_info.head_commit_id
        src_node = self.mcts.get_commit(src_head)

        with self.db.transaction():
            self.db.conn.execute(
                "UPDATE branches SET head_commit_id = ? WHERE branch_name = 'main'",
                (src_head,),
            )
            self.db.conn.execute(
                "UPDATE branches SET status = 'MERGED' WHERE branch_name = ?",
                (source_branch,),
            )
            # Promote branch-scoped active memories on source_branch so main sees them
            self.db.conn.execute(
                "UPDATE memories SET branch_scope = '*' WHERE branch_scope = ? AND tx_commit_superseded IS NULL",
                (source_branch,),
            )
            flushed = self.saga.flush_staged_on_merge(source_branch)
            prev_cid = self.current_commit_id
            self.db.set_meta("active_branch", "main")
            self.db.set_meta("active_commit_id", src_head)
            if self.config.workspace_root is not None and prev_cid != src_head:
                self.vfs.materialize_delta_to_disk(prev_cid, src_head)

        return src_node, flushed

    def abort_branch(self, branch_name: str, fallback_branch: str = "main") -> list[SagaEffect]:
        """Abort `branch_name`, run reverse Saga compensations, and checkout `fallback_branch`."""
        with self.db.transaction():
            compensated = self.saga.rollback_saga(branch_name, after_commit_id=0)
            self.db.conn.execute(
                "UPDATE branches SET status = 'ABORTED' WHERE branch_name = ?",
                (branch_name,),
            )
        if self.active_branch == branch_name:
            self.checkout(fallback_branch)
        return compensated

    def doctor(self) -> DoctorReport:
        """Run all 12 formal kernel invariant checks (`INV-01`..`INV-12`)."""
        return run_doctor_checks(self.db)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> AgFSKernel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
