"""Layer 0: Pure domain types and dataclasses (zero heavy imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class BranchInfo:
    branch_name: str
    head_commit_id: int | None
    parent_branch: str | None
    fork_commit_id: int | None
    status: str
    read_only_globs: list[str]
    allow_tools: list[str]
    max_budget_usd: float
    spent_budget_usd: float
    created_at_ns: int


@dataclass(slots=True, frozen=True)
class CommitNode:
    commit_id: int
    parent_id: int | None
    branch_name: str
    step_index: int
    message: str
    merkle_hash: str
    vfs_root_hash: str
    last_error_hash: str
    mcts_visits: int
    mcts_value_sum: float
    mcts_value_sq_sum: float
    immediate_reward: float | None
    test_pass_rate: float | None
    step_cost_usd: float
    created_at_ns: int

    @property
    def mean_value(self) -> float:
        return self.mcts_value_sum / self.mcts_visits if self.mcts_visits > 0 else 0.0


@dataclass(slots=True, frozen=True)
class UCTCandidate:
    commit_id: int
    branch_name: str
    step_index: int
    mcts_visits: int
    mean_value: float
    uct_score: float
    vfs_root_hash: str


@dataclass(slots=True, frozen=True)
class InodeEntry:
    commit_id: int
    path: str
    deleted: bool
    mode: int
    size: int
    mtime_ns: int
    file_hash: str
    chunks: list[str]


@dataclass(slots=True, frozen=True)
class MemoryFact:
    memory_id: int
    kind: str
    subject: str
    predicate: str
    content: str
    tx_commit_born: int
    tx_commit_superseded: int | None
    branch_scope: str
    valid_from_ts: float
    valid_until_ts: float | None
    trust_level: str
    source_syscall_id: int | None
    compacted_into_id: int | None
    importance: float
    created_at_ns: int


@dataclass(slots=True, frozen=True)
class RecallHit:
    memory: MemoryFact
    rrf_score: float
    vec_score: float
    bm25_score: float
    ppr_score: float
    vec_rank: int | None = None
    bm25_rank: int | None = None
    ppr_rank: int | None = None


@dataclass(slots=True, frozen=True)
class SyscallRecord:
    syscall_id: int
    commit_id: int
    branch_name: str
    tool_name: str
    mutability_class: str
    args: dict[str, Any]
    exit_code: int
    duration_ms: float
    raw_token_count: int
    masked_summary: str
    stderr_hash: str
    created_at_ns: int


@dataclass(slots=True, frozen=True)
class SagaEffect:
    effect_id: int
    commit_id: int
    branch_name: str
    idempotency_key: str
    target_service: str
    gate_policy: str
    forward_action: dict[str, Any]
    compensate_action: dict[str, Any]
    status: str
    response: dict[str, Any] | None
    created_at_ns: int
    executed_at_ns: int | None


@dataclass(slots=True, frozen=True)
class DoctorCheckResult:
    invariant_id: str
    name: str
    passed: bool
    details: str
    duration_ms: float


@dataclass(slots=True)
class DoctorReport:
    db_path: str
    checks: list[DoctorCheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)
