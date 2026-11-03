"""Layer 1: Complete 12-table STRICT SQLite 3.54+ DDL & FTS5 schema."""

from __future__ import annotations

PRAGMAS_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;
PRAGMA cache_size = -16000;
PRAGMA busy_timeout = 5000;
"""

SCHEMA_DDL = """
-- 1. Kernel Metadata & Active State Pointer
CREATE TABLE IF NOT EXISTS kernel_meta (
    key             TEXT PRIMARY KEY,
    value_json      TEXT NOT NULL,
    updated_at_ns   INTEGER NOT NULL
) STRICT;

-- 2. Branches & AIOS Capability Sandboxes
CREATE TABLE IF NOT EXISTS branches (
    branch_name         TEXT PRIMARY KEY,
    head_commit_id      INTEGER,
    parent_branch       TEXT,
    fork_commit_id      INTEGER,
    status              TEXT NOT NULL DEFAULT 'ACTIVE',
    read_only_globs     TEXT NOT NULL DEFAULT '[]',
    allow_tools_json    TEXT NOT NULL DEFAULT '["*"]',
    max_budget_usd      REAL NOT NULL DEFAULT 5.0,
    spent_budget_usd    REAL NOT NULL DEFAULT 0.0,
    created_at_ns       INTEGER NOT NULL
) STRICT;

-- 3. Merkle State-Tree Commits & MCTS Statistics (SWE-Search + DeltaBox + SentinelAgent)
CREATE TABLE IF NOT EXISTS commits (
    commit_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id           INTEGER REFERENCES commits(commit_id),
    branch_name         TEXT NOT NULL REFERENCES branches(branch_name),
    step_index          INTEGER NOT NULL,
    message             TEXT NOT NULL,
    merkle_hash         TEXT NOT NULL UNIQUE,
    vfs_root_hash       TEXT NOT NULL,
    last_error_hash     TEXT NOT NULL DEFAULT '',
    mcts_visits         INTEGER NOT NULL DEFAULT 0,
    mcts_value_sum      REAL NOT NULL DEFAULT 0.0,
    mcts_value_sq_sum   REAL NOT NULL DEFAULT 0.0,
    immediate_reward    REAL,
    test_pass_rate      REAL,
    step_cost_usd       REAL NOT NULL DEFAULT 0.0,
    created_at_ns       INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_commits_branch_hashes
    ON commits(branch_name, vfs_root_hash, last_error_hash);
CREATE INDEX IF NOT EXISTS idx_commits_parent
    ON commits(parent_id);

-- 4. Content-Addressed FastCDC Chunks (Sub-File Deduplication)
CREATE TABLE IF NOT EXISTS vfs_chunks (
    chunk_hash      TEXT PRIMARY KEY,
    raw_size        INTEGER NOT NULL,
    zstd_bytes      BLOB NOT NULL,
    created_at_ns   INTEGER NOT NULL
) STRICT;

-- 5. Copy-on-Write VFS Inode Manifests
CREATE TABLE IF NOT EXISTS vfs_inodes (
    commit_id       INTEGER NOT NULL REFERENCES commits(commit_id),
    path            TEXT NOT NULL,
    deleted         INTEGER NOT NULL DEFAULT 0,
    mode            INTEGER NOT NULL DEFAULT 33188,
    size            INTEGER NOT NULL DEFAULT 0,
    mtime_ns        INTEGER NOT NULL DEFAULT 0,
    file_hash       TEXT NOT NULL,
    chunks_json     TEXT NOT NULL,
    PRIMARY KEY (commit_id, path)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_vfs_inodes_path ON vfs_inodes(path, commit_id DESC);

-- 6. Bi-Temporal & Taint-Tracked Memory Store (Graphiti / TGMS / REMem)
CREATE TABLE IF NOT EXISTS memories (
    memory_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kind                    TEXT NOT NULL,
    subject                 TEXT NOT NULL,
    predicate               TEXT NOT NULL,
    content                 TEXT NOT NULL,
    tx_commit_born          INTEGER NOT NULL REFERENCES commits(commit_id),
    tx_commit_superseded    INTEGER REFERENCES commits(commit_id),
    branch_scope            TEXT NOT NULL,
    valid_from_ts           REAL NOT NULL,
    valid_until_ts          REAL,
    trust_level             TEXT NOT NULL DEFAULT 'AGENT_DERIVED',
    source_syscall_id       INTEGER,
    compacted_into_id       INTEGER REFERENCES memories(memory_id),
    importance              REAL NOT NULL DEFAULT 0.5,
    created_at_ns           INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_memories_active
    ON memories(branch_scope, tx_commit_superseded, tx_commit_born);
CREATE INDEX IF NOT EXISTS idx_memories_subject_pred
    ON memories(subject, predicate, tx_commit_superseded);

-- 7A. Dense Vector Table (float32 packed embeddings)
CREATE TABLE IF NOT EXISTS memories_embeddings (
    memory_id       INTEGER PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
    embedding_f32   BLOB NOT NULL,
    norm            REAL NOT NULL
) STRICT;

-- 7B. Lexical BM25 Virtual Table (SQLite FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    subject,
    predicate,
    content,
    tokenize='porter unicode61'
);

-- 8. HippoRAG 2 Entity & Relation Graph for Personalized PageRank (PPR)
CREATE TABLE IF NOT EXISTS kg_entities (
    entity_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT NOT NULL UNIQUE,
    entity_type     TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    src_entity_id           INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    dst_entity_id           INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    relation                TEXT NOT NULL,
    weight                  REAL NOT NULL DEFAULT 1.0,
    memory_id               INTEGER REFERENCES memories(memory_id),
    tx_commit_born          INTEGER NOT NULL REFERENCES commits(commit_id),
    tx_commit_superseded    INTEGER REFERENCES commits(commit_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_kg_edges_active
    ON kg_edges(tx_commit_superseded, src_entity_id, dst_entity_id);

-- 9. Flight Recorder Syscall Trace & Virtual Inode Observation Store (/.proc/obs/*)
CREATE TABLE IF NOT EXISTS syscall_trace (
    syscall_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_id           INTEGER NOT NULL REFERENCES commits(commit_id),
    branch_name         TEXT NOT NULL,
    tool_name           TEXT NOT NULL,
    mutability_class    TEXT NOT NULL,
    args_json           TEXT NOT NULL,
    exit_code           INTEGER NOT NULL DEFAULT 0,
    duration_ms         REAL NOT NULL,
    raw_token_count     INTEGER NOT NULL,
    masked_summary      TEXT NOT NULL,
    obs_zstd_blob       BLOB NOT NULL,
    stderr_hash         TEXT NOT NULL DEFAULT '',
    created_at_ns       INTEGER NOT NULL
) STRICT;

-- 10. SagaLLM + Atomix Transactional Outbox for External Side Effects
CREATE TABLE IF NOT EXISTS saga_outbox (
    effect_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_id               INTEGER NOT NULL REFERENCES commits(commit_id),
    branch_name             TEXT NOT NULL,
    idempotency_key         TEXT NOT NULL UNIQUE,
    target_service          TEXT NOT NULL,
    gate_policy             TEXT NOT NULL DEFAULT 'DEFER_UNTIL_MERGE',
    forward_action_json     TEXT NOT NULL,
    compensate_action_json  TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'STAGED',
    response_json           TEXT,
    created_at_ns           INTEGER NOT NULL,
    executed_at_ns          INTEGER
) STRICT;

-- 11. ePCA Executable State Invariants
CREATE TABLE IF NOT EXISTS state_invariants (
    invariant_id     TEXT PRIMARY KEY,
    description      TEXT NOT NULL,
    invariant_kind   TEXT NOT NULL,
    spec_json        TEXT NOT NULL,
    enabled          INTEGER NOT NULL DEFAULT 1
) STRICT;

-- 12. AIOS Preemptible Process Control Block (PCB) & PASTE Markov Transitions
CREATE TABLE IF NOT EXISTS agent_pcb (
    pcb_id               TEXT PRIMARY KEY,
    branch_name          TEXT NOT NULL REFERENCES branches(branch_name),
    commit_id            INTEGER NOT NULL REFERENCES commits(commit_id),
    active_messages_json TEXT NOT NULL,
    prefix_cache_hash    TEXT NOT NULL,
    rng_seed             INTEGER NOT NULL DEFAULT 42,
    state_status         TEXT NOT NULL DEFAULT 'RUNNING',
    updated_at_ns        INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS tool_transitions (
    prev_signature      TEXT NOT NULL,
    next_signature      TEXT NOT NULL,
    hit_count           INTEGER NOT NULL DEFAULT 1,
    last_seen_ns        INTEGER NOT NULL,
    PRIMARY KEY (prev_signature, next_signature)
) STRICT;
"""
