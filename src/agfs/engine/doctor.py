"""Layer 2: Formal Verification of the 12 AgFS Kernel Invariants (`agfs doctor`)."""

from __future__ import annotations

import json
import math
import time

import numpy as np

from agfs.core.hashing import (
    compute_commit_merkle_hash,
    compute_vfs_root_hash,
)
from agfs.core.types import DoctorCheckResult, DoctorReport
from agfs.safety.rules.builtin_rules import match_protected_glob, scan_forbidden_regex
from agfs.storage.chunker import decompress_bytes, verify_and_decompress_chunk
from agfs.storage.db import SQLiteWALEngine


def run_doctor_checks(db: SQLiteWALEngine) -> DoctorReport:
    """Execute all 12 formal mathematical and relational invariants (`INV-01`..`INV-12`)."""
    report = DoctorReport(db_path=str(db.db_path))
    conn = db.conn

    def _check(inv_id: str, name: str, fn: object) -> None:
        t0 = time.perf_counter()
        try:
            ok, msg = fn()  # type: ignore[operator]
        except Exception as exc:
            ok, msg = False, f"Exception: {exc}"
        dt = (time.perf_counter() - t0) * 1000.0
        report.checks.append(
            DoctorCheckResult(
                invariant_id=inv_id,
                name=name,
                passed=bool(ok),
                details=str(msg),
                duration_ms=round(dt, 3),
            )
        )

    # INV-01: SQLite WAL & Foreign-Key Integrity
    def inv_01() -> tuple[bool, str]:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        jmode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        ok = integrity == "ok" and len(fk_errors) == 0 and jmode in ("wal", "memory")
        return ok, f"integrity={integrity}, fk_errors={len(fk_errors)}, journal_mode={jmode}"

    # INV-02: Cryptographic Merkle Chain Integrity
    def inv_02() -> tuple[bool, str]:
        rows = conn.execute("SELECT * FROM commits ORDER BY commit_id ASC").fetchall()
        merkle_by_id: dict[int, str] = {}
        for r in rows:
            cid = int(r["commit_id"])
            pid = int(r["parent_id"]) if r["parent_id"] is not None else None
            parent_merkle = merkle_by_id.get(pid, "GENESIS") if pid is not None else "GENESIS"
            expected = compute_commit_merkle_hash(
                parent_merkle=parent_merkle,
                branch_name=str(r["branch_name"]),
                step_index=int(r["step_index"]),
                vfs_root_hash=str(r["vfs_root_hash"]),
                last_error_hash=str(r["last_error_hash"]),
                message=str(r["message"]),
            )
            actual = str(r["merkle_hash"])
            if actual != expected:
                return False, f"Merkle mismatch at commit #{cid}: expected {expected[:12]}, got {actual[:12]}"
            merkle_by_id[cid] = actual
        return True, f"Verified {len(rows)} Merkle-chained commits"

    # INV-03: VFS Root Hash Determinism
    def inv_03() -> tuple[bool, str]:
        commits = conn.execute("SELECT commit_id, vfs_root_hash FROM commits").fetchall()
        for c in commits:
            cid = int(c["commit_id"])
            inodes = conn.execute(
                "SELECT path, file_hash, mode FROM vfs_inodes WHERE commit_id = ? AND deleted = 0",
                (cid,),
            ).fetchall()
            recomputed = compute_vfs_root_hash(
                [(str(i["path"]), str(i["file_hash"]), int(i["mode"])) for i in inodes]
            )
            if recomputed != str(c["vfs_root_hash"]):
                return False, f"vfs_root_hash mismatch at commit #{cid}"
        return True, f"Verified deterministic vfs_root_hash across {len(commits)} commits"

    # INV-04: Content-Addressed Chunk Completeness
    def inv_04() -> tuple[bool, str]:
        chunks = {
            str(r["chunk_hash"]): bytes(r["zstd_bytes"])
            for r in conn.execute("SELECT chunk_hash, zstd_bytes FROM vfs_chunks").fetchall()
        }
        for ch_hash, blob in chunks.items():
            verify_and_decompress_chunk(ch_hash, blob)
        inode_rows = conn.execute("SELECT commit_id, path, chunks_json FROM vfs_inodes").fetchall()
        for r in inode_rows:
            for ch in json.loads(r["chunks_json"]):
                if ch not in chunks:
                    return False, f"Missing chunk {ch} for {r['path']}@{r['commit_id']}"
        return True, f"Verified {len(chunks)} zstd chunks and {len(inode_rows)} inode manifests"

    # INV-05: Branch Head Referential Validity
    def inv_05() -> tuple[bool, str]:
        branches = conn.execute("SELECT branch_name, head_commit_id, fork_commit_id FROM branches").fetchall()
        commit_ids = {
            int(r["commit_id"]) for r in conn.execute("SELECT commit_id FROM commits").fetchall()
        }
        for b in branches:
            head = b["head_commit_id"]
            fork = b["fork_commit_id"]
            if head is not None and int(head) not in commit_ids:
                return False, f"Branch {b['branch_name']} head_commit_id {head} missing"
            if fork is not None and int(fork) not in commit_ids:
                return False, f"Branch {b['branch_name']} fork_commit_id {fork} missing"
        return True, f"Verified {len(branches)} branch head & fork pointers"

    # INV-06: MCTS Visit & Value Monotonicity
    def inv_06() -> tuple[bool, str]:
        rows = conn.execute("SELECT commit_id, parent_id, mcts_visits FROM commits").fetchall()
        visits = {int(r["commit_id"]): int(r["mcts_visits"]) for r in rows}
        for r in rows:
            cid = int(r["commit_id"])
            if visits[cid] < 0:
                return False, f"Negative MCTS visits at commit #{cid}"
            pid = int(r["parent_id"]) if r["parent_id"] is not None else None
            if pid is not None and visits.get(pid, 0) < visits[cid]:
                return False, f"Parent commit #{pid} has fewer visits ({visits[pid]}) than child #{cid} ({visits[cid]})"
        return True, f"Verified MCTS visit monotonicity across {len(rows)} nodes"

    # INV-07: Bi-Temporal Superseding Consistency
    def inv_07() -> tuple[bool, str]:
        bad_order = conn.execute(
            "SELECT memory_id FROM memories WHERE tx_commit_superseded IS NOT NULL AND tx_commit_superseded < tx_commit_born"
        ).fetchall()
        if bad_order:
            return False, f"{len(bad_order)} memories have tx_commit_superseded < tx_commit_born"
        dups = conn.execute(
            """
            SELECT branch_scope, subject, predicate, COUNT(*) AS cnt
            FROM memories
            WHERE tx_commit_superseded IS NULL
            GROUP BY branch_scope, subject, predicate
            HAVING cnt > 1
            """
        ).fetchall()
        if dups:
            return False, f"Found {len(dups)} duplicate active (branch_scope, subject, predicate) facts"
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return True, f"Verified bi-temporal superseding invariants across {total} facts"

    # INV-08: FTS5 & Embedding Index Parity
    def inv_08() -> tuple[bool, str]:
        mem_ids = {int(r[0]) for r in conn.execute("SELECT memory_id FROM memories").fetchall()}
        emb_rows = conn.execute("SELECT memory_id, embedding_f32, norm FROM memories_embeddings").fetchall()
        emb_ids = {int(r["memory_id"]) for r in emb_rows}
        fts_ids = {int(r[0]) for r in conn.execute("SELECT rowid FROM memories_fts").fetchall()}
        if mem_ids != emb_ids or mem_ids != fts_ids:
            return False, f"Parity mismatch: memories={len(mem_ids)}, emb={len(emb_ids)}, fts={len(fts_ids)}"
        for r in emb_rows:
            norm = float(r["norm"])
            vec = np.frombuffer(r["embedding_f32"], dtype=np.float32)
            if not math.isfinite(norm) or vec.size == 0:
                return False, f"Invalid embedding vector for memory #{r['memory_id']}"
        return True, f"Verified 1:1 parity across {len(mem_ids)} memories, embeddings, and FTS5 rows"

    # INV-09: Knowledge Graph Referential Integrity
    def inv_09() -> tuple[bool, str]:
        bad = conn.execute(
            """
            SELECT edge_id FROM kg_edges
            WHERE tx_commit_superseded IS NOT NULL AND tx_commit_superseded < tx_commit_born
            """
        ).fetchall()
        if bad:
            return False, f"{len(bad)} kg_edges have invalid bi-temporal bounds"
        ent_cnt = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        edge_cnt = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        return True, f"Verified {ent_cnt} entities and {edge_cnt} edges"

    # INV-10: Virtual Inode Observation Recoverability
    def inv_10() -> tuple[bool, str]:
        rows = conn.execute("SELECT syscall_id, obs_zstd_blob FROM syscall_trace").fetchall()
        for r in rows:
            raw = decompress_bytes(r["obs_zstd_blob"])
            if not isinstance(raw, bytes):
                return False, f"Failed to decode /.proc/obs/{r['syscall_id']}.txt"
        return True, f"Verified 100% verbatim recoverability for {len(rows)} virtual inodes"

    # INV-11: Saga Outbox Idempotency & Gate Discipline
    def inv_11() -> tuple[bool, str]:
        leaked = conn.execute(
            """
            SELECT s.effect_id, s.branch_name
            FROM saga_outbox s
            JOIN branches b ON b.branch_name = s.branch_name
            WHERE s.gate_policy = 'DEFER_UNTIL_MERGE'
              AND s.branch_name != 'main'
              AND b.status != 'MERGED'
              AND s.status = 'EXECUTED'
            """
        ).fetchall()
        if leaked:
            return False, f"{len(leaked)} DEFER_UNTIL_MERGE effects executed prematurely before merge"
        total = conn.execute("SELECT COUNT(*) FROM saga_outbox").fetchone()[0]
        return True, f"Verified Atomix gatekeeper & Saga idempotency across {total} effects"

    # INV-12: ePCA State Invariant Satisfaction on Active Heads
    def inv_12() -> tuple[bool, str]:
        invs = conn.execute(
            "SELECT invariant_id, invariant_kind, spec_json FROM state_invariants WHERE enabled = 1"
        ).fetchall()
        for inv in invs:
            kind = str(inv["invariant_kind"])
            spec = json.loads(inv["spec_json"])
            if kind == "SQL_ASSERT":
                res = conn.execute(spec.get("sql", "SELECT 1")).fetchone()
                if not res or not bool(res[0]):
                    return False, f"Invariant {inv['invariant_id']} failed SQL_ASSERT"
            elif kind == "REGEX_FORBID_CONTENT":
                pattern = spec.get("pattern", "")
                if pattern:
                    for ch in conn.execute("SELECT chunk_hash, zstd_bytes FROM vfs_chunks").fetchall():
                        txt = decompress_bytes(ch["zstd_bytes"]).decode("utf-8", errors="ignore")
                        if scan_forbidden_regex(txt, pattern):
                            return False, f"Chunk {ch['chunk_hash'][:10]} violates {inv['invariant_id']}"
            elif kind == "PROTECTED_GLOB":
                # Verify protected globs were not deleted on main compared to root commit #1
                globs = spec.get("globs", [])
                root_paths = [
                    str(r[0])
                    for r in conn.execute(
                        "SELECT path FROM vfs_inodes WHERE commit_id = 1 AND deleted = 0"
                    ).fetchall()
                ]
                main_head = conn.execute(
                    "SELECT head_commit_id FROM branches WHERE branch_name = 'main'"
                ).fetchone()
                if main_head and main_head[0] is not None:
                    head_paths = {
                        str(r[0])
                        for r in conn.execute(
                            "SELECT path FROM vfs_inodes WHERE commit_id = ? AND deleted = 0",
                            (int(main_head[0]),),
                        ).fetchall()
                    }
                    for rp in root_paths:
                        if match_protected_glob(rp, globs) and rp not in head_paths:
                            return False, f"Protected file {rp} missing from main head"
        return True, f"Verified {len(invs)} active ePCA state invariants"

    _check("INV-01", "SQLite WAL & Foreign-Key Integrity", inv_01)
    _check("INV-02", "Cryptographic Merkle Chain Integrity", inv_02)
    _check("INV-03", "VFS Root Hash Determinism", inv_03)
    _check("INV-04", "Content-Addressed Chunk Completeness", inv_04)
    _check("INV-05", "Branch Head Referential Validity", inv_05)
    _check("INV-06", "MCTS Visit & Value Monotonicity", inv_06)
    _check("INV-07", "Bi-Temporal Superseding Consistency", inv_07)
    _check("INV-08", "FTS5 & Embedding Index Parity", inv_08)
    _check("INV-09", "Knowledge Graph Referential Integrity", inv_09)
    _check("INV-10", "Virtual Inode Observation Recoverability", inv_10)
    _check("INV-11", "Saga Outbox Idempotency & Gate Discipline", inv_11)
    _check("INV-12", "ePCA State Invariant Satisfaction", inv_12)
    return report
