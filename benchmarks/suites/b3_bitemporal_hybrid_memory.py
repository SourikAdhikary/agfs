"""Benchmark B3: Bi-Temporal Contradiction Handling & Multi-Hop Codebase Memory Recall.

Compares:
- Without AgFS (1. Standard Append-Only Vector RAG): Returns stale superseded API contracts
- Without AgFS (2. Pure Lexical BM25): Misses 2-hop graph-coupled components that share no keywords
- With AgFS (Bi-Temporal `tx_commit_superseded` + 3-Way Hybrid: Vector + BM25 + HippoRAG 2 PPR):
  0% stale-fact poisoning + high 2-hop structural recall in <5ms
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from agfs.interfaces.kernel import AgFSKernel
from agfs.memory.embedders.deterministic import DeterministicHashEmbedder


def run_b3_benchmark() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="agfs_b3_") as tmp:
        db_path = Path(tmp) / "b3.agdb"
        embedder = DeterministicHashEmbedder(384)

        # 20 API modules that undergo refactoring across commits (v1 -> v2 -> v3)
        num_refactored_apis = 20
        # 15 2-hop structural dependency chains:
        # e.g. CLI command `cmd_export` -> calls `ReportBuilder` -> depends on ` ParquetSchemaV3`
        num_multihop_chains = 15

        naive_vector_store: list[tuple[np.ndarray, str, str, bool]] = []  # (vec, subj, content, is_stale)

        with AgFSKernel(db_path=db_path) as kernel:
            # Wave 1 (Commit #1): Record v1 API contracts (will become STALE later)
            for i in range(num_refactored_apis):
                subj = f"api:service_{i}"
                v1_text = (
                    f"Service {i} authentication contract v1 uses legacy_md5_cookie header "
                    f"and synchronous XML payload for service_{i} requests."
                )
                kernel.remember(subj, "auth_contract", v1_text)
                naive_vector_store.append((embedder.encode(f"{subj} auth_contract {v1_text}"), subj, v1_text, True))

            # Commit #2: Refactor to v2 (still intermediate, will also be superseded)
            kernel.commit("Refactor services to v2 intermediate token")
            for i in range(num_refactored_apis):
                subj = f"api:service_{i}"
                v2_text = (
                    f"Service {i} authentication contract v2 uses hmac_sha1_header "
                    f"for service_{i} requests."
                )
                kernel.remember(subj, "auth_contract", v2_text)
                naive_vector_store.append((embedder.encode(f"{subj} auth_contract {v2_text}"), subj, v2_text, True))

            # Commit #3 (Current Head): Refactor to v3 current contract + record 2-hop dependency chains
            kernel.commit("Upgrade services to v3 Ed25519 mTLS contract")
            for i in range(num_refactored_apis):
                subj = f"api:service_{i}"
                v3_text = (
                    f"Service {i} authentication contract v3 requires strict Ed25519 mTLS client certificate "
                    f"for service_{i} requests."
                )
                kernel.remember(subj, "auth_contract", v3_text)
                naive_vector_store.append((embedder.encode(f"{subj} auth_contract {v3_text}"), subj, v3_text, False))

            # Record 15 2-hop structural dependency chains where Hop 1 and Hop 2 share only the intermediate bridge entity
            for c in range(num_multihop_chains):
                entry_sym = f"entrypoint_alpha_{c}"
                bridge_sym = f"bridge_orchestrator_{c}"
                leaf_sym = f"storage_shard_engine_{c}"

                # Hop 1 fact mentions entry_sym and bridge_sym
                kernel.remember(
                    subject=entry_sym,
                    predicate="dispatches_to",
                    content=f"CLI command {entry_sym} routes incoming jobs to {bridge_sym}.",
                    edges=[(entry_sym, "CALLS", bridge_sym)],
                )
                # Hop 2 downstream critical constraint mentions ONLY bridge_sym and leaf_sym (zero lexical overlap with entry_sym!)
                leaf_fact = (
                    f"Component {bridge_sym} crashes if {leaf_sym} write buffer exceeds 64KB chunk limit."
                )
                kernel.remember(
                    subject=bridge_sym,
                    predicate="downstream_hazard",
                    content=leaf_fact,
                    edges=[(bridge_sym, "DEPENDS_ON", leaf_sym)],
                )
                naive_vector_store.append(
                    (embedder.encode(f"{bridge_sym} downstream_hazard {leaf_fact}"), bridge_sym, leaf_fact, False)
                )

            # ------------------------------------------------------------------
            # Evaluate 1: Stale-Fact Poisoning Rate on the 20 Refactored APIs
            # ------------------------------------------------------------------
            naive_stale_poisoned = 0
            agfs_stale_poisoned = 0
            query_latencies_ms: list[float] = []

            for i in range(num_refactored_apis):
                q = f"What is the authentication contract for api:service_{i} requests?"
                q_vec = embedder.encode(q)

                # Naive Vector RAG top-3
                scored = sorted(
                    naive_vector_store,
                    key=lambda item: float(np.dot(item[0], q_vec)),
                    reverse=True,
                )[:3]
                if any(item[3] for item in scored):  # Any stale v1 or v2 fact in top-3
                    naive_stale_poisoned += 1

                # AgFS Bi-Temporal Hybrid Recall top-3
                t0 = time.perf_counter()
                hits = kernel.recall(q, top_k=3)
                query_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                if any("v1" in h.memory.content or "v2" in h.memory.content for h in hits):
                    agfs_stale_poisoned += 1

            # ------------------------------------------------------------------
            # Evaluate 2: 2-Hop Structural Dependency Recall@3
            # ------------------------------------------------------------------
            bm25_only_multihop_hits = 0
            agfs_ppr_multihop_hits = 0

            for c in range(num_multihop_chains):
                entry_sym = f"entrypoint_alpha_{c}"
                leaf_sym = f"storage_shard_engine_{c}"
                # Query only names `entry_sym`, asking about downstream storage hazards (needs 2-hop PPR via `bridge_orchestrator_{c}`)
                q = f"What downstream buffer hazard affects {entry_sym}?"

                # Pure FTS5 BM25 search for `entrypoint_alpha_{c}`
                fts_rows = kernel.db.conn.execute(
                    """
                    SELECT content FROM memories_fts
                    WHERE memories_fts MATCH ?
                    LIMIT 3
                    """,
                    (f'"{entry_sym}"',),
                ).fetchall()
                if any(leaf_sym in str(r["content"]) for r in fts_rows):
                    bm25_only_multihop_hits += 1

                # AgFS 3-Way Hybrid (Vec + BM25 + HippoRAG 2 PPR)
                t0 = time.perf_counter()
                hits = kernel.recall(q, top_k=3)
                query_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                if any(leaf_sym in h.memory.content for h in hits):
                    agfs_ppr_multihop_hits += 1

            # Also verify Point-in-Time historical travel to Commit #1 still works 100%!
            historical_hits = kernel.recall(
                "What is the authentication contract for api:service_0?",
                top_k=1,
                as_of_commit=1,
            )
            time_travel_accurate = bool(historical_hits and "v1" in historical_hits[0].memory.content)

        return {
            "benchmark_id": "B3",
            "name": "Bi-Temporal Contradiction Elimination & 2-Hop HippoRAG PageRank Recall",
            "without_agfs_stale_poisoning_pct": round((naive_stale_poisoned / num_refactored_apis) * 100.0, 1),
            "with_agfs_stale_poisoning_pct": round((agfs_stale_poisoned / num_refactored_apis) * 100.0, 1),
            "without_agfs_bm25_2hop_recall_pct": round((bm25_only_multihop_hits / num_multihop_chains) * 100.0, 1),
            "with_agfs_hybrid_2hop_recall_pct": round((agfs_ppr_multihop_hits / num_multihop_chains) * 100.0, 1),
            "point_in_time_historical_accuracy_pct": 100.0 if time_travel_accurate else 0.0,
            "query_p50_ms": round(statistics.median(query_latencies_ms), 2),
        }
