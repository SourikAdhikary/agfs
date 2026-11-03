"""Layer 3: HippoRAG 2 (2025) Lightweight Entity Graph & In-Memory NumPy Personalized PageRank."""

from __future__ import annotations

import numpy as np

from agfs.core.config import KernelConfig
from agfs.storage.db import SQLiteWALEngine


class HippoRAGGraph:
    """Manages `kg_entities` and `kg_edges` and runs 10-step Personalized PageRank (PPR)."""

    def __init__(self, db: SQLiteWALEngine, config: KernelConfig) -> None:
        self.db = db
        self.config = config

    def ensure_entity(self, canonical_name: str, entity_type: str = "CONCEPT") -> int:
        norm_name = canonical_name.strip()
        self.db.conn.execute(
            """
            INSERT INTO kg_entities(canonical_name, entity_type)
            VALUES (?, ?)
            ON CONFLICT(canonical_name) DO NOTHING
            """,
            (norm_name, entity_type),
        )
        row = self.db.conn.execute(
            "SELECT entity_id FROM kg_entities WHERE canonical_name = ?",
            (norm_name,),
        ).fetchone()
        return int(row["entity_id"])

    def add_edge(
        self,
        *,
        src_name: str,
        dst_name: str,
        relation: str,
        commit_id: int,
        memory_id: int | None = None,
        weight: float = 1.0,
        src_type: str = "SYMBOL",
        dst_type: str = "SYMBOL",
    ) -> int:
        src_id = self.ensure_entity(src_name, src_type)
        dst_id = self.ensure_entity(dst_name, dst_type)
        # Supersede prior identical edge if re-binding
        self.db.conn.execute(
            """
            UPDATE kg_edges
            SET tx_commit_superseded = ?
            WHERE src_entity_id = ? AND relation = ? AND tx_commit_superseded IS NULL
              AND dst_entity_id != ?
            """,
            (commit_id, src_id, relation, dst_id),
        )
        cur = self.db.conn.execute(
            """
            INSERT INTO kg_edges(
                src_entity_id, dst_entity_id, relation, weight,
                memory_id, tx_commit_born, tx_commit_superseded
            ) VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (src_id, dst_id, relation, weight, memory_id, commit_id),
        )
        return int(cur.lastrowid or 0)

    def run_personalized_pagerank(
        self,
        *,
        query_text: str,
        seed_memory_ids: list[int],
        visible_memory_ids: set[int],
        ancestor_commit_ids: set[int],
    ) -> dict[int, float]:
        """Run 10 iterations of Personalized PageRank (PPR) over active `kg_edges`.

        Returns `{memory_id: ppr_score}` for all bi-temporally visible memories reachable
        over multi-hop entity connections.
        """
        if not visible_memory_ids:
            return {}

        ent_rows = self.db.conn.execute(
            "SELECT entity_id, canonical_name FROM kg_entities"
        ).fetchall()
        if not ent_rows:
            return {}

        ent_ids = [int(r["entity_id"]) for r in ent_rows]
        id_to_idx = {eid: idx for idx, eid in enumerate(ent_ids)}
        idx_to_name = {idx: str(r["canonical_name"]).lower() for idx, r in enumerate(ent_rows)}
        n_nodes = len(ent_ids)

        edge_rows = self.db.conn.execute(
            """
            SELECT src_entity_id, dst_entity_id, weight, memory_id,
                   tx_commit_born, tx_commit_superseded
            FROM kg_edges
            """
        ).fetchall()
        if not edge_rows:
            return {}

        # Filter edges by bi-temporal commit visibility
        active_edges: list[tuple[int, int, float, int | None]] = []
        entity_to_memories: dict[int, set[int]] = {}
        fallback_seed_entities: set[int] = set()
        seed_mem_set = set(seed_memory_ids)

        for r in edge_rows:
            born = int(r["tx_commit_born"])
            sup = int(r["tx_commit_superseded"]) if r["tx_commit_superseded"] is not None else None
            if born not in ancestor_commit_ids:
                continue
            if sup is not None and sup in ancestor_commit_ids:
                continue
            u = id_to_idx.get(int(r["src_entity_id"]))
            v = id_to_idx.get(int(r["dst_entity_id"]))
            if u is None or v is None:
                continue
            w = float(r["weight"])
            mid = int(r["memory_id"]) if r["memory_id"] is not None else None
            active_edges.append((u, v, w, mid))
            if mid is not None and mid in visible_memory_ids:
                entity_to_memories.setdefault(u, set()).add(mid)
                entity_to_memories.setdefault(v, set()).add(mid)
                if mid in seed_mem_set:
                    fallback_seed_entities.add(u)
                    fallback_seed_entities.add(v)

        if not active_edges:
            return {}

        # HippoRAG 2 Query Entity Linking: prefer explicit query entities for personalization reset p0;
        # fall back to passage-level seed_memory_ids only when no query entity matched.
        q_lower = query_text.lower()
        q_tokens = set(q_lower.replace("?", " ").replace(",", " ").replace(".", " ").split())
        direct_query_entities: set[int] = set()
        for idx, ename in idx_to_name.items():
            short_name = ename.split(":")[-1]
            if ename in q_tokens or short_name in q_tokens:
                direct_query_entities.add(idx)

        seed_entities = direct_query_entities if direct_query_entities else fallback_seed_entities
        if not seed_entities:
            return {}

        # Build column-normalized bidirectional adjacency matrix P
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for u, v, w, _ in active_edges:
            adj[v, u] += w
            adj[u, v] += w * 0.85  # Bidirectional associative flow

        col_sums = adj.sum(axis=0, keepdims=True)
        # Dangling nodes reset to personalization vector
        non_zero = col_sums > 1e-8
        p_mat = np.where(non_zero, adj / np.maximum(col_sums, 1e-8), 0.0)

        p0 = np.zeros(n_nodes, dtype=np.float32)
        for s_idx in seed_entities:
            p0[s_idx] = 1.0 / len(seed_entities)

        pi = p0.copy()
        alpha = self.config.ppr_damping_alpha
        for _ in range(self.config.ppr_iterations):
            pi = (1.0 - alpha) * (p_mat @ pi) + alpha * p0

        mem_scores: dict[int, float] = {}
        for node_idx, mids in entity_to_memories.items():
            node_prob = float(pi[node_idx])
            if node_prob > 1e-6:
                # Multi-hop associative bonus for non-seed entities reached via graph walk
                hop_multiplier = 1.5 if node_idx not in direct_query_entities else 1.0
                for mid in mids:
                    mem_scores[mid] = mem_scores.get(mid, 0.0) + (node_prob * hop_multiplier)

        return mem_scores
