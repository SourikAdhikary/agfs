"""Layer 3: 3-Way Bi-Temporal Hybrid Recall (`Dense Cosine` + `FTS5 BM25` + `HippoRAG 2 PPR`)."""

from __future__ import annotations

import re

import numpy as np

from agfs.core.config import KernelConfig
from agfs.core.types import RecallHit
from agfs.memory.bitemporal import BiTemporalMemoryStore
from agfs.memory.ppr import HippoRAGGraph
from agfs.storage.db import SQLiteWALEngine

_FTS_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


class HybridMemoryEngine:
    """Combines Bi-Temporal visibility filtering with 3-Way Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        db: SQLiteWALEngine,
        config: KernelConfig,
        bitemporal: BiTemporalMemoryStore,
        graph: HippoRAGGraph,
    ) -> None:
        self.db = db
        self.config = config
        self.bitemporal = bitemporal
        self.graph = graph

    def recall(
        self,
        *,
        query: str,
        ancestor_commit_ids: list[int],
        active_branch: str | None = None,
        top_k: int = 5,
        world_timestamp: float | None = None,
        exclude_untrusted: bool = False,
    ) -> list[RecallHit]:
        """Run 3-way hybrid retrieval over bi-temporally visible memories."""
        visible_map = self.bitemporal.get_visible_memories(
            ancestor_commit_ids=ancestor_commit_ids,
            active_branch=active_branch,
            world_timestamp=world_timestamp,
            exclude_untrusted=exclude_untrusted,
        )
        if not visible_map:
            return []

        visible_ids = sorted(visible_map.keys())
        visible_set = set(visible_ids)

        # Channel 1: Dense Vector Cosine Similarity (NumPy / Accelerate vectorized dot product)
        q_vec = self.bitemporal.embedder.encode(query)
        placeholders = ",".join("?" for _ in visible_ids)
        emb_rows = self.db.conn.execute(
            f"SELECT memory_id, embedding_f32 FROM memories_embeddings WHERE memory_id IN ({placeholders})",
            tuple(visible_ids),
        ).fetchall()

        vec_scores: dict[int, float] = {}
        if emb_rows:
            mids = [int(r["memory_id"]) for r in emb_rows]
            mat = np.vstack(
                [np.frombuffer(r["embedding_f32"], dtype=np.float32) for r in emb_rows]
            )
            sims = mat @ q_vec
            for mid, sim in zip(mids, sims, strict=True):
                vec_scores[mid] = float(sim)

        vec_ranked = sorted(vec_scores.keys(), key=lambda m: vec_scores[m], reverse=True)
        vec_rank_map = {mid: rank + 1 for rank, mid in enumerate(vec_ranked)}

        # Channel 2: Lexical SQLite FTS5 BM25
        bm25_scores: dict[int, float] = {}
        fts_tokens = _FTS_TOKEN_RE.findall(query)
        if fts_tokens:
            fts_query = " OR ".join(f'"{t}"' for t in fts_tokens[:16])
            try:
                fts_rows = self.db.conn.execute(
                    """
                    SELECT rowid AS memory_id, -bm25(memories_fts, 2.0, 1.5, 1.0) AS score
                    FROM memories_fts
                    WHERE memories_fts MATCH ?
                    ORDER BY score DESC
                    LIMIT 100
                    """,
                    (fts_query,),
                ).fetchall()
                for r in fts_rows:
                    mid = int(r["memory_id"])
                    if mid in visible_set:
                        bm25_scores[mid] = max(0.0, float(r["score"]))
            except Exception:
                pass

        bm25_ranked = sorted(bm25_scores.keys(), key=lambda m: bm25_scores[m], reverse=True)
        bm25_rank_map = {mid: rank + 1 for rank, mid in enumerate(bm25_ranked)}

        # Channel 3: HippoRAG 2 Personalized PageRank seeded from query entities + top lexical/vector hits
        seed_mids = list(dict.fromkeys(bm25_ranked[:3] + vec_ranked[:3]))
        ppr_scores = self.graph.run_personalized_pagerank(
            query_text=query,
            seed_memory_ids=seed_mids,
            visible_memory_ids=visible_set,
            ancestor_commit_ids=set(ancestor_commit_ids),
        )
        ppr_ranked = sorted(ppr_scores.keys(), key=lambda m: ppr_scores[m], reverse=True)
        ppr_rank_map = {mid: rank + 1 for rank, mid in enumerate(ppr_ranked)}

        # 4. Weighted Reciprocal Rank Fusion (RRF)
        k0 = self.config.rrf_k0
        w_vec = self.config.rrf_weight_vec
        w_bm25 = self.config.rrf_weight_bm25
        w_ppr = self.config.rrf_weight_ppr

        hits: list[RecallHit] = []
        for mid, fact in visible_map.items():
            r_v = vec_rank_map.get(mid)
            r_b = bm25_rank_map.get(mid)
            r_p = ppr_rank_map.get(mid)

            rrf = 0.0
            if r_v is not None and vec_scores.get(mid, 0.0) > 0.01:
                rrf += w_vec / (k0 + r_v)
            if r_b is not None:
                rrf += w_bm25 / (k0 + r_b)
            if r_p is not None:
                rrf += w_ppr / (k0 + r_p)

            # Small tie-breaker bonus for direct semantic/BM25 score magnitude
            rrf += 0.001 * max(0.0, vec_scores.get(mid, 0.0)) + 0.01 * ppr_scores.get(mid, 0.0)

            hits.append(
                RecallHit(
                    memory=fact,
                    rrf_score=round(rrf, 6),
                    vec_score=round(vec_scores.get(mid, 0.0), 4),
                    bm25_score=round(bm25_scores.get(mid, 0.0), 4),
                    ppr_score=round(ppr_scores.get(mid, 0.0), 4),
                    vec_rank=r_v,
                    bm25_rank=r_b,
                    ppr_rank=r_p,
                )
            )

        hits.sort(
            key=lambda h: (h.rrf_score, h.ppr_score, h.bm25_score, h.vec_score, h.memory.memory_id),
            reverse=True,
        )
        return hits[:top_k]
