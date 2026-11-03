"""Unit tests for Bi-Temporal Memory, HippoRAG 2 PPR, and Sleep-Time Compaction."""

from __future__ import annotations

from pathlib import Path

from agfs.interfaces.kernel import AgFSKernel


def test_bitemporal_contradiction_and_point_in_time_travel(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.agdb"
    with AgFSKernel(db_path=db_path) as kernel:
        # Commit #1: Record v1 database driver
        kernel.remember("module:db", "driver", "Uses sqlite3 synchronous driver v1")
        c1 = kernel.current_commit_id

        # Commit #2: Supersede with v2 driver
        c2 = kernel.commit("Upgrade db driver", virtual_overlays={"db.py": "# v2"}).commit_id
        kernel.remember("module:db", "driver", "Uses async libSQL WAL driver v2")

        # Current head recall returns ONLY v2 (0% stale poisoning)
        head_hits = kernel.recall("What driver does module:db use?", top_k=3)
        assert len(head_hits) == 1
        assert "v2" in head_hits[0].memory.content

        # Point-in-time historical recall at Commit #1 returns v1!
        past_hits = kernel.recall("What driver does module:db use?", top_k=3, as_of_commit=c1)
        assert len(past_hits) == 1
        assert "v1" in past_hits[0].memory.content
        assert c2 > c1


def test_hipporag_two_hop_personalized_pagerank_and_sleep_compact(tmp_path: Path) -> None:
    db_path = tmp_path / "ppr.agdb"
    with AgFSKernel(db_path=db_path) as kernel:
        kernel.remember(
            "cli:deploy_cmd",
            "calls",
            "CLI command cli:deploy_cmd invokes ReleaseOrchestrator",
            edges=[("cli:deploy_cmd", "CALLS", "service:ReleaseOrchestrator")],
        )
        kernel.remember(
            "service:ReleaseOrchestrator",
            "hazard",
            "ReleaseOrchestrator deadlocks if SignerHSM pool size is under 4 threads",
            edges=[("service:ReleaseOrchestrator", "REQUIRES", "infra:SignerHSM")],
        )

        # Query mentions only `cli:deploy_cmd` and asks about thread pool deadlocks
        hits = kernel.recall("What thread pool deadlock affects cli:deploy_cmd?", top_k=2)
        assert any("SignerHSM" in h.memory.content for h in hits)

        # Test Sleep-Time Compute compaction of duplicate EPISODE_GIST rows
        kernel.remember("worker:w1", "trace", "Episode 1: saw high latency on chunk 4", kind="EPISODE_GIST")
        kernel.remember("worker:w1", "new_trace", "Episode 2: retries succeeded", kind="EPISODE_GIST")
        # Add second episode on same (subject, predicate) directly for consolidation check
        kernel.db.conn.execute(
            """
            INSERT INTO memories(
                kind, subject, predicate, content, tx_commit_born, tx_commit_superseded,
                branch_scope, valid_from_ts, trust_level, importance, created_at_ns
            ) VALUES ('EPISODE_GIST', 'worker:w1', 'new_trace', 'Episode 3: latency normalized', ?, NULL, 'main', 1.0, 'AGENT_DERIVED', 0.5, 1)
            """,
            (kernel.current_commit_id,),
        )
        compacted = kernel.sleep.sleep_compact(
            branch_name=kernel.active_branch,
            commit_id=kernel.current_commit_id,
        )
        assert compacted >= 2
