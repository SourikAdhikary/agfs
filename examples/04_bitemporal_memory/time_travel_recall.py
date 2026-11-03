"""Example 04: Bi-Temporal Point-in-Time Memory + HippoRAG 2 Multi-Hop PageRank Recall."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agfs import AgFSKernel


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agfs_ex04_") as tmp:
        with AgFSKernel(db_path=Path(tmp) / "mem.agdb") as kernel:
            # Commit #1: Record v1 API contract
            kernel.remember("api:payments", "auth", "Uses HMAC-SHA256 header v1")
            c1 = kernel.current_commit_id

            # Commit #2: Supersede with v2 mTLS contract + 2-hop graph edge
            kernel.commit("Upgrade payments auth to v2", virtual_overlays={"pay.py": "# v2"})
            kernel.remember(
                "api:payments",
                "auth",
                "Uses Ed25519 mTLS client certificates v2",
                edges=[("api:payments", "CALLS", "infra:LedgerVault")],
            )

            print("Current HEAD recall:", kernel.recall("api:payments auth", top_k=1)[0].memory.content)
            print("Time-travel Commit #1:", kernel.recall("api:payments auth", top_k=1, as_of_commit=c1)[0].memory.content)


if __name__ == "__main__":
    main()
