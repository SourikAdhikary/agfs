"""Example 03: Exploring 3 speculative coding branches with SWE-Search MCTS UCT & <1.5ms rollback."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agfs import AgFSKernel


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agfs_ex03_") as tmp:
        ws = Path(tmp) / "repo"
        ws.mkdir()
        (ws / "cache.py").write_text("BACKEND = 'none'\n")

        with AgFSKernel(db_path=Path(tmp) / "mcts.agdb", workspace_root=ws) as kernel:
            root = kernel.current_commit_id
            for name in ("try-redis", "try-memcached", "try-sqlite-wal"):
                kernel.fork(name, from_commit=root, checkout=False)

            # Score each candidate branch with MCTS UCT
            scores = {"try-redis": 0.35, "try-memcached": 0.50, "try-sqlite-wal": 0.98}
            for _ in range(3):
                cand = kernel.select_uct()[0]
                kernel.checkout(cand.branch_name)
                kernel.commit(
                    f"Implement {cand.branch_name}",
                    virtual_overlays={"cache.py": f"BACKEND = {cand.branch_name!r}\n"},
                    test_pass_rate=scores[cand.branch_name],
                )

            winner = max(kernel.select_uct(), key=lambda c: c.mean_value)
            kernel.merge_to_main(winner.branch_name)
            print(f"Merged winner: {winner.branch_name} -> {(ws / 'cache.py').read_text().strip()}")


if __name__ == "__main__":
    main()
