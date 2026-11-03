"""Unit tests for CoW VFS branching, MCTS UCT selection/backprop, and SentinelAgent StateCycleTrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from agfs.core.errors import StateCycleTrap
from agfs.interfaces.kernel import AgFSKernel


def test_cow_branch_fork_commit_and_time_travel(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "main.py").write_text("VERSION = 1\n")
    db_path = tmp_path / "state.agdb"

    with AgFSKernel(db_path=db_path, workspace_root=ws) as kernel:
        c1 = kernel.current_commit_id
        assert c1 == 1

        # Fork speculative branch `try-v2`
        kernel.fork("try-v2", from_commit=c1)
        (ws / "main.py").write_text("VERSION = 2\n")
        c2 = kernel.commit("Update to v2", test_pass_rate=0.4).commit_id
        assert (ws / "main.py").read_text() == "VERSION = 2\n"

        # Checkout back to `main` (commit #1) -> on-disk file reverts to VERSION = 1!
        kernel.checkout("main")
        assert (ws / "main.py").read_text() == "VERSION = 1\n"

        # Diff commits #1 and #2
        diffs = kernel.vfs.diff_commits(c1, c2)
        assert diffs == {"main.py": "MODIFIED"}


def test_mcts_uct_selection_and_recursive_backprop(tmp_path: Path) -> None:
    db_path = tmp_path / "mcts.agdb"
    with AgFSKernel(db_path=db_path) as kernel:
        root_cid = kernel.current_commit_id
        kernel.fork("branch-low", from_commit=root_cid, checkout=True)
        c_low = kernel.commit("Low reward step", virtual_overlays={"a.py": "x=1"}).commit_id
        kernel.backpropagate(0.1, commit_id=c_low)

        kernel.fork("branch-high", from_commit=root_cid, checkout=True)
        c_high = kernel.commit("High reward step", virtual_overlays={"a.py": "x=2"}).commit_id
        kernel.backpropagate(0.95, commit_id=c_high)

        ranking = kernel.select_uct(exploration_c=0.5)
        assert ranking[0].branch_name == "branch-high"
        assert ranking[0].mean_value > ranking[-1].mean_value

        # Root commit visits must have accumulated both backpropagations
        root_node = kernel.mcts.get_commit(root_cid)
        assert root_node.mcts_visits >= 3


def test_sentinel_state_cycle_trap(tmp_path: Path) -> None:
    db_path = tmp_path / "cycles.agdb"
    with AgFSKernel(db_path=db_path) as kernel:
        kernel.commit(
            "Step 1 broken edit",
            virtual_overlays={"auth.py": "buggy_v1"},
            stderr_text="AssertionError at line 10 in test_login (0x7fff1234abcd in 0.14s)",
        )
        kernel.commit(
            "Step 2 different edit",
            virtual_overlays={"auth.py": "buggy_v2"},
            stderr_text="TypeError: NoneType is not callable",
        )
        # Returning to buggy_v1 with the same normalized error (even with different hex address & time!) triggers StateCycleTrap
        with pytest.raises(StateCycleTrap) as exc_info:
            kernel.commit(
                "Step 3 oscillation back to buggy_v1",
                virtual_overlays={"auth.py": "buggy_v1"},
                stderr_text="AssertionError at line 10 in test_login (0x7fff9999ffff in 0.89s)",
            )
        assert exc_info.value.prior_commit_id == 2
