"""Chaos & Tamper Verification Tests (`INV-01`..`INV-12`)."""

from __future__ import annotations

from pathlib import Path

from agfs.interfaces.kernel import AgFSKernel


def test_doctor_detects_merkle_tampering_and_recovers(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "core.py").write_text("def run(): return 1\n")
    db_path = tmp_path / "chaos.agdb"

    with AgFSKernel(db_path=db_path, workspace_root=ws) as kernel:
        kernel.commit("Step 1", virtual_overlays={"core.py": "def run(): return 2\n"})
        rep_clean = kernel.doctor()
        assert rep_clean.all_passed is True
        assert len(rep_clean.checks) == 12

        # Tamper directly with a commit message in SQLite without updating its Merkle hash
        orig_msg = kernel.mcts.get_commit(2).message
        kernel.db.conn.execute("UPDATE commits SET message = 'TAMPERED_BY_ATTACKER' WHERE commit_id = 2")
        rep_tampered = kernel.doctor()
        inv02 = next(c for c in rep_tampered.checks if c.invariant_id == "INV-02")
        assert inv02.passed is False

        # Restore original message -> INV-02 passes again
        kernel.db.conn.execute("UPDATE commits SET message = ? WHERE commit_id = 2", (orig_msg,))
        assert kernel.doctor().all_passed is True
