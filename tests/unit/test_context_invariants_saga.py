"""Unit tests for Virtual Inode Observation Masking, PASTE Prefetching, ePCA Invariants & Saga Outbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from agfs.core.errors import CapabilityDeniedTrap, InvariantViolationTrap
from agfs.interfaces.kernel import AgFSKernel
from agfs.safety.connectors.mock_service import MockExternalServiceCluster


def test_virtual_inode_observation_masking_and_paste(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("print('a')\n")
    (ws / "b.py").write_text("print('b')\n")
    db_path = tmp_path / "ctx.agdb"

    with AgFSKernel(db_path=db_path, workspace_root=ws) as kernel:
        long_output = "\n".join(f"Line {i}: verbose pytest log output" for i in range(1, 150))
        rec1 = kernel.record_tool_observation("pytest", {"run": 1}, long_output, exit_code=1)

        for step in range(2, 6):
            kernel.record_tool_observation("pytest", {"run": step}, f"Short output {step}")

        msgs, _, _ = kernel.build_context(mask_after_turns=2, mask_min_tokens=50)
        assert msgs[0]["masked"] is True
        assert f"/.proc/obs/{rec1.syscall_id}.txt" in msgs[0]["content"]

        # Recover exact lines 10..12 from masked virtual inode
        sliced = kernel.read_obs(rec1.syscall_id, start_line=10, end_line=12)
        assert "Line 10:" in sliced and "Line 12:" in sliced

        # PASTE Markov transition learning: reading a.py -> b.py twice warms b.py on third read of a.py
        kernel.read_file_paste("a.py")
        kernel.read_file_paste("b.py")
        kernel.paste.prefetch_cache.clear()
        kernel.read_file_paste("a.py")
        _, was_hit = kernel.read_file_paste("b.py")
        assert was_hit is True


def test_epca_invariants_and_capability_guards(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests/test_core.py").write_text("def test_ok(): assert True\n")
    (ws / "app.py").write_text("x = 1\n")
    db_path = tmp_path / "sec.agdb"

    with AgFSKernel(db_path=db_path, workspace_root=ws, protected_globs=["tests/**"]) as kernel:
        # 1. Protected glob blocks test tampering
        with pytest.raises((InvariantViolationTrap, CapabilityDeniedTrap)):
            kernel.commit(
                "Tamper with test",
                virtual_overlays={"tests/test_core.py": "def test_ok(): pass\n"},
            )

        # 2. Secret scanner blocks API key leak
        with pytest.raises(InvariantViolationTrap):
            kernel.commit(
                "Leak secret key",
                virtual_overlays={"app.py": "OPENAI_KEY = 'sk-1234567890abcdefghijklmnop'\n"},
            )

        # 3. Test-skip scanner blocks @pytest.mark.skip
        with pytest.raises(InvariantViolationTrap):
            kernel.commit(
                "Add skip decorator",
                virtual_overlays={"app.py": "@pytest.mark.skip\ndef test_cheat(): pass\n"},
            )

        # 4. Budget cap blocks overspending
        kernel.fork("tight-budget", max_budget_usd=0.05)
        with pytest.raises(CapabilityDeniedTrap):
            kernel.commit("Expensive step", virtual_overlays={"app.py": "x = 2\n"}, step_cost_usd=0.25)


def test_atomix_gatekeeper_and_sagallm_compensation(tmp_path: Path) -> None:
    db_path = tmp_path / "saga.agdb"
    cluster = MockExternalServiceCluster()

    with AgFSKernel(db_path=db_path) as kernel:
        kernel.saga.register_connector("ext", cluster.handle_action)
        root_cid = kernel.current_commit_id

        # Speculative branch stages effect with DEFER_UNTIL_MERGE
        kernel.fork("spec-pr", from_commit=root_cid)
        eff = kernel.saga_execute(
            target_service="ext.github",
            forward_action={"op": "create", "resource_id": "pr_101"},
            compensate_action={"op": "delete", "resource_id": "pr_101"},
            gate_policy="DEFER_UNTIL_MERGE",
        )
        assert eff.status == "STAGED"
        assert "pr_101" not in cluster.active_resources

        # Merging to main flushes the staged effect!
        kernel.merge_to_main("spec-pr")
        assert "pr_101" in cluster.active_resources
