"""Integration tests for MCP Server JSON-RPC 2.0 protocol, Host Adapters, Supervisor & CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agfs.adapters.hosts import write_host_adapter_config
from agfs.interfaces.cli import app
from agfs.interfaces.kernel import AgFSKernel
from agfs.interfaces.mcp_server import AgFSMCPServer
from agfs.interfaces.supervisor import run_wrapped_command

runner = CliRunner()


def test_mcp_server_jsonrpc_lifecycle(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "service.py").write_text("VAL = 10\n")
    db_path = tmp_path / "mcp.agdb"

    with AgFSKernel(db_path=db_path, workspace_root=ws) as kernel:
        mcp = AgFSMCPServer(kernel)

        # 1. initialize
        init_resp = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init_resp is not None
        assert init_resp["result"]["serverInfo"]["name"] == "agfs-sqlite-agent-os"

        # 2. tools/list
        list_resp = mcp.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert list_resp is not None
        tool_names = {t["name"] for t in list_resp["result"]["tools"]}
        assert "agfs_fork" in tool_names and "agfs_recall" in tool_names

        # 3. tools/call -> agfs_remember & agfs_recall
        rem_resp = mcp.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "agfs_remember",
                    "arguments": {
                        "subject": "file:service.py",
                        "predicate": "invariant",
                        "content": "VAL must remain positive integer",
                        "edges": [["file:service.py", "DEFINES", "symbol:VAL"]],
                    },
                },
            }
        )
        assert rem_resp is not None and rem_resp["result"]["isError"] is False

        rec_resp = mcp.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "agfs_recall",
                    "arguments": {"query": "What invariant applies to file:service.py?"},
                },
            }
        )
        assert rec_resp is not None
        assert "VAL must remain positive" in rec_resp["result"]["content"][0]["text"]


def test_host_adapter_generation_and_supervisor_wrap(tmp_path: Path) -> None:
    for host in ("opencode", "hermes", "claude-code"):
        cfg_path = write_host_adapter_config(host, tmp_path / host)
        assert cfg_path.exists()

    # Test `agfs wrap` automatically rolling back unauthorized edits to `tests/**`
    ws = tmp_path / "wrap_ws"
    (ws / "tests").mkdir(parents=True)
    test_file = ws / "tests/test_ critical.py"
    test_file.write_text("assert 1 == 1\n")
    db_path = tmp_path / "wrap.agdb"

    code, msg, rolled_back = run_wrapped_command(
        command=["python3", "-c", f"from pathlib import Path; Path({str(test_file)!r}).write_text('TAMPERED')"],
        db_path=db_path,
        workspace_root=ws,
        protected_globs=["tests/**"],
    )
    assert rolled_back is True
    assert code == 1
    assert test_file.read_text() == "assert 1 == 1\n"


def test_cli_commands_init_status_doctor(tmp_path: Path) -> None:
    ws = tmp_path / "cli_ws"
    ws.mkdir()
    (ws / "hello.py").write_text("x = 42\n")
    db_path = tmp_path / "cli.agdb"

    r_init = runner.invoke(app, ["init", "--db", str(db_path), "--workspace", str(ws)])
    assert r_init.exit_code == 0

    r_status = runner.invoke(app, ["status", "--db", str(db_path), "--workspace", str(ws)])
    assert r_status.exit_code == 0

    r_doc = runner.invoke(app, ["doctor", "--db", str(db_path)])
    assert r_doc.exit_code == 0
    assert "12/12 kernel invariants PASSED" in r_doc.stdout
