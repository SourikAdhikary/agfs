"""Layer 5: Drop-in Host Adapter Configurations for OpenCode, Nous Hermes, and Claude Code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_opencode_config(db_path: str = ".agfs/state.agdb", workspace: str = ".") -> dict[str, Any]:
    """Generate `opencode.json` MCP configuration for OpenCode."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "agfs": {
                "type": "local",
                "command": ["agfs", "mcp", "--db", db_path, "--workspace", workspace],
                "enabled": True,
            }
        },
    }


def generate_hermes_config(db_path: str = ".agfs/state.agdb", workspace: str = ".") -> dict[str, Any]:
    """Generate Nous Hermes Agent MCP tool server configuration."""
    return {
        "mcpServers": {
            "agfs": {
                "command": "agfs",
                "args": ["mcp", "--db", db_path, "--workspace", workspace],
                "description": "AgFS SQLite-Agent-OS (CoW VFS, MCTS, Bi-Temporal + HippoRAG 2 Memory, ePCA Invariants, Saga Outbox)",
            }
        }
    }


def generate_claude_code_config(db_path: str = ".agfs/state.agdb", workspace: str = ".") -> dict[str, Any]:
    """Generate `.mcp.json` configuration for Claude Code and Cursor."""
    return {
        "mcpServers": {
            "agfs": {
                "command": "agfs",
                "args": ["mcp", "--db", db_path, "--workspace", workspace],
            }
        }
    }


def write_host_adapter_config(
    host: str,
    target_dir: Path,
    db_path: str = ".agfs/state.agdb",
) -> Path:
    """Write ready-to-use configuration file for `opencode`, `hermes`, or `claude-code`."""
    target_dir.mkdir(parents=True, exist_ok=True)
    host_norm = host.lower().strip()
    if host_norm == "opencode":
        out = target_dir / "opencode.json"
        out.write_text(json.dumps(generate_opencode_config(db_path, "."), indent=2) + "\n")
        return out
    if host_norm == "hermes":
        out = target_dir / "hermes_mcp.json"
        out.write_text(json.dumps(generate_hermes_config(db_path, "."), indent=2) + "\n")
        return out
    if host_norm in ("claude", "claude-code", "cursor"):
        out = target_dir / ".mcp.json"
        out.write_text(json.dumps(generate_claude_code_config(db_path, "."), indent=2) + "\n")
        return out
    raise ValueError(f"Unsupported host '{host}'. Choose from: opencode, hermes, claude-code")
