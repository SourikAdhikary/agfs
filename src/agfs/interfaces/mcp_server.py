"""Layer 5: Drop-in JSON-RPC 2.0 stdio MCP Server for OpenCode, Hermes, Claude Code & Cursor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from agfs.core.errors import AgFSError
from agfs.interfaces.kernel import AgFSKernel

MCP_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "agfs_fork",
        "description": "Create a <1.5ms Copy-on-Write speculative branch in AgFS (.agdb) with capability guards.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "branch_name": {"type": "string"},
                "from_commit": {"type": "integer"},
                "read_only_globs": {"type": "array", "items": {"type": "string"}},
                "max_budget_usd": {"type": "number", "default": 5.0},
            },
            "required": ["branch_name"],
        },
    },
    {
        "name": "agfs_checkout",
        "description": "Atomically switch both workspace files and active Bi-Temporal memory view to a branch or commit_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": ["string", "integer"]},
            },
            "required": ["target"],
        },
    },
    {
        "name": "agfs_commit",
        "description": "Commit workspace delta chunks via FastCDC, enforcing ePCA state invariants and SentinelAgent StateCycleTrap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "stderr_text": {"type": "string", "default": ""},
                "test_pass_rate": {"type": "number"},
                "step_cost_usd": {"type": "number", "default": 0.0},
            },
            "required": ["message"],
        },
    },
    {
        "name": "agfs_select_uct",
        "description": "Select the highest-ranked MCTS branch frontier using SWE-Search UCT scoring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "exploration_c": {"type": "number", "default": 1.414},
            },
        },
    },
    {
        "name": "agfs_backprop",
        "description": "Backpropagate a reward [0.0, 1.0] up the MCTS commit tree and optionally record a global cross-branch lesson.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reward": {"type": "number"},
                "commit_id": {"type": "integer"},
                "lesson_subject": {"type": "string"},
                "lesson_predicate": {"type": "string"},
                "lesson_content": {"type": "string"},
            },
            "required": ["reward"],
        },
    },
    {
        "name": "agfs_remember",
        "description": "Store a Bi-Temporal fact (superseding conflicting prior facts on the branch) and update HippoRAG 2 entity graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "content": {"type": "string"},
                "cross_branch_global": {"type": "boolean", "default": False},
                "edges": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            "required": ["subject", "predicate", "content"],
        },
    },
    {
        "name": "agfs_recall",
        "description": "Run 3-Way Bi-Temporal Hybrid Recall (Dense Vector + FTS5 BM25 + HippoRAG 2 Personalized PageRank).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "as_of_commit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "agfs_read_obs",
        "description": "Read verbatim lines or regex matches from a masked virtual inode (/.proc/obs/<syscall_id>.txt).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syscall_id": {"type": "integer"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer", "default": 100},
                "grep_pattern": {"type": "string"},
            },
            "required": ["syscall_id"],
        },
    },
    {
        "name": "agfs_saga_exec",
        "description": "Stage or execute an idempotent external side effect with Atomix merge gating and SagaLLM rollback.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_service": {"type": "string"},
                "forward_action": {"type": "object"},
                "compensate_action": {"type": "object"},
                "gate_policy": {"type": "string", "default": "DEFER_UNTIL_MERGE"},
            },
            "required": ["target_service", "forward_action", "compensate_action"],
        },
    },
]


class AgFSMCPServer:
    """Handles MCP JSON-RPC 2.0 requests over stdio or direct dispatch."""

    def __init__(self, kernel: AgFSKernel) -> None:
        self.kernel = kernel

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        if method == "notifications/initialized":
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "agfs-sqlite-agent-os", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": MCP_TOOLS_SCHEMA},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments") or {}
            try:
                payload = self._dispatch_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                        "isError": False,
                    },
                }
            except AgFSError as trap:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(trap)}],
                        "isError": True,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(exc)},
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "agfs_fork":
            b = self.kernel.fork(
                branch_name=str(args["branch_name"]),
                from_commit=args.get("from_commit"),
                read_only_globs=args.get("read_only_globs"),
                max_budget_usd=float(args.get("max_budget_usd", 5.0)),
            )
            return {
                "branch_name": b.branch_name,
                "head_commit_id": b.head_commit_id,
                "fork_commit_id": b.fork_commit_id,
                "status": b.status,
            }

        if name == "agfs_checkout":
            node = self.kernel.checkout(args["target"])
            return {
                "active_branch": self.kernel.active_branch,
                "commit_id": node.commit_id,
                "vfs_root_hash": node.vfs_root_hash,
            }

        if name == "agfs_commit":
            node = self.kernel.commit(
                message=str(args["message"]),
                stderr_text=str(args.get("stderr_text", "")),
                test_pass_rate=args.get("test_pass_rate"),
                step_cost_usd=float(args.get("step_cost_usd", 0.0)),
            )
            return {
                "commit_id": node.commit_id,
                "branch_name": node.branch_name,
                "merkle_hash": node.merkle_hash,
                "vfs_root_hash": node.vfs_root_hash,
            }

        if name == "agfs_select_uct":
            cands = self.kernel.select_uct(exploration_c=args.get("exploration_c"))
            return {
                "candidates": [
                    {
                        "branch_name": c.branch_name,
                        "commit_id": c.commit_id,
                        "uct_score": round(c.uct_score, 4),
                        "mean_value": round(c.mean_value, 4),
                        "visits": c.mcts_visits,
                    }
                    for c in cands
                ]
            }

        if name == "agfs_backprop":
            lesson: tuple[str, str, str] | None = None
            if args.get("lesson_subject") and args.get("lesson_content"):
                lesson = (
                    str(args["lesson_subject"]),
                    str(args.get("lesson_predicate", "lesson")),
                    str(args["lesson_content"]),
                )
            updated = self.kernel.backpropagate(
                reward=float(args["reward"]),
                commit_id=args.get("commit_id"),
                cross_branch_lesson=lesson,
            )
            return {"updated_ancestor_nodes": updated, "reward": float(args["reward"])}

        if name == "agfs_remember":
            edges_raw = args.get("edges") or []
            edges = [(str(e[0]), str(e[1]), str(e[2])) for e in edges_raw if len(e) == 3]
            fact = self.kernel.remember(
                subject=str(args["subject"]),
                predicate=str(args["predicate"]),
                content=str(args["content"]),
                cross_branch_global=bool(args.get("cross_branch_global", False)),
                edges=edges,
            )
            return {
                "memory_id": fact.memory_id,
                "tx_commit_born": fact.tx_commit_born,
                "branch_scope": fact.branch_scope,
            }

        if name == "agfs_recall":
            hits = self.kernel.recall(
                query=str(args["query"]),
                top_k=int(args.get("top_k", 5)),
                as_of_commit=args.get("as_of_commit"),
            )
            return {
                "hits": [
                    {
                        "memory_id": h.memory.memory_id,
                        "subject": h.memory.subject,
                        "predicate": h.memory.predicate,
                        "content": h.memory.content,
                        "rrf_score": h.rrf_score,
                        "vec_score": h.vec_score,
                        "bm25_score": h.bm25_score,
                        "ppr_score": h.ppr_score,
                        "tx_commit_born": h.memory.tx_commit_born,
                    }
                    for h in hits
                ]
            }

        if name == "agfs_read_obs":
            text = self.kernel.read_obs(
                syscall_id=int(args["syscall_id"]),
                start_line=int(args.get("start_line", 1)),
                end_line=int(args.get("end_line", 100)),
                grep_pattern=args.get("grep_pattern"),
            )
            return {"syscall_id": int(args["syscall_id"]), "text": text}

        if name == "agfs_saga_exec":
            eff = self.kernel.saga_execute(
                target_service=str(args["target_service"]),
                forward_action=dict(args["forward_action"]),
                compensate_action=dict(args["compensate_action"]),
                gate_policy=str(args.get("gate_policy", "DEFER_UNTIL_MERGE")),
            )
            return {
                "effect_id": eff.effect_id,
                "status": eff.status,
                "idempotency_key": eff.idempotency_key,
                "response": eff.response,
            }

        raise ValueError(f"Unknown MCP tool '{name}'")

    def serve_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": str(exc)},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def run_mcp_stdio(db_path: Path, workspace_root: Path) -> None:
    with AgFSKernel(db_path=db_path, workspace_root=workspace_root) as kernel:
        server = AgFSMCPServer(kernel)
        server.serve_stdio()
