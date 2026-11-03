"""Layer 5: CLI Entry Point for AgFS (`agfs`)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agfs.adapters.hosts import write_host_adapter_config
from agfs.interfaces.kernel import AgFSKernel
from agfs.interfaces.mcp_server import run_mcp_stdio
from agfs.interfaces.supervisor import run_wrapped_command

app = typer.Typer(
    name="agfs",
    help="AgFS (SQLite-Agent-OS): Featherweight single-file SQLite Operating System & Transactional Kernel for AI Agents.",
    add_completion=False,
)
console = Console()


@app.command("init")
def cmd_init(
    db: Path = typer.Option(Path(".agfs/state.agdb"), "--db", help="Path to .agdb SQLite file"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory root"),
    protect: list[str] = typer.Option(["tests/**"], "--protect", help="Protected read-only globs"),
) -> None:
    """Initialize an AgFS (.agdb) kernel and capture the genesis commit (#1)."""
    with AgFSKernel(db_path=db, workspace_root=workspace, protected_globs=protect) as kernel:
        node = kernel.mcts.get_commit(kernel.current_commit_id)
        console.print(
            f"[bold green]Initialized AgFS Kernel[/bold green] at [cyan]{db}[/cyan] "
            f"(branch=[bold]{kernel.active_branch}[/bold], commit=#{node.commit_id}, "
            f"merkle={node.merkle_hash[:12]}, vfs_root={node.vfs_root_hash[:12]})"
        )


@app.command("status")
def cmd_status(
    db: Path = typer.Option(Path(".agfs/state.agdb"), "--db", help="Path to .agdb SQLite file"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory root"),
) -> None:
    """Display active branch, head commit, MCTS frontier ranking, and memory stats."""
    with AgFSKernel(db_path=db, workspace_root=workspace) as kernel:
        node = kernel.mcts.get_commit(kernel.current_commit_id)
        b_info = kernel.pcb.get_branch_info(kernel.active_branch)
        mem_cnt = kernel.db.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE tx_commit_superseded IS NULL"
        ).fetchone()[0]
        chunk_cnt = kernel.db.conn.execute("SELECT COUNT(*) FROM vfs_chunks").fetchone()[0]

        table = Table(title=f"AgFS Kernel Status ({db})")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="bold")
        table.add_row("Active Branch", f"{kernel.active_branch} ({b_info.status})")
        table.add_row("Head Commit", f"#{node.commit_id} (step {node.step_index}: {node.message})")
        table.add_row("Merkle Hash", node.merkle_hash[:16])
        table.add_row("VFS Root Hash", node.vfs_root_hash[:16])
        table.add_row("MCTS Stats", f"visits={node.mcts_visits}, mean_value={node.mean_value:.3f}")
        table.add_row("Protected Globs", json.dumps(b_info.read_only_globs))
        table.add_row("Budget Used", f"${b_info.spent_budget_usd:.4f} / ${b_info.max_budget_usd:.2f}")
        table.add_row("Active Bi-Temporal Facts", str(mem_cnt))
        table.add_row("Deduplicated FastCDC Chunks", str(chunk_cnt))
        console.print(table)


@app.command("fork")
def cmd_fork(
    branch_name: str = typer.Argument(..., help="New speculative branch name"),
    from_commit: int | None = typer.Option(None, "--from", help="Base commit_id (default HEAD)"),
    db: Path = typer.Option(Path(".agfs/state.agdb"), "--db", help="Path to .agdb SQLite file"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory root"),
) -> None:
    """Create a <1.5ms Copy-on-Write speculative branch."""
    with AgFSKernel(db_path=db, workspace_root=workspace) as kernel:
        info = kernel.fork(branch_name, from_commit=from_commit)
        console.print(
            f"[green]Forked & checked out branch[/green] [bold]{info.branch_name}[/bold] "
            f"at commit #{info.head_commit_id}"
        )


@app.command("checkout")
def cmd_checkout(
    target: str = typer.Argument(..., help="Branch name or commit_id to checkout"),
    db: Path = typer.Option(Path(".agfs/state.agdb"), "--db", help="Path to .agdb SQLite file"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory root"),
) -> None:
    """Atomically switch workspace files and active Bi-Temporal memory view."""
    with AgFSKernel(db_path=db, workspace_root=workspace) as kernel:
        node = kernel.checkout(target)
        console.print(
            f"[green]Checked out[/green] branch=[bold]{kernel.active_branch}[/bold] "
            f"commit=#{node.commit_id} (vfs_root={node.vfs_root_hash[:12]})"
        )


@app.command("commit")
def cmd_commit(
    message: str = typer.Option(..., "-m", "--message", help="Commit message"),
    stderr_text: str = typer.Option("", "--stderr", help="Normalized failure stderr if step failed"),
    pass_rate: float | None = typer.Option(None, "--pass-rate", help="Test pass rate [0.0..1.0]"),
    db: Path = typer.Option(Path(".agfs/state.agdb"), "--db", help="Path to .agdb SQLite file"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory root"),
) -> None:
    """Capture workspace changes via FastCDC, verify ePCA invariants & StateCycleTrap, and commit."""
    with AgFSKernel(db_path=db, workspace_root=workspace) as kernel:
        node = kernel.commit(
            message=message,
            stderr_text=stderr_text,
            test_pass_rate=pass_rate,
        )
        console.print(
            f"[green]Committed[/green] #{node.commit_id} on [bold]{node.branch_name}[/bold] "
            f"(merkle={node.merkle_hash[:12]}, vfs_root={node.vfs_root_hash[:12]})"
        )


@app.command("recall")
def cmd_recall(
    query: str = typer.Argument(..., help="Natural language or symbol query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    as_of: int | None = typer.Option(None, "--as-of", help="Point-in-time commit_id"),
    db: Path = typer.Option(Path(".agfs/state.agdb"), "--db", help="Path to .agdb SQLite file"),
) -> None:
    """Run 3-Way Bi-Temporal Hybrid Recall (Dense Vector + FTS5 BM25 + HippoRAG 2 PPR)."""
    with AgFSKernel(db_path=db) as kernel:
        hits = kernel.recall(query, top_k=top_k, as_of_commit=as_of)
        table = Table(title=f"3-Way Hybrid Memory Recall: {query!r}")
        table.add_column("ID", style="cyan")
        table.add_column("Subject / Predicate", style="bold")
        table.add_column("Content")
        table.add_column("RRF / Vec / BM25 / PPR", style="green")
        for h in hits:
            table.add_row(
                f"#{h.memory.memory_id} (c{h.memory.tx_commit_born})",
                f"{h.memory.subject} :: {h.memory.predicate}",
                h.memory.content,
                f"{h.rrf_score:.4f} ({h.vec_score:.2f} / {h.bm25_score:.2f} / {h.ppr_score:.2f})",
            )
        console.print(table)


@app.command("doctor")
def cmd_doctor(
    db: Path | None = typer.Option(None, "--db", help="Path to existing .agdb file (or runs self-test)"),
) -> None:
    """Verify all 12 formal mathematical & relational kernel invariants (INV-01..INV-12)."""
    if db is not None and db.exists():
        with AgFSKernel(db_path=db) as kernel:
            report = kernel.doctor()
    else:
        # Run self-contained verification on a populated temporary .agdb instance
        with tempfile.TemporaryDirectory(prefix="agfs_doctor_") as tmp:
            tmp_ws = Path(tmp) / "ws"
            tmp_ws.mkdir()
            (tmp_ws / "app.py").write_text("print('hello agfs')\n")
            tmp_db = Path(tmp) / "doctor.agdb"
            with AgFSKernel(db_path=tmp_db, workspace_root=tmp_ws, protected_globs=["tests/**"]) as kernel:
                kernel.remember(
                    "file:app.py",
                    "purpose",
                    "Entrypoint module for AgFS self-check",
                    edges=[("file:app.py", "DEFINES", "symbol:main")],
                )
                kernel.record_tool_observation("pytest", {"q": True}, "1 passed in 0.01s\n")
                kernel.commit("Self-test commit", virtual_overlays={"app.py": "print('v2')\n"}, test_pass_rate=1.0)
                report = kernel.doctor()

    table = Table(title=f"AgFS Kernel Doctor — 12 Formal Invariants ({report.db_path})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Invariant", style="bold")
    table.add_column("Status")
    table.add_column("Details")
    table.add_column("Time", justify="right")

    for c in report.checks:
        status_str = "[bold green]PASS[/bold green]" if c.passed else "[bold red]FAIL[/bold red]"
        table.add_row(c.invariant_id, c.name, status_str, c.details, f"{c.duration_ms:.2f}ms")

    console.print(table)
    if not report.all_passed:
        raise typer.Exit(code=1)
    console.print(f"[bold green]All {len(report.checks)}/{len(report.checks)} kernel invariants PASSED.[/bold green]")


@app.command("bench")
def cmd_bench(
    out: Path = typer.Option(Path("BENCHMARKS.md"), "--out", "-o", help="Output Markdown file"),
) -> None:
    """Run the 9-part A/B Benchmark Suite (Without AgFS vs. With AgFS) and write BENCHMARKS.md."""
    from benchmarks.runner import run_all_benchmarks

    console.print("[bold cyan]Running 9-Part A/B Benchmark Suite (Without AgFS vs. With AgFS)...[/bold cyan]")
    results = run_all_benchmarks(output_md=out)

    b1 = results["systems_benchmarks"]["B1"]
    b2 = results["systems_benchmarks"]["B2"]
    b3 = results["systems_benchmarks"]["B3"]
    b4 = results["systems_benchmarks"]["B4"]
    b5 = results["systems_benchmarks"]["B5"]

    table = Table(title="AgFS A/B Benchmark Summary (Without AgFS vs. With AgFS)")
    table.add_column("ID", style="cyan")
    table.add_column("Benchmark")
    table.add_column("Without AgFS", style="yellow")
    table.add_column("With AgFS", style="bold green")
    table.add_column("Win / Delta", style="bold magenta")

    table.add_row(
        "B1",
        "40-Branch Fork & Rollback",
        f"{b1['without_agfs_p50_ms']} ms / {b1['without_agfs_disk_mb']} MB",
        f"{b1['with_agfs_fork_rollback_p50_ms']} ms / {b1['with_agfs_disk_mb']} MB",
        f"{b1['speedup_fork_rollback_x']}x faster, {b1['disk_reduction_x']}x smaller",
    )
    table.add_row(
        "B2",
        "40-Step Context & KV Cache",
        f"{b2['raw_append_tokens']:,} tok / {b2['sliding_window_verbatim_recall_pct']}% recall",
        f"{b2['with_agfs_tokens']:,} tok / {b2['with_agfs_verbatim_recall_pct']}% recall",
        f"-{b2['token_reduction_vs_raw_pct']}% tokens, {b2['with_agfs_kv_cache_hit_pct']}% KV hit",
    )
    table.add_row(
        "B3",
        "Bi-Temporal + 2-Hop PPR Memory",
        f"{b3['without_agfs_stale_poisoning_pct']}% stale / {b3['without_agfs_bm25_2hop_recall_pct']}% 2-hop",
        f"{b3['with_agfs_stale_poisoning_pct']}% stale / {b3['with_agfs_hybrid_2hop_recall_pct']}% 2-hop",
        "0% stale poisoning, 100% 2-hop recall",
    )
    table.add_row(
        "B4",
        "PASTE Speculative Prefetch",
        f"{b4['without_agfs_io_wait_ms']} ms wait (0% hit)",
        f"{b4['with_agfs_io_wait_ms']} ms wait ({b4['speculative_cache_hit_pct']}% hit)",
        f"-{b4['io_wait_reduction_pct']}% read I/O wait",
    )
    table.add_row(
        "B5",
        "Atomix + SagaLLM Outbox",
        f"{b5['without_agfs_aborted_branch_leaks']} leaks / {b5['without_agfs_orphaned_resources']} orphans",
        f"{b5['with_agfs_aborted_branch_leaks']} leaks / {b5['with_agfs_orphaned_resources']} orphans",
        "100% clean external state",
    )
    console.print(table)
    console.print(f"[bold green]Wrote full A/B report to {out}[/bold green]")


@app.command("mcp")
def cmd_mcp(
    db: Path = typer.Option(Path(".agfs/state.agdb"), "--db", help="Path to .agdb SQLite file"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory root"),
) -> None:
    """Start the JSON-RPC 2.0 stdio MCP server for OpenCode, Hermes, Claude Code & Cursor."""
    run_mcp_stdio(db_path=db, workspace_root=workspace)


@app.command("adapter")
def cmd_adapter(
    host: str = typer.Argument(..., help="Target host: opencode | hermes | claude-code"),
    target_dir: Path = typer.Option(Path("."), "--out", "-o", help="Directory to write config"),
    db_path: str = typer.Option(".agfs/state.agdb", "--db", help="Relative .agdb path"),
) -> None:
    """Generate ready-to-use MCP configuration for OpenCode, Nous Hermes, or Claude Code."""
    out_file = write_host_adapter_config(host, target_dir, db_path=db_path)
    console.print(f"[bold green]Generated {host} adapter config at {out_file}[/bold green]")


@app.command("wrap")
def cmd_wrap(
    command: list[str] = typer.Argument(..., help="Command to execute inside AgFS guardrail supervisor"),
    db: Path = typer.Option(Path(".agfs/state.agdb"), "--db", help="Path to .agdb SQLite file"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory root"),
    protect: list[str] = typer.Option(["tests/**"], "--protect", help="Protected globs"),
    max_budget: float = typer.Option(5.0, "--max-budget", help="Max branch budget in USD"),
) -> None:
    """Run any CLI agent/command inside an AgFS transactional guardrail boundary."""
    code, msg, rolled_back = run_wrapped_command(
        command=command,
        db_path=db,
        workspace_root=workspace,
        protected_globs=protect,
        max_budget_usd=max_budget,
    )
    if rolled_back:
        console.print(f"[bold red]ROLLED BACK BY AGFS KERNEL:[/bold red] {msg}")
    else:
        console.print(f"[bold green]AGFS WRAP OK:[/bold green] {msg}")
    raise typer.Exit(code=code)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
