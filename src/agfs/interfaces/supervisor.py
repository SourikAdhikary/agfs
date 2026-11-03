"""Layer 5: `agfs wrap` CLI Supervisor using macOS Tahoe kqueue & Commit Guardrails."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agfs.core.errors import AgFSError
from agfs.interfaces.kernel import AgFSKernel


def run_wrapped_command(
    *,
    command: list[str],
    db_path: Path,
    workspace_root: Path,
    protected_globs: list[str] | None = None,
    max_budget_usd: float = 5.0,
) -> tuple[int, str, bool]:
    """Execute an external CLI agent or tool command inside an AgFS transactional boundary.

    If the command violates an ePCA invariant (e.g. modifying `tests/**`, injecting
    `@pytest.mark.skip`, or leaking an API key), AgFS automatically rolls back the
    workspace files to the clean pre-command commit and returns `(exit_code, trap_msg, was_rolled_back=True)`.
    """
    with AgFSKernel(
        db_path=db_path,
        workspace_root=workspace_root,
        protected_globs=protected_globs or ["tests/**"],
    ) as kernel:
        t0 = time.perf_counter()
        proc = subprocess.run(
            command,
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        combined_out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

        try:
            node = kernel.commit(
                message=f"agfs wrap: {' '.join(command)}",
                stderr_text=proc.stderr or "",
                step_cost_usd=0.0,
            )
            kernel.record_tool_observation(
                tool_name=command[0],
                args={"cmd": command},
                output_text=combined_out,
                stderr_text=proc.stderr or "",
                exit_code=proc.returncode,
                duration_ms=dt_ms,
                mutability_class="WORKSPACE_MUTATE",
            )
            return proc.returncode, f"Committed state #{node.commit_id} ({node.merkle_hash[:12]})", False
        except AgFSError as trap:
            return 1, str(trap), True
