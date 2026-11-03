"""Benchmark B5: SagaLLM (VLDB 2025) + Atomix (2026) Transactional Outbox & Failure Recovery.

Compares:
- Without AgFS (Unbuffered External API Calls + Naive Retry):
  1. Speculative branches fire live external webhooks/PRs even on branches that later get aborted.
  2. When Step 4 of a 4-step deployment fails, Steps 1-3 remain orphaned in external systems, and
     retrying duplicates Steps 1-3.
- With AgFS (`DEFER_UNTIL_MERGE` Gatekeeper + `IMMEDIATE_WITH_SAGA` Compensating Rollback):
  0 premature speculative side effects, 0 duplicate mutations on retry, 0 orphaned resources on abort.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from agfs.interfaces.kernel import AgFSKernel
from agfs.safety.connectors.mock_service import MockExternalServiceCluster


def run_b5_benchmark(num_workflows: int = 10) -> dict[str, Any]:
    # ------------------------------------------------------------------
    # 1. Without AgFS: Direct Unbuffered API Calls with Naive Retry
    # ------------------------------------------------------------------
    naive_cluster = MockExternalServiceCluster()
    naive_aborted_branch_leaks = 0

    # Scenario A: 10 speculative branches where 8 are aborted and only 2 win
    for b in range(num_workflows):
        naive_cluster.handle_action(
            {"op": "create", "resource_id": f"spec_pr_{b}", "payload": {"branch": b}}
        )
        if b >= 2:
            # Branch aborted, but external PR was already created!
            naive_aborted_branch_leaks += 1

    # Scenario B: 10 4-step external workflows where Step 4 fails, triggers 1 naive retry, then aborts
    for w in range(num_workflows):
        for _attempt in range(2):
            try:
                for step in range(1, 5):
                    naive_cluster.handle_action(
                        {
                            "op": "create",
                            "resource_id": f"wf_{w}_step_{step}",
                            "inject_failure": (step == 4),
                        }
                    )
            except RuntimeError:
                pass

    naive_orphaned_resources = sum(
        1 for k in naive_cluster.active_resources if k.startswith("wf_")
    )
    naive_duplicates = naive_cluster.duplicate_mutations

    # ------------------------------------------------------------------
    # 2. With AgFS: Atomix `DEFER_UNTIL_MERGE` + SagaLLM Reverse Compensation
    # ------------------------------------------------------------------
    agfs_cluster = MockExternalServiceCluster()
    with tempfile.TemporaryDirectory(prefix="agfs_b5_") as tmp:
        db_path = Path(tmp) / "b5.agdb"
        with AgFSKernel(db_path=db_path) as kernel:
            kernel.saga.register_connector("ext", agfs_cluster.handle_action)
            root_cid = kernel.current_commit_id

            # Scenario A: 10 speculative branches using DEFER_UNTIL_MERGE (8 aborted, 2 merged)
            for b in range(num_workflows):
                b_name = f"spec-branch-{b}"
                kernel.fork(b_name, from_commit=root_cid)
                kernel.saga_execute(
                    target_service="ext.github",
                    forward_action={"op": "create", "resource_id": f"spec_pr_{b}"},
                    compensate_action={"op": "delete", "resource_id": f"spec_pr_{b}"},
                    gate_policy="DEFER_UNTIL_MERGE",
                )
                if b < 2:
                    kernel.merge_to_main(b_name)
                else:
                    kernel.abort_branch(b_name)

            agfs_aborted_branch_leaks = sum(
                1
                for b in range(2, num_workflows)
                if f"spec_pr_{b}" in agfs_cluster.active_resources
            )

            # Scenario B: 10 4-step Saga workflows where Step 4 fails after retry -> `rollback_saga`
            for w in range(num_workflows):
                wf_branch = f"deploy-wf-{w}"
                kernel.fork(wf_branch, from_commit=kernel.current_commit_id)
                for attempt in range(2):
                    try:
                        for step in range(1, 5):
                            res_id = f"wf_{w}_step_{step}"
                            kernel.saga_execute(
                                target_service="ext.cloud",
                                forward_action={
                                    "op": "create",
                                    "resource_id": res_id,
                                    "inject_failure": (step == 4),
                                },
                                compensate_action={"op": "compensate", "resource_id": res_id},
                                gate_policy="IMMEDIATE_WITH_SAGA",
                                custom_idempotency_key=f"idem_{res_id}",
                            )
                    except RuntimeError:
                        if attempt == 1:
                            kernel.abort_branch(wf_branch)

    agfs_orphaned_resources = sum(
        1 for k in agfs_cluster.active_resources if k.startswith("wf_")
    )
    agfs_duplicates = agfs_cluster.duplicate_mutations

    return {
        "benchmark_id": "B5",
        "name": "External API Transactional Reliability (Atomix Gatekeeper + SagaLLM Rollback)",
        "without_agfs_aborted_branch_leaks": naive_aborted_branch_leaks,
        "with_agfs_aborted_branch_leaks": agfs_aborted_branch_leaks,
        "without_agfs_duplicate_mutations": naive_duplicates,
        "with_agfs_duplicate_mutations": agfs_duplicates,
        "without_agfs_orphaned_resources": naive_orphaned_resources,
        "with_agfs_orphaned_resources": agfs_orphaned_resources,
    }
