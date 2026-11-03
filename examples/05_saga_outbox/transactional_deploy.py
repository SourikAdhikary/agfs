"""Example 05: Atomix Deferred Merge Gatekeeper + SagaLLM Compensating Rollback."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agfs import AgFSKernel
from agfs.safety.connectors.mock_service import MockExternalServiceCluster


def main() -> None:
    cluster = MockExternalServiceCluster()
    with tempfile.TemporaryDirectory(prefix="agfs_ex05_") as tmp:
        with AgFSKernel(db_path=Path(tmp) / "saga.agdb") as kernel:
            kernel.saga.register_connector("github", cluster.handle_action)
            kernel.fork("feature-branch")
            eff = kernel.saga_execute(
                target_service="github.pr",
                forward_action={"op": "create", "resource_id": "PR-42"},
                compensate_action={"op": "delete", "resource_id": "PR-42"},
                gate_policy="DEFER_UNTIL_MERGE",
            )
            print("On speculative branch -> status:", eff.status, "external active:", len(cluster.active_resources))
            kernel.merge_to_main("feature-branch")
            print("After merge_to_main -> external active:", list(cluster.active_resources.keys()))


if __name__ == "__main__":
    main()
