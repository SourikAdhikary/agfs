"""Layer 4: Deterministic Mock External Service Cluster for Saga Outbox tests & Benchmark B5."""

from __future__ import annotations

from typing import Any


class MockExternalServiceCluster:
    """Simulates stateful external APIs (GitHub PRs, Stripe Charges, Prod DB Records).

    Tracks active external resources so tests and benchmarks can assert:
    - 0 orphaned external records when a multi-step workflow fails at Step K
    - 0 duplicate external mutations when steps are retried
    - 0 premature side effects from aborted speculative branches
    """

    def __init__(self) -> None:
        self.active_resources: dict[str, dict[str, Any]] = {}
        self.call_log: list[dict[str, Any]] = []
        self.duplicate_mutations: int = 0

    def handle_action(self, action: dict[str, Any]) -> dict[str, Any]:
        op = str(action.get("op", "create"))
        resource_id = str(action.get("resource_id", "unknown"))
        self.call_log.append(dict(action))

        if action.get("inject_failure"):
            raise RuntimeError(f"Simulated downstream 503 failure on {resource_id}")

        if op == "create":
            if resource_id in self.active_resources:
                self.duplicate_mutations += 1
            self.active_resources[resource_id] = dict(action.get("payload", {"ok": True}))
            return {"status": "created", "resource_id": resource_id}

        if op in ("delete", "refund", "compensate", "revert"):
            self.active_resources.pop(resource_id, None)
            return {"status": "compensated", "resource_id": resource_id}

        return {"status": "ok", "resource_id": resource_id}
