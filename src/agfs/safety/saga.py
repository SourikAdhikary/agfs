"""Layer 4: SagaLLM (VLDB 2025) + Atomix (2026) Transactional Outbox & Compensating Rollback."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from agfs.core.hashing import canonical_json, sha256_text
from agfs.core.types import SagaEffect
from agfs.storage.db import SQLiteWALEngine


class SagaOutboxManager:
    """Manages external side-effect staging (`DEFER_UNTIL_MERGE`) and compensating Sagas."""

    def __init__(self, db: SQLiteWALEngine) -> None:
        self.db = db
        self._executors: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def register_connector(
        self,
        service_prefix: str,
        executor: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register an external service handler (e.g. 'github', 'stripe', 'mock')."""
        self._executors[service_prefix] = executor

    def stage_or_execute(
        self,
        *,
        commit_id: int,
        branch_name: str,
        target_service: str,
        forward_action: dict[str, Any],
        compensate_action: dict[str, Any],
        gate_policy: str = "DEFER_UNTIL_MERGE",
        custom_idempotency_key: str | None = None,
    ) -> SagaEffect:
        """Record an external side effect in `saga_outbox` with strict idempotency.

        - If an identical `idempotency_key` already exists, returns the existing effect (0 duplicate calls).
        - If `gate_policy == 'DEFER_UNTIL_MERGE'` and `branch_name != 'main'`, holds the effect
          in `STAGED` status until the branch wins and merges to `main`.
        - If `gate_policy == 'IMMEDIATE_WITH_SAGA'` (or already on `main`), executes immediately
          and stores the reverse `compensate_action` for automatic Saga rollback on failure.
        """
        idem_key = custom_idempotency_key or sha256_text(
            f"agfs:saga:{target_service}:{canonical_json(forward_action)}"
        )

        # Idempotency check: never duplicate an already staged or executed mutation
        existing = self.db.conn.execute(
            "SELECT * FROM saga_outbox WHERE idempotency_key = ?",
            (idem_key,),
        ).fetchone()
        if existing is not None:
            return self._row_to_effect(existing)

        now_ns = time.time_ns()
        should_defer = gate_policy == "DEFER_UNTIL_MERGE" and branch_name != "main"

        status = "STAGED"
        response: dict[str, Any] | None = {
            "staged": True,
            "reason": f"Deferred on branch '{branch_name}' until merge to main",
        }
        exec_ns: int | None = None

        if not should_defer:
            response = self._dispatch(target_service, forward_action)
            status = "EXECUTED"
            exec_ns = time.time_ns()

        cur = self.db.conn.execute(
            """
            INSERT INTO saga_outbox(
                commit_id, branch_name, idempotency_key, target_service,
                gate_policy, forward_action_json, compensate_action_json,
                status, response_json, created_at_ns, executed_at_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                commit_id,
                branch_name,
                idem_key,
                target_service,
                gate_policy,
                canonical_json(forward_action),
                canonical_json(compensate_action),
                status,
                canonical_json(response) if response is not None else None,
                now_ns,
                exec_ns,
            ),
        )
        effect_id = int(cur.lastrowid or 0)
        return SagaEffect(
            effect_id=effect_id,
            commit_id=commit_id,
            branch_name=branch_name,
            idempotency_key=idem_key,
            target_service=target_service,
            gate_policy=gate_policy,
            forward_action=forward_action,
            compensate_action=compensate_action,
            status=status,
            response=response,
            created_at_ns=now_ns,
            executed_at_ns=exec_ns,
        )

    def flush_staged_on_merge(self, branch_name: str) -> list[SagaEffect]:
        """Execute all `STAGED` effects from `branch_name` in commit order when merging to `main`."""
        rows = self.db.conn.execute(
            """
            SELECT * FROM saga_outbox
            WHERE branch_name = ? AND status = 'STAGED'
            ORDER BY effect_id ASC
            """,
            (branch_name,),
        ).fetchall()

        flushed: list[SagaEffect] = []
        for r in rows:
            eid = int(r["effect_id"])
            svc = str(r["target_service"])
            fwd = json.loads(r["forward_action_json"])
            resp = self._dispatch(svc, fwd)
            now_ns = time.time_ns()
            self.db.conn.execute(
                """
                UPDATE saga_outbox
                SET status = 'EXECUTED', response_json = ?, executed_at_ns = ?
                WHERE effect_id = ?
                """,
                (canonical_json(resp), now_ns, eid),
            )
            updated = self.db.conn.execute(
                "SELECT * FROM saga_outbox WHERE effect_id = ?", (eid,)
            ).fetchone()
            flushed.append(self._row_to_effect(updated))
        return flushed

    def rollback_saga(self, branch_name: str, after_commit_id: int = 0) -> list[SagaEffect]:
        """Run reverse compensating transactions for `EXECUTED` effects and drop `STAGED` effects."""
        # 1. Drop any unexecuted STAGED effects on this branch after `after_commit_id`
        self.db.conn.execute(
            """
            UPDATE saga_outbox
            SET status = 'COMPENSATED'
            WHERE branch_name = ? AND commit_id > ? AND status = 'STAGED'
            """,
            (branch_name, after_commit_id),
        )

        # 2. Compensate EXECUTED effects in reverse order (LIFO)
        rows = self.db.conn.execute(
            """
            SELECT * FROM saga_outbox
            WHERE branch_name = ? AND commit_id > ? AND status = 'EXECUTED'
            ORDER BY effect_id DESC
            """,
            (branch_name, after_commit_id),
        ).fetchall()

        compensated: list[SagaEffect] = []
        for r in rows:
            eid = int(r["effect_id"])
            svc = str(r["target_service"])
            comp_action = json.loads(r["compensate_action_json"])
            resp = self._dispatch(svc, comp_action)
            self.db.conn.execute(
                """
                UPDATE saga_outbox
                SET status = 'COMPENSATED', response_json = ?
                WHERE effect_id = ?
                """,
                (canonical_json(resp), eid),
            )
            updated = self.db.conn.execute(
                "SELECT * FROM saga_outbox WHERE effect_id = ?", (eid,)
            ).fetchone()
            compensated.append(self._row_to_effect(updated))
        return compensated

    def _dispatch(self, target_service: str, action: dict[str, Any]) -> dict[str, Any]:
        prefix = target_service.split(".")[0]
        if target_service in self._executors:
            return self._executors[target_service](action)
        if prefix in self._executors:
            return self._executors[prefix](action)
        return {"executed": True, "service": target_service, "action": action}

    @staticmethod
    def _row_to_effect(row: object) -> SagaEffect:
        r = row  # type: ignore[assignment]
        return SagaEffect(
            effect_id=int(r["effect_id"]),
            commit_id=int(r["commit_id"]),
            branch_name=str(r["branch_name"]),
            idempotency_key=str(r["idempotency_key"]),
            target_service=str(r["target_service"]),
            gate_policy=str(r["gate_policy"]),
            forward_action=json.loads(r["forward_action_json"]),
            compensate_action=json.loads(r["compensate_action_json"]),
            status=str(r["status"]),
            response=json.loads(r["response_json"]) if r["response_json"] is not None else None,
            created_at_ns=int(r["created_at_ns"]),
            executed_at_ns=int(r["executed_at_ns"]) if r["executed_at_ns"] is not None else None,
        )
