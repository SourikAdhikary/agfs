"""Layer 4: ePCA (2025-2026) Executable State Invariant Verifier."""

from __future__ import annotations

import json
from typing import Any

from agfs.core.errors import InvariantViolationTrap
from agfs.safety.rules.builtin_rules import (
    DEFAULT_SECRET_REGEX,
    DEFAULT_TEST_SKIP_REGEX,
    match_protected_glob,
    scan_forbidden_regex,
)
from agfs.storage.db import SQLiteWALEngine


class InvariantVerifier:
    """Evaluates registered `state_invariants` inside the SQLite commit hook before `COMMIT`."""

    def __init__(self, db: SQLiteWALEngine) -> None:
        self.db = db

    def register_invariant(
        self,
        *,
        invariant_id: str,
        description: str,
        invariant_kind: str,
        spec: dict[str, Any],
        enabled: bool = True,
    ) -> None:
        self.db.conn.execute(
            """
            INSERT INTO state_invariants(invariant_id, description, invariant_kind, spec_json, enabled)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(invariant_id) DO UPDATE SET
                description = excluded.description,
                invariant_kind = excluded.invariant_kind,
                spec_json = excluded.spec_json,
                enabled = excluded.enabled
            """,
            (invariant_id, description, invariant_kind, json.dumps(spec, sort_keys=True), int(enabled)),
        )

    def register_default_guardrails(
        self,
        *,
        protected_globs: list[str] | None = None,
        block_secrets: bool = True,
        block_test_skip: bool = True,
    ) -> None:
        """Install standard ePCA guardrails (protected test globs, secret scanner, no test skips)."""
        if protected_globs:
            self.register_invariant(
                invariant_id="epca_protected_globs",
                description="Prevent agent from modifying or deleting protected paths (e.g. tests/**)",
                invariant_kind="PROTECTED_GLOB",
                spec={"globs": protected_globs},
            )
        if block_secrets:
            self.register_invariant(
                invariant_id="epca_no_secret_leak",
                description="Reject commits containing high-entropy API keys or private keys",
                invariant_kind="REGEX_FORBID_CONTENT",
                spec={"pattern": DEFAULT_SECRET_REGEX},
            )
        if block_test_skip:
            self.register_invariant(
                invariant_id="epca_no_test_skip",
                description="Reject reward-hacking attempts that add @pytest.mark.skip or @unittest.skip",
                invariant_kind="REGEX_FORBID_CONTENT",
                spec={"pattern": DEFAULT_TEST_SKIP_REGEX},
            )

    def verify_before_commit(
        self,
        *,
        branch_name: str,
        changed_paths: list[str],
        changed_file_contents: dict[str, str],
        step_cost_usd: float = 0.0,
    ) -> None:
        """Evaluate all enabled invariants. Raises `InvariantViolationTrap` on any violation."""
        rows = self.db.conn.execute(
            "SELECT invariant_id, invariant_kind, spec_json FROM state_invariants WHERE enabled = 1"
        ).fetchall()

        for r in rows:
            inv_id = str(r["invariant_id"])
            kind = str(r["invariant_kind"])
            spec = json.loads(r["spec_json"])

            if kind == "PROTECTED_GLOB":
                globs: list[str] = spec.get("globs", [])
                for path in changed_paths:
                    matched_glob = match_protected_glob(path, globs)
                    if matched_glob is not None:
                        raise InvariantViolationTrap(
                            invariant_id=inv_id,
                            kind=kind,
                            details=f"Mutation of '{path}' violates protected glob '{matched_glob}'",
                        )

            elif kind == "REGEX_FORBID_CONTENT":
                pattern: str = spec.get("pattern", "")
                path_filter: str | None = spec.get("path_glob")
                if pattern:
                    for path, text_content in changed_file_contents.items():
                        if path_filter and not match_protected_glob(path, [path_filter]):
                            continue
                        hit = scan_forbidden_regex(text_content, pattern)
                        if hit is not None:
                            raise InvariantViolationTrap(
                                invariant_id=inv_id,
                                kind=kind,
                                details=f"Forbidden pattern '{hit[:32]}' detected in '{path}'",
                            )

            elif kind == "SQL_ASSERT":
                sql_query: str = spec.get("sql", "SELECT 1")
                res = self.db.conn.execute(sql_query).fetchone()
                if res is None or not bool(res[0]):
                    raise InvariantViolationTrap(
                        invariant_id=inv_id,
                        kind=kind,
                        details=f"SQL assertion failed: {sql_query}",
                    )

            elif kind == "MAX_BUDGET":
                cap = float(spec.get("max_usd", 5.0))
                b_row = self.db.conn.execute(
                    "SELECT spent_budget_usd FROM branches WHERE branch_name = ?",
                    (branch_name,),
                ).fetchone()
                spent = float(b_row["spent_budget_usd"]) if b_row else 0.0
                if spent + step_cost_usd > cap + 1e-9:
                    raise InvariantViolationTrap(
                        invariant_id=inv_id,
                        kind=kind,
                        details=f"Budget ${spent + step_cost_usd:.4f} exceeds cap ${cap:.4f}",
                    )
