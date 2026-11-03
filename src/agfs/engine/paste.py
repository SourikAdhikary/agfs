"""Layer 2: PASTE (2026) & Speculative Actions (ICLR 2026) Read Prefetcher."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from agfs.storage.db import SQLiteWALEngine

_FILE_PATH_TOKEN_RE = re.compile(r"\b([a-zA-Z0-9_\-./]+\.(?:py|ts|tsx|js|rs|go|md|sql|json|toml|yaml))\b")


class PASTEPrefetcher:
    """Pattern-Aware Speculative Tool Execution (`PASTE`) for `PURE_READ` syscalls.

    Learns a 1st-order Markov transition table (`tool_transitions`) of read/tool access
    patterns and combines it with streaming LLM thought-token path extraction to pre-warm
    likely next `PURE_READ` calls into a zero-latency RAM cache while the LLM streams.
    """

    def __init__(self, db: SQLiteWALEngine) -> None:
        self.db = db
        self._last_signature: str | None = None
        self.prefetch_cache: dict[str, bytes | str] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.speculative_prefetches: int = 0

    def record_access(self, signature: str) -> None:
        """Record a syscall access signature (e.g. `vfs_read:src/auth.py`) and update Markov counts."""
        now_ns = time.time_ns()
        if self._last_signature is not None and self._last_signature != signature:
            self.db.conn.execute(
                """
                INSERT INTO tool_transitions(prev_signature, next_signature, hit_count, last_seen_ns)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(prev_signature, next_signature) DO UPDATE SET
                    hit_count = tool_transitions.hit_count + 1,
                    last_seen_ns = excluded.last_seen_ns
                """,
                (self._last_signature, signature, now_ns),
            )
        self._last_signature = signature

    def predict_next(
        self,
        current_signature: str | None = None,
        streaming_thought_prefix: str = "",
        top_k: int = 2,
    ) -> list[str]:
        """Predict top-K most likely next `PURE_READ` signatures."""
        predictions: list[str] = []

        # 1. Extract explicit file paths mentioned early in streaming `<thought>` tokens (Spork / Speculative Actions)
        if streaming_thought_prefix:
            for match in _FILE_PATH_TOKEN_RE.finditer(streaming_thought_prefix):
                cand = f"vfs_read:{match.group(1).lstrip('./')}"
                if cand not in predictions and cand != current_signature:
                    predictions.append(cand)
                    if len(predictions) >= top_k:
                        return predictions[:top_k]

        # 2. Query Markov transition table from `tool_transitions` (PASTE)
        anchor = current_signature or self._last_signature
        if anchor:
            rows = self.db.conn.execute(
                """
                SELECT next_signature
                FROM tool_transitions
                WHERE prev_signature = ?
                ORDER BY hit_count DESC, last_seen_ns DESC
                LIMIT ?
                """,
                (anchor, top_k),
            ).fetchall()
            for r in rows:
                sig = str(r["next_signature"])
                if sig not in predictions:
                    predictions.append(sig)
                    if len(predictions) >= top_k:
                        break

        return predictions[:top_k]

    def speculate_and_warm(
        self,
        *,
        current_signature: str | None,
        streaming_thought_prefix: str,
        loader_fn: Callable[[str], bytes | str | None],
        top_k: int = 2,
    ) -> list[str]:
        """Predict and pre-execute `PURE_READ` targets into `self.prefetch_cache`."""
        predicted = self.predict_next(
            current_signature=current_signature,
            streaming_thought_prefix=streaming_thought_prefix,
            top_k=top_k,
        )
        warmed: list[str] = []
        for sig in predicted:
            if sig not in self.prefetch_cache:
                val = loader_fn(sig)
                if val is not None:
                    self.prefetch_cache[sig] = val
                    self.speculative_prefetches += 1
                    warmed.append(sig)
            else:
                warmed.append(sig)
        return warmed

    def fetch_or_load(
        self,
        signature: str,
        loader_fn: Callable[[str], bytes | str],
    ) -> tuple[bytes | str, bool]:
        """Return `(result, was_prefetch_hit)` and record Markov transition."""
        self.record_access(signature)
        if signature in self.prefetch_cache:
            self.cache_hits += 1
            return self.prefetch_cache[signature], True
        self.cache_misses += 1
        val = loader_fn(signature)
        self.prefetch_cache[signature] = val
        return val, False

    def invalidate_path(self, rel_path: str) -> None:
        sig = f"vfs_read:{rel_path.lstrip('./')}"
        self.prefetch_cache.pop(sig, None)
