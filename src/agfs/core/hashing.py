"""Layer 0: Deterministic canonical JSON, SHA-256 Merkle hashing, and stderr normalization."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,16}")
_TMP_PATH_RE = re.compile(r"(/var/folders/[^\s:]+|/tmp/[^\s:]+|/private/var/folders/[^\s:]+)")
_FLOAT_TIME_RE = re.compile(r"\b\d+\.\d{2,}s\b")
_PID_RE = re.compile(r"\bpid=\d+\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def canonical_json(obj: Any) -> str:
    """Serialize any JSON-compatible structure into deterministic canonical form."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    """Return 64-character lowercase hex SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return 64-character lowercase hex SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_vfs_root_hash(active_inodes: list[tuple[str, str, int]]) -> str:
    """Compute deterministic Merkle root hash over sorted (path, file_hash, mode) tuples."""
    sorted_items = sorted(active_inodes, key=lambda item: item[0])
    payload = "\n".join(f"{path}:{file_hash}:{mode}" for path, file_hash, mode in sorted_items)
    return sha256_text(f"agfs:vfs:v1\n{payload}")


def compute_commit_merkle_hash(
    *,
    parent_merkle: str,
    branch_name: str,
    step_index: int,
    vfs_root_hash: str,
    last_error_hash: str,
    message: str,
) -> str:
    """Compute cryptographic SHA-256 Merkle chain hash for a commit."""
    payload = canonical_json(
        {
            "parent": parent_merkle,
            "branch": branch_name,
            "step": step_index,
            "vfs_root": vfs_root_hash,
            "error_hash": last_error_hash,
            "message": message,
        }
    )
    return sha256_text(f"agfs:commit:v1:{payload}")


def normalize_stderr(stderr_text: str) -> str:
    """Strip ephemeral memory addresses, temp paths, and run durations from stderr."""
    if not stderr_text or not stderr_text.strip():
        return ""
    cleaned = _HEX_ADDR_RE.sub("0xADDR", stderr_text)
    cleaned = _TMP_PATH_RE.sub("/TMP_PATH", cleaned)
    cleaned = _FLOAT_TIME_RE.sub("0.00s", cleaned)
    cleaned = _PID_RE.sub("pid=0", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def hash_stderr(stderr_text: str) -> str:
    """Compute normalized error signature hash for SentinelAgent StateCycleTrap."""
    norm = normalize_stderr(stderr_text)
    if not norm:
        return ""
    return sha256_text(f"agfs:err:v1:{norm}")


def estimate_tokens(text: str) -> int:
    """Fast deterministic token estimator (~4 chars per token heuristic + word boundary check)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
