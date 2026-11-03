"""Layer 0: Typed exception and kernel trap hierarchy."""

from __future__ import annotations


class AgFSError(Exception):
    """Base exception for all AgFS kernel errors and traps."""


class StateCycleTrap(AgFSError):
    """Raised when SentinelAgent Merkle state-cycle detection catches a repeated broken state."""

    def __init__(
        self,
        *,
        branch_name: str,
        prior_commit_id: int,
        prior_step_index: int,
        vfs_root_hash: str,
        last_error_hash: str,
        message: str | None = None,
    ) -> None:
        self.branch_name = branch_name
        self.prior_commit_id = prior_commit_id
        self.prior_step_index = prior_step_index
        self.vfs_root_hash = vfs_root_hash
        self.last_error_hash = last_error_hash
        default_msg = (
            f"KERNEL TRAP [StateCycleTrap]: Branch '{branch_name}' returned to exact state "
            f"(vfs_root={vfs_root_hash[:12]}, err={last_error_hash[:12]}) previously visited at "
            f"Commit #{prior_commit_id} (step {prior_step_index}). Switch branches or change strategy."
        )
        super().__init__(message or default_msg)


class InvariantViolationTrap(AgFSError):
    """Raised when an ePCA commit-hook state invariant is violated."""

    def __init__(self, *, invariant_id: str, kind: str, details: str) -> None:
        self.invariant_id = invariant_id
        self.kind = kind
        self.details = details
        super().__init__(
            f"KERNEL TRAP [InvariantViolationTrap:{invariant_id} ({kind})]: {details}"
        )


class CapabilityDeniedTrap(AgFSError):
    """Raised when an AIOS branch capability mask or budget cap blocks a syscall."""

    def __init__(self, *, branch_name: str, capability: str, details: str) -> None:
        self.branch_name = branch_name
        self.capability = capability
        self.details = details
        super().__init__(
            f"KERNEL TRAP [CapabilityDeniedTrap on '{branch_name}' ({capability})]: {details}"
        )


class CorruptChunkError(AgFSError):
    """Raised when a content-addressed chunk fails zstd decompression or SHA-256 verification."""


class MerkleTamperError(AgFSError):
    """Raised when a commit's cryptographic Merkle chain fails verification."""
