"""Public Python SDK surface for AgFS (`SQLite-Agent-OS`)."""

from __future__ import annotations

from agfs.core.config import KernelConfig
from agfs.core.errors import (
    AgFSError,
    CapabilityDeniedTrap,
    CorruptChunkError,
    InvariantViolationTrap,
    MerkleTamperError,
    StateCycleTrap,
)
from agfs.core.types import (
    BranchInfo,
    CommitNode,
    DoctorCheckResult,
    DoctorReport,
    InodeEntry,
    MemoryFact,
    RecallHit,
    SagaEffect,
    SyscallRecord,
    UCTCandidate,
)
from agfs.interfaces.kernel import AgFSKernel

__all__ = [
    "AgFSKernel",
    "KernelConfig",
    "AgFSError",
    "StateCycleTrap",
    "InvariantViolationTrap",
    "CapabilityDeniedTrap",
    "CorruptChunkError",
    "MerkleTamperError",
    "BranchInfo",
    "CommitNode",
    "UCTCandidate",
    "InodeEntry",
    "MemoryFact",
    "RecallHit",
    "SyscallRecord",
    "SagaEffect",
    "DoctorCheckResult",
    "DoctorReport",
]
