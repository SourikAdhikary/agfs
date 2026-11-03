"""Layer 0: Kernel configuration & default thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class KernelConfig:
    """Configuration for an AgFS (.agdb) micro-kernel instance."""

    db_path: Path
    workspace_root: Path | None = None
    default_branch: str = "main"
    # FastCDC sub-file content-defined chunking parameters
    cdc_min_size: int = 4096
    cdc_avg_size: int = 8192
    cdc_max_size: int = 16384
    cdc_single_chunk_cutoff: int = 16384
    # Virtual Inode Observation Masking (SWE-Pruner / Context Virtualisation)
    obs_mask_after_turns: int = 3
    obs_mask_min_tokens: int = 200
    # MCTS State-Tree (SWE-Search ICLR 2025)
    uct_exploration_c: float = 1.41421356237
    # Bi-Temporal 3-Way Hybrid Memory + HippoRAG 2 PPR
    embedding_dim: int = 384
    ppr_damping_alpha: float = 0.15
    ppr_iterations: int = 10
    rrf_k0: int = 60
    rrf_weight_vec: float = 1.0
    rrf_weight_bm25: float = 1.0
    rrf_weight_ppr: float = 1.2
    # Default ignore patterns when scanning workspace directories
    ignore_names: set[str] = field(
        default_factory=lambda: {
            ".git",
            ".agfs",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".venv",
            "venv",
            "node_modules",
            ".DS_Store",
        }
    )
