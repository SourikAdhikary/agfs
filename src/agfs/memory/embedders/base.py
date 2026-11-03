"""Layer 3: Pluggable Embedder Protocol & Registry (`@register_embedder`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np


class EmbedderProtocol(Protocol):
    """Protocol for AgFS memory embedding backends."""

    @property
    def dim(self) -> int: ...

    def encode(self, text: str) -> np.ndarray:
        """Return 1-D L2-normalized float32 vector of shape `(self.dim,)`."""
        ...


_EMBEDDER_REGISTRY: dict[str, Callable[[int], EmbedderProtocol]] = {}


def register_embedder(name: str) -> Callable[[Callable[[int], EmbedderProtocol]], Callable[[int], EmbedderProtocol]]:
    """Decorator to register an embedder factory by name."""

    def _decorator(factory: Callable[[int], EmbedderProtocol]) -> Callable[[int], EmbedderProtocol]:
        _EMBEDDER_REGISTRY[name] = factory
        return factory

    return _decorator


def get_embedder(name: str = "deterministic", dim: int = 384) -> EmbedderProtocol:
    if name not in _EMBEDDER_REGISTRY:
        raise KeyError(f"Unknown embedder '{name}'. Available: {sorted(_EMBEDDER_REGISTRY)}")
    return _EMBEDDER_REGISTRY[name](dim)
