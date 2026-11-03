"""Enforces the strict 6-layer acyclic dependency hierarchy inside src/agfs/.

Layer 0: core        (cannot import storage, engine, memory, safety, adapters, interfaces)
Layer 1: storage     (can only import core)
Layer 2: engine      (can only import core, storage, memory, safety for doctor/checks as allowed)
Layer 3: memory      (can only import core, storage)
Layer 4: safety      (can only import core, storage)
Layer 5: adapters & interfaces (can import any lower layer)
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "agfs"

FORBIDDEN_IMPORTS: dict[str, set[str]] = {
    "core": {"storage", "engine", "memory", "safety", "adapters", "interfaces"},
    "storage": {"engine", "memory", "safety", "adapters", "interfaces"},
    "memory": {"engine", "safety", "adapters", "interfaces"},
    "safety": {"engine", "memory", "adapters", "interfaces"},
    "engine": {"adapters", "interfaces"},
    "adapters": {"interfaces"},
}


def _extract_agfs_submodules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "agfs":
                    imported.add(parts[1])
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if len(parts) >= 2 and parts[0] == "agfs":
                imported.add(parts[1])
    return imported


def test_six_layer_dependency_hierarchy() -> None:
    violations: list[str] = []
    for layer, forbidden in FORBIDDEN_IMPORTS.items():
        layer_dir = SRC_ROOT / layer
        if not layer_dir.exists():
            continue
        for py_file in layer_dir.rglob("*.py"):
            imported = _extract_agfs_submodules(py_file)
            bad = imported & forbidden
            if bad:
                rel = py_file.relative_to(SRC_ROOT.parent)
                violations.append(f"{rel} (in layer '{layer}') imports forbidden higher layer(s): {sorted(bad)}")
    assert not violations, "\n".join(violations)
