# Contributing to AgFS (`SQLite-Agent-OS`)

Thank you for contributing to **AgFS**! To keep the kernel featherweight, fast (`<2 ms` fork/rollback), and contributor-friendly, we follow four strict engineering rules:

## 1. The 15-Second Offline Test & Benchmark Rule
All unit, chaos, and A/B benchmark tests (`make test` and `make bench`) must run **100% offline in < 25 seconds** with **$0 API cost**, zero Docker containers, and zero external servers.

## 2. Strict 6-Layer Acyclic Architecture
Code inside `src/agfs/` is organized into 6 strict layers. Lower layers **must never** import from higher layers (enforced by `tests/unit/test_layer_boundaries.py`):

1. **`agfs.core` (Layer 0):** Pure types, typed exceptions (`StateCycleTrap`, `InvariantViolationTrap`), SHA-256 Merkle hashing, and macOS Darwin `libSystem.B.dylib` (`clonefile`, `kqueue`) bindings.
2. **`agfs.storage` (Layer 1):** SQLite 3.54+ WAL connection manager, 12 `STRICT` tables DDL, FastCDC Rolling Gear Hash chunker + `zstd`, and Copy-on-Write VFS manifest engine.
3. **`agfs.engine` (Layer 2):** `SWE-Search` MCTS (`select_uct`, recursive CTE `backprop`), `SentinelAgent` `<0.1 ms` `StateCycleTrap`, `SWE-Pruner` Virtual Inode Observation Masking (`/.proc/obs/*`), `PASTE` speculative prefetcher, `AIOS` Process Control Block, and `agfs doctor` (`INV-01`..`INV-12`).
4. **`agfs.memory` (Layer 3):** Bi-Temporal (`tx_commit_born` / `tx_commit_superseded`) + `HippoRAG 2` 3-Way Hybrid Recall (`Dense Cosine + SQLite FTS5 BM25 + NumPy Personalized PageRank`) + `Sleep-Time Compute` compactor.
5. **`agfs.safety` (Layer 4):** `ePCA` commit-hook state invariants (`protected_globs`, `secret_scanner`, `no_test_skip`, `sql_assert`) and `SagaLLM` + `Atomix` transactional outbox.
6. **`agfs.adapters` & `agfs.interfaces` (Layer 5):** Unified `AgFSKernel` Python SDK, stdio MCP server (`agfs mcp`), Darwin `kqueue` CLI supervisor (`agfs wrap`), and CLI (`agfs`).

## 3. "Drop One File" Extension Points
- **Add an Agent Host Adapter:** Create 1 file in `src/agfs/adapters/`.
- **Add an `ePCA` State Invariant Rule:** Create 1 file in `src/agfs/safety/rules/`.
- **Add an Embedding Backend:** Create 1 file in `src/agfs/memory/embedders/`.
- **Add a `SagaLLM` External Connector:** Create 1 file in `src/agfs/safety/connectors/`.

## 4. Benchmark-Driven Survival
Every kernel feature must prove its worth in `make bench` (`Without AgFS` vs. `With AgFS`). Before opening a PR, run:

```bash
make check
make test
make doctor
make bench
```
