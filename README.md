# AgFS (`SQLite-Agent-OS`)

**The featherweight, single-file SQLite Operating System & Transactional Kernel for AI Agents.**  
*Native macOS Tahoe (`APFS clonefile` + `kqueue`) · `<1.5 ms` Speculative Branching · MCTS State-Tree · Bi-Temporal + HippoRAG 2 Memory · Virtual Inode Context Masking · ePCA Commit Invariants · SagaLLM Outbox.*

---

## Why AgFS?

Terminal agents (**OpenCode**, **Nous Hermes**, **Claude Code**, **Aider**, **Cursor**) and custom SWE loops operate directly on your live working tree with fragile scratchpads:
- Exploring 3 candidate architectures requires slow `git worktree` copies (`~300 ms`) or heavy Docker containers (`~3 s`).
- ~84% of prompt tokens are verbose tool logs (`pytest`, file dumps), and rolling LLM summarization breaks Prefix KV Caching.
- Agents get trapped in infinite **edit $\rightarrow$ test fail $\rightarrow$ revert** oscillation loops or silently modify `tests/**` to pass.

**AgFS (`SQLite-Agent-OS`)** replaces ad-hoc agent scratchpads, external vector/graph DBs, and heavyweight sandboxes with **one portable `.agdb` SQLite 3 WAL file**:

1. **`<1.5 ms` Copy-on-Write VFS & FastCDC Deduplication (`DeltaBox`, 2026):** Content-defined chunking (`4 KB–16 KB` rolling Gear hash + Python 3.15 `compression.zstd`) and native macOS Tahoe APFS `clonefile(2)` zero-copy materialization.
2. **First-Class MCTS State-Tree (`SWE-Search`, ICLR 2025):** Built-in UCT branch selection (`select_uct`) and `<0.3 ms` recursive CTE reward backpropagation (`backprop`).
3. **`<0.1 ms` Merkle State-Cycle Trap (`SentinelAgent`, 2025):** Hashes `(vfs_root_hash, stderr_hash)` on every commit to trap edit-test-revert loops deterministically.
4. **Bi-Temporal 3-Way Hybrid Memory (`Graphiti` + `HippoRAG 2`, 2025):** Tracks `(tx_commit_born, tx_commit_superseded)` + `(valid_from_ts, valid_until_ts)` and fuses **Dense Cosine + SQLite FTS5 BM25 + 10-step NumPy Personalized PageRank (PPR)**.
5. **Virtual Inode Observation Masking (`SWE-Pruner`, 2026) & `PASTE` Speculative Reads:** Offloads aged tool outputs >250 tokens into `/.proc/obs/<id>.txt` pointers (cutting billed tokens by ~60% while keeping Prefix KV Cache hits >85%) and prefetches high-probability follow-up reads.
6. **`ePCA` Commit Invariants & `SagaLLM` + `Atomix` Outbox:** Kernel-enforced commit hooks prevent test tampering (`tests/**`, `@pytest.mark.skip`) and secret leaks, while staging or compensating external side effects.

---

## Quickstart (macOS Tahoe / Linux)

```bash
# 1. Initialize an AgFS kernel in any repository
agfs init --db .agfs/state.agdb --workspace . --protect "tests/**"

# 2. Verify all 12 formal kernel invariants (INV-01 .. INV-12)
agfs doctor --db .agfs/state.agdb

# 3. Run the 9-part Reproducible A/B Benchmark Suite (Without AgFS vs. With AgFS)
agfs bench --out BENCHMARKS.md

# 4. Plug into OpenCode, Hermes, Claude Code, or Cursor via MCP
agfs mcp --db .agfs/state.agdb --workspace .

# 5. Or wrap any existing CLI agent with Darwin kqueue guardrails
agfs wrap --protect "tests/**" --max-budget 2.00 -- opencode
```

---

## Documentation & Research Foundations

- **[Foundational Research Papers (2025–2026)](docs/papers.md)** — Direct arXiv/OpenReview links for all 12 papers implemented in `AgFS`.
- **[Contributing & 6-Layer Architecture](CONTRIBUTING.md)** — How to add an Agent Adapter, Invariant Rule, Embedder, or Saga Connector in 1 file.
- **[Reproducible A/B Benchmarks](BENCHMARKS.md)** — Side-by-side latency, token savings, recall, and reliability benchmarks (`Without AgFS` vs. `With AgFS`).
