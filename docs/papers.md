# Foundational Research Papers in AgFS (2025–2026)

Every subsystem in **AgFS (`SQLite-Agent-OS`)** implements a concrete algorithm from peer-reviewed 2025–2026 systems and AI agent research inside a single SQLite `.agdb` file:

| Subsystem ID | Kernel Subsystem | Paper Title & Venue | Direct Paper Link |
| :--- | :--- | :--- | :--- |
| **`K1`** | **`<2 ms` CoW Sandbox Fork/Rollback & FastCDC** | *DeltaBox: Scaling Stateful AI Agents with Millisecond-Level Sandbox Checkpoint/Rollback* (Dong et al., 2026) | [arXiv:2605.22781](https://arxiv.org/abs/2605.22781) |
| **`K2`** | **MCTS State-Tree (`select_uct` & `backprop`)** | *SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement* (**ICLR 2025**) | [arXiv:2410.20285](https://arxiv.org/abs/2410.20285) |
| **`K3`** | **Bi-Temporal Memory (`tx_commit_born` / `superseded`)** | *Zep: A Temporal Knowledge Graph Architecture for Agent Memory (Graphiti)* (2025) | [arXiv:2501.13956](https://arxiv.org/abs/2501.13956) |
| **`K4`** | **3-Way Hybrid Recall (`Vec` + `BM25` + `PageRank`)** | *HippoRAG 2 — From RAG to Memory: Non-Parametric Continual Learning for Large Language Models* (2025) | [arXiv:2502.14802](https://arxiv.org/abs/2502.14802) |
| **`K5`** | **Virtual Inode Observation Masking (`/.proc/obs/*`)** | *SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents* (Wang et al., 2026) | [arXiv:2601.16746](https://arxiv.org/abs/2601.16746) |
| **`K6a`** | **Pattern-Aware Speculative Read Prefetching** | *Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution (`PASTE`)* (Sui et al., 2026) | [arXiv:2603.18897](https://arxiv.org/abs/2603.18897) |
| **`K6b`** | **Lossless Speculative Action Execution** | *Speculative Actions: A Lossless Framework for Faster AI Agents* (**ICLR 2026 Oral**) | [arXiv:2510.04371](https://arxiv.org/abs/2510.04371) |
| **`K7`** | **Idle-Time Memory Defragmentation** | *Sleep-Time Compute: Beyond Inference-Time Scaling* (Letta / UC Berkeley, 2025) | [arXiv:2504.13171](https://arxiv.org/abs/2504.13171) |
| **`K8`** | **Merkle State-Cycle Trap (`StateCycleTrap`)** | *SentinelAgent: Graph-Based Anomaly Detection in LLM-Based Multi-Agent Systems* (He et al., 2025) | [arXiv:2505.24201](https://arxiv.org/abs/2505.24201) |
| **`K10`** | **Compensating Saga Outbox (`saga_outbox`)** | *SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning* (**VLDB 2025**) | [arXiv:2503.11951](https://arxiv.org/abs/2503.11951) |
| **`K11`** | **Preemptible PCB & Capability Sandboxing** | *AIOS: LLM Agent Operating System* (**COLM 2025** — Mei et al.) | [arXiv:2403.16971](https://arxiv.org/abs/2403.16971) |
