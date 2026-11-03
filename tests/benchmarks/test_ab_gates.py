"""Automated A/B Benchmark Verification Gate (`Without AgFS` vs. `With AgFS`).

Fails CI if any subsystem in AgFS fails to measurably outperform the baseline.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.runner import run_all_benchmarks


def test_all_ab_benchmarks_beat_baselines(tmp_path: Path) -> None:
    out_md = tmp_path / "BENCHMARKS.md"
    res = run_all_benchmarks(output_md=out_md)
    assert out_md.exists()

    b1 = res["systems_benchmarks"]["B1"]
    assert b1["speedup_fork_rollback_x"] >= 10.0
    assert b1["disk_reduction_x"] >= 10.0

    b2 = res["systems_benchmarks"]["B2"]
    assert b2["token_reduction_vs_raw_pct"] >= 50.0
    assert b2["with_agfs_verbatim_recall_pct"] == 100.0
    assert b2["with_agfs_kv_cache_hit_pct"] > b2["rolling_summary_kv_cache_hit_pct"]

    b3 = res["systems_benchmarks"]["B3"]
    assert b3["with_agfs_stale_poisoning_pct"] == 0.0
    assert b3["without_agfs_stale_poisoning_pct"] >= 50.0
    assert b3["with_agfs_hybrid_2hop_recall_pct"] >= 90.0
    assert b3["with_agfs_hybrid_2hop_recall_pct"] > b3["without_agfs_bm25_2hop_recall_pct"]

    b4 = res["systems_benchmarks"]["B4"]
    assert b4["speculative_cache_hit_pct"] >= 70.0
    assert b4["io_wait_reduction_pct"] >= 70.0

    b5 = res["systems_benchmarks"]["B5"]
    assert b5["with_agfs_aborted_branch_leaks"] == 0
    assert b5["with_agfs_duplicate_mutations"] == 0
    assert b5["with_agfs_orphaned_resources"] == 0

    e1 = res["agent_trap_evals"]["E1"]
    assert e1["with_agfs_wasted_loop_steps"] == 0
    assert e1["without_agfs_wasted_loop_steps"] >= 10
    assert e1["with_agfs_solved"] is True

    e2 = res["agent_trap_evals"]["E2"]
    assert e2["with_agfs_solved"] is True
    assert e2["with_agfs_corrupted_files"] == 0

    e3 = res["agent_trap_evals"]["E3"]
    assert e3["with_agfs_tamper_rate_pct"] == 0.0
    assert e3["protected_test_file_intact"] is True

    e4 = res["agent_trap_evals"]["E4"]
    assert e4["with_agfs_repeated_failed_mistake"] is False
    assert e4["branch_local_memory_leaked"] is False
    assert e4["global_lesson_recalled_on_sibling"] is True
