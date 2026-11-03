"""Unit tests for macOS Tahoe Darwin clonefile(2), kqueue, FastCDC chunker & zstd."""

from __future__ import annotations

from pathlib import Path

import pytest

from agfs.core.darwin import (
    DarwinKqueueWatcher,
    clone_or_copy_file,
    supports_apfs_clonefile,
)
from agfs.core.errors import CorruptChunkError
from agfs.storage.chunker import (
    fastcdc_split,
    pack_chunks_for_file,
    verify_and_decompress_chunk,
)


def test_darwin_apfs_clonefile_and_kqueue(tmp_path: Path) -> None:
    src = tmp_path / "source.txt"
    dst = tmp_path / "cloned.txt"
    src.write_text("macOS Tahoe APFS Copy-on-Write test\n" * 100)

    used_clone = clone_or_copy_file(src, dst)
    assert dst.read_text() == src.read_text()
    if supports_apfs_clonefile():
        assert used_clone is True

    # Modify cloned file and verify source is untouched (true CoW isolation)
    with DarwinKqueueWatcher([dst]) as watcher:
        dst.write_text("mutated on clone\n")
        modified = watcher.poll_modified(timeout_sec=0.05)
        assert dst in modified
    assert "Copy-on-Write" in src.read_text()


def test_fastcdc_subfile_chunk_deduplication() -> None:
    # Build a 64 KB deterministic buffer
    blocks = [f"Line {i:05d}: deterministic content block for FastCDC gear hash.\n".encode() for i in range(1200)]
    original = b"".join(blocks)
    assert len(original) > 60_000

    chunks_orig = fastcdc_split(original, min_size=4096, max_size=16384, single_chunk_cutoff=16384)
    assert len(chunks_orig) >= 4
    assert b"".join(chunks_orig) == original

    # Mutate 1 line near the middle and verify most chunks remain byte-for-byte identical
    blocks_edited = list(blocks)
    blocks_edited[600] = b"Line 00600: SINGLE EDITED LINE IN MIDDLE OF LARGE FILE!\n"
    edited = b"".join(blocks_edited)

    _, records_orig = pack_chunks_for_file(original)
    _, records_edit = pack_chunks_for_file(edited)

    hashes_orig = {r[0] for r in records_orig}
    hashes_edit = {r[0] for r in records_edit}
    shared_hashes = hashes_orig & hashes_edit
    # At least 50% of chunks must be reused across the single-line edit
    assert len(shared_hashes) >= len(hashes_orig) // 2


def test_zstd_verification_catches_tampering() -> None:
    _, records = pack_chunks_for_file(b"critical kernel state payload" * 50)
    ch_hash, _, zstd_blob, _ = records[0]
    raw = verify_and_decompress_chunk(ch_hash, zstd_blob)
    assert raw == b"critical kernel state payload" * 50

    with pytest.raises(CorruptChunkError):
        verify_and_decompress_chunk("0" * 64, zstd_blob)
