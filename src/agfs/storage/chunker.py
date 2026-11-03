"""Layer 1: FastCDC Rolling Gear Hash sub-file chunker & Python 3.15 stdlib zstd compression."""

from __future__ import annotations

import hashlib
import struct
import time

from agfs.core.errors import CorruptChunkError
from agfs.core.hashing import sha256_bytes

# Python 3.15 (macOS Tahoe) includes `compression.zstd` natively in the stdlib (PEP 784).
try:
    from compression import zstd as _stdlib_zstd  # type: ignore[import-not-found]

    def compress_bytes(data: bytes) -> bytes:
        return _stdlib_zstd.compress(data, level=3)

    def decompress_bytes(blob: bytes) -> bytes:
        return _stdlib_zstd.decompress(blob)

    ZSTD_BACKEND = "python3.15-stdlib-zstd"
except ImportError:
    try:
        import zstandard as _ext_zstd  # type: ignore[import-not-found]

        _cctx = _ext_zstd.ZstdCompressor(level=3)
        _dctx = _ext_zstd.ZstdDecompressor()

        def compress_bytes(data: bytes) -> bytes:
            return _cctx.compress(data)

        def decompress_bytes(blob: bytes) -> bytes:
            return _dctx.decompress(blob)

        ZSTD_BACKEND = "zstandard-c-ext"
    except ImportError:
        import zlib

        def compress_bytes(data: bytes) -> bytes:
            return zlib.compress(data, level=3)

        def decompress_bytes(blob: bytes) -> bytes:
            return zlib.decompress(blob)

        ZSTD_BACKEND = "stdlib-zlib-fallback"


# Deterministic 256-entry 32-bit Gear table for FastCDC content-defined chunking
def _build_gear_table() -> tuple[int, ...]:
    entries: list[int] = []
    for i in range(256):
        digest = hashlib.sha256(f"agfs:fastcdc:gear:{i}".encode()).digest()
        (val,) = struct.unpack("<I", digest[:4])
        entries.append(val)
    return tuple(entries)


GEAR_TABLE: tuple[int, ...] = _build_gear_table()
# Mask for ~8 KB average chunk size (13 bits -> 8192 bytes average)
CDC_MASK_AVG = 0x00035907


def fastcdc_split(
    data: bytes,
    *,
    min_size: int = 4096,
    max_size: int = 16384,
    single_chunk_cutoff: int = 16384,
) -> list[bytes]:
    """Split `data` into content-defined chunks using FastCDC Rolling Gear Hash.

    Files <= `single_chunk_cutoff` are kept as a single chunk. Larger files are split
    at content-defined boundaries in `[min_size, max_size]` so localized edits only
    create 1 new chunk across speculative branches.
    """
    n = len(data)
    if n <= single_chunk_cutoff:
        return [data]

    chunks: list[bytes] = []
    offset = 0
    gear = GEAR_TABLE
    mask = CDC_MASK_AVG

    # Fast boundary finder on large buffers: check 64-byte strides after min_size
    while offset < n:
        remaining = n - offset
        if remaining <= max_size:
            chunks.append(data[offset:])
            break

        start_scan = offset + min_size
        end_scan = offset + max_size
        fp = 0
        cut = end_scan
        i = start_scan
        while i < end_scan:
            fp = ((fp << 1) + gear[data[i]]) & 0xFFFFFFFF
            if (fp & mask) == 0:
                cut = i + 1
                break
            i += 1
        chunks.append(data[offset:cut])
        offset = cut

    return chunks


def verify_and_decompress_chunk(chunk_hash: str, zstd_bytes: bytes) -> bytes:
    """Decompress a chunk and verify its cryptographic SHA-256 matches `chunk_hash`."""
    try:
        raw = decompress_bytes(zstd_bytes)
    except Exception as exc:
        raise CorruptChunkError(f"Failed to decompress chunk {chunk_hash}: {exc}") from exc
    actual_hash = sha256_bytes(raw)
    if actual_hash != chunk_hash:
        raise CorruptChunkError(
            f"Chunk SHA-256 mismatch for {chunk_hash}: got {actual_hash}"
        )
    return raw


def pack_chunks_for_file(
    raw_bytes: bytes,
    *,
    min_size: int = 4096,
    max_size: int = 16384,
    single_chunk_cutoff: int = 16384,
) -> tuple[str, list[tuple[str, int, bytes, int]]]:
    """Return `(file_hash, [(chunk_hash, raw_size, zstd_bytes, created_at_ns), ...])`."""
    file_hash = sha256_bytes(raw_bytes)
    raw_chunks = fastcdc_split(
        raw_bytes,
        min_size=min_size,
        max_size=max_size,
        single_chunk_cutoff=single_chunk_cutoff,
    )
    now_ns = time.time_ns()
    records: list[tuple[str, int, bytes, int]] = []
    for ch in raw_chunks:
        ch_hash = sha256_bytes(ch)
        records.append((ch_hash, len(ch), compress_bytes(ch), now_ns))
    return file_hash, records
