"""Layer 1: Copy-on-Write VFS Manifest Engine & macOS Tahoe APFS Delta Materializer."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agfs.core.config import KernelConfig
from agfs.core.darwin import clone_or_copy_file
from agfs.core.hashing import compute_vfs_root_hash
from agfs.core.types import InodeEntry
from agfs.storage.chunker import pack_chunks_for_file, verify_and_decompress_chunk
from agfs.storage.db import SQLiteWALEngine


class CoWVFS:
    """Content-addressed Copy-on-Write Virtual Filesystem stored in `.agdb`."""

    def __init__(self, db: SQLiteWALEngine, config: KernelConfig) -> None:
        self.db = db
        self.config = config
        # Hot in-memory cache of commit_id -> {path: InodeEntry} for <0.5ms branching
        self._manifest_cache: dict[int, dict[str, InodeEntry]] = {}
        # Optional hot chunk cache for instant materialization
        self._chunk_raw_cache: dict[str, bytes] = {}
        # Optional on-disk APFS CoW blob cache directory inside `.agfs/apfs_blobs/`
        self._apfs_blob_dir = self.db.db_path.parent / "apfs_blobs"

    def get_manifest(self, commit_id: int | None) -> dict[str, InodeEntry]:
        """Return active file manifest `{path: InodeEntry}` for `commit_id`."""
        if commit_id is None:
            return {}
        if commit_id in self._manifest_cache:
            return dict(self._manifest_cache[commit_id])

        rows = self.db.conn.execute(
            """
            SELECT commit_id, path, deleted, mode, size, mtime_ns, file_hash, chunks_json
            FROM vfs_inodes
            WHERE commit_id = ? AND deleted = 0
            """,
            (commit_id,),
        ).fetchall()
        manifest: dict[str, InodeEntry] = {}
        for r in rows:
            manifest[r["path"]] = InodeEntry(
                commit_id=r["commit_id"],
                path=r["path"],
                deleted=bool(r["deleted"]),
                mode=r["mode"],
                size=r["size"],
                mtime_ns=r["mtime_ns"],
                file_hash=r["file_hash"],
                chunks=json.loads(r["chunks_json"]),
            )
        self._manifest_cache[commit_id] = dict(manifest)
        return manifest

    def store_bytes_as_chunks(self, raw_bytes: bytes) -> tuple[str, list[str]]:
        """Chunk `raw_bytes` via FastCDC, store missing chunks in `vfs_chunks`, return `(file_hash, chunk_hashes)`."""
        file_hash, chunk_records = pack_chunks_for_file(
            raw_bytes,
            min_size=self.config.cdc_min_size,
            max_size=self.config.cdc_max_size,
            single_chunk_cutoff=self.config.cdc_single_chunk_cutoff,
        )
        chunk_hashes = [rec[0] for rec in chunk_records]
        self.db.conn.executemany(
            """
            INSERT OR IGNORE INTO vfs_chunks(chunk_hash, raw_size, zstd_bytes, created_at_ns)
            VALUES (?, ?, ?, ?)
            """,
            chunk_records,
        )
        return file_hash, chunk_hashes

    def reconstruct_file_bytes(self, chunk_hashes: list[str]) -> bytes:
        """Reconstruct and verify full file bytes from ordered `chunk_hashes`."""
        parts: list[bytes] = []
        for ch_hash in chunk_hashes:
            if ch_hash in self._chunk_raw_cache:
                parts.append(self._chunk_raw_cache[ch_hash])
                continue
            row = self.db.conn.execute(
                "SELECT zstd_bytes FROM vfs_chunks WHERE chunk_hash = ?",
                (ch_hash,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Missing chunk {ch_hash} in vfs_chunks")
            raw = verify_and_decompress_chunk(ch_hash, row["zstd_bytes"])
            if len(self._chunk_raw_cache) < 4096:
                self._chunk_raw_cache[ch_hash] = raw
            parts.append(raw)
        return b"".join(parts)

    def read_file(self, commit_id: int, rel_path: str) -> bytes:
        """Read a file from `commit_id`'s VFS manifest."""
        norm_path = rel_path.lstrip("./")
        manifest = self.get_manifest(commit_id)
        if norm_path not in manifest:
            raise FileNotFoundError(f"'{norm_path}' not found in commit #{commit_id}")
        return self.reconstruct_file_bytes(manifest[norm_path].chunks)

    def scan_workspace_changes(
        self,
        parent_commit_id: int | None,
        workspace_root: Path | None = None,
        virtual_overlays: dict[str, bytes | None] | None = None,
    ) -> tuple[dict[str, InodeEntry], str, list[str]]:
        """Build a new commit's manifest by combining parent manifest + disk changes + virtual overlays.

        Uses `(mtime_ns, size)` fast-skip so unmodified files on disk are not re-read or re-hashed.
        Returns `(new_manifest_without_commit_id_bound, vfs_root_hash, changed_paths)`.
        """
        parent_manifest = self.get_manifest(parent_commit_id)
        new_manifest: dict[str, InodeEntry] = dict(parent_manifest)
        changed_paths: list[str] = []

        ws = workspace_root or self.config.workspace_root
        if ws is not None and ws.exists():
            seen_on_disk: set[str] = set()
            ignored = self.config.ignore_names
            for root, dirs, files in os.walk(ws):
                dirs[:] = [d for d in dirs if d not in ignored]
                for fname in files:
                    if fname in ignored or fname.endswith((".agdb", ".agdb-wal", ".agdb-shm")):
                        continue
                    full_path = Path(root) / fname
                    rel_path = full_path.relative_to(ws).as_posix()
                    seen_on_disk.add(rel_path)
                    try:
                        st = full_path.stat()
                    except OSError:
                        continue
                    prev = parent_manifest.get(rel_path)
                    if prev is not None and prev.mtime_ns == st.st_mtime_ns and prev.size == st.st_size:
                        continue
                    raw = full_path.read_bytes()
                    file_hash, chunk_hashes = self.store_bytes_as_chunks(raw)
                    if prev is not None and prev.file_hash == file_hash and prev.mode == st.st_mode:
                        # Update cached mtime_ns without marking content changed
                        new_manifest[rel_path] = InodeEntry(
                            commit_id=0,
                            path=rel_path,
                            deleted=False,
                            mode=st.st_mode,
                            size=st.st_size,
                            mtime_ns=st.st_mtime_ns,
                            file_hash=file_hash,
                            chunks=chunk_hashes,
                        )
                        continue
                    changed_paths.append(rel_path)
                    new_manifest[rel_path] = InodeEntry(
                        commit_id=0,
                        path=rel_path,
                        deleted=False,
                        mode=st.st_mode,
                        size=len(raw),
                        mtime_ns=st.st_mtime_ns,
                        file_hash=file_hash,
                        chunks=chunk_hashes,
                    )

            # Detect files deleted on disk
            for p_path in list(parent_manifest.keys()):
                if p_path not in seen_on_disk:
                    changed_paths.append(p_path)
                    new_manifest.pop(p_path, None)

        # Apply explicit in-memory virtual file writes/deletes
        if virtual_overlays:
            for v_path, content in virtual_overlays.items():
                norm_path = v_path.lstrip("./")
                changed_paths.append(norm_path)
                if content is None:
                    new_manifest.pop(norm_path, None)
                else:
                    file_hash, chunk_hashes = self.store_bytes_as_chunks(content)
                    new_manifest[norm_path] = InodeEntry(
                        commit_id=0,
                        path=norm_path,
                        deleted=False,
                        mode=33188,
                        size=len(content),
                        mtime_ns=0,
                        file_hash=file_hash,
                        chunks=chunk_hashes,
                    )

        vfs_root_hash = compute_vfs_root_hash(
            [(e.path, e.file_hash, e.mode) for e in new_manifest.values()]
        )
        return new_manifest, vfs_root_hash, sorted(set(changed_paths))

    def persist_manifest(self, commit_id: int, manifest: dict[str, InodeEntry]) -> None:
        """Persist `manifest` rows for `commit_id` in `vfs_inodes` and populate cache."""
        bound_manifest: dict[str, InodeEntry] = {}
        rows: list[tuple[int, str, int, int, int, int, str, str]] = []
        for path, entry in manifest.items():
            bound = InodeEntry(
                commit_id=commit_id,
                path=path,
                deleted=False,
                mode=entry.mode,
                size=entry.size,
                mtime_ns=entry.mtime_ns,
                file_hash=entry.file_hash,
                chunks=entry.chunks,
            )
            bound_manifest[path] = bound
            rows.append(
                (
                    commit_id,
                    path,
                    0,
                    entry.mode,
                    entry.size,
                    entry.mtime_ns,
                    entry.file_hash,
                    json.dumps(entry.chunks),
                )
            )
        if rows:
            self.db.conn.executemany(
                """
                INSERT INTO vfs_inodes(commit_id, path, deleted, mode, size, mtime_ns, file_hash, chunks_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        self._manifest_cache[commit_id] = bound_manifest

    def materialize_delta_to_disk(
        self,
        from_commit_id: int | None,
        to_commit_id: int,
        workspace_root: Path | None = None,
    ) -> list[str]:
        """Materialize only the delta files between `from_commit_id` and `to_commit_id` to disk.

        Uses macOS APFS `clonefile(2)` when a cached whole-file blob is present, or writes
        reconstructed FastCDC chunks directly. Returns list of modified/restored paths.
        """
        ws = workspace_root or self.config.workspace_root
        if ws is None:
            return []

        old_manifest = self.get_manifest(from_commit_id)
        new_manifest = self.get_manifest(to_commit_id)
        touched: list[str] = []

        # 1. Remove files present in old_manifest (or newly created untracked files) not in new_manifest
        for old_path in old_manifest:
            if old_path not in new_manifest:
                target_file = ws / old_path
                if target_file.exists():
                    target_file.unlink()
                touched.append(old_path)

        # 2. Write or restore files whose hash or existence differs on disk
        for path, entry in new_manifest.items():
            target_file = ws / path
            old_entry = old_manifest.get(path)
            need_write = False
            if old_entry is None or old_entry.file_hash != entry.file_hash or not target_file.exists():
                need_write = True
            else:
                try:
                    st = target_file.stat()
                    if st.st_size != entry.size or (entry.mtime_ns > 0 and st.st_mtime_ns != entry.mtime_ns):
                        need_write = True
                except OSError:
                    need_write = True

            if need_write:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                blob_cache_file = self._apfs_blob_dir / entry.file_hash[:2] / entry.file_hash
                if blob_cache_file.exists():
                    clone_or_copy_file(blob_cache_file, target_file)
                else:
                    raw = self.reconstruct_file_bytes(entry.chunks)
                    target_file.write_bytes(raw)
                try:
                    st = target_file.stat()
                    # Update cached mtime_ns so subsequent scan_workspace_changes fast-skips
                    self._manifest_cache[to_commit_id][path] = InodeEntry(
                        commit_id=to_commit_id,
                        path=path,
                        deleted=False,
                        mode=entry.mode,
                        size=entry.size,
                        mtime_ns=st.st_mtime_ns,
                        file_hash=entry.file_hash,
                        chunks=entry.chunks,
                    )
                except OSError:
                    pass
                touched.append(path)

        return sorted(touched)

    def diff_commits(self, commit_a: int, commit_b: int) -> dict[str, str]:
        """Return `{path: 'ADDED' | 'MODIFIED' | 'DELETED'}` between `commit_a` and `commit_b`."""
        man_a = self.get_manifest(commit_a)
        man_b = self.get_manifest(commit_b)
        diffs: dict[str, str] = {}
        for path, entry_b in man_b.items():
            if path not in man_a:
                diffs[path] = "ADDED"
            elif man_a[path].file_hash != entry_b.file_hash:
                diffs[path] = "MODIFIED"
        for path in man_a:
            if path not in man_b:
                diffs[path] = "DELETED"
        return diffs
