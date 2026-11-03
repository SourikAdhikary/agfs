"""Layer 0: macOS Tahoe (Darwin arm64) native APFS clonefile(2) & kqueue bindings.

Uses `/usr/lib/libSystem.B.dylib` via stdlib `ctypes` on macOS for zero-copy APFS
Copy-on-Write file cloning (`clonefile(2)`) and `select.kqueue` for instant kernel
file watching, with automatic POSIX fallback on Linux or cross-device mounts.
"""

from __future__ import annotations

import ctypes
import os
import select
import shutil
import sys
from pathlib import Path

IS_DARWIN = sys.platform == "darwin"

_clonefile_fn = None
if IS_DARWIN:
    try:
        _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        _fn = getattr(_libc, "clonefile", None)
        if _fn is not None:
            _fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
            _fn.restype = ctypes.c_int
            _clonefile_fn = _fn
    except OSError:
        _clonefile_fn = None


def supports_apfs_clonefile() -> bool:
    """Return True if macOS Darwin clonefile(2) syscall is available."""
    return _clonefile_fn is not None


def clone_or_copy_file(src: Path, dst: Path) -> bool:
    """Clone `src` to `dst` using macOS APFS CoW `clonefile(2)` in ~30us if supported.

    Returns True if APFS zero-copy `clonefile(2)` succeeded, or False if it fell back
    to `shutil.copy2`.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if _clonefile_fn is not None:
        res = _clonefile_fn(os.fsencode(src), os.fsencode(dst), 0)
        if res == 0:
            return True
    shutil.copy2(src, dst)
    return False


def clone_tree_cow(src_dir: Path, dst_dir: Path, ignore_names: set[str] | None = None) -> int:
    """Clone an entire workspace directory tree using APFS `clonefile(2)` per file."""
    ignored = ignore_names or set()
    cloned_count = 0
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ignored]
        rel_root = Path(root).relative_to(src_dir)
        target_root = dst_dir / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for fname in files:
            if fname in ignored:
                continue
            s_path = Path(root) / fname
            d_path = target_root / fname
            if clone_or_copy_file(s_path, d_path):
                cloned_count += 1
    return cloned_count


class DarwinKqueueWatcher:
    """Lightweight Darwin `select.kqueue` file watcher for `agfs wrap` supervisor."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths
        self._kq = select.kqueue() if hasattr(select, "kqueue") else None
        self._fds: dict[int, Path] = {}
        if self._kq is not None:
            for p in paths:
                if p.exists():
                    fd = os.open(str(p), os.O_EVTONLY if hasattr(os, "O_EVTONLY") else os.O_RDONLY)
                    self._fds[fd] = p
                    ev = select.kevent(
                        fd,
                        filter=select.KQ_FILTER_VNODE,
                        flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                        fflags=(
                            select.KQ_NOTE_WRITE
                            | select.KQ_NOTE_EXTEND
                            | select.KQ_NOTE_DELETE
                            | select.KQ_NOTE_RENAME
                            | select.KQ_NOTE_ATTRIB
                        ),
                    )
                    self._kq.control([ev], 0, 0)

    def poll_modified(self, timeout_sec: float = 0.0) -> list[Path]:
        """Return list of watched paths that experienced VNODE mutations."""
        if self._kq is None or not self._fds:
            return []
        events = self._kq.control(None, len(self._fds), timeout_sec)
        return [self._fds[ev.ident] for ev in events if ev.ident in self._fds]

    def close(self) -> None:
        for fd in list(self._fds.keys()):
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        if self._kq is not None:
            try:
                self._kq.close()
            except OSError:
                pass

    def __enter__(self) -> DarwinKqueueWatcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
