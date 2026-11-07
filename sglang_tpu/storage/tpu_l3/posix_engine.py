"""PosixFileEngine — whole-page L3 storage on a local POSIX filesystem.

One file per key under ``root_dir/<kk[:2]>/<kk[2:4]>/<urlsafe-key>.bin``
where ``kk`` is the URL-quoted key. Writes go to a unique tmp file and are
committed with :func:`os.replace`, so readers never observe a torn page.
The in-memory index ``{key: (relpath, size, last_access_ns)}`` is guarded by
a single lock that is only ever held for index mutation, never across IO;
the index is rebuilt by scanning ``root_dir`` on init, so the tier survives
process restarts.
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)

_pwritev = getattr(os, "pwritev", None)
_preadv = getattr(os, "preadv", None)

# posix_fadvise(POSIX_FADV_DONTNEED) — Linux only, loaded lazily best-effort.
_FADV_DONTNEED = 4
_posix_fadvise = None
if sys.platform.startswith("linux"):
    try:
        import ctypes

        try:
            _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError:
            _libc = ctypes.CDLL(None, use_errno=True)
        _posix_fadvise = _libc.posix_fadvise
        _posix_fadvise.argtypes = [
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_int,
        ]
        _posix_fadvise.restype = ctypes.c_int
    except Exception:  # pragma: no cover - no libc/ctypes
        _posix_fadvise = None


def _write_all(fd: int, mv: memoryview) -> None:
    n = mv.nbytes
    off = 0
    while off < n:
        if _pwritev is not None:
            off += _pwritev(fd, [mv[off:]], off)
        else:  # pragma: no cover - every target platform has pwritev
            off += os.write(fd, mv[off:])


def _read_into(fd: int, mv: memoryview) -> bool:
    """Fill mv exactly; False on short read."""
    n = mv.nbytes
    if _preadv is None:  # pragma: no cover - every target platform has preadv
        data = os.pread(fd, n, 0)
        if len(data) != n:
            return False
        mv[:] = data
        return True
    off = 0
    while off < n:
        got = _preadv(fd, [mv[off:]], off)
        if got == 0:
            return False
        off += got
    return True


class PosixFileEngine:
    """PageEngine over one file per page on a POSIX filesystem."""

    def __init__(
        self,
        root_dir: str,
        num_threads: int = 4,
        max_bytes: int = 0,
        eviction_ratio: float = 0.1,
        fadvise_dontneed: bool = True,
        **_ignored,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.max_bytes = max_bytes
        self.eviction_ratio = eviction_ratio
        self.fadvise_dontneed = fadvise_dontneed and _posix_fadvise is not None
        os.makedirs(self.root_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._index: Dict[str, Tuple[str, int, int]] = {}
        self._total = 0
        self._counter = itertools.count()
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="posix-engine"
        )
        self._scan()

    # -- paths -------------------------------------------------------------

    def _relpath(self, key: str) -> str:
        enc = quote(key, safe="-._~")
        return os.path.join(enc[:2] or "__", enc[2:4] or "__", enc + ".bin")

    def _scan(self) -> None:
        now = time.monotonic_ns()
        for dirpath, _dirs, files in os.walk(self.root_dir):
            for name in files:
                if not name.endswith(".bin"):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                rel = os.path.relpath(full, self.root_dir)
                key = unquote(name[: -len(".bin")])
                self._index[key] = (rel, st.st_size, now)
                self._total += st.st_size

    # -- IO workers (run on executor threads, never under the lock) --------

    def _write_one(self, key: str, page: memoryview) -> bool:
        rel = self._relpath(key)
        full = os.path.join(self.root_dir, rel)
        tmp = "%s.tmp-%d-%d" % (full, os.getpid(), next(self._counter))
        try:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                _write_all(fd, page)
                if self.fadvise_dontneed:
                    try:
                        _posix_fadvise(fd, 0, 0, _FADV_DONTNEED)
                    except Exception:
                        pass
            finally:
                os.close(fd)
            os.replace(tmp, full)
        except OSError:
            logger.warning("PosixFileEngine: failed to write key %r", key, exc_info=True)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
        now = time.monotonic_ns()
        with self._lock:
            old = self._index.get(key)
            self._index[key] = (rel, page.nbytes, now)
            self._total += page.nbytes - (old[1] if old else 0)
        return True

    def _stat_fallback(self, key: str) -> Optional[Tuple[str, int, int]]:
        """Index files written by other processes sharing this root_dir.

        MLA models store one shared copy per page (all ranks use the same
        key suffix), so a rank that never wrote the page must still see it.
        """
        rel = self._relpath(key)
        try:
            st = os.stat(os.path.join(self.root_dir, rel))
        except OSError:
            return None
        entry = (rel, st.st_size, time.monotonic_ns())
        with self._lock:
            if key not in self._index:
                self._index[key] = entry
                self._total += st.st_size
        return entry

    def _read_one(self, key: str, dst: memoryview) -> bool:
        with self._lock:
            entry = self._index.get(key)
        if entry is None:
            entry = self._stat_fallback(key)
            if entry is None:
                return False
        # A file from an incompatible layout (page bytes differ) is a miss,
        # never a partial read into live KV.
        if entry[1] != dst.nbytes:
            return False
        rel = entry[0]
        try:
            fd = os.open(os.path.join(self.root_dir, rel), os.O_RDONLY)
            try:
                ok = _read_into(fd, dst)
            finally:
                os.close(fd)
        except OSError:
            return False
        if ok:
            now = time.monotonic_ns()
            with self._lock:
                cur = self._index.get(key)
                if cur is not None and cur[0] == rel:
                    self._index[key] = (cur[0], cur[1], now)
        return ok

    # -- eviction ----------------------------------------------------------

    def _evict_if_needed(self) -> None:
        if self.max_bytes <= 0:
            return
        victims: List[str] = []
        with self._lock:
            if self._total <= self.max_bytes:
                return
            target = int((1.0 - self.eviction_ratio) * self.max_bytes)
            for key, (rel, size, _ns) in sorted(
                self._index.items(), key=lambda kv: kv[1][2]
            ):
                if self._total <= target:
                    break
                del self._index[key]
                self._total -= size
                victims.append(rel)
        for rel in victims:
            try:
                os.unlink(os.path.join(self.root_dir, rel))
            except OSError:
                pass

    # -- PageEngine API ----------------------------------------------------

    def set_pages(self, keys: List[str], pages: List[memoryview]) -> List[bool]:
        results = list(self._executor.map(self._write_one, keys, pages))
        self._evict_if_needed()
        return results

    def get_pages(self, keys: List[str], dsts: List[memoryview]) -> List[bool]:
        return list(self._executor.map(self._read_one, keys, dsts))

    def exists(self, keys: List[str]) -> int:
        n = 0
        for key in keys:
            with self._lock:
                hit = key in self._index
            if not hit:
                hit = self._stat_fallback(key) is not None
            if not hit:
                break
            n += 1
        return n

    def clear(self) -> bool:
        """Delete every file this engine owns and reset the index."""
        with self._lock:
            rels = [rel for rel, _s, _n in self._index.values()]
            self._index.clear()
            self._total = 0
        ok = True
        for rel in rels:
            try:
                os.unlink(os.path.join(self.root_dir, rel))
            except OSError:
                ok = False
        return ok

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        with self._lock:
            entries = len(self._index)
            total = self._total
        cap = "unbounded" if self.max_bytes <= 0 else str(self.max_bytes)
        return (
            f"PosixFileEngine(root={self.root_dir!r}, "
            f"threads={self._executor._max_workers}, max_bytes={cap}, "
            f"entries={entries}, bytes={total})"
        )
