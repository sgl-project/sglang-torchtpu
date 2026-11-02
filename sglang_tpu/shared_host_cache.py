"""Process-shared immutable CPU buffers with a byte-bounded LRU."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import torch

_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )
}


@dataclass
class HostCacheLease:
    buffers: list[torch.Tensor]
    spans: tuple[tuple[int, int, int], ...] | None
    nbytes: int
    fd: int

    def close(self) -> None:
        if self.fd >= 0:
            self.buffers = []
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SharedHostCache:
    """All clients of a directory must agree on its total byte budget.

    A lease protects an entry until its final asynchronous reader completes.
    Namespace identifies checkpoint, preprocessing and encoder implementation.
    """

    def __init__(self, directory: str | Path, cap_bytes: int, namespace: str):
        if cap_bytes <= 0 or not namespace:
            raise ValueError("A shared cache requires a positive budget and namespace")
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.cap = cap_bytes
        self.namespace = namespace
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS config (version INTEGER, cap INTEGER)"
            )
            config = db.execute("SELECT version, cap FROM config").fetchone()
            if config is None:
                db.execute("INSERT INTO config VALUES (1, ?)", (self.cap,))
            elif config != (1, self.cap):
                raise ValueError(f"Shared cache configuration differs: {config}")
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "key TEXT PRIMARY KEY, nbytes INTEGER NOT NULL, "
                "ready INTEGER NOT NULL, metadata TEXT NOT NULL, accessed INTEGER NOT NULL)"
            )
            known = {row[0] for row in db.execute("SELECT key FROM entries")}
            # A writer can die after creating a file but before committing its reservation.
            for path in self.directory.glob("*.bin"):
                if path.stem not in known:
                    fd = self._try_exclusive(path)
                    if fd is not None:
                        try:
                            path.unlink()
                        finally:
                            os.close(fd)

    @contextlib.contextmanager
    def _transaction(self):
        # Each operation opens a distinct description: flock must also exclude threads.
        with (self.directory / "index.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = sqlite3.connect(self.directory / "index.sqlite3", isolation_level=None)
            try:
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.execute("COMMIT")
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
            finally:
                db.close()

    def _key(self, key) -> str:
        if key[0] is None:
            raise ValueError("A shared embedding needs a content identity")
        payload = json.dumps([self.namespace, key], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.bin"

    @staticmethod
    def _try_exclusive(path: Path) -> int | None:
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        return fd

    def _remove(self, db, key: str, fd: int) -> None:
        try:
            self._path(key).unlink()
            db.execute("DELETE FROM entries WHERE key = ?", (key,))
        finally:
            os.close(fd)

    def get(self, key) -> HostCacheLease | None:
        digest = self._key(key)
        fd = None
        try:
            with self._transaction() as db:
                row = db.execute(
                    "SELECT nbytes, ready, metadata FROM entries WHERE key = ?",
                    (digest,),
                ).fetchone()
                if row is None:
                    return None
                nbytes, ready, metadata = row
                if not ready:
                    stale_fd = self._try_exclusive(self._path(digest))
                    if stale_fd is not None:
                        self._remove(db, digest, stale_fd)
                    return None
                fd = os.open(self._path(digest), os.O_RDONLY)
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH)
                    if os.fstat(fd).st_size != nbytes:
                        raise RuntimeError(
                            "Shared embedding file has an invalid length"
                        )
                    spec = json.loads(metadata)
                    storage = torch.from_file(
                        str(self._path(digest)),
                        shared=False,
                        size=nbytes,
                        dtype=torch.uint8,
                    )
                    buffers = []
                    for desc in spec["buffers"]:
                        offset, size = desc["offset"], desc["nbytes"]
                        dtype = _DTYPES[desc["dtype"]]
                        shape = desc["shape"]
                        if (
                            offset < 0
                            or size < 0
                            or offset + size > nbytes
                            or any(d < 0 for d in shape)
                            or size
                            != math.prod(shape)
                            * torch.empty((), dtype=dtype).element_size()
                        ):
                            raise RuntimeError(
                                "Shared embedding metadata has an invalid shape"
                            )
                        buffers.append(
                            storage[offset : offset + size].view(dtype).reshape(shape)
                        )
                    spans = spec["spans"]
                    if spans is not None:
                        spans = tuple(tuple(span) for span in spans)
                        for i, start, length in spans:
                            if not (
                                0 <= i < len(buffers)
                                and start >= 0
                                and length >= 0
                                and start + length <= buffers[i].shape[0]
                            ):
                                raise RuntimeError(
                                    "Shared embedding metadata has an invalid span"
                                )
                    db.execute(
                        "UPDATE entries SET accessed = ? WHERE key = ?",
                        (time.monotonic_ns(), digest),
                    )
                    lease = HostCacheLease(buffers, spans, nbytes, fd)
                except BaseException:
                    os.close(fd)
                    fd = None
                    raise
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        return lease

    def put(self, key, buffers: list[torch.Tensor], spans=None) -> str:
        """Return inserted, present, pending, too_big, or busy (all space leased)."""
        digest = self._key(key)
        descriptors = []
        nbytes = 0
        for buffer in buffers:
            if buffer.device.type != "cpu" or str(buffer.dtype) not in _DTYPES:
                raise ValueError(
                    "Shared entries require CPU buffers with supported dtypes"
                )
            offset = (nbytes + 7) // 8 * 8
            size = buffer.numel() * buffer.element_size()
            descriptors.append(
                dict(
                    offset=offset,
                    nbytes=size,
                    dtype=str(buffer.dtype),
                    shape=list(buffer.shape),
                )
            )
            nbytes = offset + size
        if nbytes <= 0:
            raise ValueError("A shared entry needs nonempty physical buffers")
        if nbytes > self.cap:
            return "too_big"
        metadata = json.dumps(dict(buffers=descriptors, spans=spans))
        fd = None
        try:
            with self._transaction() as db:
                old = db.execute(
                    "SELECT ready FROM entries WHERE key = ?", (digest,)
                ).fetchone()
                if old is not None:
                    if old[0]:
                        return "present"
                    old_fd = self._try_exclusive(self._path(digest))
                    if old_fd is None:
                        return "pending"
                    self._remove(db, digest, old_fd)
                used = db.execute(
                    "SELECT coalesce(sum(nbytes), 0) FROM entries"
                ).fetchone()[0]
                for other, size in db.execute(
                    "SELECT key, nbytes FROM entries ORDER BY accessed, key"
                ).fetchall():
                    if used + nbytes <= self.cap:
                        break
                    evict_fd = self._try_exclusive(self._path(other))
                    if evict_fd is not None:
                        self._remove(db, other, evict_fd)
                        used -= size
                if used + nbytes > self.cap:
                    return "busy"
                fd = os.open(
                    self._path(digest), os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
                )
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    os.ftruncate(fd, nbytes)
                    db.execute(
                        "INSERT INTO entries VALUES (?, ?, 0, ?, ?)",
                        (digest, nbytes, metadata, time.monotonic_ns()),
                    )
                except BaseException:
                    os.close(fd)
                    fd = None
                    self._path(digest).unlink()
                    raise
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        try:
            storage = torch.from_file(
                str(self._path(digest)), shared=True, size=nbytes, dtype=torch.uint8
            )
            for desc, buffer in zip(descriptors, buffers):
                raw = buffer.contiguous().view(torch.uint8).reshape(-1)
                offset, size = desc["offset"], desc["nbytes"]
                storage[offset : offset + size].copy_(raw)
            # Publish only after all CPU copies complete. Readers never see a partial entry.
            with self._transaction() as db:
                db.execute(
                    "UPDATE entries SET ready = 1, accessed = ? WHERE key = ?",
                    (time.monotonic_ns(), digest),
                )
        except BaseException:
            with self._transaction() as db:
                self._path(digest).unlink()
                db.execute("DELETE FROM entries WHERE key = ?", (digest,))
            raise
        finally:
            os.close(fd)
        return "inserted"

    def stats(self) -> dict[str, int]:
        with self._transaction() as db:
            entries, nbytes, pending = db.execute(
                "SELECT count(*), coalesce(sum(nbytes), 0), "
                "coalesce(sum(1 - ready), 0) FROM entries"
            ).fetchone()
        return dict(entries=entries, nbytes=nbytes, pending=pending)
